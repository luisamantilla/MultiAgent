# mcp/mcp_client.py
"""
Modern MCP Client with enhanced capabilities for biology agents.
Supports async operations, connection pooling, and robust error handling.
"""

import asyncio
import json
import logging
import time
from typing import Dict, Any, List, Optional, Union, Callable
from dataclasses import dataclass, field
from contextlib import asynccontextmanager
import aiohttp
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class MCPRequest:
    """Enhanced MCP request with validation and metadata."""
    method: str
    params: Dict[str, Any] = field(default_factory=dict)
    id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.id is None:
            self.id = f"req_{int(time.time() * 1000)}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "jsonrpc": "2.0",
            "method": self.method,
            "params": self.params,
            "id": self.id
        }


@dataclass
class MCPResponse:
    """Enhanced MCP response with error handling."""
    id: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MCPResponse':
        return cls(
            id=data.get("id", ""),
            result=data.get("result"),
            error=data.get("error"),
            metadata=data.get("metadata", {})
        )

    @property
    def is_success(self) -> bool:
        return self.error is None

    @property
    def error_message(self) -> str:
        if self.error:
            return self.error.get("message", "Unknown error")
        return ""


class MCPClient:
    """
    Modern MCP Client with connection pooling and async support.
    Maintains backward compatibility while adding advanced features.
    """

    def __init__(self,
                 server_url: str = "http://localhost:8000",
                 timeout: int = 30,
                 max_connections: int = 10,
                 retry_attempts: int = 3):
        self.server_url = server_url
        self.timeout = timeout
        self.max_connections = max_connections
        self.retry_attempts = retry_attempts

        # Connection management
        self.session: Optional[aiohttp.ClientSession] = None
        self._connection_pool = None

        # Request tracking
        self.pending_requests: Dict[str, asyncio.Future] = {}
        self.request_history: List[Dict[str, Any]] = []

        # Statistics
        self.stats = {
            "requests_sent": 0,
            "responses_received": 0,
            "errors": 0,
            "avg_response_time": 0.0
        }

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.disconnect()

    async def connect(self):
        """Establish connection to MCP server."""
        if self.session is None:
            connector = aiohttp.TCPConnector(
                limit=self.max_connections,
                limit_per_host=self.max_connections,
                keepalive_timeout=30
            )

            self.session = aiohttp.ClientSession(
                connector=connector,
                timeout=aiohttp.ClientTimeout(total=self.timeout),
                headers={"Content-Type": "application/json"}
            )

            # Test connection
            try:
                await self._health_check()
                logger.info(f"Connected to MCP server at {self.server_url}")
            except Exception as e:
                logger.warning(f"Could not reach MCP server: {e}")

    async def disconnect(self):
        """Close connection to MCP server."""
        if self.session:
            await self.session.close()
            self.session = None
            logger.info("Disconnected from MCP server")

    async def _health_check(self) -> bool:
        """Check if server is responsive."""
        try:
            request = MCPRequest(method="health_check")
            response = await self._send_request(request)
            return response.is_success
        except:
            return False

    async def _send_request(self, request: MCPRequest) -> MCPResponse:
        """Send request with retry logic and error handling."""
        if not self.session:
            await self.connect()

        start_time = time.time()
        last_error = None

        for attempt in range(self.retry_attempts + 1):
            try:
                self.stats["requests_sent"] += 1

                async with self.session.post(
                        f"{self.server_url}/mcp",
                        json=request.to_dict()
                ) as resp:
                    response_data = await resp.json()
                    response = MCPResponse.from_dict(response_data)

                    # Update statistics
                    duration = time.time() - start_time
                    self._update_stats(duration, response.is_success)

                    # Track request
                    self.request_history.append({
                        "request_id": request.id,
                        "method": request.method,
                        "duration": duration,
                        "success": response.is_success,
                        "attempt": attempt + 1
                    })

                    if response.is_success:
                        self.stats["responses_received"] += 1
                        return response
                    else:
                        self.stats["errors"] += 1
                        if attempt == self.retry_attempts:
                            logger.error(f"Request failed after {attempt + 1} attempts: {response.error_message}")
                            return response

                        await asyncio.sleep(0.5 * (2 ** attempt))  # Exponential backoff

            except Exception as e:
                last_error = e
                if attempt == self.retry_attempts:
                    logger.error(f"Request failed permanently: {e}")
                    break
                await asyncio.sleep(0.5 * (2 ** attempt))

        # Return error response
        return MCPResponse(
            id=request.id,
            error={"message": str(last_error) if last_error else "Unknown error"}
        )

    def _update_stats(self, duration: float, success: bool):
        """Update client statistics."""
        total_requests = self.stats["requests_sent"]
        if total_requests > 0:
            current_avg = self.stats["avg_response_time"]
            self.stats["avg_response_time"] = (current_avg * (total_requests - 1) + duration) / total_requests

    # Core MCP methods
    async def run_agent(self,
                        agent_id: str,
                        prompt: str,
                        context: Optional[Dict[str, Any]] = None) -> MCPResponse:
        """Run a specific agent with enhanced error handling."""
        request = MCPRequest(
            method=f"agent_{agent_id}",
            params={
                "prompt": prompt,
                "context": context or {}
            },
            metadata={"agent_id": agent_id}
        )

        return await self._send_request(request)

    async def run_workflow(self,
                           user_request: str,
                           agents: Optional[List[str]] = None,
                           config: Optional[Dict[str, Any]] = None) -> MCPResponse:
        """Run a complete biology modeling workflow."""
        request = MCPRequest(
            method="run_biology_workflow",
            params={
                "user_request": user_request,
                "agents": agents or ["pi", "biologist", "parameter_extractor", "model_builder"],
                "config": config or {}
            },
            metadata={"workflow": True}
        )

        return await self._send_request(request)

    async def get_server_status(self) -> MCPResponse:
        """Get comprehensive server status."""
        request = MCPRequest(method="get_server_status")
        return await self._send_request(request)

    async def list_available_agents(self) -> MCPResponse:
        """List all available agents and their capabilities."""
        request = MCPRequest(method="list_agents")
        return await self._send_request(request)

    async def search_papers(self,
                            query: str,
                            top_k: int = 5) -> MCPResponse:
        """Search papers in the vector database."""
        request = MCPRequest(
            method="search_papers",
            params={
                "query": query,
                "top_k": top_k
            }
        )

        return await self._send_request(request)

    # Batch operations
    async def run_multiple_agents(self,
                                  requests: List[Dict[str, Any]]) -> List[MCPResponse]:
        """Run multiple agents concurrently."""
        tasks = []

        for req in requests:
            task = self.run_agent(
                agent_id=req.get("agent_id"),
                prompt=req.get("prompt"),
                context=req.get("context")
            )
            tasks.append(task)

        return await asyncio.gather(*tasks, return_exceptions=True)

    # Streaming support (for future real-time updates)
    async def stream_workflow(self,
                              user_request: str,
                              callback: Callable[[Dict[str, Any]], None]) -> None:
        """Stream workflow progress (placeholder for future implementation)."""
        # For now, run regular workflow and call callback with final result
        response = await self.run_workflow(user_request)
        if response.is_success and callback:
            callback(response.result)

    # Utility methods
    def get_statistics(self) -> Dict[str, Any]:
        """Get client performance statistics."""
        return {
            **self.stats,
            "success_rate": (
                (self.stats["responses_received"] / self.stats["requests_sent"])
                if self.stats["requests_sent"] > 0 else 0
            ),
            "total_requests": len(self.request_history)
        }

    def clear_history(self):
        """Clear request history."""
        self.request_history.clear()

    async def export_session_data(self, file_path: str):
        """Export session data for analysis."""
        session_data = {
            "statistics": self.get_statistics(),
            "request_history": self.request_history,
            "server_url": self.server_url,
            "export_time": time.time()
        }

        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(session_data, f, indent=2)


# Convenience functions for backward compatibility
async def create_mcp_client(server_url: str = "http://localhost:8000") -> MCPClient:
    """Factory function to create and connect MCP client."""
    client = MCPClient(server_url)
    await client.connect()
    return client


# Context manager for easy usage
@asynccontextmanager
async def mcp_session(server_url: str = "http://localhost:8000"):
    """Context manager for MCP client sessions."""
    client = MCPClient(server_url)
    try:
        await client.connect()
        yield client
    finally:
        await client.disconnect()