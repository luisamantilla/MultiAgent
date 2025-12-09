# mcp/mcp_server.py
"""
Enhanced MCP Server with modern architecture, real-time capabilities,
and comprehensive integration with the biology agents ecosystem.
"""

import asyncio
import json
import logging
import sys
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Callable
from dataclasses import dataclass, field
from contextlib import asynccontextmanager
import uvicorn
from fastapi import FastAPI, WebSocket, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
import aioredis
from prometheus_client import Counter, Histogram, Gauge, generate_latest

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)


# Pydantic models for API
class AgentRequest(BaseModel):
    agent_id: str
    prompt: str
    context: Optional[Dict[str, Any]] = None
    priority: int = Field(default=5, ge=1, le=10)
    timeout: Optional[int] = Field(default=300, le=600)


class WorkflowRequest(BaseModel):
    user_request: str
    template: Optional[str] = None
    agents: Optional[List[str]] = None
    context: Optional[Dict[str, Any]] = None
    priority: int = Field(default=5, ge=1, le=10)
    async_execution: bool = False


class AgentResponse(BaseModel):
    agent_id: str
    status: str
    result: Optional[Any] = None
    error: Optional[str] = None
    execution_time: float
    confidence: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class WorkflowResponse(BaseModel):
    workflow_id: str
    status: str
    results: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    execution_time: Optional[float] = None
    template_used: Optional[str] = None
    progress: Optional[float] = None


# Metrics
REQUEST_COUNTER = Counter('mcp_requests_total', 'Total requests', ['endpoint', 'status'])
REQUEST_DURATION = Histogram('mcp_request_duration_seconds', 'Request duration', ['endpoint'])
ACTIVE_CONNECTIONS = Gauge('mcp_active_connections', 'Active WebSocket connections')
QUEUE_SIZE = Gauge('mcp_queue_size', 'Request queue size')


@dataclass
class ConnectionManager:
    """Manages WebSocket connections for real-time updates."""
    active_connections: List[WebSocket] = field(default_factory=list)
    connection_metadata: Dict[WebSocket, Dict[str, Any]] = field(default_factory=dict)

    async def connect(self, websocket: WebSocket, metadata: Optional[Dict[str, Any]] = None):
        await websocket.accept()
        self.active_connections.append(websocket)
        self.connection_metadata[websocket] = metadata or {}
        ACTIVE_CONNECTIONS.set(len(self.active_connections))
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            self.connection_metadata.pop(websocket, None)
            ACTIVE_CONNECTIONS.set(len(self.active_connections))
            logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")

    async def broadcast(self, message: Dict[str, Any], filter_fn: Optional[Callable] = None):
        """Broadcast message to all or filtered connections."""
        if not self.active_connections:
            return

        connections_to_notify = self.active_connections
        if filter_fn:
            connections_to_notify = [
                conn for conn in self.active_connections
                if filter_fn(self.connection_metadata.get(conn, {}))
            ]

        disconnect_list = []
        for connection in connections_to_notify:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.warning(f"Failed to send message to WebSocket: {e}")
                disconnect_list.append(connection)

        # Clean up failed connections
        for connection in disconnect_list:
            self.disconnect(connection)


class EnhancedMCPServer:
    """
    Enhanced MCP Server with modern architecture and comprehensive features.
    """

    def __init__(self,
                 host: str = "127.0.0.1",
                 port: int = 8000,
                 papers_dir: str = "./papers",
                 output_dir: str = "./output",
                 redis_url: Optional[str] = None,
                 enable_auth: bool = False,
                 api_key: Optional[str] = None):

        self.host = host
        self.port = port
        self.papers_dir = papers_dir
        self.output_dir = output_dir
        self.redis_url = redis_url
        self.enable_auth = enable_auth
        self.api_key = api_key

        # Core components
        self.app = FastAPI(
            title="Biology Agents MCP Server",
            description="Enhanced MCP server for biological modeling agents",
            version="2.0.0",
            docs_url="/docs",
            redoc_url="/redoc"
        )

        self.connection_manager = ConnectionManager()
        self.orchestrator = None
        self.vector_db = None
        self.rag_pipeline = None
        self.redis_client = None

        # Request queue and processing
        self.request_queue: asyncio.Queue = asyncio.Queue()
        self.background_tasks: Set[asyncio.Task] = set()
        self.active_workflows: Dict[str, Dict[str, Any]] = {}

        # Initialize server
        self._setup_middleware()
        self._setup_routes()
        self._setup_background_tasks()

    def _setup_middleware(self):
        """Setup FastAPI middleware."""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Configure appropriately for production
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        @self.app.middleware("http")
        async def add_process_time_header(request, call_next):
            start_time = time.time()
            response = await call_next(request)
            process_time = time.time() - start_time
            response.headers["X-Process-Time"] = str(process_time)
            return response

    def _setup_routes(self):
        """Setup FastAPI routes."""

        # Health check
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint."""
            try:
                health_status = {
                    "status": "healthy",
                    "timestamp": time.time(),
                    "version": "2.0.0",
                    "components": {
                        "orchestrator": "healthy" if self.orchestrator else "not_initialized",
                        "vector_db": "healthy" if self.vector_db else "not_initialized",
                        "rag_pipeline": "healthy" if self.rag_pipeline else "not_initialized",
                        "redis": "healthy" if self.redis_client else "not_available",
                        "active_connections": len(self.connection_manager.active_connections),
                        "queue_size": self.request_queue.qsize(),
                        "active_workflows": len(self.active_workflows)
                    }
                }

                if self.orchestrator:
                    orchestrator_health = await self.orchestrator.health_check()
                    health_status["agents"] = orchestrator_health.get("agents", {})

                return health_status
            except Exception as e:
                logger.error(f"Health check failed: {e}")
                raise HTTPException(status_code=503, detail=f"Service unhealthy: {e}")

        # Agent execution
        @self.app.post("/agents/{agent_id}/run", response_model=AgentResponse)
        async def run_agent(
                agent_id: str,
                request: AgentRequest,
                background_tasks: BackgroundTasks,
                credentials: Optional[HTTPAuthorizationCredentials] = Depends(HTTPBearer(auto_error=False))
        ):
            """Run a specific agent."""
            if self.enable_auth and not self._verify_credentials(credentials):
                raise HTTPException(status_code=401, detail="Invalid authentication")

            start_time = time.time()

            try:
                REQUEST_COUNTER.labels(endpoint=f"agent_{agent_id}", status="started").inc()

                if not self.orchestrator:
                    raise HTTPException(status_code=503, detail="Orchestrator not initialized")

                # Add to request queue for processing
                task_id = f"agent_{agent_id}_{int(time.time() * 1000)}"

                # Execute agent
                if agent_id not in self.orchestrator.agents:
                    raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")

                agent = self.orchestrator.agents[agent_id]

                # Build conversation
                conversation = [agent.system_message()]
                if request.context:
                    context_msg = "Additional context:\n" + json.dumps(request.context, indent=2)
                    conversation.append({"role": "system", "content": context_msg})

                conversation.append({"role": "user", "content": request.prompt})

                # Execute with MCP compatibility
                if hasattr(agent, 'run_mcp'):
                    result = agent.run_mcp(conversation, request.context)
                    content = result.content if hasattr(result, 'content') else str(result)
                    confidence = getattr(result, 'confidence', None)
                    metadata = getattr(result, 'metadata', {})
                else:
                    content = agent.run(conversation)
                    confidence = None
                    metadata = {}

                execution_time = time.time() - start_time

                response = AgentResponse(
                    agent_id=agent_id,
                    status="success",
                    result=content,
                    execution_time=execution_time,
                    confidence=confidence,
                    metadata=metadata
                )

                REQUEST_COUNTER.labels(endpoint=f"agent_{agent_id}", status="success").inc()
                REQUEST_DURATION.labels(endpoint=f"agent_{agent_id}").observe(execution_time)

                # Broadcast to WebSocket connections
                await self.connection_manager.broadcast({
                    "type": "agent_completed",
                    "agent_id": agent_id,
                    "task_id": task_id,
                    "status": "success",
                    "execution_time": execution_time
                })

                return response

            except Exception as e:
                execution_time = time.time() - start_time
                REQUEST_COUNTER.labels(endpoint=f"agent_{agent_id}", status="error").inc()

                logger.error(f"Agent {agent_id} execution failed: {e}")
                return AgentResponse(
                    agent_id=agent_id,
                    status="error",
                    error=str(e),
                    execution_time=execution_time
                )

        # Workflow execution
        @self.app.post("/workflows/run", response_model=WorkflowResponse)
        async def run_workflow(
                request: WorkflowRequest,
                background_tasks: BackgroundTasks,
                credentials: Optional[HTTPAuthorizationCredentials] = Depends(HTTPBearer(auto_error=False))
        ):
            """Run a complete workflow."""
            if self.enable_auth and not self._verify_credentials(credentials):
                raise HTTPException(status_code=401, detail="Invalid authentication")

            start_time = time.time()
            workflow_id = f"workflow_{int(time.time() * 1000)}"

            try:
                REQUEST_COUNTER.labels(endpoint="workflow", status="started").inc()

                if not self.orchestrator:
                    raise HTTPException(status_code=503, detail="Orchestrator not initialized")

                if request.async_execution:
                    # Start workflow in background
                    task = asyncio.create_task(
                        self._run_workflow_async(workflow_id, request, start_time)
                    )
                    self.background_tasks.add(task)
                    task.add_done_callback(self.background_tasks.discard)

                    return WorkflowResponse(
                        workflow_id=workflow_id,
                        status="started",
                        progress=0.0
                    )
                else:
                    # Run workflow synchronously
                    result = await self.orchestrator.run_adaptive_workflow(
                        user_request=request.user_request,
                        template_name=request.template,
                        constraints={"agents": request.agents} if request.agents else None,
                        rag_context=request.context
                    )

                    execution_time = time.time() - start_time
                    REQUEST_COUNTER.labels(endpoint="workflow", status="success").inc()
                    REQUEST_DURATION.labels(endpoint="workflow").observe(execution_time)

                    return WorkflowResponse(
                        workflow_id=result.get("workflow_id", workflow_id),
                        status=result["status"],
                        results=result.get("results"),
                        error=result.get("error"),
                        execution_time=execution_time,
                        template_used=result.get("template_used"),
                        progress=100.0 if result["status"] == "success" else 0.0
                    )

            except Exception as e:
                execution_time = time.time() - start_time
                REQUEST_COUNTER.labels(endpoint="workflow", status="error").inc()

                logger.error(f"Workflow execution failed: {e}")
                return WorkflowResponse(
                    workflow_id=workflow_id,
                    status="error",
                    error=str(e),
                    execution_time=execution_time
                )

        # Workflow status
        @self.app.get("/workflows/{workflow_id}/status", response_model=WorkflowResponse)
        async def get_workflow_status(workflow_id: str):
            """Get workflow status."""
            if workflow_id in self.active_workflows:
                workflow = self.active_workflows[workflow_id]
                return WorkflowResponse(
                    workflow_id=workflow_id,
                    status=workflow["status"],
                    progress=workflow.get("progress", 0.0),
                    template_used=workflow.get("template")
                )

            if self.orchestrator:
                status = await self.orchestrator.get_workflow_status(workflow_id)
                if "error" not in status:
                    return WorkflowResponse(**status)

            raise HTTPException(status_code=404, detail=f"Workflow {workflow_id} not found")

        # Agent management
        @self.app.get("/agents")
        async def list_agents():
            """List available agents and their capabilities."""
            if not self.orchestrator:
                return {"error": "Orchestrator not initialized"}

            agents_info = {}
            for agent_id, agent in self.orchestrator.agents.items():
                capabilities = []
                if hasattr(agent, 'get_capabilities'):
                    capabilities = agent.get_capabilities()

                agents_info[agent_id] = {
                    "title": getattr(agent, 'title', agent_id),
                    "capabilities": capabilities,
                    "model": getattr(agent, 'model', 'unknown'),
                    "dependencies": getattr(agent, 'dependencies', [])
                }

            return {
                "agents": agents_info,
                "total_count": len(agents_info)
            }

        # Paper search and RAG
        @self.app.post("/papers/search")
        async def search_papers(
                query: str,
                top_k: int = 5,
                search_online: bool = False
        ):
            """Search papers in vector database and optionally online."""
            if not self.vector_db:
                raise HTTPException(status_code=503, detail="Vector database not initialized")

            try:
                if search_online and self.rag_pipeline:
                    # Use RAG pipeline for comprehensive search
                    results = await self.rag_pipeline.run_rag_pipeline(
                        query, self.vector_db, search_online=True, chunk_top_k=top_k
                    )
                    return results
                else:
                    # Search existing database
                    results = self.vector_db.search_content(query, top_k=top_k)
                    return {
                        "query": query,
                        "results": results,
                        "search_online": False
                    }

            except Exception as e:
                logger.error(f"Paper search failed: {e}")
                raise HTTPException(status_code=500, detail=f"Search failed: {e}")

        # Database management
        @self.app.get("/database/stats")
        async def get_database_stats():
            """Get vector database statistics."""
            if not self.vector_db:
                return {"error": "Vector database not initialized"}

            try:
                stats = self.vector_db.get_database_stats()
                return stats
            except Exception as e:
                return {"error": str(e)}

        @self.app.post("/database/add_papers")
        async def add_papers_to_database(
                papers_directory: str,
                background_tasks: BackgroundTasks
        ):
            """Add papers from directory to vector database."""
            if not self.vector_db:
                raise HTTPException(status_code=503, detail="Vector database not initialized")

            if not Path(papers_directory).exists():
                raise HTTPException(status_code=404, detail="Papers directory not found")

            # Process in background
            background_tasks.add_task(self._process_papers_background, papers_directory)

            return {
                "message": f"Started processing papers from {papers_directory}",
                "status": "processing"
            }

        # Templates and workflows
        @self.app.get("/templates")
        async def list_workflow_templates():
            """List available workflow templates."""
            if not self.orchestrator:
                return {"error": "Orchestrator not initialized"}

            templates = {}
            for template_name in self.orchestrator.get_available_templates():
                description = self.orchestrator.get_template_description(template_name)
                templates[template_name] = description

            return {"templates": templates}

        # Performance metrics
        @self.app.get("/metrics")
        async def get_performance_metrics():
            """Get performance metrics."""
            if not self.orchestrator:
                return {"error": "Orchestrator not initialized"}

            return self.orchestrator.get_performance_summary()

        # Prometheus metrics endpoint
        @self.app.get("/metrics/prometheus")
        async def prometheus_metrics():
            """Prometheus metrics endpoint."""
            from fastapi import Response
            return Response(generate_latest(), media_type="text/plain")

        # WebSocket endpoint
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket endpoint for real-time updates."""
            await self.connection_manager.connect(websocket)
            try:
                while True:
                    # Keep connection alive and handle client messages
                    data = await websocket.receive_text()
                    try:
                        message = json.loads(data)
                        await self._handle_websocket_message(websocket, message)
                    except json.JSONDecodeError:
                        await websocket.send_json({
                            "type": "error",
                            "message": "Invalid JSON format"
                        })
            except Exception as e:
                logger.info(f"WebSocket connection closed: {e}")
            finally:
                self.connection_manager.disconnect(websocket)

    def _verify_credentials(self, credentials: Optional[HTTPAuthorizationCredentials]) -> bool:
        """Verify API credentials if authentication is enabled."""
        if not self.enable_auth:
            return True

        if not credentials or not self.api_key:
            return False

        return credentials.credentials == self.api_key

    async def _handle_websocket_message(self, websocket: WebSocket, message: Dict[str, Any]):
        """Handle incoming WebSocket messages."""
        message_type = message.get("type")

        if message_type == "ping":
            await websocket.send_json({"type": "pong"})

        elif message_type == "subscribe_workflow":
            workflow_id = message.get("workflow_id")
            if workflow_id:
                # Update connection metadata for filtering
                self.connection_manager.connection_metadata[websocket] = {
                    "subscribed_workflows": [workflow_id]
                }
                await websocket.send_json({
                    "type": "subscription_confirmed",
                    "workflow_id": workflow_id
                })

        elif message_type == "get_status":
            status = await self.get_server_status()
            await websocket.send_json({
                "type": "status_update",
                "data": status
            })

    async def _run_workflow_async(self, workflow_id: str, request: WorkflowRequest, start_time: float):
        """Run workflow asynchronously and broadcast updates."""
        self.active_workflows[workflow_id] = {
            "status": "running",
            "progress": 0.0,
            "template": request.template,
            "start_time": start_time
        }

        try:
            # Broadcast start
            await self.connection_manager.broadcast({
                "type": "workflow_started",
                "workflow_id": workflow_id,
                "template": request.template
            })

            result = await self.orchestrator.run_adaptive_workflow(
                user_request=request.user_request,
                template_name=request.template,
                constraints={"agents": request.agents} if request.agents else None,
                rag_context=request.context
            )

            execution_time = time.time() - start_time

            self.active_workflows[workflow_id].update({
                "status": result["status"],
                "progress": 100.0,
                "results": result.get("results"),
                "execution_time": execution_time
            })

            # Broadcast completion
            await self.connection_manager.broadcast({
                "type": "workflow_completed",
                "workflow_id": workflow_id,
                "status": result["status"],
                "execution_time": execution_time,
                "results": result.get("results")
            })

            REQUEST_COUNTER.labels(endpoint="workflow", status="success").inc()

        except Exception as e:
            execution_time = time.time() - start_time

            self.active_workflows[workflow_id].update({
                "status": "error",
                "error": str(e),
                "execution_time": execution_time
            })

            await self.connection_manager.broadcast({
                "type": "workflow_failed",
                "workflow_id": workflow_id,
                "error": str(e),
                "execution_time": execution_time
            })

            REQUEST_COUNTER.labels(endpoint="workflow", status="error").inc()

        finally:
            # Clean up after some time
            await asyncio.sleep(300)  # Keep for 5 minutes
            self.active_workflows.pop(workflow_id, None)

    async def _process_papers_background(self, papers_directory: str):
        """Process papers in background."""
        try:
            self.vector_db.add_pdf_directory(papers_directory)

            await self.connection_manager.broadcast({
                "type": "papers_processed",
                "directory": papers_directory,
                "status": "completed"
            })

            logger.info(f"Completed processing papers from {papers_directory}")

        except Exception as e:
            logger.error(f"Failed to process papers: {e}")

            await self.connection_manager.broadcast({
                "type": "papers_processed",
                "directory": papers_directory,
                "status": "failed",
                "error": str(e)
            })

    def _setup_background_tasks(self):
        """Setup background task processing."""

        async def queue_processor():
            """Process requests from queue."""
            while True:
                try:
                    # Get request from queue
                    request_data = await self.request_queue.get()
                    QUEUE_SIZE.set(self.request_queue.qsize())

                    # Process request
                    await self._process_queued_request(request_data)

                    # Mark task as done
                    self.request_queue.task_done()

                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Queue processor error: {e}")

        # Add queue processor to background tasks
        task = asyncio.create_task(queue_processor())
        self.background_tasks.add(task)

    async def _process_queued_request(self, request_data: Dict[str, Any]):
        """Process a queued request."""
        # Implementation for processing queued requests
        # This can be extended for batch processing, rate limiting, etc.
        pass

    async def initialize_components(self):
        """Initialize all server components."""
        logger.info("Initializing MCP server components...")

        try:
            # Initialize Redis if URL provided
            if self.redis_url:
                self.redis_client = await aioredis.from_url(self.redis_url)
                logger.info("Redis client initialized")

            # Initialize vector database
            try:
                from memory.pdf_vector_db import PDFVectorDB
                if Path(self.papers_dir).exists():
                    self.vector_db = PDFVectorDB(papers_dir=self.papers_dir)
                    logger.info(f"Vector database initialized with papers from {self.papers_dir}")
                else:
                    logger.warning(f"Papers directory not found: {self.papers_dir}")
            except Exception as e:
                logger.warning(f"Could not initialize vector database: {e}")

            # Initialize RAG pipeline
            try:
                from tools.tools import RAGPipeline
                email = os.getenv("EMAIL")
                llamaparse_key = os.getenv("LLAMAPARSE_API_KEY")
                if email:
                    self.rag_pipeline = RAGPipeline(email=email, llamaparse_api_key=llamaparse_key)
                    logger.info("RAG pipeline initialized")
            except Exception as e:
                logger.warning(f"Could not initialize RAG pipeline: {e}")

            # Initialize orchestrator with agents
            try:
                from agents.specialized_agents import create_all_agents
                from tools.orchestration import AdaptiveOrchestrator

                agents = create_all_agents()

                # Attach vector database to agents if available
                if self.vector_db:
                    for agent in agents.values():
                        if hasattr(agent, 'attach_runtime'):
                            agent.attach_runtime(db=self.vector_db)

                self.orchestrator = AdaptiveOrchestrator(agents)
                logger.info(f"Orchestrator initialized with {len(agents)} agents")

            except Exception as e:
                logger.error(f"Failed to initialize orchestrator: {e}")
                raise

            logger.info("All components initialized successfully")

        except Exception as e:
            logger.error(f"Component initialization failed: {e}")
            raise

    async def get_server_status(self) -> Dict[str, Any]:
        """Get comprehensive server status."""
        status = {
            "server": {
                "status": "running",
                "version": "2.0.0",
                "uptime": time.time() - getattr(self, '_start_time', time.time()),
                "host": self.host,
                "port": self.port
            },
            "components": {
                "orchestrator": "available" if self.orchestrator else "unavailable",
                "vector_db": "available" if self.vector_db else "unavailable",
                "rag_pipeline": "available" if self.rag_pipeline else "unavailable",
                "redis": "available" if self.redis_client else "unavailable"
            },
            "metrics": {
                "active_connections": len(self.connection_manager.active_connections),
                "queue_size": self.request_queue.qsize(),
                "active_workflows": len(self.active_workflows),
                "background_tasks": len(self.background_tasks)
            }
        }

        if self.orchestrator:
            orchestrator_health = await self.orchestrator.health_check()
            status["agents"] = orchestrator_health.get("agents", {})
            status["templates"] = len(self.orchestrator.get_available_templates())

        if self.vector_db:
            db_stats = self.vector_db.get_database_stats()
            status["database"] = db_stats

        return status

    async def shutdown(self):
        """Graceful shutdown."""
        logger.info("Shutting down MCP server...")

        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()

        # Close Redis connection
        if self.redis_client:
            await self.redis_client.close()

        # Close WebSocket connections
        for connection in self.connection_manager.active_connections.copy():
            await connection.close()

        logger.info("MCP server shutdown complete")

    def run(self):
        """Run the server."""
        self._start_time = time.time()

        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        # Run with uvicorn
        config = uvicorn.Config(
            self.app,
            host=self.host,
            port=self.port,
            log_level="info",
            loop="asyncio"
        )

        server = uvicorn.Server(config)

        # Add startup and shutdown events
        @self.app.on_event("startup")
        async def startup_event():
            await self.initialize_components()

        @self.app.on_event("shutdown")
        async def shutdown_event():
            await self.shutdown()

        # Run server
        try:
            server.run()
        except KeyboardInterrupt:
            logger.info("Server interrupted by user")


# Factory functions and convenience wrappers
def create_mcp_server(
        host: str = "127.0.0.1",
        port: int = 8000,
        papers_dir: str = "./papers",
        output_dir: str = "./output",
        **kwargs
) -> EnhancedMCPServer:
    """Factory function to create MCP server."""
    return EnhancedMCPServer(
        host=host,
        port=port,
        papers_dir=papers_dir,
        output_dir=output_dir,
        **kwargs
    )


async def run_server_async(server: EnhancedMCPServer):
    """Run server asynchronously."""
    await server.initialize_components()

    config = uvicorn.Config(
        server.app,
        host=server.host,
        port=server.port,
        log_level="info"
    )

    uvicorn_server = uvicorn.Server(config)
    await uvicorn_server.serve()


# Command line interface
async def main():
    """Main entry point for command line usage."""
    import argparse

    parser = argparse.ArgumentParser(description="Enhanced Biology Agents MCP Server")
    parser.add_argument("--host", default="127.0.0.1", help="Server host")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--papers-dir", default="./papers", help="Papers directory")
    parser.add_argument("--output-dir", default="./output", help="Output directory")
    parser.add_argument("--redis-url", help="Redis URL for caching")
    parser.add_argument("--enable-auth", action="store_true", help="Enable API authentication")
    parser.add_argument("--api-key", help="API key for authentication")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    server = EnhancedMCPServer(
        host=args.host,
        port=args.port,
        papers_dir=args.papers_dir,
        output_dir=args.output_dir,
        redis_url=args.redis_url,
        enable_auth=args.enable_auth,
        api_key=args.api_key
    )

    server.run()


if __name__ == "__main__":
    asyncio.run(main())