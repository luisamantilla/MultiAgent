# agents/mcp_adapters.py
"""
Fixed MCP Adapters with proper protocol implementation.
Resolves async/sync issues and implements proper MCP message handling.
"""

import asyncio
import json
import logging
import time
import uuid
from typing import Dict, Any, List, Optional, Union, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


@dataclass
class MCPMessage:
    """Proper MCP message following JSON-RPC 2.0 specification."""
    jsonrpc: str = "2.0"
    method: Optional[str] = None
    params: Optional[Dict[str, Any]] = None
    id: Optional[Union[str, int]] = None
    result: Optional[Any] = None
    error: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.id is None and self.method:
            self.id = str(uuid.uuid4())

    def to_dict(self) -> Dict[str, Any]:
        data = {"jsonrpc": self.jsonrpc}

        if self.method:
            data["method"] = self.method
            data["id"] = self.id
            if self.params:
                data["params"] = self.params
        else:
            # Response message
            data["id"] = self.id
            if self.result is not None:
                data["result"] = self.result
            elif self.error:
                data["error"] = self.error

        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MCPMessage':
        return cls(
            jsonrpc=data.get("jsonrpc", "2.0"),
            method=data.get("method"),
            params=data.get("params"),
            id=data.get("id"),
            result=data.get("result"),
            error=data.get("error")
        )

    def create_response(self, result: Any = None, error: Dict[str, Any] = None) -> 'MCPMessage':
        return MCPMessage(
            jsonrpc=self.jsonrpc,
            id=self.id,
            result=result,
            error=error
        )


class MCPTool(ABC):
    """Abstract base class for MCP tools."""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    @abstractmethod
    async def call(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the tool with given arguments."""
        pass

    @abstractmethod
    def get_schema(self) -> Dict[str, Any]:
        """Get the JSON schema for this tool."""
        pass


class AgentMCPTool(MCPTool):
    """MCP tool wrapper for biology agents."""

    def __init__(self, agent_id: str, agent, capabilities: List[str]):
        super().__init__(
            name=f"agent_{agent_id}",
            description=f"Execute {agent.title}: {agent.goal}"
        )
        self.agent_id = agent_id
        self.agent = agent
        self.capabilities = capabilities
        self.execution_history: List[Dict[str, Any]] = []

    async def call(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the agent tool."""
        start_time = time.time()

        try:
            # Extract prompt and context
            prompt = arguments.get("prompt", "")
            context = arguments.get("context", {})

            if not prompt:
                raise ValueError("Prompt is required")

            # Build conversation
            conversation = [self.agent.system_message()]

            # Add context if provided
            if context:
                context_str = self._format_context(context)
                if context_str:
                    conversation.append({"role": "system", "content": context_str})

            # Add user prompt
            conversation.append({"role": "user", "content": prompt})

            # Execute agent (handle both sync and async)
            if hasattr(self.agent, 'run_mcp'):
                # Use MCP method if available
                result = self.agent.run_mcp(conversation, context)
                if hasattr(result, 'content'):
                    content = result.content
                    confidence = getattr(result, 'confidence', None)
                    metadata = getattr(result, 'metadata', {})
                else:
                    content = str(result)
                    confidence = None
                    metadata = {}
            else:
                # Fallback to regular run
                content = self.agent.run(conversation)
                confidence = None
                metadata = {}

            execution_time = time.time() - start_time

            # Process result
            processed_result = self._process_agent_result(content)

            result_data = {
                "content": content,
                "processed_result": processed_result,
                "agent_id": self.agent_id,
                "agent_title": self.agent.title,
                "execution_time": execution_time,
                "confidence": confidence,
                "metadata": metadata,
                "status": "success"
            }

            # Log execution
            self.execution_history.append({
                "timestamp": time.time(),
                "prompt_preview": prompt[:100],
                "execution_time": execution_time,
                "success": True,
                "output_length": len(content)
            })

            return result_data

        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = str(e)

            # Log error
            self.execution_history.append({
                "timestamp": time.time(),
                "prompt_preview": prompt[:100] if 'prompt' in locals() else "",
                "execution_time": execution_time,
                "success": False,
                "error": error_msg
            })

            logger.error(f"Agent {self.agent_id} execution failed: {error_msg}")

            return {
                "status": "error",
                "error": error_msg,
                "agent_id": self.agent_id,
                "execution_time": execution_time
            }

    def _format_context(self, context: Dict[str, Any]) -> str:
        """Format context for agent consumption."""
        if not context:
            return ""

        context_parts = []

        # Add previous results
        if "previous_results" in context:
            context_parts.append("Previous step results:")
            for step_id, result in context["previous_results"].items():
                if isinstance(result, dict):
                    content = result.get("content", "")
                    preview = content[:150] + "..." if len(content) > 150 else content
                    context_parts.append(f"- {step_id}: {preview}")

        # Add vector search results
        if "vector_search_results" in context:
            results = context["vector_search_results"]
            if results:
                context_parts.append("Relevant research findings:")
                for i, result in enumerate(results[:3], 1):
                    content = result.get("content", "")
                    preview = content[:200] + "..." if len(content) > 200 else content
                    context_parts.append(f"{i}. {preview}")

        # Add other context items
        for key, value in context.items():
            if key not in ["previous_results", "vector_search_results"] and value:
                if isinstance(value, (str, int, float)):
                    context_parts.append(f"{key}: {value}")

        return "\n".join(context_parts)

    def _process_agent_result(self, content: str) -> Optional[Union[Dict[str, Any], str]]:
        """Process agent result for structured data extraction."""
        agent_title = self.agent.title.lower()

        # Try to parse JSON for PI agent
        if 'principal investigator' in agent_title:
            try:
                import re
                json_match = re.search(r'\{[\s\S]*\}', content)
                if json_match:
                    return json.loads(json_match.group())
            except:
                pass

        # Try to parse TSV for parameter extractor
        elif 'parameter' in agent_title and '\t' in content:
            try:
                lines = [line.strip() for line in content.split('\n') if line.strip() and '\t' in line]
                if len(lines) >= 2:
                    headers = lines[0].split('\t')
                    data = []
                    for line in lines[1:]:
                        values = line.split('\t')
                        if len(values) >= len(headers):
                            row = {headers[i]: values[i] for i in range(len(headers))}
                            data.append(row)
                    return {"format": "tsv", "headers": headers, "data": data}
            except:
                pass

        return content

    def get_schema(self) -> Dict[str, Any]:
        """Get MCP tool schema for this agent."""
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "The task or question for the agent"
                    },
                    "context": {
                        "type": "object",
                        "description": "Additional context for the agent",
                        "properties": {
                            "previous_results": {
                                "type": "object",
                                "description": "Results from previous workflow steps"
                            },
                            "vector_search_results": {
                                "type": "array",
                                "description": "Relevant research findings from vector search"
                            },
                            "user_request": {
                                "type": "string",
                                "description": "Original user request"
                            }
                        }
                    }
                },
                "required": ["prompt"]
            }
        }

    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics for this agent tool."""
        if not self.execution_history:
            return {"total_executions": 0}

        successful = sum(1 for h in self.execution_history if h.get("success", False))
        total = len(self.execution_history)
        avg_time = sum(h.get("execution_time", 0) for h in self.execution_history) / total

        return {
            "total_executions": total,
            "successful_executions": successful,
            "success_rate": successful / total,
            "average_execution_time": avg_time,
            "capabilities": self.capabilities
        }


class MCPServer:
    """Simple MCP server for biology agents."""

    def __init__(self):
        self.tools: Dict[str, MCPTool] = {}
        self.request_handlers: Dict[str, Callable] = {}
        self._setup_default_handlers()

    def _setup_default_handlers(self):
        """Setup default MCP request handlers."""
        self.request_handlers = {
            "tools/list": self._handle_list_tools,
            "tools/call": self._handle_call_tool,
            "server/info": self._handle_server_info
        }

    def register_tool(self, tool: MCPTool):
        """Register a new tool."""
        self.tools[tool.name] = tool
        logger.info(f"Registered MCP tool: {tool.name}")

    def register_agent_tools(self, agents: Dict[str, Any]):
        """Register all agents as MCP tools."""
        for agent_id, agent in agents.items():
            # Get agent capabilities
            capabilities = []
            if hasattr(agent, 'get_capabilities'):
                capabilities = agent.get_capabilities()

            # Create MCP tool wrapper
            tool = AgentMCPTool(agent_id, agent, capabilities)
            self.register_tool(tool)

        logger.info(f"Registered {len(agents)} agent tools")

    async def handle_request(self, message: MCPMessage) -> MCPMessage:
        """Handle incoming MCP request."""
        if not message.method:
            return message.create_response(
                error={"code": -32600, "message": "Invalid request - missing method"}
            )

        if message.method not in self.request_handlers:
            return message.create_response(
                error={"code": -32601, "message": f"Method not found: {message.method}"}
            )

        try:
            handler = self.request_handlers[message.method]
            result = await handler(message.params or {})
            return message.create_response(result=result)

        except Exception as e:
            logger.error(f"Handler error for {message.method}: {e}")
            return message.create_response(
                error={"code": -32603, "message": f"Internal error: {str(e)}"}
            )

    async def _handle_list_tools(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle tools/list request."""
        tools_list = []
        for tool in self.tools.values():
            tools_list.append(tool.get_schema())

        return {"tools": tools_list}

    async def _handle_call_tool(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle tools/call request."""
        tool_name = params.get("name")
        arguments = params.get("arguments", {})

        if not tool_name:
            raise ValueError("Tool name is required")

        if tool_name not in self.tools:
            raise ValueError(f"Tool not found: {tool_name}")

        tool = self.tools[tool_name]
        result = await tool.call(arguments)

        return {"content": [{"type": "text", "text": json.dumps(result, indent=2)}]}

    async def _handle_server_info(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle server/info request."""
        return {
            "name": "Biology Agents MCP Server",
            "version": "1.0.0",
            "description": "MCP server for biological modeling agents",
            "tools_count": len(self.tools)
        }


class SimpleOrchestrator:
    """Fixed orchestrator with proper MCP integration."""

    def __init__(self):
        self.mcp_server = MCPServer()
        self.agents: Dict[str, Any] = {}
        self.workflow_history: List[Dict[str, Any]] = []

    def register_agents(self, agents: Dict[str, Any]):
        """Register agents and create MCP tools."""
        self.agents = agents
        self.mcp_server.register_agent_tools(agents)

    async def execute_workflow(self,
                               workflow_steps: List[Dict[str, Any]],
                               context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute a workflow using MCP protocol."""
        results = {}
        current_context = context or {}

        for step in workflow_steps:
            step_id = step.get("step_id", f"step_{len(results)}")
            agent_tool = step.get("tool_name", step.get("agent_id"))
            prompt = step.get("prompt", step.get("task"))

            # Prepare MCP message
            message = MCPMessage(
                method="tools/call",
                params={
                    "name": f"agent_{agent_tool}" if not agent_tool.startswith("agent_") else agent_tool,
                    "arguments": {
                        "prompt": prompt,
                        "context": current_context.copy()
                    }
                }
            )

            # Execute via MCP
            response = await self.mcp_server.handle_request(message)

            if response.error:
                results[step_id] = {
                    "status": "error",
                    "error": response.error["message"]
                }
                logger.error(f"Step {step_id} failed: {response.error['message']}")
            else:
                # Parse result
                content = response.result.get("content", [{}])[0].get("text", "{}")
                try:
                    step_result = json.loads(content)
                    results[step_id] = step_result

                    # Update context for next steps
                    current_context["previous_results"] = results

                except json.JSONDecodeError:
                    results[step_id] = {"content": content, "status": "success"}

            logger.info(f"Completed step: {step_id}")

        # Log workflow
        self.workflow_history.append({
            "timestamp": time.time(),
            "steps_count": len(workflow_steps),
            "results_count": len(results),
            "success": all(r.get("status") != "error" for r in results.values())
        })

        return {
            "status": "completed",
            "results": results,
            "context": current_context
        }

    def get_available_tools(self) -> List[str]:
        """Get list of available tools."""
        return list(self.mcp_server.tools.keys())

    def get_workflow_stats(self) -> Dict[str, Any]:
        """Get workflow execution statistics."""
        if not self.workflow_history:
            return {"total_workflows": 0}

        total = len(self.workflow_history)
        successful = sum(1 for w in self.workflow_history if w.get("success", False))

        return {
            "total_workflows": total,
            "successful_workflows": successful,
            "success_rate": successful / total,
            "available_tools": len(self.mcp_server.tools)
        }


# Factory functions for easy setup
def create_mcp_server() -> MCPServer:
    """Factory function to create MCP server."""
    return MCPServer()


def setup_mcp_orchestrator(agents: Dict[str, Any]) -> SimpleOrchestrator:
    """Setup MCP orchestrator with agents."""
    orchestrator = SimpleOrchestrator()
    orchestrator.register_agents(agents)
    return orchestrator


def create_agent_mcp_tool(agent_id: str, agent: Any) -> AgentMCPTool:
    """Create MCP tool wrapper for a single agent."""
    capabilities = agent.get_capabilities() if hasattr(agent, 'get_capabilities') else []
    return AgentMCPTool(agent_id, agent, capabilities)