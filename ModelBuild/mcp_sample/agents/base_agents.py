# agents/base_agents.py
import os
import json
import time
import uuid
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

try:
    # OpenAI SDK v1.x
    from openai import OpenAI

    _OPENAI_CLIENT = OpenAI(api_key=os.getenv("OPENAI_API_KEY")) if os.getenv("OPENAI_API_KEY") else None
except Exception:
    _OPENAI_CLIENT = None
    logger.warning("OpenAI client not available; Agent.run() will return stubbed output.")


# ---------- Execution tracking ----------

@dataclass
class LLMCallRecord:
    agent: str
    model: str
    prompt_chars: int
    output_chars: int
    started_at: float
    finished_at: float
    meta: Dict[str, Any] = field(default_factory=dict)


class ExecutionTracker:
    """Simple execution tracker that writes JSON logs to ./output/logs/"""

    def __init__(self):
        self.execution_id = f"exec_{uuid.uuid4().hex[:8]}"
        self.records: List[LLMCallRecord] = []
        self.meta: Dict[str, Any] = {}
        Path("./output/logs").mkdir(parents=True, exist_ok=True)

    def log_llm(self, rec: LLMCallRecord):
        self.records.append(rec)

    def save_execution_log(self) -> str:
        data = {
            "execution_id": self.execution_id,
            "records": [r.__dict__ for r in self.records],
            "meta": self.meta,
            "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        path = Path("./output/logs") / f"{self.execution_id}.json"
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return str(path)

    def print_summary(self):
        total = len(self.records)
        logger.info(f"[TRACKER] Execution {self.execution_id} | LLM calls: {total}")
        for i, r in enumerate(self.records, 1):
            dur = r.finished_at - r.started_at
            logger.info(
                f"  {i:02d}. {r.agent} | {r.model} | {dur:.2f}s | in {r.prompt_chars} chars → out {r.output_chars} chars")


_TRACKER: Optional[ExecutionTracker] = None


def get_tracker() -> ExecutionTracker:
    global _TRACKER
    if _TRACKER is None:
        _TRACKER = ExecutionTracker()
    return _TRACKER


def reset_tracker():
    global _TRACKER
    _TRACKER = ExecutionTracker()


# ---------- MCP Compatible Result ----------

@dataclass
class MCPResult:
    """Structured result for MCP compatibility."""
    content: str
    agent_id: str
    status: str = "success"
    confidence: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    structured_data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


# ---------- Base Agent ----------

class Agent:
    """
    Enhanced base agent with MCP compatibility.
    - Maintains backward compatibility with existing code
    - Adds MCP-compatible methods for structured results
    - Integrates with vector databases and workflow managers
    """

    def __init__(
            self,
            title: str,
            expertise: str,
            goal: str,
            role: str,
            model: str = "gpt-4o",
            output_format: Optional[str] = None,
            enable_confidence: bool = False,
            dependencies: Optional[List[str]] = None,
            max_retries: int = 2,
            timeout: int = 120,
            mcp_enabled: bool = True,  # NEW: MCP compatibility flag
    ):
        self.title = title
        self.expertise = expertise
        self.goal = goal
        self.role = role
        self.model = model
        self.output_format = output_format or ""
        self.enable_confidence = enable_confidence
        self.dependencies = dependencies or []
        self.max_retries = max_retries
        self.timeout = timeout
        self.mcp_enabled = mcp_enabled

        # runtime attachments (may be set by orchestrators)
        self.config: Dict[str, Any] = {}
        self.db: Any = None
        self.peers: Dict[str, "Agent"] = {}

        # NEW: MCP-specific attributes
        self.agent_id = self._generate_agent_id()
        self.last_result: Optional[MCPResult] = None
        self.execution_history: List[MCPResult] = []

    def _generate_agent_id(self) -> str:
        """Generate a unique agent ID based on title."""
        return self.title.lower().replace(" ", "_").replace("-", "_")

    # ----- Orchestrator hooks -----
    def attach_runtime(self, *, config: Optional[Dict[str, Any]] = None, db: Any = None,
                       peers: Optional[Dict[str, "Agent"]] = None):
        if config is not None:
            self.config = config
        if db is not None:
            self.db = db
        if peers is not None:
            self.peers = peers

    # ----- Chat primitives -----
    def system_message(self) -> Dict[str, str]:
        base = (
            f"You are {self.title}.\n"
            f"Role: {self.role}\n"
            f"Goal: {self.goal}\n"
            f"Expertise: {self.expertise}\n"
        )
        if self.output_format:
            base += "\nOutput format:\n" + self.output_format.strip()

        # NEW: Add vector database context if available
        if self.db and hasattr(self.db, 'get_context'):
            try:
                db_context = self.db.get_context()
                if db_context:
                    base += f"\n\nAvailable knowledge base: {db_context}"
            except:
                pass  # Graceful degradation if DB context fails

        return {"role": "system", "content": base}

    def run(self, conversation: List[Dict[str, str]]) -> str:
        """
        Original run method - maintains backward compatibility.
        Run a synchronous LLM call. If OpenAI is not configured, returns a minimal stub to keep the pipeline moving.
        `conversation` is a list of dicts: [{"role": "...", "content": "..."}].
        """
        # Defensive copy and small cleanup
        conversation = [dict(m) for m in conversation if m.get("content")]
        if not conversation or conversation[0].get("role") != "system":
            conversation.insert(0, self.system_message())

        # Attempt OpenAI call
        tracker = get_tracker()
        started = time.time()
        prompt_chars = sum(len(m.get("content", "")) for m in conversation)

        result_text = None
        error = None

        if _OPENAI_CLIENT:
            # Use Chat Completions (responses API can be swapped in if preferred)
            try:
                # truncate to keep requests sane if needed
                trimmed: List[Dict[str, str]] = []
                running = 0
                for msg in reversed(conversation):
                    running += len(msg["content"])
                    trimmed.append(msg)
                    if running > 12000:  # rough cap
                        break
                trimmed.reverse()

                resp = _OPENAI_CLIENT.chat.completions.create(
                    model=self.model,
                    messages=trimmed,
                    temperature=0.2,
                    timeout=self.timeout,
                )
                result_text = (resp.choices[0].message.content or "").strip()
            except Exception as e:
                error = str(e)

        if result_text is None:
            # Fallback stub so the workflow doesn't break if no API key
            result_text = (
                    f"[Stub output from {self.title}] No OpenAI client detected or call failed."
                    + (f" Error: {error}" if error else "")
            )

        finished = time.time()
        tracker.log_llm(
            LLMCallRecord(
                agent=self.title,
                model=self.model,
                prompt_chars=prompt_chars,
                output_chars=len(result_text),
                started_at=started,
                finished_at=finished,
                meta={"dependencies": self.dependencies, "enable_confidence": self.enable_confidence},
            )
        )

        # NEW: Store MCP-compatible result if enabled
        if self.mcp_enabled:
            self._store_mcp_result(result_text, error, conversation)

        return result_text

    # ----- NEW: MCP-compatible methods -----

    def run_mcp(self, conversation: List[Dict[str, str]], context: Optional[Dict[str, Any]] = None) -> MCPResult:
        """
        MCP-compatible run method that returns structured results.
        """
        try:
            # Add context to conversation if provided
            if context and context.get('vector_search'):
                conversation = self._enhance_with_vector_search(conversation, context)

            # Run the original method
            result_text = self.run(conversation)

            # Process result for structured data
            structured_data = self._extract_structured_data(result_text)

            # Create MCP result
            mcp_result = MCPResult(
                content=result_text,
                agent_id=self.agent_id,
                status="success",
                confidence=self._extract_confidence(result_text) if self.enable_confidence else None,
                metadata={
                    "model": self.model,
                    "dependencies": self.dependencies,
                    "execution_time": time.time(),
                    "context_used": bool(context)
                },
                structured_data=structured_data
            )

            self.last_result = mcp_result
            self.execution_history.append(mcp_result)

            return mcp_result

        except Exception as e:
            error_result = MCPResult(
                content="",
                agent_id=self.agent_id,
                status="error",
                error=str(e),
                metadata={"model": self.model}
            )
            self.last_result = error_result
            return error_result

    def _store_mcp_result(self, result_text: str, error: Optional[str], conversation: List[Dict[str, str]]):
        """Store result in MCP format for tracking."""
        mcp_result = MCPResult(
            content=result_text,
            agent_id=self.agent_id,
            status="error" if error else "success",
            error=error,
            metadata={
                "model": self.model,
                "conversation_length": len(conversation),
                "timestamp": time.time()
            }
        )
        self.last_result = mcp_result
        self.execution_history.append(mcp_result)

    def _enhance_with_vector_search(self, conversation: List[Dict[str, str]], context: Dict[str, Any]) -> List[
        Dict[str, str]]:
        """Enhance conversation with vector database search results."""
        if not self.db or not hasattr(self.db, 'search'):
            return conversation

        try:
            # Extract user query from conversation
            user_messages = [msg["content"] for msg in conversation if msg.get("role") == "user"]
            if user_messages:
                query = user_messages[-1]  # Use the last user message

                # Search vector database
                search_results = self.db.search(query, k=context.get('search_k', 3))

                if search_results:
                    # Add search results as system context
                    search_context = "Relevant research findings:\n"
                    for i, result in enumerate(search_results, 1):
                        search_context += f"{i}. {result.get('content', result)}\n"

                    # Insert after system message but before user messages
                    enhanced = conversation.copy()
                    enhanced.insert(1, {"role": "system", "content": search_context})
                    return enhanced
        except Exception as e:
            logger.warning(f"Vector search failed: {e}")

        return conversation

    def _extract_structured_data(self, result_text: str) -> Optional[Dict[str, Any]]:
        """Extract structured data from result text (JSON, TSV, etc.)."""
        try:
            # Try to extract JSON
            import re
            json_match = re.search(r'\{[\s\S]*\}', result_text)
            if json_match:
                return json.loads(json_match.group())
        except:
            pass

        try:
            # Try to parse TSV format
            if '\t' in result_text and any(
                    keyword in result_text.lower() for keyword in ['parameter', 'value', 'units']):
                return self._parse_tsv_data(result_text)
        except:
            pass

        return None

    def _parse_tsv_data(self, text: str) -> Dict[str, Any]:
        """Parse TSV text into structured data."""
        lines = [line.strip() for line in text.split('\n') if line.strip() and '\t' in line]
        if len(lines) < 2:
            return None

        headers = lines[0].split('\t')
        data = []

        for line in lines[1:]:
            values = line.split('\t')
            if len(values) >= len(headers):
                row = {}
                for i, header in enumerate(headers):
                    row[header] = values[i] if i < len(values) else ""
                data.append(row)

        return {"format": "tsv", "headers": headers, "data": data}

    def _extract_confidence(self, result_text: str) -> Optional[float]:
        """Extract confidence score from result text."""
        import re
        # Look for confidence patterns like "confidence: 0.85" or "85% confident"
        patterns = [
            r'confidence[:\s]+([0-9.]+)',
            r'([0-9.]+)%?\s*confident',
            r'certainty[:\s]+([0-9.]+)'
        ]

        for pattern in patterns:
            match = re.search(pattern, result_text.lower())
            if match:
                try:
                    value = float(match.group(1))
                    return value if value <= 1.0 else value / 100.0
                except:
                    continue
        return None

    # ----- NEW: MCP utility methods -----

    def get_capabilities(self) -> List[str]:
        """Get agent capabilities for MCP tool registration."""
        capabilities = ["text_processing", "conversation"]

        # Add agent-specific capabilities
        title_lower = self.title.lower()
        if 'principal investigator' in title_lower or 'pi' in title_lower:
            capabilities.extend(["task_planning", "orchestration", "json_output"])
        elif 'biologist' in title_lower:
            capabilities.extend(["biological_analysis", "quantitative_modeling"])
        elif 'parameter' in title_lower:
            capabilities.extend(["parameter_extraction", "literature_analysis", "tsv_output"])
        elif 'model builder' in title_lower:
            capabilities.extend(["vivarium_modeling", "system_design"])
        elif 'code generation' in title_lower:
            capabilities.extend(["code_generation", "repository_creation", "file_management"])
        elif 'critique' in title_lower:
            capabilities.extend(["quality_assurance", "validation"])

        # Add database capabilities if available
        if self.db:
            capabilities.append("vector_search")

        return capabilities

    def get_mcp_schema(self) -> Dict[str, Any]:
        """Get MCP tool schema for this agent."""
        return {
            "name": f"agent_{self.agent_id}",
            "description": f"{self.title}: {self.goal}",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "The task or question for the agent"
                    },
                    "context": {
                        "type": "object",
                        "description": "Additional context or parameters",
                        "properties": {
                            "vector_search": {"type": "boolean", "description": "Enable vector database search"},
                            "search_k": {"type": "integer", "description": "Number of search results to include"},
                            "previous_results": {"type": "object", "description": "Results from previous agents"},
                            "project_name": {"type": "string", "description": "Project name for code generation"},
                            "output_dir": {"type": "string", "description": "Output directory for files"}
                        }
                    }
                },
                "required": ["prompt"]
            }
        }

    def get_execution_summary(self) -> Dict[str, Any]:
        """Get summary of agent execution history."""
        if not self.execution_history:
            return {"total_executions": 0}

        successful = sum(1 for r in self.execution_history if r.status == "success")
        failed = len(self.execution_history) - successful

        avg_confidence = None
        confidences = [r.confidence for r in self.execution_history if r.confidence is not None]
        if confidences:
            avg_confidence = sum(confidences) / len(confidences)

        return {
            "total_executions": len(self.execution_history),
            "successful": successful,
            "failed": failed,
            "success_rate": successful / len(self.execution_history),
            "average_confidence": avg_confidence,
            "last_execution": self.last_result.status if self.last_result else None
        }