# agents/base_agents.py
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
from openai import OpenAI
import os


class AgentStatus(Enum):
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class ExecutionMetrics:
    start_time: float
    end_time: Optional[float] = None
    token_usage: Optional[Dict[str, int]] = None
    confidence_score: Optional[float] = None

    @property
    def duration(self) -> Optional[float]:
        if self.end_time:
            return self.end_time - self.start_time
        return None


class Agent:
    """
    Enhanced base class for agents in the Multi-Agent system.
    """

    def __init__(
            self,
            title: str,
            expertise: str,
            goal: str,
            role: str,
            model: str = "gpt-4",
            input_format: str = None,
            output_format: str = None,
            enable_confidence: bool = False,
            dependencies: List[str] = None,
            max_retries: int = 3,
            timeout: int = 300,
    ) -> None:
        """
        Initializes the agent with enhanced capabilities.
        """
        self.title = title
        self.name = title  # Add name property for consistency
        self.expertise = expertise
        self.goal = goal
        self.role = role
        self.model = model
        self.input_format = input_format
        self.output_format = output_format
        self.enable_confidence = enable_confidence
        self.dependencies = dependencies or []
        self.max_retries = max_retries
        self.timeout = timeout

        # Execution state
        self.status = AgentStatus.IDLE
        self.metrics = None
        self.last_output = None
        self.error_history = []

        # Initialize OpenAI client if API key is available
        self.client = None
        if os.environ.get("OPENAI_API_KEY"):
            self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    @property
    def prompt(self) -> str:
        base = (
            f"You are a {self.title}.\n"
            f"Expertise: {self.expertise}.\n"
            f"Goal: {self.goal}.\n"
            f"Role: {self.role}."
        )

        if self.input_format:
            base += f"\n\nInput Format:\n{self.input_format}"
        if self.output_format:
            base += f"\n\nOutput Format:\n{self.output_format}"
        if self.enable_confidence:
            base += (
                "\n\nAt the end of your response, provide a confidence score between 0 and 1 "
                "formatted as: CONFIDENCE: <score>"
            )

        # Add context about the Vivarium framework
        base += self._get_vivarium_context()

        return base

    def _get_vivarium_context(self) -> str:
        """Add Vivarium-specific context to the prompt."""
        return (
            "\n\nVivarium Framework Context:\n"
            "- Vivarium is a simulation framework for modeling biological systems\n"
            "- Processes define how entities evolve over time\n"
            "- Composites organize and connect multiple processes\n"
            "- State represents the current condition of all entities\n"
            "- Stores handle state management and data flow\n"
            "- Use proper Vivarium imports: from vivarium import Process, Composite\n"
            "- Follow Vivarium naming conventions and patterns"
        )

    def system_message(self) -> dict:
        """Returns a formatted system message for the LLM chat API."""
        return {"role": "system", "content": self.prompt}

    def run(self, conversation: List[Dict[str, str]]) -> str:
        """
        Default run method for agents. Should be overridden by subclasses.

        Args:
            conversation: List of messages in OpenAI chat format

        Returns:
            Agent response as string
        """
        try:
            self.start_execution()

            if not self.client:
                raise Exception("OpenAI client not initialized. Please set OPENAI_API_KEY environment variable.")

            # Ensure system message is included
            if not conversation or conversation[0].get("role") != "system":
                conversation.insert(0, self.system_message())

            response = self.client.chat.completions.create(
                model=self.model,
                messages=conversation,
                temperature=0.3
            )

            result = response.choices[0].message.content
            self.complete_execution(result)
            return result

        except Exception as e:
            self.error_execution(e)
            return f"Error in {self.title}: {str(e)}"

    def start_execution(self):
        """Mark the start of agent execution."""
        self.status = AgentStatus.RUNNING
        self.metrics = ExecutionMetrics(start_time=time.time())

    def complete_execution(self, output: Any, confidence: Optional[float] = None):
        """Mark successful completion of agent execution."""
        self.status = AgentStatus.COMPLETED
        self.metrics.end_time = time.time()
        self.metrics.confidence_score = confidence
        self.last_output = output

    def error_execution(self, error: Exception):
        """Mark failed execution of agent."""
        self.status = AgentStatus.ERROR
        if self.metrics:
            self.metrics.end_time = time.time()
        self.error_history.append(str(error))

    def can_execute(self, completed_agents: List[str]) -> bool:
        """Check if this agent can execute based on dependencies."""
        return all(dep in completed_agents for dep in self.dependencies)

    def __str__(self) -> str:
        return self.title

    def __repr__(self) -> str:
        return f"Agent({self.title}, status={self.status.value})"