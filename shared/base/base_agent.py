import os
from openai import OpenAI

# BaseAgent: Abstract base class for all agents. Handles prompt construction, API initialization, and reporting.
class BaseAgent:
    """
    Abstract base class for all agents. Handles prompt construction, API initialization, and reporting.
    """
    def __init__(self, title: str, expertise: str, goal: str, role: str, model: str):
        """Initialize agent metadata and OpenAI client"""
        self.title = title
        self.expertise = expertise
        self.goal = goal
        self.role = role
        self.model = model
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    def construct_prompt(self, task_instruction: str) -> str:
        """Build a prompt string for the agent's LLM"""
        return (
            f"You are a {self.title}.\n"
            f"Expertise: {self.expertise}.\n"
            f"Goal: {self.goal}.\n"
            f"Role: {self.role}.\n"
            f"Task: {task_instruction}"
        )

    def perform_task(self, *args, **kwargs):
        """Abstract method to be implemented by subclasses"""
        raise NotImplementedError("This method should be implemented by subclasses.")

    def report(self):
        """Return agent metadata as a dictionary"""
        return {
            "title": self.title,
            "expertise": self.expertise,
            "goal": self.goal,
            "role": self.role,
            "model": self.model,
        }