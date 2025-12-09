# agents/base_agent.py

class Agent:
    """
    Base class for agents in the Multi-Agent system.
    """

    def __init__(self, title: str, expertise: str, goal: str, role: str, model: str = "gpt-4-1106-preview") -> None:
        """
        Initializes the agent with its title, area of expertise, goal, role, and chosen LLM model.
        :param title: The title of the agent.
        :param expertise: The area of expertise.
        :param goal: The primary goal of this agent.
        :param role: The responsibilities or duties.
        :param model: The name of the LLM to be used.
        """
        self.title = title
        self.expertise = expertise
        self.goal = goal
        self.role = role
        self.model = model

    @property
    def prompt(self) -> str:
        """
        Returns the prompt defining the agent's identity and goal.
        """
        return (
            f"You are a {self.title}. "
            f"Your expertise is in {self.expertise}. "
            f"Your goal is to {self.goal}. "
            f"Your role is to {self.role}."
        )

    def system_message(self) -> dict:
        """
        Returns a formatted system message for the LLM chat API.
        """
        return {"role": "system", "content": self.prompt}

    def __str__(self) -> str:
        return self.title

    def __repr__(self) -> str:
        return f"Agent({self.title})"

