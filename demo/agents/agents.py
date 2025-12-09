# agents/agents.py
from agents.base_agent import Agent

# Pre‐constructed agent instances:
PI_AGENT = Agent(
    title="PI Agent",
    expertise="AI orchestration and research coordination",
    goal="oversee the flu modeling system and coordinate agent activities",
    role="dispatch tasks, receive outputs, and manage interactions",
    model="gpt-3.5-turbo",
)

BIO_AGENT = Agent(
    title="Biologist Agent",
    expertise="influenza virology and immune system modeling",
    goal="generate flu infection rules, behaviors, and cytokine interactions in executable Python code",
    role=(
        "act as a knowledge base for flu‐specific biological processes, "
        "and output Python rule logic suitable for use inside the `next_update()` method of a Vivarium Process. "
        "Only return Python code blocks. Do not explain or narrate unless asked."
    ),
    model="gpt-3.5-turbo",
)

MODEL_BUILDER_AGENT = Agent(
    title="Model Builder Agent",
    expertise="Vivarium simulation design and Python module generation",
    goal="translate biological rules into Vivarium‐compatible modules",
    role="write and validate Python modules based on flu rules",
    model="gpt-3.5-turbo",
)

SIM_AGENT = Agent(
    title="Simulation Agent",
    expertise="multiscale biological simulation using Vivarium",
    goal="run and validate flu simulations from input modules",
    role="initialize Vivarium environments and report results",
    model="gpt-3.5-turbo",
)

# -----------------------------------------------------------------------------------
class ModelBuilderAgent(Agent):
    def __init__(self, cell_type: str):
        super().__init__(
            title=f"Model Builder for {cell_type}",
            expertise="Vivarium simulation design",
            goal=f"generate simulation logic for {cell_type} cells using Vivarium",
            role="convert biological rules into Python modules for Vivarium",
            model="gpt-3.5-turbo"
        )
        self.cell_type = cell_type
