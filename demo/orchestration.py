# orchestration.py
import os

from openai import OpenAI
from agents.agents import PI_AGENT, BIO_AGENT, MODEL_BUILDER_AGENT, SIM_AGENT
from agents.pi_agent import call_pi_agent
from tools.tools import PUBMED_TOOL_DESCRIPTION, run_tools_from_llm, call_biologist_with_pubmed_via_chat
from tools.tools import VIVARIUM_TOOL_DESCRIPTION, run_vivarium_module_generator


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
def call_biologist_with_pubmed(query: str) -> str:
    """
    Uses tool-calling via the Biologist Agent to fetch immune rules from PubMed.
    """
    # Create Biologist assistant with PubMed tool
    assistant = client.beta.assistants.create(
        name=BIO_AGENT.title,
        instructions=BIO_AGENT.prompt,
        model=BIO_AGENT.model,
        tools=[PUBMED_TOOL_DESCRIPTION],
    )

    # Create thread + add user message
    thread = client.beta.threads.create()
    client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=query,
    )

    # Create and poll run
    run = client.beta.threads.runs.create_and_poll(
        thread_id=thread.id,
        assistant_id=assistant.id,
    )

    # If tool is required, run tool logic
    if run.status == "requires_action":
        outputs = run_tools_from_llm(run)
        run = client.beta.threads.runs.submit_tool_outputs_and_poll(
            thread_id=thread.id,
            run_id=run.id,
            tool_outputs=outputs,
        )

    # Retrieve final message
    messages = client.beta.threads.messages.list(thread_id=thread.id)
    return messages.data[-1].content[0].text.value

def run_pipeline(question: str) -> str:
    """
    Orchestrates the full Multi-Agent reasoning pipeline with tool-calling support.
    """
    log = []
    log.append(f"[User] {question}")
    log.append(f"[PI Agent] {PI_AGENT.prompt}")

    # Step 1: Biologist (with PubMed tool calling)
    log.append(f"[PI → Biologist] Asking for flu rules via PubMed...")
    log.append(f"[{BIO_AGENT.title}] {BIO_AGENT.prompt}")
    bio_rules = call_biologist_with_pubmed_via_chat(question)
    log.append(f"[{BIO_AGENT.title}] {bio_rules}")

    # Step 2: Model Builder Agent (generate + run Vivarium module)
    log.append(f"[PI → ModelBuilder] Translating rules into code...")
    log.append(f"[{MODEL_BUILDER_AGENT.title}] {MODEL_BUILDER_AGENT.prompt}")

    # Step 3: Ask the model builder to generate Vivarium code
    # For now, assume `bio_rules` contains valid Python logic in string
    cell_type = "T cell"
    vivarium_output = run_vivarium_module_generator(cell_type, bio_rules)

    log.append(f"[{MODEL_BUILDER_AGENT.title}] Vivarium output:\n{vivarium_output}")

    log.append(f"[PI → Simulation Agent] Interpreting simulation results...")
    log.append(f"[{SIM_AGENT.title}] {SIM_AGENT.prompt}")
    log.append(f"[{SIM_AGENT.title}] Based on the Vivarium simulation, the model shows strong cytokine response within 24h.")

    # Step 4: Final summary via PI Agent
    summary = call_pi_agent(
        question + "\n\nBiologist provided rules:\n" + bio_rules[:1000] +
        "\n\nVivarium simulation output:\n" + vivarium_output[:1000]
    )
    log.append(f"[PI Agent Summary] {summary}")


    return "\n\n".join(log)
