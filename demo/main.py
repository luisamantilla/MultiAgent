# main.py
from agents.pi_agent import get_pi_agent
from agents.biologist_agent import get_biologist_agent
from agents.model_builder_agent import get_model_builder_agent
from agents.simulation_agent import get_simulation_agent


def run_pipeline():
    print("[SYSTEM] Starting Multi-Agent Flu Modeling Pipeline...")

    # Initialize agents
    pi_agent = get_pi_agent()
    biologist = get_biologist_agent()
    model_builder = get_model_builder_agent()
    simulator = get_simulation_agent()

    # Step 1: PI defines the task
    print(f"[PI Agent] Prompt: {pi_agent.prompt}")
    question = "What biological rules define flu virus infection in the lungs?"

    # Step 2: PI queries Biologist Agent
    print("\n[PI → Biologist] Asking for flu immune rules...")
    print(f"[Biologist Agent] Prompt: {biologist.prompt}")
    # Placeholder: response = query_llm(biologist, question)
    print("[Biologist Agent] (Stubbed response) Returning example flu rules.")
    flu_rules = {
        "cell_types": ["epithelial", "macrophage", "T-cell"],
        "cytokines": ["IFN-alpha", "IL-6", "TNF-alpha"],
        "rules": [
            "virus infects epithelial cells",
            "infected cells release IFN-alpha",
            "macrophages are recruited to IFN-alpha",
            "T-cells kill infected cells"
        ]
    }

    # Step 3: PI sends rules to Model Builder Agent
    print("\n[PI → ModelBuilder] Translating flu rules to code...")
    print(f"[ModelBuilder Agent] Prompt: {model_builder.prompt}")
    # Placeholder: modules = translate_rules_to_modules(flu_rules)
    print("[ModelBuilder Agent] (Stubbed) Created Python modules from rules.")

    # Step 4: PI instructs Simulation Agent to simulate
    print("\n[PI → Simulator] Running simulation with generated model...")
    print(f"[Simulation Agent] Prompt: {simulator.prompt}")
    # Placeholder: simulation_result = run_simulation(modules)
    print("[Simulation Agent] (Stubbed) Simulation complete. Output: 80% infection clearance in 72h")

    print("\n[SYSTEM] Pipeline complete.")


if __name__ == "__main__":
    run_pipeline()