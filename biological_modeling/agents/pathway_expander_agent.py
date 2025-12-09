import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List
from agents.base_agent import BaseAgent

# --- Data Structures ---

@dataclass
class HighLevelInteraction:
    """Represents a single high-level interaction to be expanded."""
    interaction_id: str
    description: str
    # Add other relevant fields from BiologistAgent output as needed

@dataclass
class DetailedStep:
    """A single, granular step within an expanded interaction."""
    step: int
    description: str
    species: List[str] = field(default_factory=list)
    genes: List[str] = field(default_factory=list)
    molecules: List[str] = field(default_factory=list)

@dataclass
class ExpandedInteraction:
    """The output of this agent: an interaction with full detail."""
    interaction_id: str
    detailed_steps: List[DetailedStep]

# --- Agent Definition ---

class PathwayExpanderAgent(BaseAgent):
    """
    Agent responsible for expanding a high-level biological interaction into its
    detailed, step-by-step mechanistic cascade.
    """

    def __init__(self, model: str = "gpt-4.1", base_output_dir: str = None):
        super().__init__(
            title="Pathway Expander Agent",
            expertise="Expanding high-level biological concepts into detailed mechanistic steps, including genes, proteins, and signaling events.",
            goal="To generate the maximum level of mechanistic detail for every biological interaction.",
            role="Mechanistic Detail Generator",
            model=model
        )
        # Optional: output directory for saving intermediate/final results (like BiologistAgent)
        from pathlib import Path
        from datetime import datetime
        if base_output_dir is None:
            base_output_dir = "/home/labuser/Desktop/lab_member_projects/Bobby_Ni/tumor-tcell-exp/biological_modeling_agents_design"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(base_output_dir) / f"pathway_expander_run_{timestamp}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.intermediate_dir = self.output_dir / "intermediate"
        self.intermediate_dir.mkdir(exist_ok=True)

    def run(self, interaction: HighLevelInteraction) -> ExpandedInteraction:
        """
        Takes a single high-level interaction and generates a detailed, step-by-step breakdown using LLM.
        This version always splits the interaction into two LLM calls and merges the results for maximum detail and reliability.
        """
        print(f"Expanding interaction: {interaction.interaction_id} - '{interaction.description}' (split mode)")

        # Split the description in half (by sentences if possible)
        import re
        sentences = re.split(r'(?<=[.!?]) +', interaction.description.strip())
        if len(sentences) > 1:
            mid = len(sentences) // 2
            desc1 = ' '.join(sentences[:mid])
            desc2 = ' '.join(sentences[mid:])
        else:
            # fallback: split by length
            half = len(interaction.description) // 2
            desc1 = interaction.description[:half]
            desc2 = interaction.description[half:]

        # Helper to call LLM and parse steps for a given description
        def expand_half(desc, suffix):
            system_message = (
                f"You are {self.role}. "
                f"Expertise: {self.expertise} "
                f"Goal: {self.goal}"
            )
            prompt = self._create_prompt(HighLevelInteraction(interaction_id=interaction.interaction_id, description=desc))
            try:
                print(f"About to make API call for {suffix}...")
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.2,
                    response_format={"type": "json_object"}
                )
                print(f"API call for {suffix} completed successfully")
                content = response.choices[0].message.content
                print(f"LLM Response for {suffix} received: {len(content)} characters")
                print(f"=== FULL LLM RESPONSE {suffix} START ===")
                print(content)
                print(f"=== FULL LLM RESPONSE {suffix} END ===")
                data = json.loads(content)
                steps = data.get("detailed_steps", [])
                print(f"Number of steps in LLM response {suffix}: {len(steps)}")
                return steps
            except Exception as e:
                print(f"Error expanding {suffix} of interaction {interaction.interaction_id}: {e}")
                import traceback
                print(f"Full traceback: {traceback.format_exc()}")
                return []

        steps1 = expand_half(desc1, "part1")
        steps2 = expand_half(desc2, "part2")

        # Merge and renumber steps
        detailed_steps = []
        step_num = 1
        for step_data in steps1 + steps2:
            detailed_steps.append(DetailedStep(
                step=step_num,
                description=step_data.get("description", ""),
                species=step_data.get("species", []),
                genes=step_data.get("genes", []),
                molecules=step_data.get("molecules", [])
            ))
            step_num += 1

        expanded_interaction = ExpandedInteraction(
            interaction_id=interaction.interaction_id,
            detailed_steps=detailed_steps
        )
        print(f"Successfully expanded interaction {interaction.interaction_id} into {len(detailed_steps)} steps (merged from two calls).")
        self._save_intermediate(f"expanded_{interaction.interaction_id}", asdict(expanded_interaction))
        return expanded_interaction

    def _save_intermediate(self, name: str, data, format: str = "json"):
        """Save intermediate results for debugging and traceability."""
        from datetime import datetime
        timestamp = datetime.now().strftime("%H%M%S")
        filename = f"{name}_{timestamp}"
        if format == "json":
            filepath = self.intermediate_dir / f"{filename}.json"
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        print(f"💾 Saved intermediate: {filepath.name}")

    def _create_prompt(self, interaction: HighLevelInteraction) -> str:
        """Creates the prompt for the LLM to expand the interaction."""
        prompt = f"""
You are an expert molecular biologist. Your task is to expand a high-level biological interaction into a series of detailed, numbered mechanistic steps.

High-Level Interaction: "{interaction.description}"

Provide a comprehensive, step-by-step breakdown of this process. For each step, include the specific species (proteins, cells), genes, and molecules involved.

Output the result as a JSON object with a single key, "detailed_steps". Each item in the list should have the following structure:
{{
  "step": <number>,
  "description": "<Detailed description of the step>",
  "species": ["<list of species>"],
  "genes": ["<list of relevant genes>"],
  "molecules": ["<list of signaling molecules, etc.>"]
}}

Be exhaustive. It is critical to capture every known step in the cascade.
"""
        return prompt
