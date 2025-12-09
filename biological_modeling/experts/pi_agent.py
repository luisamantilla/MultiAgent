from typing import List, Dict
from pathlib import Path
import sys

# Add the shared directory to Python path to import BaseAgent
_PROJECT_ROOT = Path(__file__).resolve().parents[2]  # Go up to MultiAgent/
_SHARED_DIR = _PROJECT_ROOT / "shared"
if str(_SHARED_DIR) not in sys.path:
    sys.path.insert(0, str(_SHARED_DIR))

from base.base_agent import BaseAgent


class PIAgent(BaseAgent):
    """LLM-backed PI that opens rounds and synthesizes briefly."""

    def __init__(self, model: str = "gpt-4o-mini"):
        super().__init__(
            title="PI",
            expertise="Synthesis and decision-making",
            goal="Coordinate discussion and deliver a concise plan",
            role="Principal Investigator",
            model=model,
        )
        self.system_prompt = (
            "You are a concise PI. Synthesize inputs, resolve obvious conflicts, and propose next steps in 2-3 sentences."
        )

    def open(self, topic: str) -> str:
        return f"Team, let's briefly discuss: {topic}"

    def synthesize(self, topic: str, history: List[Dict[str, str]]) -> str:
        messages = [{"role": "system", "content": self.system_prompt}]
        for m in history[-9:]:
            messages.append({"role": "user", "content": f"{m['speaker']}: {m['content']}"})
        messages.append({"role": "user", "content": f"PI: Summarize current consensus on: {topic}. Keep it brief."})
        resp = self.client.chat.completions.create(model=self.model, messages=messages, temperature=0.4)
        return resp.choices[0].message.content.strip()

    def finalize_vivarium_minimal_model(self, premise: str, bio_eval: Dict, exp_eval: Dict, comp_proposal: Dict) -> Dict:
        """Return JSON-only final Vivarium selection with rationale and next coding steps."""
        schema = {
            "selected": {"cell_types": ["string"], "molecules": ["string"], "interactions": ["string"]},
            "justification": ["string"],
            "tradeoffs": ["string"],
            "next_coding_steps": ["string"],
        }
        messages = [
            {"role": "system", "content": self.system_prompt + " Output JSON only."},
            {"role": "user", "content": (
                "Goal: Choose the smallest set that is biologically defensible and computationally lean for Vivarium coding.\n"
                "Premise: " + premise + "\n"
                "Schema (JSON): " + str(schema)
            )},
            {"role": "user", "content": "Biologist proposal:\n" + str(bio_eval)},
            {"role": "user", "content": "Modelability assessment:\n" + str(exp_eval)},
            {"role": "user", "content": "Vivarium plan:\n" + str(comp_proposal)},
        ]
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.3,
            response_format={"type": "json_object"},
        )
        import json as _json
        try:
            return _json.loads(resp.choices[0].message.content)
        except Exception:
            return {"error": "Non-JSON response", "raw": resp.choices[0].message.content}