from typing import List, Dict
from pathlib import Path
import sys

# Add the shared directory to Python path to import BaseAgent
_PROJECT_ROOT = Path(__file__).resolve().parents[2]  # Go up to MultiAgent/
_SHARED_DIR = _PROJECT_ROOT / "shared"
if str(_SHARED_DIR) not in sys.path:
    sys.path.insert(0, str(_SHARED_DIR))

from base.base_agent import BaseAgent


class BiologicalExpertAgent(BaseAgent):
    """LLM-backed biological expert with a brief reply style."""

    def __init__(self, model: str = "gpt-4o-mini"):
        super().__init__(
            title="Biological Expert",
            expertise="Cellular and molecular biology; pathways and mechanisms",
            goal="Provide biologically sound, concise input",
            role="Biology domain expert",
            model=model,
        )
        self.system_prompt = (
            "You are a concise biological expert. Focus on core mechanisms, "
            "state assumptions, and keep answers in 2-3 sentences."
        )

    def reply(self, topic: str, history: List[Dict[str, str]]) -> str:
        messages = [{"role": "system", "content": self.system_prompt}]
        # Include a little context
        for m in history[-6:]:
            messages.append({"role": "user", "content": f"{m['speaker']}: {m['content']}"})
        messages.append({"role": "user", "content": f"Biologist: Weigh in on: {topic} (2-3 sentences)."})
        resp = self.client.chat.completions.create(model=self.model, messages=messages, temperature=0.5)
        return resp.choices[0].message.content.strip()

    def evaluate_requirements(self, requirements: Dict, premise: str, max_interactions: int | None = None) -> Dict:
        """Return JSON-only, code-oriented minimal selection for modeling under compute constraints.
        If max_interactions is provided, keep interactions <= that number.
        """
        schema = {
            "keep_cell_types": ["string"],
            "drop_cell_types": ["string"],
            "keep_interactions": ["string"],
            "drop_interactions": [{"name": "string", "reason": "string"}],
            "merge_candidates": [{"from": "string", "into": "string", "reason": "string"}],
            "keep_molecules": ["string"],
            "drop_molecules": ["string"],
            "assumptions": ["string"],
            "notes": ["string"],
        }
        # Safely summarize requirements whether it's a dict or a list
        def _summarize_requirements(req):
            try:
                if isinstance(req, dict):
                    return {k: req.get(k) for k in ["cell_types", "interactions", "molecules"]}
                elif isinstance(req, list):
                    # Trim to avoid overly long prompts
                    return {"interactions": req[: min(20, len(req))]}
                else:
                    return {"raw_type": type(req).__name__}
            except Exception:
                return {"error": "could_not_parse_requirements"}

        messages = [
            {"role": "system", "content": self.system_prompt + " Output JSON only."},
            {"role": "user", "content": (
                "Goal: Select the minimal, code-ready subset (cell_types, molecules, interactions) that preserves causal validity for the premise.\n"
                "Premise: " + premise + "\n"
                "Instructions: Prioritize merges/simplifications; keep interaction count as low as possible"
                + (f" (<= {max_interactions} interactions)." if max_interactions else ".") + "\n"
                "Schema (JSON): " + str(schema) + "\n"
                "Important: Use names from the provided requirements where possible."
            )},
            {"role": "user", "content": "Requirements snippet:\n" + str(_summarize_requirements(requirements))},
        ]
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.4,
            response_format={"type": "json_object"},
        )
        import json as _json
        try:
            return _json.loads(resp.choices[0].message.content)
        except Exception:
            return {"error": "Non-JSON response", "raw": resp.choices[0].message.content}