from typing import List, Dict
from pathlib import Path
import sys

# Add the shared directory to Python path to import BaseAgent
_PROJECT_ROOT = Path(__file__).resolve().parents[2]  # Go up to MultiAgent/
_SHARED_DIR = _PROJECT_ROOT / "shared"
if str(_SHARED_DIR) not in sys.path:
    sys.path.insert(0, str(_SHARED_DIR))

from base.base_agent import BaseAgent


class ComputationalExpertAgent(BaseAgent):
    """LLM-backed computational modeling expert with simple abstractions."""

    def __init__(self, model: str = "gpt-4o-mini"):
        super().__init__(
            title="Computationalist",
            expertise="Modeling simplifications, parameters, data needs",
            goal="Suggest lean computational abstractions",
            role="Computational modeling expert",
            model=model,
        )
        self.system_prompt = (
            "You are a concise computational modeler. Offer simple abstractions, key parameters, "
            "and feasibility notes in 2-3 sentences."
        )

    def reply(self, topic: str, history: List[Dict[str, str]]) -> str:
        messages = [{"role": "system", "content": self.system_prompt}]
        for m in history[-6:]:
            messages.append({"role": "user", "content": f"{m['speaker']}: {m['content']}"})
        messages.append({"role": "user", "content": f"Computationalist: Suggest a simple abstraction for: {topic} (2-3 sentences)."})
        resp = self.client.chat.completions.create(model=self.model, messages=messages, temperature=0.5)
        return resp.choices[0].message.content.strip()

    def propose_vivarium_plan(self, requirements: Dict, premise: str, bio_eval: Dict, exp_eval: Dict, max_interactions: int | None = None) -> Dict:
        """Return JSON-only Vivarium-oriented plan with minimal processes/stores and interactions.
        Include a rough complexity score and flags if exceeding the budget.
        """
        schema = {
            "entities": {"cell_types": ["string"], "molecules": ["string"]},
            "interactions": ["string"],
            "vivarium": {"processes": ["string"], "stores": ["string"], "topology_notes": ["string"]},
            "parameters": [{"name": "string", "value": "number|string", "source": "string|unknown"}],
            "simplifications": ["string"],
            "complexity_score": "number",
            "exceeds_budget": "boolean",
            "notes": ["string"],
        }
        # Safely summarize requirements whether it's a dict or a list
        def _summarize_requirements(req):
            try:
                if isinstance(req, dict):
                    return {k: req.get(k) for k in ["cell_types", "interactions", "molecules"]}
                elif isinstance(req, list):
                    return {"interactions": req[: min(20, len(req))]}
                else:
                    return {"raw_type": type(req).__name__}
            except Exception:
                return {"error": "could_not_parse_requirements"}

        messages = [
            {"role": "system", "content": self.system_prompt + " Output JSON only."},
            {"role": "user", "content": (
                "Goal: Propose the smallest viable Vivarium model to test the premise with minimal interactions and entities.\n"
                "Premise: " + premise + "\n"
                "Constraint: Keep interaction count as low as possible"
                + (f" (<= {max_interactions} interactions)." if max_interactions else ".") + "\n"
                "Schema (JSON): " + str(schema)
            )},
            {"role": "user", "content": "Biologist proposal:\n" + str(bio_eval)},
            {"role": "user", "content": "Modelability assessment:\n" + str(exp_eval)},
            {"role": "user", "content": "Requirements snippet:\n" + str(_summarize_requirements(requirements))},
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