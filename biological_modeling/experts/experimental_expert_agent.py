from typing import List, Dict
from pathlib import Path
import sys

# Add the shared directory to Python path to import BaseAgent
_PROJECT_ROOT = Path(__file__).resolve().parents[2]  # Go up to MultiAgent/
_SHARED_DIR = _PROJECT_ROOT / "shared"
if str(_SHARED_DIR) not in sys.path:
    sys.path.insert(0, str(_SHARED_DIR))

from base.base_agent import BaseAgent


class ExperimentalExpertAgent(BaseAgent):
    """LLM-backed experimental design expert with concise suggestions."""

    def __init__(self, model: str = "gpt-4o-mini"):
        super().__init__(
            title="Experimentalist",
            expertise="Experimental design, controls, measurable readouts",
            goal="Propose practical assays and controls",
            role="Experimental design expert",
            model=model,
        )
        self.system_prompt = (
            "You are a concise experimentalist. Suggest practical assays, controls, and measurable readouts "
            "in 2-3 sentences. Prefer straightforward setups."
        )

    def reply(self, topic: str, history: List[Dict[str, str]]) -> str:
        messages = [{"role": "system", "content": self.system_prompt}]
        for m in history[-6:]:
            messages.append({"role": "user", "content": f"{m['speaker']}: {m['content']}"})
        messages.append({"role": "user", "content": f"Experimentalist: Propose a simple assay for: {topic} (2-3 sentences)."})
        resp = self.client.chat.completions.create(model=self.model, messages=messages, temperature=0.5)
        return resp.choices[0].message.content.strip()

    def assess_modelability(self, requirements: Dict, premise: str, bio_eval: Dict, max_interactions: int | None = None) -> Dict:
        """Return JSON-only assessment of modelability: feasibility, simplifications, and keep/drop under compute limits."""
        schema = {
            "feasibility_flags": [{"name": "string", "issue": "string"}],
            "simplifications": ["string"],
            "keep_interactions": ["string"],
            "drop_interactions": [{"name": "string", "reason": "string"}],
            "parameter_availability": [{"name": "string", "source": "string|unknown"}],
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
                "Goal: Determine what is straightforward to model and what to drop to meet compute constraints.\n"
                "Premise: " + premise + "\n"
                "Constraint: Keep interaction count as low as possible"
                + (f" (<= {max_interactions} interactions)." if max_interactions else ".") + "\n"
                "Schema (JSON): " + str(schema)
            )},
            {"role": "user", "content": "Biologist selection:\n" + str(bio_eval)},
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