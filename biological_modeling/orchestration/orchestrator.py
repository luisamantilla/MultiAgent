from typing import List, Dict
from ..experts.pi_agent import PIAgent
from ..experts.biological_expert_agent import BiologicalExpertAgent
from ..experts.experimental_expert_agent import ExperimentalExpertAgent
from ..experts.computational_expert_agent import ComputationalExpertAgent


def run_conversation(topic: str, model: str = "gpt-4o-mini", rounds: int = 2) -> List[Dict[str, str]]:
    pi = PIAgent(model=model)
    bio = BiologicalExpertAgent(model=model)
    exp = ExperimentalExpertAgent(model=model)
    comp = ComputationalExpertAgent(model=model)

    transcript: List[Dict[str, str]] = []
    # PI opens
    transcript.append({"speaker": "PI", "content": pi.open(topic)})

    for _ in range(rounds):
        bio_msg = bio.reply(topic, transcript)
        transcript.append({"speaker": "Biologist", "content": bio_msg})

        exp_msg = exp.reply(topic, transcript)
        transcript.append({"speaker": "Experimentalist", "content": exp_msg})

        comp_msg = comp.reply(topic, transcript)
        transcript.append({"speaker": "Computationalist", "content": comp_msg})

        pi_msg = pi.synthesize(topic, transcript)
        transcript.append({"speaker": "PI", "content": pi_msg})

    return transcript


def run_minimal_model_crosstalk(requirements: Dict, premise: str, model: str = "gpt-4o-mini", max_interactions: int | None = None) -> Dict:
    """Orchestrate a structured, single-pass crosstalk aimed at minimizing interaction count under compute constraints.
    Returns a dict with all agent JSON outputs and the PI final selection.
    """
    pi = PIAgent(model=model)
    bio = BiologicalExpertAgent(model=model)
    exp = ExperimentalExpertAgent(model=model)
    comp = ComputationalExpertAgent(model=model)

    bio_eval = bio.evaluate_requirements(requirements=requirements, premise=premise, max_interactions=max_interactions)
    exp_eval = exp.assess_modelability(requirements=requirements, premise=premise, bio_eval=bio_eval, max_interactions=max_interactions)
    comp_prop = comp.propose_vivarium_plan(requirements=requirements, premise=premise, bio_eval=bio_eval, exp_eval=exp_eval, max_interactions=max_interactions)
    final_sel = pi.finalize_vivarium_minimal_model(premise=premise, bio_eval=bio_eval, exp_eval=exp_eval, comp_proposal=comp_prop)

    return {
        "biologist": bio_eval,
        "experimentalist": exp_eval,
        "computationalist": comp_prop,
        "pi": final_sel,
    }