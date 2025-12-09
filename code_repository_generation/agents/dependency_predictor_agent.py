from agents.base_agent import BaseAgent
import json
from typing import Dict, List, Optional, Any

class DependencyPredictorAgent(BaseAgent):
    # --- Defaults ---
    DEFAULT_TITLE = "Dependency Graph Predictor"
    DEFAULT_EXPERTISE = (
        "Expert in Python project dependency analysis and restructuring codebases based on requirements."
    )
    DEFAULT_GOAL = (
        "Produce a new dependency graph based on an old one and specified requirements."
    )
    DEFAULT_ROLE_FOR_BASE_AGENT = ( # Descriptive role for BaseAgent
        "Python architect specializing in dependency graph generation."
    )
    # DEFAULT_PERSONA_FOR_LLM removed
    DEFAULT_PROMPT_TEMPLATE = (
        "Given the OLD DEPENDENCY GRAPH (see below), produce a NEW dependency graph that reflects the "
        "following requirements: {requirements_description}\n\n"
        "The new dependency graph should include all original utility, setup, analysis, experiment, "
        "and miscellaneous files (e.g., setup.py, __init__.py, plotting scripts, data processing, "
        "configuration files) that are required to run, analyze, or support the project, "
        "even if not directly mentioned in the new requirements, unless explicitly told to exclude them or "
        "they are clearly irrelevant to the new requirements.\n\n"
        "The new dependency graph should be a JSON mapping from filename to a list of dependencies. "
        "Keys (filenames) should be in a valid topological order if possible.\n"
        "Output ONLY the new dependency graph as a valid JSON object, with no explanation, markdown, or commentary.\n"
        "You do NOT need to generate any code yet—just the new dependency structure.\n\n"
        "OLD DEPENDENCY GRAPH:\n{old_dep_graph_json}"
    )

    def __init__(self, 
                 model: str = "gpt-4.1",
                 title: Optional[str] = None,
                 expertise: Optional[str] = None,
                 goal: Optional[str] = None,
                 role_for_base_agent: Optional[str] = None,
                 # persona_for_llm: Optional[str] = None, # Removed
                 prompt_template_override: Optional[str] = None):
        super().__init__(
            title=title or self.DEFAULT_TITLE,
            expertise=expertise or self.DEFAULT_EXPERTISE,
            goal=goal or self.DEFAULT_GOAL,
            role=role_for_base_agent or self.DEFAULT_ROLE_FOR_BASE_AGENT,
            model=model
        )
        # self.actual_persona_for_llm = persona_for_llm or self.DEFAULT_PERSONA_FOR_LLM # Removed
        self.prompt_template_to_use = prompt_template_override or self.DEFAULT_PROMPT_TEMPLATE


    def predict_new_dependencies(self, 
                                 old_dep_graph: Dict[str, List[str]], 
                                 requirements_description: str) -> str:
        """
        Predicts a new dependency graph based on an old one and a description of requirements.

        Args:
            old_dep_graph: The existing dependency graph.
            requirements_description: A description of the changes or focus for the new graph.
            system_persona_override: Optional override for the LLM's persona for this specific call.
        
        Returns:
            A string containing the predicted dependency graph in JSON format.
        """
        
        prompt_format_kwargs = {
            "requirements_description": requirements_description,
            "old_dep_graph_json": json.dumps(old_dep_graph, indent=2)
        }

        try:
            user_prompt_content = self.prompt_template_to_use.format(**prompt_format_kwargs)
        except KeyError as e:
            raise ValueError(
                f"Missing a required formatting argument for the prompt template: {e}. "
                f"The template expects 'requirements_description' and 'old_dep_graph_json'. "
                f"Provided kwargs for formatting: {list(prompt_format_kwargs.keys())}"
            ) from e
        
        system_message_content = (
            "You are a helpful assistant. You are precise and only output what is requested. Specifically, you must output ONLY a valid JSON object "
            "representing the new dependency graph, with no explanations, markdown, or any other text."
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_message_content},
                {"role": "user", "content": user_prompt_content}
            ]
        )
        return response.choices[0].message.content
