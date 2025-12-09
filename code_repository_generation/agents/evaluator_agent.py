from ..base_agent import BaseAgent
from ...memory.unified_chroma_memory_manager import UnifiedChromaMemoryManager
from typing import Optional, Dict, Any
import os

class EvaluatorAgent(BaseAgent):
    """
    Agent for evaluating simulation outcomes and providing structured feedback.
    Capable of handling text, code, images, and other artifact types.
    """
    def __init__(self, 
                 model: str = "gpt-4.1-nano", 
                 output_dir: Optional[str] = None,
                 memory_manager: Optional[UnifiedChromaMemoryManager] = None):
        super().__init__(
            title="Evaluator Agent",
            expertise="simulation evaluation",
            goal="Judge simulation outcomes and suggest improvements",
            role="Analyze simulation results and provide feedback",
            model=model
        )
        self.memory_manager = memory_manager if memory_manager else UnifiedChromaMemoryManager(
            db_dir="/home/labuser/Desktop/lab_member_projects/Bobby_Ni/agent-testing-framework-output/chroma_memory/evaluator_agent_memory"
        )
        self.last_summary = ""
        self.output_dir = output_dir or "./outputs"

    def evaluate_results(self, 
                        outputs: Dict[str, Any], 
                        expected_goal: str, 
                        code: Optional[str] = None) -> str:
        """
        Evaluate outputs from the simulation and suggest improvements.

        Args:
            outputs (Dict[str, Any]): Dict with keys for each artifact type, e.g., {'stdout': ..., 'images': [...], 'data': [...]}
            expected_goal (str): Description of the desired goal or success criteria.
            code (Optional[str]): (Optional) Code that produced the outputs.

        Returns:
            str: Evaluation feedback from the agent.
        """
        # Build a summary of all output artifacts
        output_summary = self.summarize_outputs(outputs)
        relevant_memories = self.memory_manager.retrieve_memories(str(outputs))
        memory_context = "\n".join(relevant_memories)
        code_section = f"\nCode:\n{code}\n" if code else ""
        prompt = self.construct_prompt(
            f"Simulation outputs summary:\n{output_summary}\nGoal: {expected_goal}\n"
            f"{code_section}"
            f"Relevant previous evaluations:\n{memory_context}\n"
            "Did the simulation achieve the goal? Explain and suggest improvements.\n"
            "If and only if the goal is fully achieved, include the word 'success' or 'achieved' in your evaluation."
        )
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": "Evaluate the above simulation results and all provided artifacts."}
            ]
        )
        evaluation = response.choices[0].message.content
        self.memory_manager.add_memory(f"Evaluation for outputs:\n{output_summary}\nFeedback:\n{evaluation}")
        self.last_summary = self.summarize_evaluation(output_summary, evaluation)
        return evaluation

    def summarize_outputs(self, outputs: Dict[str, Any]) -> str:
        """
        Summarize all types of output artifacts for inclusion in evaluation prompt.

        Args:
            outputs (Dict[str, Any]): Simulation outputs.

        Returns:
            str: Summary description of outputs.
        """
        summary_lines = []
        if "stdout" in outputs and outputs["stdout"]:
            summary_lines.append(f"Text output:\n{outputs['stdout']}\n")
        if "images" in outputs and outputs["images"]:
            summary_lines.append(f"Images produced: {', '.join([os.path.basename(img) for img in outputs['images']])}")
        if "data" in outputs and outputs["data"]:
            # You can enhance this to preview CSV, etc.
            summary_lines.append(f"Data files: {', '.join([os.path.basename(d) for d in outputs['data']])}")
        if "other" in outputs and outputs["other"]:
            summary_lines.append(f"Other artifacts: {', '.join([os.path.basename(o) for o in outputs['other']])}")
        return "\n".join(summary_lines) or "No outputs produced."

    def summarize_evaluation(self, outputs_summary: str, evaluation: str) -> str:
        """
        Use the LLM to summarize important findings and suggestions from the evaluation.

        Args:
            outputs_summary (str): The summary of simulation outputs.
            evaluation (str): The raw feedback from the evaluation.

        Returns:
            str: Concise summary of findings and suggestions.
        """
        prompt = (
            "Summarize the most important findings and suggestions from the following evaluation.\n\n"
            f"Simulation outputs:\n{outputs_summary}\n\nEvaluation:\n{evaluation}"
        )
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that summarizes evaluation feedback."},
                {"role": "user", "content": prompt}
            ]
        )
        summary = response.choices[0].message.content
        self.memory_manager.add_memory(f"Summary: {summary}")
        return summary
