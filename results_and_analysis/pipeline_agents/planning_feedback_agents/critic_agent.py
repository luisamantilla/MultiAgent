from openai import OpenAI # Will be handled by BaseAgent
import os
from ...base_agent import BaseAgent # Added import for BaseAgent
from ....memory.unified_chroma_memory_manager import UnifiedChromaMemoryManager # Added import for UnifiedChromaMemoryManager

class CriticAgent(BaseAgent): # Inherit from BaseAgent
    """
    Reviews the analysis report, provides feedback, and can suggest or generate code edits for new experiments.
    """
    def __init__(self, model="gpt-4.1", memory_manager: UnifiedChromaMemoryManager = None, iteration: int = 0, goal: str = None, expertise: str = None, db_dir: str = None, collection_name: str = "critic_agent_memory"):
        title = "Critic Agent"
        resolved_expertise = expertise or "Computational Biology, Code Review, Experiment Design"
        resolved_goal = goal or "Critique analysis, suggest improvements, and generate new code for experiments."
        
        super().__init__( # Call BaseAgent's __init__
            title=title,
            expertise=resolved_expertise,
            goal=resolved_goal,
            role="Analysis Critic and Experiment Design Advisor", # Added a role
            model=model
        )
        # self.client is initialized by BaseAgent
        if memory_manager is None:
            default_db_dir = db_dir or os.path.expanduser("~/.agent_framework_memory/critic_agent_chroma")
            os.makedirs(default_db_dir, exist_ok=True)
            self.memory_manager = UnifiedChromaMemoryManager(
                db_dir=default_db_dir,
                collection_name=collection_name
            )
        else:
            self.memory_manager = memory_manager
        self.iteration = iteration

    def critique_and_edit(self, analysis_report, original_code_path, output_code_path=None, custom_instruction=None):
        """
        Review the analysis report, provide feedback, and suggest or generate code edits for a new experiment.
        analysis_report: The full analysis report from the AnalysisAgent (long summary).
        original_code_path: Path to the original experiment code to be critiqued/edited.
        output_code_path: Where to save the new/edited code (optional).
        custom_instruction: Optional extra prompt for the LLM.
        Returns: (critique_feedback, new_code)
        """
        with open(original_code_path, "r") as f:
            original_code = f.read()
        
        # Specific instructions regarding data saving and analysis
        # Ensure this is passed effectively into the main prompt.
        data_handling_instructions = """
        IMPORTANT DATA HANDLING FOR GENERATED SCRIPT:
        1.  The script must save raw simulation data (e.g., using `pickle.dump`).
        2.  It MUST call the `individual_analysis` function for EACH simulation run within a batch to generate plots/CSVs.
        3.  All outputs for EACH run must be saved in a UNIQUE subdirectory. If sweeping a parameter, include the parameter value in the subdirectory name (e.g., `output_base_dir/experiment_name_param_value/exp_id_param_value/`).
            The `outdir` for `pickle.dump` and `analysis_dir` for `individual_analysis` must be correctly constructed for each run.
            The `experiment_name` and `exp_id` should also reflect the specific parameters of each run.
        """

        # New prompt focusing on creating a batch script based on analysis
        prompt = (
            f"You are an expert computational biology code reviewer and experiment designer.\\n"
            f"Your task is to analyze the provided 'Analysis report' (which includes summaries of previous simulation outputs) and the 'Original code'.\\n"
            f"Based on this, generate a NEW Python script that performs a parameter sweep for 1-2 key parameters identified as interesting from the analysis or original code structure.\\n"
            f"The new script should iterate through a list of values for the chosen parameter(s) and run the simulation for each value.\\n\\n"
            
            f"REQUIREMENTS FOR THE GENERATED SCRIPT:\\n"
            f"- It MUST be a single, complete, runnable Python script including all necessary imports.\\n"
            f"- It MUST accept a command-line argument `--output_base_dir` for the top-level output directory.\\n"
            f"- For each simulation run in the sweep, it must create unique subdirectories under `--output_base_dir` incorporating the parameter values being tested to avoid overwriting data (e.g., `output_base_dir/experiment_name_paramX_valueA/exp_id_paramX_valueA/`).\\n"
            f"- It MUST call `individual_analysis` for each simulation run, saving its outputs into the correct unique subdirectory.\\n"
            f"- It MUST save the raw simulation data (e.g., pickle dump) for each run into its unique subdirectory.\\n"
            f"- Identify a sensible small set of values for the sweep (e.g., 3-4 values). For example, if original `n_tcells` was 12, a sweep could be `[6, 12, 24]` or similar.\\n"
            f"- The script should print a clear message indicating the start and end of each simulation run within the sweep, and where its outputs are saved.\\n\\n"
            
            f"{data_handling_instructions}\\n\\n"
            
            f"Analysis report (contains summaries from SummarizerAgent):\n{analysis_report}\n\\n"
            f"---\nOriginal code (for context of structure, typical parameters, and functions like `tumor_tcell_abm` and `individual_analysis`):\n{original_code}\n\\n"
            f"---\n"
            f"Provide ONLY the full new Python script within a single ```python ... ``` block. Do not include any other explanatory text before or after the code block."
        )

        if custom_instruction: # Allow override or prepend if needed, though the new prompt is quite specific
            prompt = custom_instruction + "\n" + prompt
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10000 # Increased max_tokens for more complete script generation
        )
        output = response.choices[0].message.content
        # Try to extract code block if present
        new_code = None
        if '```python' in output:
            try:
                new_code = output.split('```python')[1].split('```')[0].strip()
            except Exception:
                new_code = None
        # Save new code if requested
        if new_code and output_code_path:
            with open(output_code_path, "w") as f:
                f.write(new_code)
        # Save short summary to memory if manager is provided
        if self.memory_manager:
            summary_prompt = (
                f"Summarize the following critique and code suggestion in ≤512 tokens (about 2000 characters). "
                f"Highlight the main feedback and any code changes.\n\n---\n{output}\n---\nSummary (≤512 tokens):"
            )
            summary_response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": summary_prompt}],
                max_tokens=512
            )
            short_summary = summary_response.choices[0].message.content[:2000]
            # Ensure this uses self.memory_manager.write_memory
            self.memory_manager.write_memory(
                iteration=self.iteration, 
                agent=self.title, 
                summary=short_summary,
                output_file=output_code_path or "" 
            )
        return output, new_code
