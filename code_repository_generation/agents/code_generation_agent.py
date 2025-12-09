from ..base_agent import BaseAgent  # Corrected import
from ...memory.unified_chroma_memory_manager import UnifiedChromaMemoryManager  # Corrected import
from ...utils.file_utils import save_generated_files  # Corrected import
from typing import List, Optional
import re

class CodeGenerationAgent(BaseAgent):
    """
    Agent responsible for generating code files based on a provided task.
    Can handle single or multi-file projects, and is designed for extensibility.
    """
    def __init__(self, 
                 title: str, 
                 expertise: str, 
                 model: str = "gpt-4.1-nano", 
                 output_dir: Optional[str] = None,
                 memory_manager: Optional[UnifiedChromaMemoryManager] = None):  # Added memory_manager
        super().__init__(
            title, 
            expertise, 
            goal="Generate code files based on specified criteria", 
            role="Create and manage code generation tasks", 
            model=model
        )
        # Updated to use UnifiedChromaMemoryManager
        self.memory_manager = memory_manager if memory_manager else UnifiedChromaMemoryManager(
            db_dir="/home/labuser/Desktop/lab_member_projects/Bobby_Ni/agent-testing-framework-output/chroma_memory_unified",
            collection_name="code_generation_memory"
        )
        self.last_summary = ""
        self.output_dir = output_dir or "./outputs"

    def generate_code(self, new_filename: str, dependencies: list, old_filename: str = None, old_content: str = None, memory_context: str = "", extra_instructions: str = "") -> str:
        """
        Generate code for a new file, given dependencies, old file info, and memory context.
        
        Args:
            new_filename (str): The name of the new file to generate.
            dependencies (list): List of intended dependencies for the new file.
            old_filename (str, optional): The old file this is adapted from.
            old_content (str, optional): The content of the old file.
            memory_context (str, optional): Project or memory context.
            extra_instructions (str, optional): Any extra instructions for the LLM.
        
        Returns:
            str: The raw generated code string, with file marker.
        """
        prompt = self.construct_prompt(
            f"You are an expert code generation agent.\n"
            f"You are generating a new file: {new_filename}\n"
            f"- Intended dependencies for this file: {dependencies}\n"
            + (f"- This file is adapted from: {old_filename}\n" if old_filename else "")
            + (f"- Old file content (if available):\n{old_content}\n" if old_content else "")
            + (f"\nProject context:\n{memory_context}\n" if memory_context else "")
            + (f"\nAdditional instructions:\n{extra_instructions}\n" if extra_instructions else "") +
            "\nInstructions:\n"
            "- Write the complete code for {new_filename}, adapted as needed for the new dependencies and project context.\n"
            "- Output ONLY the code, starting with a file marker: '# file: {new_filename}'\n"
            "- Do NOT include markdown formatting, explanations, or extra comments outside the code.\n"
            "- If the file is a module, include docstrings and type hints.\n"
            "- If the file is an entry point, include a main guard and example usage.\n"
            "- If you generate multiple files, use '# file: filename.py' for each, and '# main: main_filename.py' at the top if needed.\n"
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"Please write the code for {new_filename}."}
            ]
        )
        code = response.choices[0].message.content
        # Updated to use self.memory_manager.add_memory
        self.memory_manager.add_memory(f"Generated code for: {new_filename}\\n{code}", metadata={"filename": new_filename, "agent": self.title})
        # New: Parse the main file and save to memory
        main_file_match = re.search(r'#\s*main:\s*([^\s]+)', code)
        if main_file_match:
            main_file_name = main_file_match.group(1).strip()
            # Updated to use self.memory_manager.add_memory
            self.memory_manager.add_memory(f"Main file: {main_file_name}", metadata={"main_file": main_file_name, "agent": self.title})

        self.last_summary = self.summarize_change(new_filename, code)
        return code

    def save_generated_files(self, generated_code: str) -> List[str]:
        """
        Save generated code files to disk using the utility function.

        Args:
            generated_code (str): The code with file markers.

        Returns:
            List[str]: List of saved file paths.
        """
        return save_generated_files(generated_code, self.output_dir)

    def summarize_change(self, task_instruction: str, code: str) -> str:
        """
        Use LLM to summarize the changes made by the generated code.

        Args:
            task_instruction (str): The original task.
            code (str): The generated code.

        Returns:
            str: Summary of important changes.
        """
        prompt = f"Summarize the important changes made in the following code for the task: {task_instruction}\n\n{code}"
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that summarizes code changes."},
                {"role": "user", "content": prompt}
            ]
        )
        summary = response.choices[0].message.content
        self.memory.add(f"Summary: {summary}")
        return summary
