import os
import re
from typing import List

def save_generated_files(generated_code: str, output_dir: str) -> List[str]:
    """
    Saves generated multi-file code outputs to disk.
    Splits code blocks based on `# file: filename.ext` comments.

    Args:
        generated_code (str): The code as a string, with file markers.
        output_dir (str): Where to save the files.

    Returns:
        List[str]: List of saved file paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    # Matches: # file: filename.ext\n<content>
    file_blocks = re.split(r'# file: (.+)', generated_code)
    files = []
    for i in range(1, len(file_blocks), 2):
        filename = file_blocks[i].strip()
        code = file_blocks[i+1].strip()
        file_path = os.path.join(output_dir, filename)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)  # Ensure parent dirs exist
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(code)
        files.append(file_path)
    return files

def get_new_files(directory: str, before_files: set) -> List[str]:
    """
    Return a list of new files in `directory` compared to the `before_files` set.
    """
    after_files = set(os.listdir(directory))
    return [os.path.join(directory, f) for f in after_files - before_files]