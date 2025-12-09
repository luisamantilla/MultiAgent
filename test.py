#!/usr/bin/env python3
"""
summarize_tumortcell_structure.py

Walks through the "tumor-tcell" directory and prints a hierarchical summary
of all subdirectories and files.
"""

import os

def summarize_directory(root_path: str, indent: str = ""):
    """
    Recursively prints the directory structure of root_path.
    """
    try:
        entries = sorted(os.listdir(root_path))
    except FileNotFoundError:
        print(f"Error: Directory '{root_path}' does not exist.")
        return
    except PermissionError:
        print(f"Error: Permission denied for '{root_path}'.")
        return

    for entry in entries:
        full_path = os.path.join(root_path, entry)
        if os.path.isdir(full_path):
            print(f"{indent}{entry}/")
            summarize_directory(full_path, indent + "    ")
        else:
            print(f"{indent}{entry}")

if __name__ == "__main__":
    # Replace this path with the actual path to your tumor-tcell folder
    ModelBuild_folder = "ModelBuild"

    print(f"Directory summary for '{ModelBuild_folder}':\n")
    summarize_directory(ModelBuild_folder)
