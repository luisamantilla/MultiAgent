import os
import json
from agents.code_understanding_agents.code_generation_agent import CodeGenerationAgent
from memory.unified_chroma_memory_manager import UnifiedChromaMemoryManager


def generate_project_from_mapping(predicted_to_actual_path, predicted_dep_graph_path, old_project_root, new_project_root, memory_db_dir=None, collection_name="project_generation_memory"):
    os.makedirs(new_project_root, exist_ok=True)
    with open(predicted_to_actual_path) as f:
        predicted_to_actual = json.load(f)
    with open(predicted_dep_graph_path) as f:
        predicted_dep_graph = json.load(f)

    # Initialize UnifiedChromaMemoryManager
    memory_manager = None
    if memory_db_dir:
        memory_manager = UnifiedChromaMemoryManager(db_dir=memory_db_dir, collection_name=collection_name)

    code_gen_agent = CodeGenerationAgent(
        title="Code Generator",
        expertise="Expert in code adaptation and simulation frameworks"
    )

    for new_file in predicted_dep_graph.keys():
        old_file = predicted_to_actual.get(new_file)
        dependencies = predicted_dep_graph[new_file]
        old_content = None
        old_file_path = os.path.join(old_project_root, old_file) if old_file else None
        if old_file_path and os.path.exists(old_file_path):
            with open(old_file_path, "r") as f:
                old_content = f.read()
        
        # Add a summary to memory for this file using UnifiedChromaMemoryManager
        if memory_manager and old_content:
            # UnifiedChromaMemoryManager expects iteration, agent name, summary, and optional output_file
            # For this context, we might not have a formal iteration or agent name in the same way a pipeline does.
            # We'll use a generic agent name and iteration 0, or adapt as needed.
            summary_text = f"Content summary for old file: {old_file}\n{old_content[:500]}..."
            memory_manager.write_memory(
                iteration=0, # Or a relevant iteration/step counter if available
                agent="ProjectGenerator", 
                summary=summary_text,
                output_file=old_file_path 
            )
            
        # Generate new file content using the code generation agent
        # The agent's generate method might need to be aware of how to use UnifiedChromaMemoryManager
        # or it might expect memories to be passed in a specific format.
        # For now, assuming the agent's generate method can take the memory_manager directly
        # or that relevant memories are fetched and passed.
        
        # Example: Fetching relevant memories for the agent (if needed by its generate method)
        relevant_memories = []
        if memory_manager:
            # This is a placeholder for how you might query memories.
            # The actual query would depend on what context the CodeGenerationAgent needs.
            # For instance, it might search for memories related to dependencies.
            for dep in dependencies:
                relevant_memories.extend(memory_manager.semantic_search(query=f"dependency: {dep}", top_k=1))
            # Or just pass the manager and let the agent handle it
            
        new_content = code_gen_agent.generate(
            new_filename=new_file,
            old_content=old_content,
            dependencies=dependencies,
            # Pass memory_manager or fetched memories to the agent
            # This depends on how CodeGenerationAgent is implemented
            memory=memory_manager # or relevant_memories
        )
        new_file_path = os.path.join(new_project_root, new_file)
        os.makedirs(os.path.dirname(new_file_path), exist_ok=True)
        with open(new_file_path, "w") as f:
            f.write(new_content)
        print(f"Generated: {new_file_path}")
    print(f"\nAll files generated in {new_project_root}")


if __name__ == "__main__":
    import sys
    import argparse
    parser = argparse.ArgumentParser(description="Generate new project code from mapping and dependency graph.")
    parser.add_argument("--predicted_to_actual", required=True, help="Path to predicted_to_actual_mapping.json")
    parser.add_argument("--predicted_dep_graph", required=True, help="Path to predicted_dependency_graph.json")
    parser.add_argument("--old_project_root", required=True, help="Path to old project root directory")
    parser.add_argument("--new_project_root", required=True, help="Path to new project root directory")
    parser.add_argument("--memory_db_dir", default=None, help="(Optional) Path to Chroma memory DB directory")
    args = parser.parse_args()

    generate_project_from_mapping(
        predicted_to_actual_path=args.predicted_to_actual,
        predicted_dep_graph_path=args.predicted_dep_graph,
        old_project_root=args.old_project_root,
        new_project_root=args.new_project_root,
        memory_db_dir=args.memory_db_dir,
    )
