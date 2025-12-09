import os
import json
from agents.code_understanding_agents.dependency_predictor_agent import DependencyPredictorAgent
from agents.code_understanding_agents.structure_agent import StructureAnalyzerAgent
from agents.code_understanding_agents.template_mapping_agent import TemplateMappingAgent

def compare_dep_graphs_with_prediction(input_folder, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Analyze folder and build the actual dependency graph
    structure_agent = StructureAnalyzerAgent()
    structure_agent.analyze_folder(input_folder)
    dep_graph = structure_agent.file_deps
    dep_graph_path = os.path.join(output_dir, "dependency_graph.json")
    with open(dep_graph_path, "w") as f:
        json.dump(dep_graph, f, indent=2)
    print(f"Dependency graph saved to {dep_graph_path}")

    # 2. Predict new dependency graph from actual dependency graph
        # --- Influenza Model Specific Setup for DependencyPredictorAgent ---
    influenza_title = "Influenza Infection Model Dependency Architect"
    influenza_expertise = (
        "Expert in agent-based modeling for viral infection, computational immunology, "
        "and Python project architecture, with a specialty in analyzing and reconstructing "
        "code dependency graphs. Highly experienced in designing simulation frameworks "
        "where core infrastructure (utility, setup, analysis, and configuration files) is reused unchanged, "
        "while biological model files (cells/virus) are adapted or newly created."
    )
    influenza_goal = (
        "To generate a new, accurate dependency graph for a simulation project modeling human influenza A infection. "
        "In the new graph, all utility, infrastructure, setup, analysis, and miscellaneous support files must be "
        "PRESERVED as-is from the old project. "
        "Only files corresponding to the biological agents (cell types and virus) may be new or edited. "
        "No other files should be added, removed, or changed."
    )
    influenza_role_for_base_agent = (
        "You are a senior computational virologist and Python software architect. "
        "Your job is to meticulously adapt an existing project’s dependency graph for a human influenza A agent-based simulation."
        "Strictly reuse all utility, setup, analysis, runner, configuration, and support scripts as they appear in the old project. "
        "Do NOT create, rename, or edit these files."
        "Only cell type and virus agent files may be changed or newly added, "
        "reflecting the new biological focus (NK cells, CD4+ T cells, CD8+ T cells, B cells, Macrophages, Dendritic Cells, Lung Epithelial Cells, virus)."
        "Your output is a new dependency graph in which all non-biology-specific files match the old graph exactly (both filename and dependency links)."
    )

    influenza_prompt_template = (
        "Given the OLD DEPENDENCY GRAPH below, your task is to produce a NEW dependency graph"
        "for an agent-based simulation of human influenza A infection."
        "CRITICAL INSTRUCTIONS:"
        "- All utility, infrastructure, setup, data processing, runner, configuration, and analysis files"
        "(such as local_field.py, snapshot.py, location.py, math_utils.py, setup.py, __init__.py, config.yaml, etc.)"
        "MUST be preserved exactly as they appear in the old graph—including their dependencies and filenames."
        "- ONLY files directly modeling the biological agents (NK cells, CD4+ T cells, CD8+ T cells,"
        "B cells, Macrophages, Dendritic Cells, Lung Epithelial Cells, and virus) can be new or changed."
        "- Do not add, remove, or rename any other files. Do not generate any code. Do not modify the"
        "dependency structure of utility/support files."
        "Requirements to reflect in the new graph:"
        "1. Cellular Players: NK cells, CD4+ T cells, CD8+ T cells, B cells, Macrophages, Dendritic Cells,"
        "Lung Epithelial Cells (infected and uninfected), and viruses."
        "2. All supporting infrastructure files from the old graph must be reused unchanged."
        "- The new dependency graph should be a JSON mapping from filename to a list of its direct dependencies."
        "- Arrange keys (filenames) in valid topological order if possible (dependencies before dependents)."
        "- Output ONLY the new dependency graph as a valid JSON object. No explanations, markdown, or extra text."
        "OLD DEPENDENCY GRAPH:\n{old_dep_graph_json}"
    )

    dep_predictor = DependencyPredictorAgent(
        title=influenza_title,
        expertise=influenza_expertise,
        goal=influenza_goal,
        role_for_base_agent=influenza_role_for_base_agent,
        prompt_template_override=influenza_prompt_template
        # model="gpt-4.1" # Or your preferred model
    )

    influenza_requirements_description = (
        "The project aims to simulate the early innate and adaptive immune response to Influenza A virus infection "
        "in the human respiratory tract. Focus on the interactions between epithelial cells, NK cells, macrophages, "
        "dendritic cells, T cells (CD4+ and CD8+), and B cells. Model virus replication within epithelial cells and "
        "its spread. Include key antiviral cytokines like IFN-alpha/beta and IFN-gamma, and pro-inflammatory "
        "cytokines such as TNF-alpha and IL-6. The simulation should track cell populations (including differentiation "
        "and activation states where appropriate), virus load, and cytokine concentrations over time. Ensure all "
        "necessary helper functions for the simulation (e.g., grid management, agent scheduling, parameter loading, "
        "ODE solvers if used for intracellular models) and analysis scripts (e.g., for plotting time courses, "
        "spatial visualization, endpoint statistical analysis) are included in the dependency graph."
    )
    
    predicted_dep_graph_str = dep_predictor.predict_new_dependencies(
        dep_graph,
        requirements_description=influenza_requirements_description
    )
    predicted_dep_graph_path = os.path.join(output_dir, "predicted_dependency_graph.json")
    with open(predicted_dep_graph_path, "w") as f:
        f.write(predicted_dep_graph_str)
    print(f"Predicted dependency graph saved to {predicted_dep_graph_path}")

    # 3. Parse predicted dependency graph
    try:
        predicted_dep_graph = json.loads(predicted_dep_graph_str)
    except Exception as e:
        print("Error parsing predicted dependency graph:", e)
        predicted_dep_graph = {}

    # 4. Map each predicted file to its closest actual file using TemplateMappingAgent
    template_mapper = TemplateMappingAgent(dep_graph) # treat dep_graph as "old"
    template_mapper.new_dep_graph = predicted_dep_graph
    predicted_file_list = list(predicted_dep_graph.keys())
    file_mapping = template_mapper.map_all(predicted_file_list)

    # 5. Also map each actual file to its closest predicted file (optional, reverse mapping)
    reverse_mapper = TemplateMappingAgent(predicted_dep_graph) # treat predicted as "old"
    reverse_mapper.new_dep_graph = dep_graph
    actual_file_list = list(dep_graph.keys())
    reverse_mapping = reverse_mapper.map_all(actual_file_list)

    # 6. Save results
    mapping_results_path = os.path.join(output_dir, "predicted_to_actual_mapping.json")
    with open(mapping_results_path, "w") as f:
        json.dump(file_mapping, f, indent=2)
    reverse_mapping_results_path = os.path.join(output_dir, "actual_to_predicted_mapping.json")
    with open(reverse_mapping_results_path, "w") as f:
        json.dump(reverse_mapping, f, indent=2)
    print(f"Predicted-to-actual mapping saved to {mapping_results_path}")
    print(f"Actual-to-predicted mapping saved to {reverse_mapping_results_path}")

    # 7. Print mapping results
    print("\nPredicted → Actual File Mapping:")
    for pred_file, actual_file in file_mapping.items():
        print(f"Predicted: {pred_file} --> Actual: {actual_file}")
    print("\nActual → Predicted File Mapping:")
    for actual_file, pred_file in reverse_mapping.items():
        print(f"Actual: {actual_file} --> Predicted: {pred_file}")

if __name__ == "__main__":
    input_folder = "/home/labuser/Desktop/lab_member_projects/Bobby_Ni/tumor-tcell/tumor_tcell"
    # Define the base output directory
    base_output_dir = "/home/labuser/Desktop/lab_member_projects/Bobby_Ni/tumor-tcell-exp"
    # Define the specific sub-folder for this script's output
    analysis_output_folder_name = "dependency_analysis_output"
    output_dir = os.path.join(base_output_dir, analysis_output_folder_name)
    
    # Ensure the specific output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Using Input Folder: {input_folder}")
    print(f"Using Output Directory: {output_dir}")
    
    compare_dep_graphs_with_prediction(input_folder, output_dir)
