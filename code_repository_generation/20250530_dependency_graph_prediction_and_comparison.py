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
    dep_predictor = DependencyPredictorAgent()
    predicted_dep_graph_str = dep_predictor.predict_new_dependencies(dep_graph)
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
