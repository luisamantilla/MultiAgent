\
import os
import json
import re # Added for extracting code
import shutil # Added for directory cleanup
from ..agents.pipeline_agents.analysis_agents.summarizer_agent import SummarizerAgent
from ..agents.pipeline_agents.analysis_agents.result_analysis_agent import AnalysisAgent
from ..agents.pipeline_agents.planning_feedback_agents.critic_agent import CriticAgent
from ..agents.core_utility_agents.code_execution_agent import CodeExecutionAgent
from ..memory.unified_chroma_memory_manager import UnifiedChromaMemoryManager

# --- Config ---
# Path to the simulation script within the MultiAgent GitHub repo
simulation_script_path = "/home/labuser/Desktop/lab_member_projects/Bobby_Ni/MultiAgent/agent-testing-framework/src/simulations/abm_scripts/abm_simulation.py"
iteration = 1 # Define iteration number
DOCKER_IMAGE_NAME = "code-execution-env" # Define Docker image name

# --- Define Base Directories ---
# 1. Local base directory for non-coding outputs (within tumor-tcell-exp)
# This is also where the main simulation script will save its artifacts on the host.
local_outputs_base_dir = "/home/labuser/Desktop/lab_member_projects/Bobby_Ni/tumor-tcell-exp/simulation_runs"

# 2. Base directory for generated code (within MultiAgent GitHub repo)
generated_code_base_dir = "/home/labuser/Desktop/lab_member_projects/Bobby_Ni/MultiAgent/agent-testing-framework/src/simulations/generated_experiments"

# 3. Base directory for agent memories (local within tumor-tcell-exp)
local_memory_db_base_dir = "/home/labuser/Desktop/lab_member_projects/Bobby_Ni/tumor-tcell-exp/agent_memories_db"

# --- Iteration-Specific Output Paths ---

# Non-coding outputs (manifest, summaries, reports, logs) for the initial simulation
iteration_local_output_dir = os.path.join(
    local_outputs_base_dir,
    f"iteration_{iteration}",
    "data_modeling"
)
# Clean up the directory for initial simulation outputs
if os.path.exists(iteration_local_output_dir):
    shutil.rmtree(iteration_local_output_dir)
os.makedirs(iteration_local_output_dir, exist_ok=True)

# Generated code output (for critic's script)
iteration_generated_code_dir = os.path.join(
    generated_code_base_dir,
    f"iteration_{iteration}"
)
os.makedirs(iteration_generated_code_dir, exist_ok=True)
critic_code_path = os.path.join(iteration_generated_code_dir, "suggested_experiment.py")


# Paths for CodeExecutionAgent's own internal logs/manifest (if any)
# This is for the agent's own operational logs, not the script's outputs.
code_execution_agent_internal_output_dir = os.path.join(
    local_outputs_base_dir, 
    f"iteration_{iteration}",
    "agent_internals", 
    "code_execution_agent"
)
os.makedirs(code_execution_agent_internal_output_dir, exist_ok=True)

# Manifest, summary, report, and conversation log for the initial simulation
manifest_path = os.path.join(iteration_local_output_dir, "run_manifest.json") # Default, might be overwritten by CodeExecutionAgent
data_summary_path = os.path.join(iteration_local_output_dir, "data_summary.json")
analysis_report_path = os.path.join(iteration_local_output_dir, "analysis_report.md")
conversation_log = os.path.join(iteration_local_output_dir, "multiagent_conversation.json")

# --- Memory manager ---
memory_manager_db_path = os.path.join(local_memory_db_base_dir, f"iteration_{iteration}_chroma")
os.makedirs(memory_manager_db_path, exist_ok=True) 
memory_manager = UnifiedChromaMemoryManager(
    db_dir=memory_manager_db_path, 
    collection_name=f"abm_pipeline_iter_{iteration}" 
)

# --- Initialize agents ---
summarizer = SummarizerAgent(model="gpt-4.1", memory_manager=memory_manager, iteration=iteration)
analysis = AnalysisAgent(model="gpt-4.1", memory_manager=memory_manager, iteration=iteration)
critic = CriticAgent(model="gpt-4.1", memory_manager=memory_manager, iteration=iteration)
code_executor = CodeExecutionAgent(
    docker_image_name=DOCKER_IMAGE_NAME,
    model="gpt-4.1", 
    memory_manager=memory_manager, 
    iteration=iteration, 
    output_dir=code_execution_agent_internal_output_dir 
)

# --- Conversation log ---
conversation = []
def log(role, content):
    conversation.append({"role": role, "content": content})
    # Ensure the directory for conversation_log exists (it should from above)
    os.makedirs(os.path.dirname(conversation_log), exist_ok=True)
    try:
        with open(conversation_log, "w") as f:
            json.dump(conversation, f, indent=2)
    except Exception as e:
        print(f"Error writing to conversation log {conversation_log}: {e}")


log("PipelineStart", f"Starting iteration {iteration} of ABM results pipeline.")

# --- 1. CodeExecutionAgent runs the initial simulation script in Docker ---
host_dir_for_simulation_artifacts = iteration_local_output_dir 
simulation_args_for_docker = [
    "--output_base_dir", "/app/outputs" # Internal container path
]

print(f"Attempting to run simulation script: {simulation_script_path}")
print(f"Host output directory for simulation artifacts: {host_dir_for_simulation_artifacts}")
print(f"Simulation arguments for Docker: {simulation_args_for_docker}")

exec_result = code_executor.execute_script_in_docker(
    script_file_path_on_host=simulation_script_path,
    script_args=simulation_args_for_docker,
    host_output_dir_for_script_artifacts=host_dir_for_simulation_artifacts,
)

actual_manifest_path_for_summarizer = None
if exec_result.get("success"):
    actual_manifest_path_for_summarizer = exec_result.get("manifest_path")
    if not actual_manifest_path_for_summarizer or not os.path.exists(actual_manifest_path_for_summarizer):
        error_msg = f"CodeExecutionAgent succeeded but manifest_path is missing or invalid: {actual_manifest_path_for_summarizer}. Expected in {host_dir_for_simulation_artifacts} or subdirs."
        log("CodeExecutionAgent_Error", error_msg)
        print(f"ERROR: {error_msg}")
        exit(1) 
    log("CodeExecutionAgent_SimulationRun", exec_result.get("message_to_analysis_agent", "No LLM summary provided by agent."))
    print("\\n=== CodeExecutionAgent (Simulation Run) Output ===\\n")
    print(f"STDOUT:\\n{exec_result.get('stdout', 'N/A')}")
    print(f"STDERR:\\n{exec_result.get('stderr', 'N/A')}")
    print(f"Manifest for simulation run at: {actual_manifest_path_for_summarizer}")
    print(f"LLM Summary: {exec_result.get('message_to_analysis_agent', 'N/A')}")
else:
    error_msg = f"CodeExecutionAgent failed to execute simulation script. Error: {exec_result.get('error')}. Stderr: {exec_result.get('stderr')}"
    log("CodeExecutionAgent_Error", error_msg)
    print(f"ERROR: {error_msg}")
    exit(1)

# --- 2. SummarizerAgent summarizes outputs ---
sum_path, sum_obj = summarizer.summarize_manifest(actual_manifest_path_for_summarizer, output_summary_path=data_summary_path)
log("SummarizerAgent", sum_obj["long_summary"])
print("\\n=== SummarizerAgent Output ===\\n")
print(sum_obj["long_summary"])

# --- 3. AnalysisAgent analyzes results ---
analysis_report_content, analysis_short_summary = analysis.perform_task(
    experiment_summary=exec_result.get("message_to_analysis_agent", "No summary from simulation execution."),
    textual_data_summary=sum_obj["long_summary"], 
    structured_file_summaries=sum_obj["file_summaries"], 
    output_path=analysis_report_path 
)
log("AnalysisAgent", analysis_report_content)
print("\\n=== AnalysisAgent Output ===\\n")
print(analysis_report_content)

# --- 4. CriticAgent critiques and suggests code ---
critic_feedback, new_code_generated_by_critic = critic.critique_and_edit(
    analysis_report=analysis_report_content, # Use the content string
    original_code_path=simulation_script_path, 
    output_code_path=critic_code_path 
)
log("CriticAgent", critic_feedback) # Log the full feedback
print("\\n=== CriticAgent Output ===\\n")
print(critic_feedback) # Print the full feedback

# --- 5. CodeExecutionAgent runs the script generated by CriticAgent ---
if new_code_generated_by_critic and critic_code_path and os.path.exists(critic_code_path):
    print(f"\\nAttempting to run new simulation script generated by CriticAgent: {critic_code_path}")
    
    critic_script_output_dir_name = f"iteration_{iteration}_critic_run"
    host_dir_for_critic_script_artifacts = os.path.join(
        local_outputs_base_dir, 
        critic_script_output_dir_name,
        "data_modeling" 
    )
    # Clean up the directory for critic script artifacts
    if os.path.exists(host_dir_for_critic_script_artifacts):
        shutil.rmtree(host_dir_for_critic_script_artifacts)
    os.makedirs(host_dir_for_critic_script_artifacts, exist_ok=True)
    print(f"Host output directory for critic script artifacts: {host_dir_for_critic_script_artifacts}")

    critic_script_args_for_docker = [
        "--output_base_dir", "/app/outputs" 
    ]
    print(f"Critic script arguments for Docker: {critic_script_args_for_docker}")

    critic_exec_result = code_executor.execute_script_in_docker(
        script_file_path_on_host=critic_code_path,
        script_args=critic_script_args_for_docker,
        host_output_dir_for_script_artifacts=host_dir_for_critic_script_artifacts
    )

    if critic_exec_result.get("success"):
        log("CodeExecutionAgent_CriticScriptRun", critic_exec_result.get("message_to_analysis_agent", "No LLM summary from critic script execution."))
        print("\\n=== CodeExecutionAgent (Critic Script Run) Output ===\\n")
        print(f"STDOUT:\\n{critic_exec_result.get('stdout', 'N/A')}")
        print(f"STDERR:\\n{critic_exec_result.get('stderr', 'N/A')}")
        if critic_exec_result.get("manifest_path"):
            print(f"Manifest for critic script run at: {critic_exec_result.get('manifest_path')}")
            # If we needed to analyze the critic's script outputs, we'd use this manifest.
        print(f"LLM Summary: {critic_exec_result.get('message_to_analysis_agent', 'N/A')}")
    else:
        error_msg = f"CodeExecutionAgent failed to execute critic-generated script. Error: {critic_exec_result.get('error')}. Stderr: {critic_exec_result.get('stderr')}"
        log("CodeExecutionAgent_CriticScriptError", error_msg)
        print(f"ERROR executing critic script: {error_msg}")
elif new_code_generated_by_critic and not (critic_code_path and os.path.exists(critic_code_path)):
    no_critic_script_msg = "CriticAgent generated new code, but the output path is invalid or file does not exist."
    log("CodeExecutionAgent_CriticScriptError", {"status": "skipped", "reason": no_critic_script_msg})
    print(f"\\n=== CodeExecutionAgent (Critic Script Run) SKIPPED ===\\n{no_critic_script_msg}")
else:
    no_critic_code_msg = "CriticAgent did not generate new code to execute."
    log("CodeExecutionAgent_CriticScriptRun", {"status": "skipped", "reason": no_critic_code_msg})
    print(f"\\n=== CodeExecutionAgent (Critic Script Run) SKIPPED ===\\n{no_critic_code_msg}")

print(f"\\nPipeline complete. Conversation log saved at: {conversation_log}")

