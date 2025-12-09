#!/bin/bash

# Activate the Python environment
source /home/labuser/Desktop/lab_member_projects/Bobby_Ni/bobby/bin/activate

# Change to the agent-testing-framework directory
cd /home/labuser/Desktop/lab_member_projects/Bobby_Ni/MultiAgent/agent-testing-framework/

# Run the pipeline script
python -m src.framework.20250530_abm_results_multi_agent_orchestration_pipeline

echo "Pipeline execution finished."
