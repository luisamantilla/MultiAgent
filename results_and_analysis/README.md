# Results and Analysis Module

## Purpose
Comprehensive analysis, visualization, and reporting of results from simulations, experiments, and biological modeling workflows.

## Structure

### `/agents/` - Analysis Agents
- `analysis_agent.py` - Coordinates data analysis workflows
- `visualization_agent.py` - Automated visualization generation
- `summarizer_agent.py` - Results summarization and reporting
- `critic_agent.py` - Critical analysis and iterative improvement suggestions

### `/analyzers/` - Specialized Analysis Tools
- `simulation_analyzer.py` - ABM/simulation result analysis
- `statistical_analyzer.py` - Statistical analysis and hypothesis testing
- `pathway_analyzer.py` - Biological pathway analysis
- `comparative_analyzer.py` - Multi-experiment and multi-condition comparison

### `/visualizers/` - Visualization Tools
- `pathway_visualizer.py` - Biological pathway visualization
- `simulation_visualizer.py` - Simulation result plots and animations
- `report_generator.py` - Automated report and figure generation

### `/workflows/` - Analysis Workflows
- `standard_analysis.py` - Standard analysis pipeline for simulation results
- `comparative_analysis.py` - Multi-condition comparison workflows
- `iterative_analysis.py` - Iterative refinement and improvement analysis

### `/notebooks/` - Analysis Notebooks
- `template_analysis.ipynb` - Template for new analysis notebooks
- `examples/` - Example analysis notebooks for different data types

### `/tools/` - Analysis Utilities
- `data_loaders.py` - Data loading utilities for various formats
- `metrics_calculator.py` - Standard biological and computational metrics
- `export_tools.py` - Data export utilities (CSV, JSON, plots)

## Key Files to Migrate Here
- From `agent-testing-framework/src/agents/pipeline_agents/analysis_agents/`
- From `agent-testing-framework/src/simulations/notebooks/`
- Analysis scripts from `agent-testing-framework/src/framework/`
- Scattered Jupyter notebooks throughout the repository