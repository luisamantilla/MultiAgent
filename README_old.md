# Multi-Agent Framework for Biological Modeling and Code Generation

## Project Vision & Current Progress

This project is an extensible, modular multi-agent framework designed to automate biological pathway modeling, code generation, execution, evaluation, and iterative refinement. The system specializes in translating high-level biological concepts into detailed mechanistic pathways and computational implementations through collaborative AI agents.

### Core Capabilities
- **Biological Pathway Expansion**: Transform high-level biological interactions into detailed mechanistic steps with species, genes, and molecules
- **Multi-Expert Review Systems**: Coordinate biological, computational, and experimental experts for pathway validation
- **Automated Code Generation**: Generate simulation-ready code from biological specifications with dependency management
- **Secure Code Execution**: Run generated code in containerized Docker environments with artifact management
- **Iterative Refinement**: Continuously improve outputs through multi-agent feedback loops and memory-driven context sharing
- **Expert Collaboration**: Systematic consensus building across multiple domains of expertise

### What I've Built

#### Biological Modeling Pipeline
- **BiologicalRequirementAgent**: Generates comprehensive biological requirements for any tissue/system/pathway
- **PathwayExpanderAgent**: Expands high-level interactions into detailed mechanistic steps with molecular specificity
- **OverlapPrunerAgent**: Detects and removes duplicate/similar mechanistic steps across pathways
- **DetailPrunerAgent**: Prunes and summarizes detailed steps based on user preferences and focus areas
- **ParameterMapperAgent**: Maps biological parameters to computational values for simulation readiness

#### Expert Review System
- **PIAgent**: Principal Investigator agent that orchestrates multi-expert review cycles and makes informed decisions
- **BiologicalExpertAgent**: Specialized biological experts including:
  - ImmunologyExpertAgent: Immune system processes and interactions
  - CellBiologyExpertAgent: Cellular processes and organelle function
  - MolecularBiologyExpertAgent: Protein interactions and gene regulation
  - PathologyExpertAgent: Disease mechanisms and clinical relevance
- **ComputationalExpertAgent**: Implementation experts for different modeling approaches
- **ExperimentalExpertAgent**: Validation experts for different experimental methodologies

#### Code Understanding & Generation
- **CodeGenerationAgent**: Generates Python code from specifications with dependency management
- **StructureAnalyzerAgent**: Analyzes project structure and builds dependency graphs
- **DependencyPredictorAgent**: Predicts project dependencies for adaptation tasks
- **TemplateMappingAgent**: Maps between old and new project structures for code adaptation
- **EvaluatorAgent**: Evaluates generated code against success criteria

#### Core Utilities & Framework
- **CodeExecutionAgent**: Secure Docker-based code execution with artifact management
- **BaseAgent**: Foundation class with LLM integration and standardized interfaces
- **UnifiedChromaMemoryManager**: ChromaDB-based persistent memory with semantic search capabilities
- **Framework modules**: Orchestration pipelines for dependency analysis, project generation, and multi-agent workflows

#### Infrastructure & Tools
- **Jupyter notebook pipelines**: Interactive agent testing and pathway expansion/pruning workflows
- **Testing infrastructure**: Comprehensive validation for framework components and generated code
- **File utilities**: Management of generated files, outputs, and iterative run cleanup
- **Archived experiments**: Historical implementations and rapid prototyping scripts

### What I'm Envisioning
- **Enhanced Expert Systems**: More specialized domain experts and decision algorithms for multi-expert consensus
- **Real-time Collaboration**: Live human-AI collaboration interfaces for pathway refinement and validation
- **Advanced Memory Systems**: Long-term learning and knowledge accumulation across experimental sessions
- **Integration Ecosystem**: Connections to major biological databases, simulation platforms, and experimental tools
- **Autonomous Scientific Workflows**: End-to-end automation from hypothesis generation to experimental design and analysis
- **Cross-Domain Applications**: Extension beyond biology to other scientific domains requiring multi-expert validation

### Current Experiments
- **Full ABM Pipeline Automation:** Successfully running an end-to-end pipeline where:
    1.  An initial Agent-Based Model (ABM) simulation (`abm_simulation.py`) is executed by `CodeExecutionAgent` in Docker.
    2.  `SummarizerAgent` and `AnalysisAgent` process the simulation outputs.
    3.  `CriticAgent` reviews the analysis and the original simulation code, then **generates a new Python script** (e.g., `suggested_experiment.py`) designed to perform a parameter sweep or other follow-up experiment.
    4.  `CodeExecutionAgent` executes this newly generated experiment script in Docker, with its outputs saved to a unique, iteration-specific directory.
- **Iterative Refinement of Experiments:** Exploring how the `CriticAgent` can leverage feedback from multiple cycles to propose increasingly sophisticated experimental designs.
- **Secure and Reproducible Code Execution:** Utilizing Docker via `CodeExecutionAgent` to ensure that both initial simulations and critic-generated scripts run in a consistent and isolated environment.
- **Automated Output Management:** Implemented automated cleanup of output directories to facilitate rapid iteration and testing of the pipeline.
- Evaluating the effectiveness of memory sharing and context retrieval in improving the quality and relevance of suggestions and generated code from the `CriticAgent`.

---

## Project Structure

```
agent-testing-framework/
├── README.md
├── requirements.txt
└── src/
    ├── agents/
    │   ├── __init__.py
    │   ├── base_agent.py                    # Foundation agent class
    │   ├── biological_modeling_agents/      # Biological pathway modeling
    │   │   ├── __init__.py
    │   │   ├── biological_requirement_agent.py  # BiologicalRequirementAgent
    │   │   ├── pathway_expander_agent.py    # Pathway expansion
    │   │   ├── overlap_pruner_agent.py      # Overlap detection/removal
    │   │   ├── detail_pruner_agent.py       # Detail management
    │   │   ├── parameter_mapper_agent.py    # Parameter mapping
    │   │   ├── test_agents_pipeline.ipynb   # Interactive testing pipeline
    │   │   └── crosstalk/                   # Expert review system
    │   │       ├── pi_agent.py              # PI orchestrator
    │   │       ├── biological_expert_agent.py
    │   │       ├── computational_expert_agent.py
    │   │       └── experimental_expert_agent.py
    │   ├── code_understanding_agents/       # Code analysis & generation
    │   │   ├── __init__.py
    │   │   ├── code_generation_agent.py
    │   │   ├── structure_agent.py
    │   │   ├── dependency_predictor_agent.py
    │   │   ├── template_mapping_agent.py
    │   │   └── evaluator_agent.py
    │   ├── core_utility_agents/             # Core functionalities
    │   │   ├── __init__.py
    │   │   ├── code_execution_agent.py
    │   │   └── Dockerfile.code_execution
    │   └── pipeline_agents/                 # Workflow orchestration
    │       ├── __init__.py
    │       ├── analysis_agents/
    │       │   ├── __init__.py
    │       │   ├── summarizer_agent.py
    │       │   └── result_analysis_agent.py
    │       └── planning_feedback_agents/
    │           ├── __init__.py
    │           └── critic_agent.py
    ├── archived/                            # Historical implementations
    │   ├── RAM_memory/                      # Legacy memory systems
    │   ├── simulation_agents/               # Legacy simulation agents
    │   ├── 20250528_*.py                    # Dated experimental scripts
    │   ├── 20250529_*.py                    # Iterative refinement experiments
    │   └── test_*.py                        # Historical test files
    ├── framework/                           # Core orchestration logic
    │   ├── __init__.py
    │   ├── 20250530_dependency_graph_prediction_and_comparison.py
    │   ├── 20250530_project_generation_from_dependency_map.py
    │   └── 20250602_*.py                    # Updated pipeline scripts
    ├── memory/                              # Memory management systems
    │   ├── __init__.py
    │   └── unified_chroma_memory_manager.py # Primary memory system
    ├── simulations/                         # Simulation environments
    │   ├── __init__.py
    │   ├── abm_scripts/
    │   │   └── abm_simulation.py
    │   └── generated_experiments/
    │       └── iteration_*/
    ├── tests/                               # Testing infrastructure
    │   ├── __init__.py
    │   └── test_*.py
    └── utils/                               # Utility functions
        ├── __init__.py
        └── file_utils.py
```

- **agents/**: Contains all agent implementations, organized by their role:
    - **biological_modeling_agents/**: Specialized agents for biological pathway modeling and expert review
    - **code_understanding_agents/**: Agents focused on code generation, analysis, and adaptation
    - **core_utility_agents/**: Fundamental agents like secure code executors
    - **pipeline_agents/**: Agents involved in workflow orchestration (simulation, analysis, planning)
- **framework/**: Core logic for running agent pipelines, refinement loops, and orchestration
- **memory/**: Implements memory modules for sharing context and feedback between agents
- **simulations/**: Houses simulation scripts (e.g., for ABM) and generated experiments
- **utils/**: Helper functions and utilities for file management and operations
- **tests/**: Unit and integration tests for framework components
- **archived/**: Historical implementations and experimental scripts for reference

---

## Setup Instructions

1. **Clone the repository**:
    ```bash
    git clone <repo-url>
    cd MultiAgent/agent-testing-framework
    ```

2. **(Optional) Create and activate a virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4. **Set up environment variables:**
    ```bash
    export OPENAI_API_KEY="your-openai-api-key"
    ```

---

## Usage

#### Biological Pathway Modeling
```bash
# Run the interactive biological modeling pipeline
jupyter notebook src/agents/biological_modeling_agents/test_agents_pipeline.ipynb
```

#### Code Generation Pipeline
```bash
# Run framework components
python -c "from src.framework.20250602_abm_results_multi_agent_orchestration_pipeline import *"
```

#### Expert Review System
```bash
# Initialize expert review for biological pathways
python -c "from src.agents.biological_modeling_agents.crosstalk.pi_agent import PIAgent"
```

---

## Features

- **Multi-agent collaboration**: Code generation, execution, and evaluation agents work together
- **Memory sharing**: Agents share context and feedback to improve results over iterations
- **Automated refinement**: The system iteratively improves code until the task is achieved
- **Multi-file support**: Handles tasks requiring multiple Python files and cross-file imports
- **Extensible**: Easily add new agent types or integrate with larger multi-agent systems
- **Biological expertise**: Specialized agents for pathway modeling, biological validation, and expert review
- **Secure execution**: Docker-based code execution with comprehensive artifact management
- **Expert consensus**: Multi-expert review system with systematic conflict resolution

---

## Testing

Run the included tests with:

```bash
pytest src/tests/
```

---

## Contribution

Contributions are welcome! Please open an issue or submit a pull request for bug fixes, new features, or improvements.

---

## License

This project is licensed under the MIT License. See the LICENSE file for details.
