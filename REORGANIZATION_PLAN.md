# Multi-Agent Biological Modeling System

## 📁 **Proposed Repository Reorganization**

### **New Structure Overview**
```
MultiAgent/
├── biological_modeling/           # Module 1: Biological pathway modeling and expert systems
├── parameter_extraction/          # Module 2: PDF-to-parameter extraction pipeline  
├── code_repository_generation/    # Module 3: Code generation and project scaffolding
├── results_and_analysis/          # Module 4: Analysis, visualization, and reporting
├── shared/                        # Common utilities, base classes, memory management
├── examples/                      # End-to-end examples and demos
├── tests/                         # Comprehensive test suite
├── archived/                      # Historical/deprecated code (cleanly separated)
└── README.md                      # Main project documentation
```

## 🧬 **Module 1: Biological Modeling**
**Purpose**: Biological pathway modeling, expert agent systems, multi-agent coordination

### **Structure**:
```
biological_modeling/
├── agents/                        # Core biological agents
│   ├── __init__.py
│   ├── base_biological_agent.py   # Base class for all biological agents
│   ├── biologist_agent.py         # Main BiologistAgent (consolidated)
│   ├── pathway_expander_agent.py  # PathwayExpanderAgent
│   ├── parameter_mapper_agent.py  # Parameter mapping to computational values
│   └── overlap_pruner_agent.py    # Deduplication and pruning
├── experts/                       # Specialized expert agents
│   ├── __init__.py
│   ├── biological_expert_agent.py # Domain-specific biological experts
│   ├── experimental_expert_agent.py # Experimental design experts
│   ├── computational_expert_agent.py # Computational modeling experts
│   └── pi_agent.py                # Principal Investigator coordination
├── orchestration/                 # Multi-agent coordination systems
│   ├── __init__.py
│   ├── review_orchestration.py    # Multi-expert review coordination
│   ├── decision_analysis.py       # Decision-making logic
│   └── agent_selection.py         # Dynamic agent selection
├── models/                        # Data models and schemas
│   ├── __init__.py
│   ├── biological_entities.py     # Molecules, genes, species, interactions
│   ├── pathway_models.py          # Pathway representations
│   └── review_models.py           # Review and consensus models
├── workflows/                     # Pre-defined biological workflows
│   ├── __init__.py
│   ├── standard_workflow.py       # Standard biological modeling workflow
│   ├── literature_workflow.py     # Literature-intensive workflow
│   └── rapid_workflow.py          # Fast prototyping workflow
├── tools/                         # Biological modeling utilities
│   ├── __init__.py
│   ├── pubmed_search.py           # PubMed integration
│   ├── pathway_validation.py      # Pathway quality checks
│   └── interaction_analysis.py    # Interaction analysis tools
├── tests/                         # Module-specific tests
├── examples/                      # Usage examples
└── README.md                      # Module documentation
```

### **Key Files to Migrate Here**:
- `src/agents/biological_modeling_agents/*` (main content)
- `src/agents/biological_modeling_agents/crosstalk/*` (expert systems)
- `demo/agents/` (demo agents - consolidate with main)
- Parts of `src/framework/` (workflow orchestration)

---

## 📊 **Module 2: Parameter Extraction**
**Purpose**: PDF-to-text-to-parameters pipeline (already well organized!)

### **Structure**:
```
parameter_extraction/              # ✅ Already well-structured!
├── agents/
│   ├── __init__.py
│   ├── parameter_extraction_agent.py
│   └── vision_extraction_agent.py
├── processors/
│   ├── __init__.py
│   ├── pdf_processor.py
│   ├── clean_text_extractor.py
│   ├── vision_pdf_parser.py
│   └── organized_vision_extractor.py
├── utils/
│   ├── __init__.py
│   └── output_manager.py
├── tests/
├── examples/
└── README.md
```

### **Key Files to Migrate Here**:
- `src/agents/parameter_fetching_agents/*` (move entire folder, it's perfect!)

---

## 🏗️ **Module 3: Code Repository Generation**
**Purpose**: Generate complete computational repositories from biological specifications

### **Structure**:
```
code_repository_generation/
├── agents/                        # Code generation agents
│   ├── __init__.py
│   ├── base_code_agent.py         # Base class for code agents
│   ├── code_generation_agent.py   # Main code generator
│   ├── structure_agent.py         # Project structure generation
│   ├── dependency_predictor_agent.py # Dependency analysis
│   ├── evaluator_agent.py         # Code quality evaluation
│   └── template_mapping_agent.py  # Template-based generation
├── templates/                     # Code templates and scaffolding
│   ├── __init__.py
│   ├── vivarium_template/         # Vivarium project template
│   ├── abm_template/             # Agent-based model template
│   ├── python_package_template/  # Generic Python package
│   └── jupyter_template/         # Analysis notebook template
├── generators/                    # Specialized generators
│   ├── __init__.py
│   ├── vivarium_generator.py     # Vivarium-specific code generation
│   ├── dependency_generator.py   # Requirements and setup generation
│   └── documentation_generator.py # Auto-documentation
├── execution/                     # Code execution and validation
│   ├── __init__.py
│   ├── code_execution_agent.py   # Safe code execution in Docker
│   ├── validation_tools.py       # Code validation utilities
│   └── docker_manager.py         # Docker container management
├── tools/                         # Code generation utilities
│   ├── __init__.py
│   ├── ast_analysis.py           # Abstract syntax tree analysis
│   ├── import_resolver.py        # Import dependency resolution
│   └── quality_checker.py        # Code quality metrics
├── tests/
├── examples/
└── README.md
```

### **Key Files to Migrate Here**:
- `src/agents/code_understanding_agents/*`
- `src/agents/core_utility_agents/code_execution_agent.py`
- `src/framework/*` (project generation scripts)
- `ModelBuild/` (if applicable - advanced MCP integration)

---

## 📈 **Module 4: Results and Analysis**
**Purpose**: Analysis, visualization, and reporting of results from simulations and experiments

### **Structure**:
```
results_and_analysis/
├── agents/                        # Analysis agents
│   ├── __init__.py
│   ├── analysis_agent.py          # Data analysis coordination
│   ├── visualization_agent.py     # Automated visualization
│   ├── summarizer_agent.py        # Results summarization
│   └── critic_agent.py            # Critical analysis and suggestions
├── analyzers/                     # Specific analysis tools
│   ├── __init__.py
│   ├── simulation_analyzer.py     # ABM/simulation result analysis
│   ├── statistical_analyzer.py   # Statistical analysis tools
│   ├── pathway_analyzer.py        # Biological pathway analysis
│   └── comparative_analyzer.py    # Multi-experiment comparison
├── visualizers/                   # Visualization tools
│   ├── __init__.py
│   ├── pathway_visualizer.py      # Biological pathway plots
│   ├── simulation_visualizer.py   # Simulation result plots
│   └── report_generator.py        # Automated report generation
├── workflows/                     # Analysis workflows
│   ├── __init__.py
│   ├── standard_analysis.py       # Standard analysis pipeline
│   ├── comparative_analysis.py    # Multi-condition comparison
│   └── iterative_analysis.py      # Iterative refinement analysis
├── notebooks/                     # Analysis notebooks and templates
│   ├── template_analysis.ipynb    # Template for new analyses
│   └── examples/                  # Example analysis notebooks
├── tools/                         # Analysis utilities
│   ├── __init__.py
│   ├── data_loaders.py           # Data loading utilities
│   ├── metrics_calculator.py     # Standard metrics
│   └── export_tools.py           # Data export utilities
├── tests/
├── examples/
└── README.md
```

### **Key Files to Migrate Here**:
- `src/agents/pipeline_agents/analysis_agents/*`
- `src/simulations/notebooks/*`
- Analysis-related scripts from `src/framework/`
- Jupyter notebooks scattered throughout the repository

---

## 🔧 **Module 5: Shared (Common Infrastructure)**
**Purpose**: Common utilities, base classes, memory management, configuration

### **Structure**:
```
shared/
├── base/                          # Base classes and interfaces
│   ├── __init__.py
│   ├── base_agent.py             # Core agent base class
│   ├── base_workflow.py          # Workflow base class
│   └── interfaces.py             # Common interfaces
├── memory/                        # Memory and context management
│   ├── __init__.py
│   ├── unified_chroma_memory_manager.py
│   ├── conversation_memory.py
│   └── context_manager.py
├── utils/                         # Common utilities
│   ├── __init__.py
│   ├── file_utils.py
│   ├── logging_utils.py
│   ├── config_manager.py
│   └── validation_utils.py
├── orchestration/                 # General orchestration framework
│   ├── __init__.py
│   ├── workflow_orchestrator.py  # General workflow execution
│   ├── agent_registry.py         # Agent discovery and registration
│   └── task_planner.py           # Task planning and scheduling
├── models/                        # Common data models
│   ├── __init__.py
│   ├── workflow_models.py        # Workflow and task definitions
│   ├── result_models.py          # Standard result formats
│   └── config_models.py          # Configuration schemas
└── README.md
```

### **Key Files to Migrate Here**:
- `src/agents/base_agent.py`
- `src/memory/*`
- `src/utils/*`
- Common orchestration logic from various modules

---

## 🌟 **Module 6: Examples and Demos**
**Purpose**: End-to-end examples, tutorials, and demonstrations

### **Structure**:
```
examples/
├── end_to_end/                    # Complete workflow examples
│   ├── influenza_modeling/        # Full influenza modeling example
│   ├── tumor_tcell_interaction/   # Tumor-T cell example
│   └── pathway_discovery/         # Pathway discovery example
├── tutorials/                     # Step-by-step tutorials
│   ├── getting_started.ipynb      # Basic usage tutorial
│   ├── biological_modeling_101.ipynb
│   ├── parameter_extraction_guide.ipynb
│   └── code_generation_tutorial.ipynb
├── demos/                         # Interactive demos
│   ├── web_demo/                  # Flask web interface
│   ├── jupyter_demo/              # Jupyter-based demo
│   └── cli_demo/                  # Command-line demo
└── README.md
```

### **Key Files to Migrate Here**:
- `demo/*` (current demo folder)
- Key examples from each module
- New tutorial content

---

## 🗃️ **Module 7: Tests**
**Purpose**: Comprehensive testing across all modules

### **Structure**:
```
tests/
├── unit/                          # Unit tests by module
│   ├── test_biological_modeling/
│   ├── test_parameter_extraction/
│   ├── test_code_generation/
│   └── test_results_analysis/
├── integration/                   # Integration tests
│   ├── test_end_to_end_workflows/
│   └── test_multi_agent_coordination/
├── performance/                   # Performance and load tests
├── fixtures/                      # Test data and fixtures
├── utils/                         # Testing utilities
└── README.md
```

---

## 📚 **Migration Priority and Plan**

### **Phase 1: Foundation (High Priority)**
1. ✅ Create new directory structure (DONE)
2. Migrate `shared/` module first (base classes, utilities)
3. Move `parameter_extraction/` (already well-organized)

### **Phase 2: Core Modules (Medium Priority)**  
1. Migrate `biological_modeling/` (consolidate scattered agents)
2. Move `code_repository_generation/` 
3. Organize `results_and_analysis/`

### **Phase 3: Polish (Lower Priority)**
1. Create comprehensive `examples/`
2. Migrate and organize `tests/`
3. Clean up `archived/`
4. Update documentation

### **Phase 4: Cleanup**
1. Remove old `agent-testing-framework/src/` structure
2. Update all imports and references
3. Rename root directory (optional): `MultiAgent` → `BiologicalModelingSystem`

---

## 🎯 **Benefits of This Reorganization**

1. **Clear Separation of Concerns**: Each module has a specific, well-defined purpose
2. **Scalability**: Easy to add new agents/tools to the appropriate module
3. **Reusability**: Shared utilities avoid code duplication  
4. **Maintainability**: Related functionality is grouped together
5. **Discoverability**: Clear structure makes finding code intuitive
6. **Testing**: Organized test structure by module
7. **Documentation**: Each module can have focused documentation

This structure transforms your research codebase into a production-ready, modular system that clearly separates biological modeling, parameter extraction, code generation, and analysis capabilities while maintaining clean interfaces between modules.

Would you like me to start with the migration of specific modules or create the detailed folder structure for any particular module first?