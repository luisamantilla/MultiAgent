# ModelBuild

*Multi-Agent Vivarium Pipeline for Influenza Modeling*

**Author**: Faye Guo

---

## Overview

ModelBuild is a multi-agent framework that transforms biological questions into executable Vivarium simulation models. It orchestrates specialized AI agents to automate the entire pipeline from biological scope definition to tested, documented simulation code.

(The details of the system can be found in ModelBuild/modelbuild_system_design.md.)

### Key Features

- **Multi-Agent Collaboration**: Specialized agents work together to handle different aspects of model building
- **Automated Parameter Extraction**: Uses PaperQA to extract quantitative parameters from scientific literature
- **Vivarium Integration**: Generates production-ready Vivarium simulation code
- **Quality Assurance**: Built-in validation and critique workflows
- **Extensible Architecture**: Modular design supports future framework integration

---

## Quick Start

### Prerequisites

- Python 3.8+
- OpenAI API key
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone git@github.com:YourOrg/MultiAgent.git
   cd MultiAgent/ModelBuild
   ```

2. **Set up virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate    # macOS/Linux
   venv\Scripts\activate       # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure API key**
   ```bash
   export OPENAI_API_KEY="sk-your-key-here"   # macOS/Linux
   set OPENAI_API_KEY=sk-your-key-here        # Windows
   ```

### Basic Usage

```bash
# Run the full pipeline
python main.py --input "How do T-cells interact with tumor cells?"

# Or with a biological mapping file
python main.py --mapping biological_mapping.json
```

---

## System Architecture

### Agent Workflow

```
User Question/Biological Mapping
           ↓
      [PI Agent] ← Orchestrates the entire workflow
           ↓
    ┌─────────────┬─────────────────┬──────────────────┐
    │             │                 │                  │
    ▼             ▼                 ▼                  ▼
Biologist    ModelBuilder    ParameterExtractor   Critique Agent
  Agent        Agent              Agent
    │             │                 │
    ▼             ▼                 ▼
(Biological   (Vivarium        (Parameter
 Mapping)      Code)            Values)
    │             │                 │
    └─────────────┼─────────────────┘
                  ▼
           [Validation & Merge]
                  ↓
           [Simulation Agent]*

* Developed by Bobby (external component)
```

### Core Agents

| Agent | Role | Input | Output |
|-------|------|-------|---------|
| **PI Agent** | Task orchestration and workflow management | User prompt | Task assignments, workflow plan |
| **Biologist Agent** | Biological knowledge interpretation | Biological question | Biological mapping (JSON) |
| **Parameter Extractor** | Literature parameter extraction | Paper list, context | Structured parameter tables |
| **Model Builder** | Vivarium code generation | Biological mapping, parameters | Vivarium Python modules |
| **Critique Agent** | Model validation and QA | Generated models | Validation reports, improvements |

---

## Project Structure

```
ModelBuild/
├── agents/                     # Agent implementations
│   ├── base_agents.py         # Base agent classes
│   ├── specialized_agents.py  # Domain-specific agents
│   └── agent_registry.py      # Agent management
├── memory/                     # Shared memory system
│   ├── pdf_vector_db.py       # Literature database
│   └── knowledge_graph.py     # Structured knowledge
├── tools/                      # Utility tools
│   ├── parameter_extraction.py # PaperQA integration
│   ├── code_generation.py     # Vivarium templates
│   └── model_validation.py    # Validation utilities
├── templates/                  # Code templates
│   ├── vivarium_base.py       # Base templates
│   ├── cell_types/            # Cell-specific templates
│   └── processes/             # Process templates
├── orchestration/              # Workflow management
│   ├── workflow_manager.py    # Main orchestration
│   └── task_scheduler.py      # Task management
├── config/                     # Configuration files
│   ├── agent_config.yaml      # Agent settings
│   └── system_config.yaml     # System settings
├── output/                     # Generated outputs
│   ├── models/                # Vivarium models
│   ├── parameters/            # Extracted parameters
│   └── reports/               # Validation reports
└── main.py                    # Entry point
```

---

## Configuration

### Agent Configuration (`config/agent_config.yaml`)

```yaml
agents:
  pi_agent:
    model: "gpt-4"
    temperature: 0.3
    max_iterations: 5
  
  biologist_agent:
    model: "gpt-4"
    expertise_domains: ["immunology", "cell_biology"]
    confidence_threshold: 0.8
```

### System Configuration (`config/system_config.yaml`)

```yaml
system:
  database:
    type: "chromadb"
    path: "./multi_agent_db"
  
  embedding:
    model: "text-embedding-3-large"
    dimensions: 1536
  
  validation:
    confidence_threshold: 0.7
    max_iterations: 10
```

---

## Usage Examples

### Example 1: Basic T-cell Tumor Interaction

```bash
python main.py --input "Model the interaction between CD8+ T-cells and tumor cells in the tumor microenvironment"
```

### Example 2: With Biological Mapping

```bash
python main.py --mapping examples/tcell_tumor_mapping.json --parameters examples/known_parameters.json
```

### Example 3: Custom Configuration

```bash
python main.py --input "Influenza infection dynamics" --config custom_config.yaml
```

---

## Input Formats

### Biological Question (Text)
Simple natural language description of the biological system you want to model.

### Biological Mapping (JSON)
```json
{
  "cell_types": ["T-cell", "Tumor-cell"],
  "interactions": [
    {
      "type": "cytotoxic_killing",
      "source": "T-cell",
      "target": "Tumor-cell",
      "parameters": ["kill_rate", "contact_probability"]
    }
  ],
  "processes": ["proliferation", "apoptosis", "migration"]
}
```

---

## Output

ModelBuild generates:

1. **Vivarium Models** (`output/models/`)
   - Complete Python modules for each cell type
   - Process implementations
   - Integration code

2. **Parameter Files** (`output/parameters/`)
   - Extracted parameter values with confidence scores
   - Source citations and metadata

3. **Validation Reports** (`output/reports/`)
   - Model validation results
   - Biological plausibility assessments
   - Improvement suggestions

---

## Development

### Current Status

- Core agent framework
- Vector database integration
- Parameter extraction pipeline
- Vivarium code generation (in progress)
- Validation workflows (in progress)
- MCP integration (planned)

### Testing

The system is being tested with the tumor-tcell Vivarium repository reconstruction, using ground truth parameter tables for validation.

### Contributing

1. Fork the repository
2. Create a feature branch
3. Follow the existing code style
4. Add tests for new functionality
5. Submit a pull request

---

## Roadmap

### Phase 1: Core Functionality
- [x] Multi-agent framework
- [x] Parameter extraction
- [ ] Code generation
- [ ] Basic validation

### Phase 2: Advanced Features
- [ ] Active learning
- [ ] Uncertainty quantification
- [ ] Multi-modal input support

### Phase 3: Integration
- [ ] MCP compatibility
- [ ] API endpoints
- [ ] Container deployment

---

## Support

### Team
- **Faye Guo**: ModelBuild system developing
- **Bobby**: Biological mapping and Simulation Agent
- **Kevin**: Vector database and paper curation

### Issues and Questions
- Create an issue on GitHub for bugs or feature requests
- Check the documentation in `docs/` for detailed guides
- Review the ground truth validation in `tests/validation/`

---

## License

[Add your license information here]

---

## Acknowledgments

- Built on the Vivarium simulation framework
- Uses PaperQA for parameter extraction
- Integrates with OpenAI language models