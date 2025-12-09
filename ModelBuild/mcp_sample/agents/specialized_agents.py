# agents/specialized_agents.py
import re
import json
import time
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from .base_agents import Agent

logger = logging.getLogger(__name__)


# -------- Principal Investigator --------

class PIAgent(Agent):
    """Principal Investigator Agent with task orchestration capabilities (LLM planning only)."""

    def __init__(self):
        super().__init__(
            title="Principal Investigator Agent",
            expertise="Biological modeling, task orchestration, project management",
            goal="Interpret user prompts and create comprehensive task plans",
            role="Project orchestrator and strategic planner",
            model="gpt-4o",
            output_format=(
                "Return a concise JSON object with fields:\n"
                "{\n"
                '  "project_overview": str,\n'
                '  "key_tasks": [str],\n'
                '  "risks": [str],\n'
                '  "handoffs": [{"to": "biologist|parameter_extractor|model_builder|code_generator", "what": str}]\n'
                "}\n"
            ),
            enable_confidence=True,
            mcp_enabled=True,  # Enable MCP features
        )

    @staticmethod
    def extract_task_plan(response: str) -> Dict[str, Any]:
        """Best-effort JSON extraction from an LLM response."""
        try:
            json_match = re.search(r"\{[\s\S]*\}", response)
            if json_match:
                return json.loads(json_match.group())
        except Exception as e:
            logger.error(f"PIAgent.extract_task_plan error: {e}")
        return {
            "project_overview": "Vivarium model generation task",
            "key_tasks": [],
            "risks": [],
            "handoffs": [],
            "raw_response": response,
        }

    def get_capabilities(self) -> List[str]:
        """Override to provide specific PI capabilities."""
        base_caps = super().get_capabilities()
        return base_caps + [
            "task_planning", "project_orchestration", "json_output",
            "workflow_design", "risk_assessment"
        ]


# -------- Biologist --------

class BiologistAgent(Agent):
    """Biologist that produces quantitative biological mappings."""

    def __init__(self):
        super().__init__(
            title="Biologist Agent",
            expertise="Quantitative cell biology, biochemical kinetics, systems biology",
            goal="Provide detailed biological context with quantitative insights",
            role="Quantitative biological context expert",
            model="gpt-4o",
            output_format=(
                "Write a compact technical brief that includes:\n"
                "1) Molecular components and interactions\n"
                "2) Typical concentration ranges and timescales\n"
                "3) Quantitative relations (stoichiometry, cooperativity)\n"
                "4) Systems-level behavior and regulation\n"
            ),
            enable_confidence=True,
            mcp_enabled=True,
        )

    def provide_quantitative_context(self, user_request: str) -> str:
        """Legacy method for backward compatibility."""
        convo = [
            self.system_message(),
            {
                "role": "user",
                "content": (
                    "Provide a quantitative biological context summary for:\n"
                    f"{user_request}\n\n"
                    "Be concrete with numbers (units!), typical ranges, and key mechanisms."
                ),
            },
        ]
        return self.run(convo)

    def provide_enhanced_context(self, user_request: str, paper_context: Optional[str] = None) -> str:
        """Enhanced method that can use paper context from vector database."""
        conversation = [self.system_message()]

        if paper_context:
            conversation.append({
                "role": "system",
                "content": f"Additional context from research literature:\n{paper_context}"
            })

        conversation.append({
            "role": "user",
            "content": (
                "Provide a quantitative biological context summary for:\n"
                f"{user_request}\n\n"
                "Be concrete with numbers (units!), typical ranges, and key mechanisms.\n"
                "If research literature context is provided, integrate relevant findings."
            ),
        })

        return self.run(conversation)

    def get_capabilities(self) -> List[str]:
        """Override to provide specific biologist capabilities."""
        base_caps = super().get_capabilities()
        return base_caps + [
            "biological_analysis", "quantitative_modeling", "biochemical_kinetics",
            "systems_biology", "literature_integration", "concentration_estimation"
        ]


# -------- Parameter Extractor --------

class ParameterExtractorAgent(Agent):
    """Extracts or infers modeling parameters."""

    def __init__(self):
        super().__init__(
            title="Parameter Extractor Agent",
            expertise="Scientific literature analysis, parameter extraction, biological inference",
            goal="Extract parameters from literature or infer from biological principles",
            role="Scientific data extraction and inference",
            model="gpt-4o",
            output_format=(
                "Return a list of parameters in this TSV format:\n"
                "parameter\tvalue\tunits\tsource\tmethod\tconfidence(0-1)\trange\n"
            ),
            enable_confidence=True,
            dependencies=["BiologistAgent"],
            max_retries=3,
            timeout=180,
            mcp_enabled=True,
        )

    def infer_parameters_from_biology(self, bio_mapping: str, user_request: str) -> str:
        """Helper that asks the LLM to infer values when literature is scarce."""
        prompt = f"""
BIOLOGICAL CONTEXT
------------------
{bio_mapping}

TASK
----
Infer reasonable parameter values for: {user_request}

GUIDELINES
----------
- Provide numeric values with units.
- Include a confidence score (0.0-1.0).
- Add a typical range and a 1-line method/reasoning (e.g., first-principles, scaling, typical enzyme kinetics).

OUTPUT FORMAT (TSV)
parameter\tvalue\tunits\tsource\tmethod\tconfidence(0-1)\trange
"""
        return self.run([self.system_message(), {"role": "user", "content": prompt.strip()}])

    def extract_with_papers(self, user_request: str, bio_context: str = "",
                            paper_results: Optional[List[Dict]] = None) -> str:
        """Enhanced extraction method that uses paper search results."""
        conversation = [self.system_message()]

        # Add biological context if provided
        if bio_context:
            conversation.append({
                "role": "system",
                "content": f"BIOLOGICAL CONTEXT:\n{bio_context}"
            })

        # Add paper search results if available
        if paper_results:
            paper_context = "RELEVANT RESEARCH FINDINGS:\n"
            for i, paper in enumerate(paper_results, 1):
                if isinstance(paper, dict):
                    source = paper.get('source', f'Paper {i}')
                    content = paper.get('content', str(paper))
                    paper_context += f"\n{i}. [{source}]\n{content}\n"
                else:
                    paper_context += f"\n{i}. {str(paper)}\n"

            conversation.append({"role": "system", "content": paper_context})

        # Main extraction prompt
        prompt = f"""
TASK: Extract quantitative parameters for: {user_request}

GUIDELINES:
- Extract specific numeric values with units from the provided research context
- When literature values are available, prioritize them over inference
- Include confidence scores (0.0-1.0) based on source quality
- Add typical ranges and cite sources when possible
- If no literature values found, infer reasonable estimates

OUTPUT FORMAT (TSV):
parameter\tvalue\tunits\tsource\tmethod\tconfidence(0-1)\trange
"""

        conversation.append({"role": "user", "content": prompt.strip()})
        return self.run(conversation)

    def get_capabilities(self) -> List[str]:
        """Override to provide specific parameter extraction capabilities."""
        base_caps = super().get_capabilities()
        return base_caps + [
            "parameter_extraction", "literature_analysis", "tsv_output",
            "confidence_scoring", "value_inference", "range_estimation"
        ]


# -------- Model Builder --------

class ModelBuilderAgent(Agent):
    """Converts biological mapping + parameters into a Vivarium spec following established patterns."""

    def __init__(self):
        super().__init__(
            title="Model Builder Agent",
            expertise="Vivarium simulation framework, biological modeling, software architecture patterns",
            goal="Convert biological mappings and parameters to Vivarium model specifications following best practices",
            role="Simulation architecture specialist with pattern recognition",
            model="gpt-4o",
            output_format="""
            Create a detailed Vivarium model specification that includes:

            MODEL ARCHITECTURE:
            - Process hierarchy and dependencies
            - State variable definitions with units
            - Parameter mappings to biological processes
            - Port specifications and data flow
            - Composite structure and organization

            VIVARIUM PATTERNS:
            - Follow established Vivarium conventions
            - Use proper process inheritance patterns
            - Implement standard port naming conventions
            - Apply consistent parameter handling
            - Include proper initialization and update methods

            IMPLEMENTATION DETAILS:
            - Specify exact class names and file organization
            - Define process schemas and default parameters
            - Map biological concepts to Vivarium constructs
            - Include validation and error handling approaches
            """,
            enable_confidence=True,
            dependencies=["BiologistAgent", "ParameterExtractorAgent"],
            mcp_enabled=True,
        )
        self._load_vivarium_patterns()

    def _load_vivarium_patterns(self):
        """Load and analyze Vivarium patterns from scaffold templates."""
        self.vivarium_patterns = {
            "vivarium_process_structure": """
            Vivarium Process Pattern (from generic_cell.py):
            - Simple class definition (no inheritance from Process base class)
            - __init__ with parameters dictionary
            - ports_schema() with 'internal' and 'boundary' ports
            - '_default', '_emit', '_updater' configuration for each variable
            - next_update() with threshold-based logic pattern
            - Simple parameter access with .get() and defaults
            """,
            "vivarium_field_structure": """
            Vivarium Field Pattern (from field.py):
            - Similar simple class structure
            - 'field' port with 'accumulate' updater
            - Decay pattern: value * (1 - decay_rate)
            - Simple environmental variable handling
            """,
            "vivarium_port_patterns": """
            Vivarium Port Schema Patterns:
            - 'internal': for cellular/process internal states
            - 'boundary': for external signals/interactions  
            - 'field': for environmental/spatial variables
            - '_updater' types: 'set', 'accumulate'
            - '_emit': True for most variables
            - '_default': initial values
            """,
            "vivarium_update_patterns": """
            Vivarium Update Logic Patterns:
            - Threshold-based responses: if signal > threshold
            - Simple decay: value * (1 - decay_rate)
            - Conditional updates with parameter checks
            - Return dictionary with port->variable->value structure
            """,
            "file_organization": """
            Vivarium Repository Structure (standard):
            - Simple process files (generic_cell.py, field.py)
            - Class-per-file organization
            - Minimal import dependencies
            - Template-based approach for reusability
            """
        }

    def build_enhanced_spec(self, user_request: str, bio_context: str = "", parameters: str = "", **kwargs) -> str:
        """Enhanced model building with Vivarium pattern awareness."""
        conversation = [self.system_message()]

        # Add Vivarium patterns as context
        pattern_context = "VIVARIUM BEST PRACTICES AND PATTERNS:\n"
        for pattern_type, pattern_desc in self.vivarium_patterns.items():
            pattern_context += f"\n{pattern_type.upper()}:\n{pattern_desc}\n"

        conversation.append({"role": "system", "content": pattern_context})

        # Build comprehensive prompt
        prompt_parts = [f"BUILD VIVARIUM MODEL SPECIFICATION FOR: {user_request}"]

        if bio_context:
            prompt_parts.append(f"\nBIOLOGICAL CONTEXT:\n{bio_context}")

        if parameters:
            prompt_parts.append(f"\nEXTRACTED PARAMETERS:\n{parameters}")

        # Add any additional context
        for key, value in kwargs.items():
            if value:
                prompt_parts.append(f"\n{key.upper().replace('_', ' ')}:\n{value}")

        prompt_parts.append("""
REQUIREMENTS:
- Follow established Vivarium patterns and conventions
- Define clear process hierarchy with proper inheritance
- Specify state variables with appropriate units and ranges
- Map extracted parameters to specific processes
- Define process ports and connections using Vivarium standards
- Include proper schema definitions for all processes
- Specify composite topology and wiring
- Include validation, testing, and experiment approaches
- Use consistent naming conventions throughout
- Provide implementation guidance for complex biological mechanisms
""")

        conversation.append({"role": "user", "content": "\n".join(prompt_parts)})
        return self.run(conversation)

    def analyze_biological_processes(self, bio_context: str) -> Dict[str, Any]:
        """Analyze biological context to identify Vivarium processes."""
        analysis_prompt = f"""
        Analyze this biological context and identify how to map it to Vivarium processes:

        {bio_context}

        Identify:
        1. Core biological processes that need separate Vivarium Process classes
        2. State variables that need to be tracked
        3. Parameters that control each process
        4. Connections/dependencies between processes
        5. Hierarchical organization possibilities
        """

        response = self.run([
            self.system_message(),
            {"role": "user", "content": analysis_prompt}
        ])

        return {"analysis": response}

    def get_capabilities(self) -> List[str]:
        """Override to provide specific model building capabilities."""
        base_caps = super().get_capabilities()
        return base_caps + [
            "vivarium_modeling", "system_design", "process_architecture",
            "parameter_mapping", "port_specification", "composite_design",
            "pattern_recognition", "biological_mapping", "schema_definition"
        ]


# -------- Code Generator --------

class CodeGenerationAgent(Agent):
    """Creates runnable Vivarium repositories following established patterns and conventions."""

    def __init__(self):
        super().__init__(
            title="Code Generation Agent",
            expertise="Vivarium repository structure, Python packaging, biological simulation code patterns",
            goal="Generate complete, executable Vivarium repositories following best practices and conventions",
            role="Software architect and Vivarium pattern specialist",
            model="gpt-4o",
            output_format=(
                "Generate complete Vivarium repository files using this exact pattern:\n"
                "FILE: relative/path/filename.ext\n"
                "```python\n"
                "# Complete, production-ready code following Vivarium patterns\n"
                "```\n\n"
                "Ensure all files follow established Vivarium conventions and patterns."
            ),
            dependencies=["ModelBuilderAgent"],
            max_retries=2,
            timeout=300,
            mcp_enabled=True,
        )
        self._load_vivarium_templates()

    def _load_vivarium_templates(self):
        """Load Vivarium code templates and patterns from Bobby's scaffold."""
        self.templates = {
            "process_template": '''"""
Vivarium Process Pattern (based on generic_cell.py):

class {ProcessName}:
    \"\"\"
    {Process description and biological context}

    This process implements {biological_mechanism} following Vivarium conventions.
    \"\"\"

    def __init__(self, parameters=None):
        self.parameters = parameters or {}
        # Add any process-specific initialization

    def ports_schema(self):
        \"\"\"
        Defines the structure of state variables (ports) for this process.
        Follow Vivarium pattern: internal states and boundary interactions.
        \"\"\"
        return {
            'internal': {
                # Internal cellular/process states
                '{internal_var}': {
                    '_default': {default_value},
                    '_emit': True,
                    '_updater': '{updater_type}'  # 'set', 'accumulate', etc.
                }
            },
            'boundary': {
                # External signals/interactions
                '{external_var}': {
                    '_default': {default_value},
                    '_emit': True,
                    '_updater': '{updater_type}'
                }
            }
        }

    def next_update(self, timestep, states):
        \"\"\"
        Implement the biological mechanism update logic.
        Follow Vivarium pattern: check conditions, compute updates.
        \"\"\"
        update = {}

        # Example biological logic following threshold pattern
        if states['boundary']['{trigger_var}'] > self.parameters.get('{threshold_param}', {default_threshold}):
            # Implement biological response
            update['internal'] = {
                '{response_var}': {computation}
            }

        return update
"""''',

            "field_template": '''"""
Vivarium Field/Environment Process Pattern (based on field.py):

class {FieldName}:
    \"\"\"
    {Field description and environmental context}

    Handles spatial or environmental variables following Vivarium field pattern.
    \"\"\"

    def __init__(self, parameters=None):
        self.parameters = parameters or {}

    def ports_schema(self):
        \"\"\"
        Define field/environment schema following Vivarium accumulate pattern.
        \"\"\"
        return {
            'field': {
                '{field_variable}': {
                    '_default': {default_value},
                    '_emit': True,
                    '_updater': 'accumulate'  # Standard updater for fields
                }
            }
        }

    def next_update(self, timestep, states):
        \"\"\"
        Update field variables following Vivarium decay pattern.
        \"\"\"
        # Standard decay pattern
        decay = self.parameters.get('{decay_param}', {default_decay})
        new_value = states['field']['{field_variable}'] * (1 - decay)

        return {
            'field': {
                '{field_variable}': new_value
            }
        }
"""''',

            "composite_template": '''"""
Vivarium Composite Pattern (inferred from process structure):

class {CompositeName}:
    \"\"\"
    {Composite description and biological system overview}

    Integrates multiple processes following Vivarium organizational pattern.
    \"\"\"

    def __init__(self, config=None):
        config = config or {}

        # Vivarium pattern: simple initialization with parameter passing
        self.processes = {
            '{process_name}': {ProcessClass}(config.get('{process_name}', {}))
        }

        # Define topology following Vivarium internal/boundary pattern
        self.topology = {
            '{process_name}': {
                'internal': ('global_state', 'internal', '{process_name}'),
                'boundary': ('global_state', 'boundary', '{process_name}')
            }
        }

    def initial_state(self):
        \"\"\"Initialize state following Vivarium schema patterns.\"\"\"
        state = {
            'global_state': {
                'internal': {},
                'boundary': {}
            }
        }

        # Initialize from process schemas
        for process_name, process in self.processes.items():
            schema = process.ports_schema()
            for port_name, port_schema in schema.items():
                if port_name not in state['global_state']:
                    state['global_state'][port_name] = {}
                state['global_state'][port_name][process_name] = {}
                for var_name, var_config in port_schema.items():
                    state['global_state'][port_name][process_name][var_name] = var_config['_default']

        return state
"""''',

            "experiment_template": '''"""
Vivarium Experiment Pattern (following simple structure):

def run_{experiment_name}(config=None, total_time=100):
    \"\"\"
    Run {experiment_description} following Vivarium simple pattern.

    Args:
        config: Configuration dictionary for processes
        total_time: Simulation time

    Returns:
        Simulation results
    \"\"\"

    # Initialize following Vivarium pattern
    composite = {CompositeClass}(config)

    # Simple simulation loop (Vivarium approach)
    timestep = 1.0
    time = 0
    results = {
        'time': [],
        'states': []
    }

    # Initial state
    state = composite.initial_state()

    while time < total_time:
        # Record state
        results['time'].append(time)
        results['states'].append(state.copy())

        # Update all processes (Vivarium simple pattern)
        updates = {}
        for process_name, process in composite.processes.items():
            # Extract process-specific state
            process_state = extract_process_state(state, process_name, composite.topology)

            # Get process update
            process_update = process.next_update(timestep, process_state)

            # Apply update following Vivarium pattern
            apply_update(updates, process_update, process_name, composite.topology)

        # Apply all updates to state
        state = apply_state_update(state, updates)

        time += timestep

    # Generate plots following Vivarium simple approach
    plot_results(results)

    return results

def plot_results(results):
    \"\"\"Simple plotting following Vivarium approach.\"\"\"
    import matplotlib.pyplot as plt

    time = results['time']

    # Extract key variables for plotting
    # (This would be specific to the biological system)

    plt.figure(figsize=(10, 6))
    # Add specific plots based on the biological system
    plt.xlabel('Time')
    plt.ylabel('State Variables')
    plt.title('Simulation Results')
    plt.show()

if __name__ == '__main__':
    results = run_{experiment_name}()
"""'''
        }

    def generate_repository(self, model_specs: Dict[str, Any], parameters: Any, output_dir: str) -> str:
        """Generate repository using enhanced patterns and templates."""
        project_name = model_specs.get("project_name", "generated_model")

        # Enhanced prompt with pattern awareness
        prompt = f"""
Generate a complete, production-ready Vivarium repository following established patterns.

PROJECT NAME: {project_name}

MODEL SPECIFICATIONS:
{model_specs.get('description', 'Biological model')}

EXTRACTED PARAMETERS:
{parameters}

VIVARIUM PATTERNS TO FOLLOW:
{self._get_pattern_instructions()}

REQUIRED FILES WITH FULL IMPLEMENTATION:
1. setup.py - Complete package setup with proper dependencies
2. {project_name}/__init__.py - Package initialization
3. {project_name}/processes/ - All biological processes as separate classes
4. {project_name}/composites/ - System-level composite organization
5. {project_name}/experiments/ - Runnable experiments with plots
6. requirements.txt - All required packages
7. README.md - Complete documentation with usage examples
8. tests/ - Basic unit tests for processes

QUALITY REQUIREMENTS:
- Follow Vivarium process and composite patterns exactly
- Include proper schema definitions for all ports
- Implement meaningful biological mechanisms
- Add comprehensive docstrings and type hints
- Include parameter validation and error handling
- Generate publication-quality plots in experiments
- Ensure all code is immediately runnable

EMIT FILES USING THIS EXACT PATTERN:
FILE: relative/path/filename.ext
```python
# Complete, production-ready code
```
"""

        response = self.run([self.system_message(), {"role": "user", "content": prompt.strip()}])
        repo_path = self._create_and_populate_repository(response, output_dir, project_name)
        return str(repo_path)

    def _get_pattern_instructions(self) -> str:
        """Get formatted pattern instructions based on Vivarium templates."""
        return """
VIVARIUM PROCESS PATTERN (from generic_cell.py):
- Simple class definition (no Process inheritance)
- __init__(self, parameters=None) with parameters = parameters or {}
- ports_schema() method returning 'internal' and 'boundary' ports
- Each variable has '_default', '_emit': True, '_updater': 'set'/'accumulate'
- next_update(self, timestep, states) with threshold logic
- Use self.parameters.get('param_name', default_value) pattern

VIVARIUM FIELD PATTERN (from field.py):
- Same class structure as processes
- Use 'field' port with '_updater': 'accumulate'
- Implement decay pattern: new_value = old_value * (1 - decay_rate)
- Environmental/spatial variables go in 'field' port

VIVARIUM PORT ORGANIZATION:
- 'internal': cellular/process internal states
- 'boundary': external signals and interactions
- 'field': environmental/spatial variables
- Use meaningful variable names reflecting biology

VIVARIUM UPDATE LOGIC:
- Threshold responses: if states['boundary']['signal'] > threshold
- Parameter access: self.parameters.get('param', default)
- Return format: {'port_name': {'var_name': new_value}}
- Simple, readable conditional logic
"""

    def generate_enhanced_repository(self, user_request: str, model_specs: str, parameters: str,
                                     output_dir: str, project_name: Optional[str] = None) -> str:
        """Enhanced repository generation with pattern-aware templates."""
        if not project_name:
            # Generate project name from user request
            words = re.findall(r'\b[a-zA-Z]+\b', user_request.lower())
            bio_words = [w for w in words if len(w) > 3][:3]
            project_name = "_".join(bio_words) if bio_words else "biology_model"

        # Use the enhanced generate_repository method
        model_specs_dict = {
            "project_name": project_name,
            "description": f"Original Request: {user_request}\n\nModel Specifications:\n{model_specs}"
        }

        return self.generate_repository(model_specs_dict, parameters, output_dir)

    def _create_and_populate_repository(self, response: str, output_dir: str, project_name: str) -> Path:
        """Create repository with enhanced error handling and pattern validation."""
        repo_path = Path(output_dir) / f"vivarium_{project_name}"
        (repo_path).mkdir(parents=True, exist_ok=True)
        (repo_path / project_name / "processes").mkdir(parents=True, exist_ok=True)
        (repo_path / project_name / "composites").mkdir(parents=True, exist_ok=True)
        (repo_path / project_name / "experiments").mkdir(parents=True, exist_ok=True)
        (repo_path / "tests").mkdir(parents=True, exist_ok=True)

        # Parse "FILE: <path>\n```...```" blocks
        files = re.findall(r'FILE:\s*([^\n]+)\n```(?:python|txt|md|yaml)?\n(.*?)\n```', response, re.DOTALL)

        if not files:
            logger.warning("No FILE blocks found in LLM output. Falling back to enhanced skeleton.")
            self._create_enhanced_skeleton(repo_path, project_name)
        else:
            for rel_path, content in files:
                rel_path = rel_path.strip().replace("\\", "/")
                if not rel_path:
                    continue
                full_path = repo_path / rel_path
                full_path.parent.mkdir(parents=True, exist_ok=True)

                # Enhanced content processing
                processed_content = self._process_file_content(content, project_name, rel_path)
                full_path.write_text(processed_content, encoding="utf-8")

        self._ensure_enhanced_essentials(repo_path, project_name)
        self._validate_vivarium_patterns(repo_path, project_name)
        return repo_path

    def _process_file_content(self, content: str, project_name: str, file_path: str) -> str:
        """Process file content to ensure proper patterns and naming."""
        processed = content.strip()

        # Replace placeholder names
        processed = processed.replace("generated_model", project_name)
        processed = processed.replace("GENERATED_MODEL", project_name.upper())
        processed = processed.replace("{project_name}", project_name)

        # Add proper imports for Python files
        if file_path.endswith('.py') and 'processes/' in file_path:
            if 'from vivarium.core.process import Process' not in processed:
                processed = "from vivarium.core.process import Process\n" + processed

        return processed

    def _create_enhanced_skeleton(self, repo_path: Path, project_name: str):
        """Create enhanced skeleton with proper Vivarium patterns."""
        # Enhanced setup.py
        setup_py = f'''from setuptools import setup, find_packages

setup(
    name="vivarium-{project_name}",
    version="0.1.0",
    description="Vivarium biological model: {project_name}",
    author="Generated by Biology Agents",
    packages=find_packages(),
    install_requires=[
        "vivarium-core>=0.3.0",
        "numpy>=1.20.0",
        "matplotlib>=3.3.0",
        "scipy>=1.6.0",
        "pandas>=1.2.0"
    ],
    python_requires=">=3.8",
    entry_points={{
        'console_scripts': [
            '{project_name}='{project_name}.experiments.run_simulation:main',
        ],
    }},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
'''
        (repo_path / "setup.py").write_text(setup_py, encoding="utf-8")

        # Enhanced package __init__.py
        (repo_path / project_name / "__init__.py").write_text(
            f'"""\nVivarium biological model: {project_name}\n\nGenerated by Biology Agents Multi-Agent System\n"""\n\n__version__ = "0.1.0"\n__author__ = "Biology Agents"\n',
            encoding="utf-8"
        )

        # Enhanced README with usage examples
        readme_content = f"""# Vivarium {project_name}

A biological simulation model generated by the Biology Agents Multi-Agent System.

## Installation

```bash
cd vivarium_{project_name}
pip install -e .
```

## Quick Start

```python
from {project_name}.experiments.run_simulation import run_simulation

# Run default simulation
results = run_simulation(total_time=100)
```

## Structure

- `{project_name}/processes/` - Individual biological processes
- `{project_name}/composites/` - System-level model organization  
- `{project_name}/experiments/` - Simulation experiments and analysis
- `tests/` - Unit tests for model components

## Generated on

{time.strftime('%Y-%m-%d %H:%M:%S')}

## Next Steps

1. Review the generated processes in `{project_name}/processes/`
2. Customize parameters in the experiment files
3. Run simulations and analyze results
4. Extend the model with additional biological mechanisms
"""
        (repo_path / "README.md").write_text(readme_content, encoding="utf-8")

    def _ensure_enhanced_essentials(self, repo_path: Path, project_name: str):
        """Ensure all essential files exist with proper content."""
        # Enhanced requirements.txt
        requirements = """vivarium-core>=0.3.0
numpy>=1.20.0
matplotlib>=3.3.0
scipy>=1.6.0
pandas>=1.2.0
jupyter>=1.0.0
pytest>=6.0.0
"""
        (repo_path / "requirements.txt").write_text(requirements, encoding="utf-8")

        # Add __init__.py files to all package directories
        for subdir in ["processes", "composites", "experiments"]:
            init_file = repo_path / project_name / subdir / "__init__.py"
            if not init_file.exists():
                init_file.write_text(f'"""{subdir.title()} module for {project_name}"""\n', encoding="utf-8")

    def _validate_vivarium_patterns(self, repo_path: Path, project_name: str):
        """Validate that generated code follows Vivarium patterns."""
        validation_log = []

        # Check for process files
        processes_dir = repo_path / project_name / "processes"
        process_files = list(processes_dir.glob("*.py"))
        if not process_files:
            validation_log.append("Warning: No process files found")

        # Check for composite files
        composites_dir = repo_path / project_name / "composites"
        composite_files = list(composites_dir.glob("*.py"))
        if not composite_files:
            validation_log.append("Warning: No composite files found")

        # Log validation results
        if validation_log:
            logger.warning(f"Validation issues: {validation_log}")
        else:
            logger.info("Repository structure validation passed")

    def get_capabilities(self) -> List[str]:
        """Override to provide specific code generation capabilities."""
        base_caps = super().get_capabilities()
        return base_caps + [
            "code_generation", "repository_creation", "file_management",
            "python_packaging", "vivarium_structure", "documentation_generation",
            "pattern_implementation", "template_processing", "code_validation"
        ]


# -------- Critique / QA --------

class CritiqueAgent(Agent):
    """Simple critique/QA agent."""

    def __init__(self):
        super().__init__(
            title="Critique Agent",
            expertise="Scientific validation, quality assurance",
            goal="Evaluate and validate extracted parameters and model specifications",
            role="Quality assurance specialist",
            model="gpt-4o",
            output_format="Return bullet points: issues found, suggested fixes, and confidence per point.",
            enable_confidence=True,
            dependencies=["ParameterExtractorAgent"],
            mcp_enabled=True,
        )

    def comprehensive_critique(self, user_request: str, **components) -> str:
        """Critique that evaluates all workflow components."""
        conversation = [self.system_message()]

        critique_prompt = [f"COMPREHENSIVE QUALITY REVIEW FOR: {user_request}\n"]

        # Add each component for review
        component_map = {
            "task_plan": "TASK PLANNING",
            "biological_context": "BIOLOGICAL CONTEXT",
            "parameters": "EXTRACTED PARAMETERS",
            "model_specification": "MODEL SPECIFICATION",
            "generated_code": "GENERATED CODE"
        }

        for key, title in component_map.items():
            if key in components and components[key]:
                critique_prompt.append(f"\n{title}:\n{components[key]}")

        critique_prompt.append("""
EVALUATION CRITERIA:
- Scientific accuracy and biological plausibility
- Parameter completeness and reasonable values
- Model architecture soundness
- Code quality and executability
- Overall workflow coherence

PROVIDE:
- Major issues (if any)
- Minor concerns
- Strengths of the approach
- Specific recommendations for improvement
- Overall confidence score (0-1)
""")

        conversation.append({"role": "user", "content": "\n".join(critique_prompt)})
        return self.run(conversation)

    def get_capabilities(self) -> List[str]:
        """Override to provide specific critique capabilities."""
        base_caps = super().get_capabilities()
        return base_caps + [
            "quality_assurance", "validation", "scientific_review",
            "parameter_validation", "code_review", "workflow_assessment"
        ]


# -------- Utility Functions --------

def create_all_agents() -> Dict[str, Agent]:
    """Factory function to create all specialized agents."""
    return {
        "pi": PIAgent(),
        "biologist": BiologistAgent(),
        "parameter_extractor": ParameterExtractorAgent(),
        "model_builder": ModelBuilderAgent(),
        "code_generator": CodeGenerationAgent(),
        "critique": CritiqueAgent(),
    }


def get_agent_by_capability(capability: str) -> List[str]:
    """Get agents that have a specific capability."""
    agents = create_all_agents()
    matching_agents = []

    for agent_id, agent in agents.items():
        if hasattr(agent, 'get_capabilities') and capability in agent.get_capabilities():
            matching_agents.append(agent_id)

    return matching_agents