# Code Repository Generation Module

## Purpose
Generate complete computational repositories from biological specifications, including project scaffolding, dependency management, and code execution validation.

## Structure

### `/agents/` - Code Generation Agents
- `code_generation_agent.py` - Main code generator from biological specifications
- `structure_agent.py` - Project structure and scaffolding generation
- `dependency_predictor_agent.py` - Dependency analysis and requirements generation
- `evaluator_agent.py` - Code quality evaluation and validation
- `template_mapping_agent.py` - Template-based code generation

### `/templates/` - Code Templates
- `vivarium_template/` - Complete Vivarium project template
- `abm_template/` - Agent-based model template
- `python_package_template/` - Generic Python package structure
- `jupyter_template/` - Analysis notebook templates

### `/generators/` - Specialized Generators
- `vivarium_generator.py` - Vivarium-specific code generation
- `dependency_generator.py` - Requirements and setup.py generation
- `documentation_generator.py` - Automated documentation generation

### `/execution/` - Code Execution and Validation
- `code_execution_agent.py` - Safe code execution in Docker containers
- `validation_tools.py` - Code validation utilities
- `docker_manager.py` - Docker container management

### `/tools/` - Code Generation Utilities
- `ast_analysis.py` - Abstract syntax tree analysis
- `import_resolver.py` - Import dependency resolution
- `quality_checker.py` - Code quality metrics and linting

## Key Files to Migrate Here
- From `agent-testing-framework/src/agents/code_understanding_agents/`
- From `agent-testing-framework/src/agents/core_utility_agents/code_execution_agent.py`
- From `agent-testing-framework/src/framework/` (project generation scripts)
- From `ModelBuild/` (advanced MCP integration, if applicable)