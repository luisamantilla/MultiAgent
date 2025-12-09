# Biological Modeling Module

## Purpose
This module contains all components for biological pathway modeling, multi-agent expert systems, and biological reasoning.

## Structure

### `/agents/` - Core Biological Agents
- `biologist_agent.py` - Main biological reasoning agent (consolidates current scattered versions)
- `pathway_expander_agent.py` - Expands high-level interactions into detailed mechanistic steps
- `parameter_mapper_agent.py` - Maps biological parameters to computational values
- `overlap_pruner_agent.py` - Removes duplicate/overlapping mechanistic steps

### `/experts/` - Specialized Expert Agents
- `biological_expert_agent.py` - Domain-specific biological expertise (immunology, cell biology, etc.)
- `experimental_expert_agent.py` - Experimental design and validation expertise
- `computational_expert_agent.py` - Computational modeling expertise
- `pi_agent.py` - Principal Investigator coordination and decision-making

### `/orchestration/` - Multi-Agent Coordination
- `review_orchestration.py` - Coordinates multi-expert review cycles
- `decision_analysis.py` - Consensus building and decision analysis
- `agent_selection.py` - Dynamic selection of appropriate experts

### `/models/` - Data Models
- `biological_entities.py` - Molecules, genes, species, interactions
- `pathway_models.py` - Biological pathway representations
- `review_models.py` - Review and consensus data models

### `/workflows/` - Predefined Workflows
- `standard_workflow.py` - Standard biological modeling pipeline
- `literature_workflow.py` - Literature-intensive analysis workflow
- `rapid_workflow.py` - Fast prototyping workflow

### `/tools/` - Utilities
- `pubmed_search.py` - PubMed literature search integration
- `pathway_validation.py` - Quality checks and validation
- `interaction_analysis.py` - Biological interaction analysis

## Key Files to Migrate Here
- From `agent-testing-framework/src/agents/biological_modeling_agents/`
- From `agent-testing-framework/src/agents/biological_modeling_agents/crosstalk/`
- From `demo/agents/` (consolidate)
- Workflow orchestration from `agent-testing-framework/src/framework/`