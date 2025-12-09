# Shared Infrastructure Module

## Purpose
Common utilities, base classes, memory management, and orchestration framework shared across all modules.

## Structure

### `/base/` - Base Classes and Interfaces
- `base_agent.py` - Core agent base class with common functionality
- `base_workflow.py` - Workflow base class for all workflow types
- `interfaces.py` - Common interfaces and abstract classes

### `/memory/` - Memory and Context Management
- `unified_chroma_memory_manager.py` - Vector database memory management
- `conversation_memory.py` - Conversation history management
- `context_manager.py` - Context and state management across agents

### `/utils/` - Common Utilities
- `file_utils.py` - File operations and path management
- `logging_utils.py` - Standardized logging configuration
- `config_manager.py` - Configuration management
- `validation_utils.py` - Common validation functions

### `/orchestration/` - General Orchestration Framework
- `workflow_orchestrator.py` - General-purpose workflow execution engine
- `agent_registry.py` - Agent discovery and registration system
- `task_planner.py` - Task planning and scheduling

### `/models/` - Common Data Models
- `workflow_models.py` - Workflow and task definitions
- `result_models.py` - Standard result formats
- `config_models.py` - Configuration schemas

## Key Files to Migrate Here
- From `agent-testing-framework/src/agents/base_agent.py`
- From `agent-testing-framework/src/memory/`
- From `agent-testing-framework/src/utils/`
- Common orchestration logic from various modules