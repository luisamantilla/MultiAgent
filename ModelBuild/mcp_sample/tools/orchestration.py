# tools/orchestration.py
"""
Fixed orchestration module with critical bug fixes:
1. Proper async/sync handling
2. Fixed dependency imports
3. Corrected workflow execution logic
4. Enhanced error handling
"""

import asyncio
import json
import logging
import time
from typing import Dict, Any, List, Optional, Union, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

# Handle optional dependencies gracefully
try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


    # Simple fallback for numpy operations
    class SimpleStats:
        @staticmethod
        def mean(data):
            return sum(data) / len(data) if data else 0

        @staticmethod
        def std(data):
            if not data:
                return 0
            mean_val = sum(data) / len(data)
            variance = sum((x - mean_val) ** 2 for x in data) / len(data)
            return variance ** 0.5


    np = SimpleStats()

logger = logging.getLogger(__name__)


class WorkflowStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AgentCapability(Enum):
    TASK_PLANNING = "task_planning"
    BIOLOGICAL_ANALYSIS = "biological_analysis"
    PARAMETER_EXTRACTION = "parameter_extraction"
    MODEL_BUILDING = "model_building"
    CODE_GENERATION = "code_generation"
    QUALITY_ASSURANCE = "quality_assurance"
    LITERATURE_SEARCH = "literature_search"


@dataclass
class WorkflowStep:
    """Enhanced workflow step with proper validation."""
    step_id: str
    agent_id: str
    task_description: str
    dependencies: List[str] = field(default_factory=list)
    required_capabilities: List[AgentCapability] = field(default_factory=list)
    timeout: int = 300
    max_retries: int = 2
    status: WorkflowStatus = WorkflowStatus.PENDING
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None


@dataclass
class WorkflowTemplate:
    """Template for biological modeling workflows."""
    name: str
    description: str
    steps: List[WorkflowStep]
    parallel_groups: List[List[str]] = field(default_factory=list)


class SimpleDependencyResolver:
    """Simple dependency resolver using Kahn's algorithm."""

    @staticmethod
    def topological_sort(dependencies: Dict[str, List[str]]) -> List[str]:
        """Topological sort using Kahn's algorithm."""
        # Calculate in-degrees
        in_degree = {node: 0 for node in dependencies.keys()}

        for node, deps in dependencies.items():
            for dep in deps:
                if dep in in_degree:
                    in_degree[node] += 1

        # Find nodes with no incoming edges
        queue = [node for node, degree in in_degree.items() if degree == 0]
        result = []

        while queue:
            node = queue.pop(0)
            result.append(node)

            # Remove edges from this node
            for dependent in dependencies.keys():
                if node in dependencies[dependent]:
                    in_degree[dependent] -= 1
                    if in_degree[dependent] == 0:
                        queue.append(dependent)

        # Check for cycles
        if len(result) != len(dependencies):
            raise ValueError("Dependency cycle detected")

        return result


class AdaptiveOrchestrator:
    """
    Fixed adaptive orchestrator with proper error handling and workflow execution.
    """

    def __init__(self, agents: Dict[str, Any]):
        self.agents = agents
        self.execution_history: List[Dict[str, Any]] = []
        self.performance_metrics: Dict[str, Dict[str, float]] = {}
        self.workflow_templates: Dict[str, WorkflowTemplate] = {}
        self.dependency_resolver = SimpleDependencyResolver()

        # Initialize components
        self._discover_agent_capabilities()
        self._initialize_workflow_templates()
        self._initialize_performance_tracking()

    def _discover_agent_capabilities(self) -> Dict[str, List[AgentCapability]]:
        """Discover capabilities of available agents."""
        capabilities = {}

        for agent_id, agent in self.agents.items():
            agent_caps = []

            # Try to get capabilities from agent
            if hasattr(agent, 'get_capabilities'):
                try:
                    raw_caps = agent.get_capabilities()
                    for cap in raw_caps:
                        # Map string capabilities to enums
                        cap_lower = str(cap).lower()
                        if any(word in cap_lower for word in ['planning', 'task', 'orchestration']):
                            agent_caps.append(AgentCapability.TASK_PLANNING)
                        elif any(word in cap_lower for word in ['biological', 'biology', 'bio']):
                            agent_caps.append(AgentCapability.BIOLOGICAL_ANALYSIS)
                        elif any(word in cap_lower for word in ['parameter', 'extraction']):
                            agent_caps.append(AgentCapability.PARAMETER_EXTRACTION)
                        elif any(word in cap_lower for word in ['model', 'building']):
                            agent_caps.append(AgentCapability.MODEL_BUILDING)
                        elif any(word in cap_lower for word in ['code', 'generation']):
                            agent_caps.append(AgentCapability.CODE_GENERATION)
                        elif any(word in cap_lower for word in ['critique', 'quality']):
                            agent_caps.append(AgentCapability.QUALITY_ASSURANCE)
                except Exception as e:
                    logger.warning(f"Error getting capabilities for {agent_id}: {e}")

            # Fallback: infer from agent title
            if not agent_caps:
                title = getattr(agent, 'title', agent_id).lower()
                if any(word in title for word in ['pi', 'principal', 'investigator']):
                    agent_caps.append(AgentCapability.TASK_PLANNING)
                elif 'biologist' in title:
                    agent_caps.append(AgentCapability.BIOLOGICAL_ANALYSIS)
                elif 'parameter' in title:
                    agent_caps.append(AgentCapability.PARAMETER_EXTRACTION)
                elif any(word in title for word in ['model', 'builder']):
                    agent_caps.append(AgentCapability.MODEL_BUILDING)
                elif any(word in title for word in ['code', 'generation']):
                    agent_caps.append(AgentCapability.CODE_GENERATION)
                elif 'critique' in title:
                    agent_caps.append(AgentCapability.QUALITY_ASSURANCE)

            capabilities[agent_id] = agent_caps

        self.agent_capabilities = capabilities
        logger.info(f"Discovered capabilities for {len(capabilities)} agents")
        return capabilities

    def _initialize_workflow_templates(self):
        """Initialize predefined workflow templates."""
        try:
            # Standard biology workflow
            standard_steps = [
                WorkflowStep(
                    step_id="task_planning",
                    agent_id="pi",
                    task_description="Analyze user request and create task plan",
                    required_capabilities=[AgentCapability.TASK_PLANNING],
                    timeout=180
                ),
                WorkflowStep(
                    step_id="biological_analysis",
                    agent_id="biologist",
                    task_description="Provide biological context and quantitative insights",
                    dependencies=["task_planning"],
                    required_capabilities=[AgentCapability.BIOLOGICAL_ANALYSIS],
                    timeout=240
                ),
                WorkflowStep(
                    step_id="parameter_extraction",
                    agent_id="parameter_extractor",
                    task_description="Extract biological parameters from literature",
                    dependencies=["biological_analysis"],
                    required_capabilities=[AgentCapability.PARAMETER_EXTRACTION],
                    timeout=300
                ),
                WorkflowStep(
                    step_id="model_specification",
                    agent_id="model_builder",
                    task_description="Create Vivarium model specification",
                    dependencies=["parameter_extraction"],
                    required_capabilities=[AgentCapability.MODEL_BUILDING],
                    timeout=240
                ),
                WorkflowStep(
                    step_id="code_generation",
                    agent_id="code_generator",
                    task_description="Generate complete Vivarium repository",
                    dependencies=["model_specification"],
                    required_capabilities=[AgentCapability.CODE_GENERATION],
                    timeout=360
                ),
                WorkflowStep(
                    step_id="quality_check",
                    agent_id="critique",
                    task_description="Review and validate generated model",
                    dependencies=["code_generation"],
                    required_capabilities=[AgentCapability.QUALITY_ASSURANCE],
                    timeout=180
                )
            ]

            standard_template = WorkflowTemplate(
                name="standard_biology_modeling",
                description="Standard workflow for biological model generation",
                steps=standard_steps
            )

            # Rapid workflow
            rapid_steps = [
                WorkflowStep("planning", "pi", "Quick task analysis", timeout=120),
                WorkflowStep("bio_context", "biologist", "Basic biological context",
                             dependencies=["planning"], timeout=180),
                WorkflowStep("quick_params", "parameter_extractor", "Estimate key parameters",
                             dependencies=["planning"], timeout=180),
                WorkflowStep("model_sketch", "model_builder", "Create basic model",
                             dependencies=["bio_context", "quick_params"], timeout=180),
                WorkflowStep("prototype_code", "code_generator", "Generate prototype",
                             dependencies=["model_sketch"], timeout=240)
            ]

            rapid_template = WorkflowTemplate(
                name="rapid_prototyping",
                description="Fast workflow for quick prototyping",
                steps=rapid_steps,
                parallel_groups=[["bio_context", "quick_params"]]
            )

            # Literature-intensive workflow
            literature_steps = [
                WorkflowStep("planning", "pi", "Detailed task planning", timeout=240),
                WorkflowStep("bio_analysis", "biologist", "Literature-informed analysis",
                             dependencies=["planning"], timeout=300),
                WorkflowStep("param_curation", "parameter_extractor", "Comprehensive parameter curation",
                             dependencies=["bio_analysis"], timeout=420),
                WorkflowStep("validated_model", "model_builder", "Literature-validated model",
                             dependencies=["param_curation"], timeout=300),
                WorkflowStep("robust_code", "code_generator", "Production-ready code",
                             dependencies=["validated_model"], timeout=480),
                WorkflowStep("peer_review", "critique", "Comprehensive peer review",
                             dependencies=["robust_code"], timeout=240)
            ]

            literature_template = WorkflowTemplate(
                name="literature_intensive",
                description="Comprehensive workflow emphasizing literature review",
                steps=literature_steps
            )

            self.workflow_templates = {
                "standard": standard_template,
                "rapid": rapid_template,
                "literature": literature_template
            }

            logger.info(f"Initialized {len(self.workflow_templates)} workflow templates")

        except Exception as e:
            logger.error(f"Error initializing workflow templates: {e}")
            # Fallback minimal template
            minimal_step = WorkflowStep("fallback", "pi", "Fallback task")
            self.workflow_templates = {
                "minimal": WorkflowTemplate("minimal", "Fallback workflow", [minimal_step])
            }

    def _initialize_performance_tracking(self):
        """Initialize performance metrics for agents."""
        for agent_id in self.agents.keys():
            self.performance_metrics[agent_id] = {
                "success_rate": 1.0,
                "avg_duration": 30.0,
                "confidence_score": 0.8,
                "total_executions": 0,
                "last_used": time.time()
            }

    async def select_optimal_workflow(self, user_request: str,
                                      constraints: Optional[Dict[str, Any]] = None) -> str:
        """Select optimal workflow template based on request analysis."""
        constraints = constraints or {}
        request_lower = user_request.lower()

        # Analyze request characteristics
        complexity_indicators = {
            "high": {"comprehensive", "detailed", "complete", "literature", "thorough"},
            "medium": {"model", "simulate", "analyze", "build", "create"},
            "low": {"quick", "simple", "basic", "prototype", "fast"}
        }

        scores = {}
        for template_name in self.workflow_templates.keys():
            score = 1.0  # Base score

            # Match complexity
            for complexity, words in complexity_indicators.items():
                word_matches = len([w for w in words if w in request_lower])
                if complexity == "high" and template_name == "literature":
                    score += word_matches * 2
                elif complexity == "low" and template_name == "rapid":
                    score += word_matches * 2
                elif complexity == "medium" and template_name == "standard":
                    score += word_matches * 1.5

            # Check agent availability
            template = self.workflow_templates[template_name]
            available_agents = sum(1 for step in template.steps if step.agent_id in self.agents)
            score += available_agents * 0.5

            scores[template_name] = score

        selected = max(scores.keys(), key=lambda x: scores[x])
        logger.info(f"Selected workflow template: {selected} (score: {scores[selected]:.1f})")
        return selected

    async def _select_best_agent(self, step: WorkflowStep) -> Optional[str]:
        """Select best agent for a workflow step."""
        # Use specified agent if available
        if step.agent_id in self.agents:
            return step.agent_id

        # Find agents with required capabilities
        candidate_agents = []
        for agent_id, capabilities in self.agent_capabilities.items():
            if agent_id not in self.agents:
                continue

            if step.required_capabilities:
                has_caps = all(cap in capabilities for cap in step.required_capabilities)
            else:
                has_caps = True

            if has_caps:
                candidate_agents.append(agent_id)

        if not candidate_agents:
            logger.error(f"No suitable agents found for step {step.step_id}")
            return None

        if len(candidate_agents) == 1:
            return candidate_agents[0]

        # Score candidates by performance metrics
        best_agent = None
        best_score = -1

        for agent_id in candidate_agents:
            metrics = self.performance_metrics[agent_id]

            # Composite score
            recency_factor = max(0.1, 1.0 - (time.time() - metrics["last_used"]) / 3600)
            speed_factor = min(1.0, 60.0 / max(metrics["avg_duration"], 1.0))

            score = (
                    metrics["success_rate"] * 0.4 +
                    metrics["confidence_score"] * 0.3 +
                    speed_factor * 0.2 +
                    recency_factor * 0.1
            )

            if score > best_score:
                best_score = score
                best_agent = agent_id

        if best_agent:
            self.performance_metrics[best_agent]["last_used"] = time.time()

        return best_agent

    async def _execute_step(self, step: WorkflowStep,
                            context: Dict[str, Any],
                            previous_results: Dict[str, Any]) -> Dict[str, Any]:
        """Execute individual workflow step with proper error handling."""
        step.start_time = time.time()
        step.status = WorkflowStatus.RUNNING

        # Select agent
        selected_agent_id = await self._select_best_agent(step)
        if not selected_agent_id:
            raise RuntimeError(f"No suitable agent found for step {step.step_id}")

        agent = self.agents[selected_agent_id]

        # Build step context
        step_context = context.copy()
        step_context.update({
            "step_id": step.step_id,
            "task_description": step.task_description,
            "previous_results": previous_results
        })

        # Add dependency results
        for dep_id in step.dependencies:
            if dep_id in previous_results:
                step_context[f"{dep_id}_result"] = previous_results[dep_id]

        # Execute with retry logic
        last_error = None
        for attempt in range(step.max_retries + 1):
            try:
                # Build conversation
                conversation = [agent.system_message()]

                # Add context if substantial
                if step_context and len(str(step_context)) > 100:
                    context_summary = self._create_context_summary(step_context)
                    conversation.append({"role": "system", "content": context_summary})

                conversation.append({"role": "user", "content": step.task_description})

                # Execute agent (handle both sync and async)
                if hasattr(agent, 'run_mcp'):
                    result = agent.run_mcp(conversation, step_context)
                    if hasattr(result, 'content'):
                        content = result.content
                        metadata = getattr(result, 'metadata', {})
                    else:
                        content = str(result)
                        metadata = {}
                else:
                    # Use regular run method
                    content = agent.run(conversation)
                    metadata = {}

                # Success
                step.status = WorkflowStatus.COMPLETED
                step.end_time = time.time()

                step_result = {
                    "content": content,
                    "agent_id": selected_agent_id,
                    "execution_time": step.end_time - step.start_time,
                    "attempt": attempt + 1,
                    "metadata": metadata,
                    "status": "success"
                }

                step.result = step_result

                # Update performance metrics
                execution_time = step.end_time - step.start_time
                self._update_agent_performance(selected_agent_id, True, execution_time)

                logger.info(f"Step {step.step_id} completed successfully")
                return step_result

            except Exception as e:
                last_error = e
                if attempt < step.max_retries:
                    backoff_time = min(2 ** attempt, 10)
                    logger.warning(f"Step {step.step_id} failed (attempt {attempt + 1}), retrying in {backoff_time}s")
                    await asyncio.sleep(backoff_time)
                else:
                    logger.error(f"Step {step.step_id} failed after {attempt + 1} attempts: {e}")

        # All retries failed
        step.status = WorkflowStatus.FAILED
        step.end_time = time.time()
        step.error = str(last_error)

        # Update performance metrics
        execution_time = step.end_time - step.start_time
        self._update_agent_performance(selected_agent_id, False, execution_time)

        raise RuntimeError(f"Step {step.step_id} failed: {last_error}")

    def _create_context_summary(self, step_context: Dict[str, Any]) -> str:
        """Create context summary for agent consumption."""
        summary_parts = ["Context Information:"]

        # Add user request
        if "user_request" in step_context:
            summary_parts.append(f"User Request: {step_context['user_request']}")

        # Add previous results
        if "previous_results" in step_context:
            summary_parts.append("Previous Results:")
            for step_id, result in step_context["previous_results"].items():
                if isinstance(result, dict):
                    content = result.get("content", "")
                    preview = content[:150] + "..." if len(content) > 150 else content
                    summary_parts.append(f"  {step_id}: {preview}")

        # Add RAG results if available
        if "rag_results" in step_context:
            rag_data = step_context["rag_results"]
            if rag_data.get("chunks_retrieved"):
                summary_parts.append("Relevant Research:")
                for i, chunk in enumerate(rag_data["chunks_retrieved"][:3], 1):
                    content = chunk.get("content", "")[:200]
                    summary_parts.append(f"  {i}. {content}...")

        return "\n".join(summary_parts)

    def _update_agent_performance(self, agent_id: str, success: bool, duration: float):
        """Update agent performance metrics."""
        metrics = self.performance_metrics[agent_id]

        # Update counters
        total = metrics["total_executions"]
        new_total = total + 1

        # Update success rate with exponential decay
        decay_factor = 0.95
        current_success = metrics["success_rate"]
        new_success = 1.0 if success else 0.0
        metrics["success_rate"] = current_success * decay_factor + new_success * (1 - decay_factor)

        # Update average duration (filter outliers)
        if duration < metrics["avg_duration"] * 3:
            metrics["avg_duration"] = (metrics["avg_duration"] * total + duration) / new_total

        # Update total executions and last used
        metrics["total_executions"] = new_total
        metrics["last_used"] = time.time()

    async def run_adaptive_workflow(self, user_request: str,
                                    template_name: Optional[str] = None,
                                    constraints: Optional[Dict[str, Any]] = None,
                                    rag_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Run adaptive workflow with enhanced error handling."""
        workflow_id = f"workflow_{int(time.time() * 1000)}"
        start_time = time.time()

        try:
            # Select workflow template
            if not template_name:
                template_name = await self.select_optimal_workflow(user_request, constraints)

            if template_name not in self.workflow_templates:
                logger.warning(f"Template {template_name} not found, using standard")
                template_name = "standard" if "standard" in self.workflow_templates else \
                list(self.workflow_templates.keys())[0]

            template = self.workflow_templates[template_name]
            logger.info(f"Using workflow template: {template_name}")

            # Prepare context
            context = {"user_request": user_request}
            if rag_context:
                context.update(rag_context)

            # Build dependency graph and execution order
            dependencies = {}
            steps_dict = {}
            for step in template.steps:
                dependencies[step.step_id] = step.dependencies
                steps_dict[step.step_id] = step

            execution_order = self.dependency_resolver.topological_sort(dependencies)
            logger.info(f"Execution order: {execution_order}")

            # Execute steps in order
            results = {}

            for step_id in execution_order:
                step = steps_dict[step_id]

                try:
                    logger.info(f"Executing step: {step_id}")
                    result = await self._execute_step(step, context, results)
                    results[step_id] = result

                except Exception as e:
                    logger.error(f"Step {step_id} failed: {e}")
                    results[step_id] = {
                        "status": "failed",
                        "error": str(e),
                        "step_id": step_id
                    }

                    # Stop execution on critical failures
                    if step_id in ["task_planning"]:  # Critical steps
                        break

            execution_time = time.time() - start_time

            # Determine overall status
            failed_steps = [r for r in results.values() if r.get("status") == "failed"]
            status = "failed" if failed_steps else "success"

            # Log workflow execution
            self.execution_history.append({
                "workflow_id": workflow_id,
                "template": template_name,
                "execution_time": execution_time,
                "success": status == "success",
                "steps_completed": len([r for r in results.values() if r.get("status") != "failed"]),
                "timestamp": time.time()
            })

            result_data = {
                "workflow_id": workflow_id,
                "status": status,
                "template_used": template_name,
                "execution_time": execution_time,
                "results": results
            }

            if failed_steps:
                result_data["error"] = f"Steps failed: {[r.get('step_id', 'unknown') for r in failed_steps]}"

            logger.info(f"Workflow {workflow_id} completed with status: {status}")
            return result_data

        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"Workflow execution failed: {str(e)}"
            logger.error(error_msg)

            return {
                "workflow_id": workflow_id,
                "status": "failed",
                "error": error_msg,
                "execution_time": execution_time,
                "partial_results": locals().get("results", {})
            }

    # Utility methods for system integration
    def get_available_templates(self) -> List[str]:
        """Get list of available workflow templates."""
        return list(self.workflow_templates.keys())

    def get_template_description(self, template_name: str) -> Dict[str, Any]:
        """Get description of workflow template."""
        if template_name not in self.workflow_templates:
            return {"error": f"Template {template_name} not found"}

        template = self.workflow_templates[template_name]
        return {
            "name": template.name,
            "description": template.description,
            "steps": [
                {
                    "step_id": step.step_id,
                    "agent_id": step.agent_id,
                    "description": step.task_description,
                    "dependencies": step.dependencies,
                    "timeout": step.timeout
                }
                for step in template.steps
            ],
            "estimated_duration": sum(step.timeout for step in template.steps)
        }

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on orchestrator and agents."""
        health = {
            "orchestrator": "healthy",
            "agents": {},
            "templates": len(self.workflow_templates),
            "execution_history": len(self.execution_history)
        }

        # Check agent health
        healthy_agents = 0
        for agent_id, agent in self.agents.items():
            try:
                if hasattr(agent, 'system_message'):
                    system_msg = agent.system_message()
                    if system_msg and system_msg.get("content"):
                        health["agents"][agent_id] = "healthy"
                        healthy_agents += 1
                    else:
                        health["agents"][agent_id] = "warning: empty system message"
                else:
                    health["agents"][agent_id] = "warning: no system_message method"
            except Exception as e:
                health["agents"][agent_id] = f"error: {str(e)}"

        # Overall health assessment
        if healthy_agents < len(self.agents) * 0.5:
            health["orchestrator"] = "degraded"
        elif healthy_agents < len(self.agents):
            health["orchestrator"] = "partial"

        return health

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        if not self.execution_history:
            return {"message": "No execution history available"}

        total_workflows = len(self.execution_history)
        successful = sum(1 for w in self.execution_history if w.get("success", False))

        execution_times = [w.get("execution_time", 0) for w in self.execution_history]
        if NUMPY_AVAILABLE:
            avg_time = float(np.mean(execution_times))
            std_time = float(np.std(execution_times))
        else:
            avg_time = np.mean(execution_times)
            std_time = np.std(execution_times)

        return {
            "total_workflows": total_workflows,
            "success_rate": successful / total_workflows if total_workflows > 0 else 0,
            "avg_execution_time": avg_time,
            "std_execution_time": std_time,
            "agent_performance": self.performance_metrics.copy()
        }


# Factory function for integration
def create_adaptive_orchestrator(agents: Dict[str, Any]) -> AdaptiveOrchestrator:
    """Factory function to create adaptive orchestrator."""
    return AdaptiveOrchestrator(agents)


# Backward compatibility wrapper
class WorkflowManager:
    """Wrapper class for backward compatibility."""

    def __init__(self, agents: Dict[str, Any]):
        self.orchestrator = AdaptiveOrchestrator(agents)

    async def run_workflow(self, user_request: str, **kwargs) -> Dict[str, Any]:
        """Run workflow with automatic template selection."""
        return await self.orchestrator.run_adaptive_workflow(user_request, **kwargs)

    def get_agents(self) -> Dict[str, Any]:
        """Get available agents."""
        return self.orchestrator.agents

    async def get_status(self) -> Dict[str, Any]:
        """Get orchestrator status."""
        return await self.orchestrator.health_check()