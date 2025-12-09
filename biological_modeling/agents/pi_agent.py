import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path
import sys

# Add the shared directory to Python path to import BaseAgent
_PROJECT_ROOT = Path(__file__).resolve().parents[2]  # Go up to MultiAgent/
_SHARED_DIR = _PROJECT_ROOT / "shared"
if str(_SHARED_DIR) not in sys.path:
    sys.path.insert(0, str(_SHARED_DIR))

from base.base_agent import BaseAgent

@dataclass
class ReviewDecision:
    """Represents a decision made by the PI agent"""
    step_id: str
    decision: str  # "keep", "modify", "remove", "merge"
    reasoning: str
    priority: int  # 1-5, where 5 is highest priority
    suggested_modifications: Optional[str] = None
    consensus_level: Optional[float] = None  # 0-1, based on expert agreement

@dataclass
class IterationSummary:
    """Summary of a complete iteration cycle"""
    iteration_number: int
    total_steps_reviewed: int
    decisions: List[ReviewDecision]
    expert_feedback_summary: Dict[str, Any]
    pi_rationale: str
    next_actions: List[str]
    timestamp: str

class PIAgent(BaseAgent):
    """
    Principal Investigator Agent - Orchestrates iterative review cycles with expert agents
    and makes final decisions on biological step refinement based on collective expertise.
    """

    def __init__(self, model: str = "gpt-4.1", base_output_dir: str = None):
        super().__init__(
            title="Principal Investigator Agent",
            expertise="Synthesizing diverse expert opinions, making strategic research decisions, balancing biological accuracy with computational feasibility",
            goal="To lead iterative refinement of biological pathways by coordinating expert feedback and making informed decisions on step inclusion/modification",
            role="Research Leadership and Decision Making",
            model=model
        )
        
        # Set up output directory
        if base_output_dir is None:
            base_output_dir = "/home/labuser/Desktop/lab_member_projects/Bobby_Ni/tumor-tcell-exp/biological_modeling_agents_design"
        
        self.base_output_dir = Path(base_output_dir)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = self.base_output_dir / f"pi_review_run_{timestamp}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"📁 PI Agent output directory: {self.output_dir}")

    def orchestrate_review_cycle(self, 
                                 steps_to_review: List[Any],
                                 expert_agents: List[Any],
                                 iteration_number: int = 1,
                                 review_criteria: Dict[str, float] = None) -> IterationSummary:
        """
        Orchestrates a complete review cycle with expert agents.
        
        Args:
            steps_to_review: List of biological steps to review
            expert_agents: List of expert agent instances
            iteration_number: Current iteration number
            review_criteria: Weights for different review criteria
            
        Returns:
            IterationSummary with decisions and rationale
        """
        
        print(f"\n🔄 Starting PI Review Cycle - Iteration {iteration_number}")
        print(f"📊 Reviewing {len(steps_to_review)} steps with {len(expert_agents)} expert agents")
        
        # Default review criteria if not provided
        if review_criteria is None:
            review_criteria = {
                "biological_accuracy": 0.3,
                "computational_feasibility": 0.2,
                "experimental_support": 0.2,
                "model_relevance": 0.2,
                "implementation_complexity": 0.1
            }
        
        # Collect expert feedback for all steps
        expert_feedback = self._collect_expert_feedback(steps_to_review, expert_agents)
        
        # Synthesize feedback and make decisions
        decisions = self._make_informed_decisions(steps_to_review, expert_feedback, review_criteria)
        
        # Generate iteration summary
        summary = self._generate_iteration_summary(
            iteration_number, steps_to_review, decisions, expert_feedback, review_criteria
        )
        
        # Save results
        self._save_iteration_results(summary)
        
        print(f"✅ PI Review Cycle {iteration_number} completed")
        print(f"📈 Decisions: {len([d for d in decisions if d.decision == 'keep'])} keep, "
              f"{len([d for d in decisions if d.decision == 'modify'])} modify, "
              f"{len([d for d in decisions if d.decision == 'remove'])} remove")
        
        return summary

    def _collect_expert_feedback(self, steps: List[Any], expert_agents: List[Any]) -> Dict[str, Any]:
        """Collect feedback from all expert agents"""
        
        feedback = {
            "by_expert": {},
            "by_step": {},
            "consensus_metrics": {}
        }
        
        print(f"📋 Collecting expert feedback...")
        
        for expert in expert_agents:
            expert_name = expert.__class__.__name__
            print(f"   Consulting {expert_name}...")
            
            expert_reviews = []
            for i, step in enumerate(steps):
                try:
                    review = expert.review_step(step, step_index=i)
                    expert_reviews.append(review)
                except Exception as e:
                    print(f"     ⚠️ Error getting review from {expert_name}: {e}")
                    expert_reviews.append({"error": str(e)})
            
            feedback["by_expert"][expert_name] = expert_reviews
        
        # Organize feedback by step
        for i, step in enumerate(steps):
            step_feedback = {}
            for expert_name, reviews in feedback["by_expert"].items():
                if i < len(reviews):
                    review = reviews[i]
                    # Check if it's an error dict or a successful review object
                    if isinstance(review, dict) and "error" in review:
                        # Skip error reviews
                        continue
                    else:
                        # It's a successful review object
                        step_feedback[expert_name] = review
            feedback["by_step"][f"step_{i}"] = step_feedback
        
        return feedback

    def _make_informed_decisions(self, 
                                steps: List[Any], 
                                feedback: Dict[str, Any],
                                criteria: Dict[str, float]) -> List[ReviewDecision]:
        """Make informed decisions based on expert feedback"""
        
        decisions = []
        
        print(f"🤔 Making informed decisions...")
        
        for i, step in enumerate(steps):
            step_feedback = feedback["by_step"].get(f"step_{i}", {})
            
            # Analyze expert consensus
            decision_analysis = self._analyze_step_consensus(step, step_feedback, criteria)
            
            decision = ReviewDecision(
                step_id=f"step_{i}",
                decision=decision_analysis["decision"],
                reasoning=decision_analysis["reasoning"],
                priority=decision_analysis["priority"],
                suggested_modifications=decision_analysis.get("modifications"),
                consensus_level=decision_analysis.get("consensus_level")
            )
            
            decisions.append(decision)
        
        return decisions

    def _analyze_step_consensus(self, step: Any, feedback: Dict[str, Any], criteria: Dict[str, float]) -> Dict[str, Any]:
        """Analyze expert consensus for a single step using LLM"""
        
        # Prepare feedback summary for LLM
        feedback_summary = []
        for expert_name, review in feedback.items():
            feedback_summary.append(f"{expert_name}: {review}")
        
        step_description = getattr(step, 'description', str(step))
        
        prompt = f"""
        As a Principal Investigator, analyze the following biological step and expert feedback to make an informed decision.

        STEP TO REVIEW:
        {step_description}

        EXPERT FEEDBACK:
        {chr(10).join(feedback_summary)}

        DECISION CRITERIA (weights):
        {json.dumps(criteria, indent=2)}

        Please provide your analysis in the following JSON format:
        {{
            "decision": "keep|modify|remove|merge",
            "reasoning": "detailed rationale for the decision",
            "priority": 1-5,
            "consensus_level": 0.0-1.0,
            "modifications": "suggested changes if decision is modify",
            "key_concerns": ["list", "of", "main concerns"],
            "supporting_evidence": ["list", "of", "supporting points"]
        }}

        Consider:
        - Biological accuracy and experimental support
        - Computational feasibility for modeling
        - Relevance to research objectives
        - Expert consensus level
        - Implementation complexity
        """

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            
            result = json.loads(response.choices[0].message.content)
            return result
            
        except Exception as e:
            print(f"⚠️ Error in decision analysis: {e}")
            return {
                "decision": "keep",
                "reasoning": f"Default decision due to analysis error: {e}",
                "priority": 3,
                "consensus_level": 0.5
            }

    def _generate_iteration_summary(self, 
                                   iteration_num: int,
                                   steps: List[Any],
                                   decisions: List[ReviewDecision],
                                   feedback: Dict[str, Any],
                                   criteria: Dict[str, float]) -> IterationSummary:
        """Generate comprehensive iteration summary"""
        
        # Summarize expert feedback
        expert_summary = {}
        for expert_name, reviews in feedback["by_expert"].items():
            expert_summary[expert_name] = {
                "total_reviews": len(reviews),
                "errors": len([r for r in reviews if isinstance(r, dict) and "error" in r])
            }
        
        # Generate PI rationale using LLM
        pi_rationale = self._generate_pi_rationale(decisions, criteria)
        
        # Determine next actions
        next_actions = self._determine_next_actions(decisions)
        
        return IterationSummary(
            iteration_number=iteration_num,
            total_steps_reviewed=len(steps),
            decisions=decisions,
            expert_feedback_summary=expert_summary,
            pi_rationale=pi_rationale,
            next_actions=next_actions,
            timestamp=datetime.now().isoformat()
        )

    def _generate_pi_rationale(self, decisions: List[ReviewDecision], criteria: Dict[str, float]) -> str:
        """Generate overall PI rationale for the iteration"""
        
        decision_summary = {}
        for decision in decisions:
            if decision.decision not in decision_summary:
                decision_summary[decision.decision] = 0
            decision_summary[decision.decision] += 1
        
        prompt = f"""
        As a Principal Investigator, provide a comprehensive rationale for this iteration's decisions.

        DECISION SUMMARY:
        {json.dumps(decision_summary, indent=2)}

        REVIEW CRITERIA USED:
        {json.dumps(criteria, indent=2)}

        HIGH PRIORITY CONCERNS:
        {[d.reasoning for d in decisions if d.priority >= 4][:5]}

        Provide a 2-3 paragraph summary explaining:
        1. Overall approach and strategy for this iteration
        2. Key factors influencing decisions
        3. Expected impact on model quality and research objectives
        """

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"PI Rationale generation error: {e}"

    def _determine_next_actions(self, decisions: List[ReviewDecision]) -> List[str]:
        """Determine next actions based on decisions"""
        
        actions = []
        
        modify_count = len([d for d in decisions if d.decision == "modify"])
        remove_count = len([d for d in decisions if d.decision == "remove"])
        high_priority = len([d for d in decisions if d.priority >= 4])
        
        if modify_count > 0:
            actions.append(f"Implement modifications for {modify_count} steps")
        
        if remove_count > 0:
            actions.append(f"Remove {remove_count} steps from pathway")
        
        if high_priority > 5:
            actions.append("Schedule focused review for high-priority items")
        
        if modify_count + remove_count > len(decisions) * 0.3:
            actions.append("Consider additional expert consultation")
        
        actions.append("Prepare refined pathway for next iteration")
        
        return actions

    def _save_iteration_results(self, summary: IterationSummary):
        """Save iteration results to files"""
        
        # Save main summary
        summary_dict = {
            "iteration_number": summary.iteration_number,
            "total_steps_reviewed": summary.total_steps_reviewed,
            "expert_feedback_summary": summary.expert_feedback_summary,
            "pi_rationale": summary.pi_rationale,
            "next_actions": summary.next_actions,
            "timestamp": summary.timestamp,
            "decisions": [
                {
                    "step_id": d.step_id,
                    "decision": d.decision,
                    "reasoning": d.reasoning,
                    "priority": d.priority,
                    "suggested_modifications": d.suggested_modifications,
                    "consensus_level": d.consensus_level
                }
                for d in summary.decisions
            ]
        }
        
        summary_file = self.output_dir / f"iteration_{summary.iteration_number}_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary_dict, f, indent=2)
        
        # Save decisions only
        decisions_file = self.output_dir / f"iteration_{summary.iteration_number}_decisions.json"
        with open(decisions_file, 'w') as f:
            json.dump([d.__dict__ for d in summary.decisions], f, indent=2)
        
        print(f"💾 Iteration results saved to {self.output_dir}")

    def generate_final_recommendations(self, all_iterations: List[IterationSummary]) -> Dict[str, Any]:
        """Generate final recommendations based on all iterations"""
        
        prompt = f"""
        As a Principal Investigator, analyze the complete iterative review process and provide final recommendations.

        ITERATION HISTORY:
        {json.dumps([{
            "iteration": s.iteration_number,
            "steps_reviewed": s.total_steps_reviewed,
            "decisions": len(s.decisions),
            "rationale": s.pi_rationale[:200] + "..."
        } for s in all_iterations], indent=2)}

        Provide comprehensive final recommendations in JSON format:
        {{
            "pathway_quality_assessment": "overall assessment",
            "key_improvements_made": ["list of improvements"],
            "remaining_concerns": ["list of concerns"],
            "implementation_readiness": "assessment of readiness",
            "next_research_priorities": ["list of priorities"],
            "methodological_insights": ["lessons learned"]
        }}
        """

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            
            recommendations = json.loads(response.choices[0].message.content)
            
            # Save final recommendations
            final_file = self.output_dir / "final_recommendations.json"
            with open(final_file, 'w') as f:
                json.dump(recommendations, f, indent=2)
            
            return recommendations
            
        except Exception as e:
            return {"error": f"Failed to generate final recommendations: {e}"}
