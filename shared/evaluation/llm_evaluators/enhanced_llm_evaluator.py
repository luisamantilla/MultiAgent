"""
Enhanced LLM Evaluator for Biological Interactions
Implements step-level analysis and comprehensive entity coverage tracking
"""

import os
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import importlib.util

# Import BaseAgent using the same pattern as other agents
_REPO_SRC = Path(__file__).resolve().parents[6]
_BASE_AGENT_PATH = _REPO_SRC / "agents" / "base_agent.py"
_spec = importlib.util.spec_from_file_location("repo_base_agent", str(_BASE_AGENT_PATH))
repo_base_agent = importlib.util.module_from_spec(_spec)  # type: ignore
assert _spec and _spec.loader
_spec.loader.exec_module(repo_base_agent)  # type: ignore
BaseAgent = repo_base_agent.BaseAgent  # type: ignore


@dataclass
class StepLevelResult:
    """Individual step evaluation result"""
    process_name: str
    step_detail: str
    matched_prediction: Optional[str]
    human_judgment: str = "pending"  # "good" | "bad" | "pending"
    evidence_comparison: Dict[str, str] = None
    
    def to_dict(self):
        return {
            "process_name": self.process_name,
            "step_detail": self.step_detail,
            "matched_prediction": self.matched_prediction,
            "human_judgment": self.human_judgment,
            "evidence_comparison": self.evidence_comparison or {}
        }


@dataclass
class EntityCoverage:
    """Entity coverage tracking"""
    entity_type: str  # "cell_types" | "molecules" | "other_entities"
    entity_name: str
    found_in_predictions: bool
    prediction_variants: List[str]
    synonym_matches: List[str]
    
    def to_dict(self):
        return {
            "entity_type": self.entity_type,
            "entity_name": self.entity_name,
            "found_in_predictions": self.found_in_predictions,
            "prediction_variants": self.prediction_variants,
            "synonym_matches": self.synonym_matches
        }


@dataclass
class ProcessCoverage:
    """Process-level coverage summary"""
    process_name: str
    total_steps: int
    matched_steps: int
    missing_steps: List[str]
    extra_predictions: List[str]
    
    def to_dict(self):
        return {
            "process_name": self.process_name,
            "total_steps": self.total_steps,
            "matched_steps": self.matched_steps,
            "missing_steps": self.missing_steps,
            "extra_predictions": self.extra_predictions
        }


class StepDecomposerAgent(BaseAgent):
    """LLM agent that extracts individual steps from BiologicalProcesses"""
    
    def __init__(self, model: str = "gpt-4o"):
        super().__init__(
            title="Step Decomposer",
            expertise="Biological process step extraction and standardization",
            goal="Break complex biological processes into individual, evaluable steps",
            role="Process Analyst",
            model=model,
        )
        self.system_prompt = (
            "You are a biological process analyst. Your role is to extract individual, "
            "concrete steps from complex biological process descriptions. Each step should "
            "describe a specific biological interaction, mechanism, or event that can be "
            "independently evaluated."
        )
    
    def extract_steps(self, biological_processes: List[Dict]) -> List[Dict]:
        """Extract individual steps from BiologicalProcesses"""
        
        prompt = f"""
{self.construct_prompt("Extract individual steps from biological processes")}

For each biological process, extract ALL individual steps as separate, evaluable units.
Each step should be a specific biological interaction or mechanism.

Input biological processes:
{json.dumps(biological_processes, indent=2)}

Return JSON with this structure:
{{
  "extracted_steps": [
    {{
      "process_name": "name of the biological process",
      "step_detail": "specific biological step or interaction",
      "step_type": "interaction" | "mechanism" | "event",
      "entities_involved": ["cell type", "molecule", "etc"],
      "biological_context": "brief context if needed"
    }}
  ]
}}

Be comprehensive - extract every meaningful biological step, interaction, and mechanism.
Use clear, specific language for each step.
"""
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.2,
            response_format={"type": "json_object"}
        )
        
        try:
            result = json.loads(response.choices[0].message.content)
            return result.get("extracted_steps", [])
        except json.JSONDecodeError:
            return []


class EntityTrackerAgent(BaseAgent):
    """LLM agent that tracks entity coverage with synonym detection"""
    
    def __init__(self, model: str = "gpt-4o"):
        super().__init__(
            title="Entity Coverage Tracker",
            expertise="Biological entity identification and synonym matching",
            goal="Track coverage of specific cell types, molecules, and entities",
            role="Entity Analyst",
            model=model,
        )
        self.system_prompt = (
            "You are a biological entity analyst. Your role is to identify and match "
            "biological entities (cell types, molecules, etc.) across different texts, "
            "accounting for synonyms, abbreviations, and variant naming conventions."
        )
    
    def analyze_entity_coverage(
        self, 
        ground_truth_entities: Dict[str, List[str]], 
        prediction_texts: List[str]
    ) -> List[EntityCoverage]:
        """Analyze entity coverage with synonym detection"""
        
        prompt = f"""
{self.construct_prompt("Analyze entity coverage with synonym detection")}

Ground truth entities to track:
{json.dumps(ground_truth_entities, indent=2)}

Prediction texts to search:
{json.dumps(prediction_texts, indent=2)}

For each ground truth entity, determine if it appears in the predictions (including synonyms).

Return JSON with this structure:
{{
  "entity_coverage": [
    {{
      "entity_type": "cell_types" | "molecules" | "other_entities",
      "entity_name": "canonical name from ground truth",
      "found_in_predictions": true | false,
      "prediction_variants": ["how it appears in predictions"],
      "synonym_matches": ["alternative names found"],
      "biological_reasoning": "why this entity is/isn't represented"
    }}
  ]
}}

Consider biological synonyms like:
- "NK cell" = "Natural Killer cell" = "Natural Killer"
- "IFNg" = "IFN-γ" = "interferon gamma" = "interferon-gamma"
- "CD8+ T" = "CD8 T cell" = "cytotoxic T lymphocyte" = "CTL"
- "TNF-a" = "TNF-α" = "tumor necrosis factor alpha"

Be thorough in synonym detection and provide biological reasoning.
"""
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.2,
            response_format={"type": "json_object"}
        )
        
        try:
            result = json.loads(response.choices[0].message.content)
            coverage_list = []
            
            for item in result.get("entity_coverage", []):
                coverage_list.append(EntityCoverage(
                    entity_type=item.get("entity_type", ""),
                    entity_name=item.get("entity_name", ""),
                    found_in_predictions=item.get("found_in_predictions", False),
                    prediction_variants=item.get("prediction_variants", []),
                    synonym_matches=item.get("synonym_matches", [])
                ))
            
            return coverage_list
        except json.JSONDecodeError:
            return []


class StepMatcherAgent(BaseAgent):
    """LLM agent that matches ground truth steps with predictions using biological reasoning"""
    
    def __init__(self, model: str = "gpt-4o"):
        super().__init__(
            title="Step Matcher",
            expertise="Biological interaction matching and semantic comparison", 
            goal="Match ground truth steps with predicted interactions using biological understanding",
            role="Semantic Matcher",
            model=model,
        )
        self.system_prompt = (
            "You are a biological interaction matcher. Your role is to determine if "
            "predicted biological interactions represent the same biological mechanisms "
            "as ground truth steps, considering semantic equivalence, biological context, "
            "and mechanistic similarity."
        )
    
    def match_steps_to_predictions(
        self,
        ground_truth_steps: List[Dict],
        prediction_interactions: List[str],
        evaluation_mode: str = "lenient"
    ) -> List[StepLevelResult]:
        """Match each ground truth step with predictions"""
        
        mode_instructions = {
            "strict": "Only match if the biological mechanisms are nearly identical",
            "lenient": "Allow partial matches and reasonable biological equivalences", 
            "process_aware": "Consider broader biological pathway context when matching"
        }
        
        prompt = f"""
{self.construct_prompt("Match ground truth biological steps with predictions")}

Evaluation Mode: {evaluation_mode.upper()}
Instructions: {mode_instructions.get(evaluation_mode, mode_instructions["lenient"])}

Ground Truth Steps:
{json.dumps(ground_truth_steps, indent=2)}

Predicted Interactions:
{json.dumps(prediction_interactions, indent=2)}

For each ground truth step, find the best matching prediction (if any).

Return JSON with this structure:
{{
  "step_matches": [
    {{
      "process_name": "name of biological process",
      "step_detail": "ground truth step description",
      "matched_prediction": "best matching prediction text" | null,
      "biological_reasoning": "why this is/isn't a match",
      "evidence_comparison": {{
        "truth": "relevant ground truth text",
        "pred": "relevant prediction text" | null
      }}
    }}
  ],
  "unmatched_predictions": [
    {{
      "prediction": "prediction text",
      "reason": "why this doesn't match any ground truth step"
    }}
  ]
}}

Consider biological equivalences:
- Same mechanism with different cell types may still match
- Directional relationships: "A activates B" ≈ "B is activated by A"
- Mechanistic equivalence: "cytotoxic killing" ≈ "CD8 T cell elimination" 
- Pathway context: related steps in same biological pathway

Provide clear biological reasoning for each match/non-match decision.
"""
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.2,
            response_format={"type": "json_object"}
        )
        
        try:
            result = json.loads(response.choices[0].message.content)
            matches = []
            
            for match in result.get("step_matches", []):
                matches.append(StepLevelResult(
                    process_name=match.get("process_name", ""),
                    step_detail=match.get("step_detail", ""),
                    matched_prediction=match.get("matched_prediction"),
                    evidence_comparison=match.get("evidence_comparison", {})
                ))
            
            return matches, result.get("unmatched_predictions", [])
        except json.JSONDecodeError:
            return [], []


class EnhancedLLMEvaluator:
    """Enhanced LLM evaluator with step-level analysis and entity coverage"""
    
    def __init__(self, model: str = "gpt-4o", output_dir: str = None):
        self.model = model
        self.step_decomposer = StepDecomposerAgent(model)
        self.entity_tracker = EntityTrackerAgent(model)
        self.step_matcher = StepMatcherAgent(model)
        
        if output_dir:
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)
        else:
            eval_dir = Path(__file__).parent
            self.output_dir = eval_dir / "enhanced_llm_eval_outputs"
            self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def evaluate(
        self,
        pred_file: str,
        truth_file: str,
        evaluation_mode: str = "lenient",
        context: str = "",
        save_intermediate: bool = True
    ) -> Dict[str, Any]:
        """Full enhanced evaluation pipeline"""
        
        # Load files
        pred_data = json.loads(Path(pred_file).read_text())
        truth_data = json.loads(Path(truth_file).read_text())
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print("Step 1: Extracting steps from biological processes...")
        # Extract individual steps from biological processes
        biological_processes = truth_data.get("BiologicalProcesses", [])
        extracted_steps = self.step_decomposer.extract_steps(biological_processes)
        
        print("Step 2: Analyzing entity coverage...")
        # Analyze entity coverage
        ground_truth_entities = {
            "cell_types": truth_data.get("cell_types", []),
            "molecules": truth_data.get("molecules", []),
            "other_entities": truth_data.get("other_entities", [])
        }
        
        prediction_texts = self._extract_prediction_texts(pred_data)
        entity_coverage = self.entity_tracker.analyze_entity_coverage(
            ground_truth_entities, prediction_texts
        )
        
        print("Step 3: Matching steps with predictions...")
        # Match steps with predictions
        step_matches, unmatched_predictions = self.step_matcher.match_steps_to_predictions(
            extracted_steps, prediction_texts, evaluation_mode
        )
        
        print("Step 4: Generating process coverage summary...")
        # Generate process coverage summaries
        process_coverage = self._generate_process_coverage(
            biological_processes, step_matches, unmatched_predictions
        )
        
        # Compute overall metrics
        metrics = self._compute_enhanced_metrics(step_matches, entity_coverage)
        
        # Generate comprehensive report
        report = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "model": self.model,
                "pred_file": pred_file,
                "truth_file": truth_file,
                "evaluation_mode": evaluation_mode,
                "context": context,
                "step_decomposer_agent": self.step_decomposer.report(),
                "entity_tracker_agent": self.entity_tracker.report(),
                "step_matcher_agent": self.step_matcher.report()
            },
            "metrics": metrics,
            "step_level_results": [step.to_dict() for step in step_matches],
            "entity_coverage": [entity.to_dict() for entity in entity_coverage],
            "process_coverage": [proc.to_dict() for proc in process_coverage],
            "unmatched_predictions": unmatched_predictions,
            "extracted_steps_count": len(extracted_steps),
            "prediction_texts_count": len(prediction_texts)
        }
        
        if save_intermediate:
            # Save detailed report
            report_file = self.output_dir / f"enhanced_evaluation_{timestamp}.json"
            report_file.write_text(json.dumps(report, indent=2))
            print(f"Saved detailed report to {report_file}")
            
            # Save human-readable summary
            summary_file = self.output_dir / f"enhanced_summary_{timestamp}.md"
            summary_file.write_text(self._generate_markdown_summary(report))
            print(f"Saved summary to {summary_file}")
        
        return report
    
    def _extract_prediction_texts(self, pred_data: Dict) -> List[str]:
        """Extract prediction texts from PI selected interactions"""
        pi_selected = pred_data.get("pi", {}).get("selected", {})
        return [str(x) for x in pi_selected.get("interactions", [])]
    
    def _generate_process_coverage(
        self, 
        biological_processes: List[Dict],
        step_matches: List[StepLevelResult],
        unmatched_predictions: List[Dict]
    ) -> List[ProcessCoverage]:
        """Generate process-level coverage summaries"""
        
        process_coverage = []
        
        # Group step matches by process
        process_steps = {}
        for step in step_matches:
            if step.process_name not in process_steps:
                process_steps[step.process_name] = []
            process_steps[step.process_name].append(step)
        
        for process in biological_processes:
            process_name = process.get("name", "Unknown Process")
            steps = process_steps.get(process_name, [])
            
            total_steps = len(process.get("steps", []))
            matched_steps = len([s for s in steps if s.matched_prediction is not None])
            missing_steps = [s.step_detail for s in steps if s.matched_prediction is None]
            
            process_coverage.append(ProcessCoverage(
                process_name=process_name,
                total_steps=total_steps,
                matched_steps=matched_steps,
                missing_steps=missing_steps,
                extra_predictions=[p.get("prediction", "") for p in unmatched_predictions]
            ))
        
        return process_coverage
    
    def _compute_enhanced_metrics(
        self, 
        step_matches: List[StepLevelResult],
        entity_coverage: List[EntityCoverage]
    ) -> Dict:
        """Compute enhanced evaluation metrics"""
        
        # Step-level metrics
        total_steps = len(step_matches)
        matched_steps = len([s for s in step_matches if s.matched_prediction is not None])
        
        step_precision = matched_steps / total_steps if total_steps > 0 else 0
        step_recall = matched_steps / total_steps if total_steps > 0 else 0
        step_f1 = (2 * step_precision * step_recall) / (step_precision + step_recall) if (step_precision + step_recall) > 0 else 0
        
        # Entity coverage metrics
        total_entities = len(entity_coverage)
        found_entities = len([e for e in entity_coverage if e.found_in_predictions])
        entity_coverage_rate = found_entities / total_entities if total_entities > 0 else 0
        
        return {
            "step_level": {
                "total_steps": total_steps,
                "matched_steps": matched_steps,
                "precision": round(step_precision, 3),
                "recall": round(step_recall, 3), 
                "f1": round(step_f1, 3)
            },
            "entity_coverage": {
                "total_entities": total_entities,
                "found_entities": found_entities,
                "coverage_rate": round(entity_coverage_rate, 3),
                "by_type": {
                    entity_type: {
                        "total": len([e for e in entity_coverage if e.entity_type == entity_type]),
                        "found": len([e for e in entity_coverage if e.entity_type == entity_type and e.found_in_predictions])
                    }
                    for entity_type in ["cell_types", "molecules", "other_entities"]
                }
            }
        }
    
    def _generate_markdown_summary(self, report: Dict) -> str:
        """Generate human-readable markdown summary"""
        
        metrics = report["metrics"]
        step_metrics = metrics["step_level"]
        entity_metrics = metrics["entity_coverage"]
        
        md = f"""# Enhanced LLM Evaluation Report

Generated: {report['metadata']['timestamp']}
Model: {report['metadata']['model']}
Evaluation Mode: {report['metadata']['evaluation_mode']}

## Step-Level Analysis

- **Total Steps Extracted**: {step_metrics['total_steps']}
- **Steps Matched**: {step_metrics['matched_steps']}
- **Step-Level Precision**: {step_metrics['precision']}
- **Step-Level Recall**: {step_metrics['recall']}
- **Step-Level F1**: {step_metrics['f1']}

## Entity Coverage Analysis

- **Overall Entity Coverage**: {entity_metrics['coverage_rate']} ({entity_metrics['found_entities']}/{entity_metrics['total_entities']})

### By Entity Type:
"""
        for entity_type, stats in entity_metrics["by_type"].items():
            coverage_pct = stats["found"] / stats["total"] if stats["total"] > 0 else 0
            md += f"- **{entity_type.replace('_', ' ').title()}**: {coverage_pct:.1%} ({stats['found']}/{stats['total']})\n"
        
        md += f"""

## Process Coverage Summary

"""
        for process in report["process_coverage"]:
            coverage_pct = process["matched_steps"] / process["total_steps"] if process["total_steps"] > 0 else 0
            md += f"- **{process['process_name']}**: {coverage_pct:.1%} ({process['matched_steps']}/{process['total_steps']} steps)\n"
        
        md += f"""

## Key Findings

### Well-Represented Entities:
"""
        well_represented = [e for e in report["entity_coverage"] if e["found_in_predictions"]][:5]
        for entity in well_represented:
            md += f"- {entity['entity_name']} ({entity['entity_type']})\n"
        
        md += f"""

### Missing Entities:
"""
        missing = [e for e in report["entity_coverage"] if not e["found_in_predictions"]][:5]
        for entity in missing:
            md += f"- {entity['entity_name']} ({entity['entity_type']})\n"
        
        return md


# Convenience function for CLI usage
def run_enhanced_evaluation(
    pred_file: str,
    truth_file: str,
    evaluation_mode: str = "lenient",
    context: str = "",
    model: str = "gpt-4o",
    output_dir: str = None
) -> Dict[str, Any]:
    """Run enhanced LLM evaluation pipeline"""
    
    evaluator = EnhancedLLMEvaluator(model=model, output_dir=output_dir)
    return evaluator.evaluate(
        pred_file, truth_file, evaluation_mode, context
    )
