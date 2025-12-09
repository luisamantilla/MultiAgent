import os
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import importlib.util

# Import BaseAgent using the same pattern as pi_agent
_REPO_SRC = Path(__file__).resolve().parents[6]
_BASE_AGENT_PATH = _REPO_SRC / "agents" / "base_agent.py"
_spec = importlib.util.spec_from_file_location("repo_base_agent", str(_BASE_AGENT_PATH))
repo_base_agent = importlib.util.module_from_spec(_spec)  # type: ignore
assert _spec and _spec.loader
_spec.loader.exec_module(repo_base_agent)  # type: ignore
BaseAgent = repo_base_agent.BaseAgent  # type: ignore


@dataclass
class AtomicInteraction:
    """Minimal unit of biological interaction"""
    source: str  # Actor entity
    action: str  # Verb/relation
    target: str  # Object entity
    mediators: List[str] = None  # Via molecules
    context: Dict[str, str] = None  # Qualifiers
    evidence: str = ""  # Original text
    confidence: float = 1.0  # Extraction confidence
    
    def to_dict(self):
        return {
            "source": self.source,
            "action": self.action,
            "target": self.target,
            "mediators": self.mediators or [],
            "context": self.context or {},
            "evidence": self.evidence,
            "confidence": self.confidence
        }


class LLMDecomposerAgent(BaseAgent):
    """LLM agent that decomposes complex interaction descriptions into atomic parts"""
    
    def __init__(self, model: str = "gpt-4o-mini"):
        super().__init__(
            title="Interaction Decomposer",
            expertise="Biological interaction parsing and semantic analysis",
            goal="Break complex interaction descriptions into atomic, standardized components",
            role="Semantic Parser",
            model=model,
        )
        self.system_prompt = (
            "You are a biological interaction parser. Your role is to decompose complex interaction "
            "descriptions into atomic parts. Each atomic interaction should have exactly one source "
            "entity acting on one target entity via a standardized action verb."
        )

    def decompose_interactions(self, texts: List[str], context: str = "") -> List[AtomicInteraction]:
        """Break down interaction texts into atomic components"""
        
        prompt = f"""
{self.construct_prompt("Decompose biological interactions into atomic components")}

Context: {context if context else "General biological system"}

For each text, extract ALL atomic interactions (one source → one target via action).
Complex interactions should be split. For example:
- "A and B activate C" → two atomics: "A activates C", "B activates C"
- "A activates B leading to C inhibition" → two atomics: "A activates B", "B inhibits C"
- "A recruits B via X and Y" → one atomic with two mediators

Use standardized action verbs: activates, inhibits, secretes, recruits, binds, kills, differentiates, presents, phagocytoses

Texts to decompose:
{json.dumps(texts, indent=2)}

Return JSON with this structure:
{{
  "interactions": [
    {{
      "source": "entity name",
      "action": "standardized verb",
      "target": "entity name", 
      "mediators": ["molecule1", "molecule2"],
      "context": {{"tissue": "...", "time": "...", "condition": "..."}},
      "evidence": "exact original text this came from",
      "confidence": 0.95
    }}
  ]
}}

Be exhaustive - capture every relationship. Use standard entity names where possible.
"""
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.3,
            response_format={"type": "json_object"}
        )
        
        try:
            result = json.loads(response.choices[0].message.content)
        except json.JSONDecodeError:
            return []
        
        atomics = []
        for item in result.get("interactions", []):
            atomics.append(AtomicInteraction(
                source=item.get("source", ""),
                action=item.get("action", ""),
                target=item.get("target", ""),
                mediators=item.get("mediators"),
                context=item.get("context"),
                evidence=item.get("evidence", ""),
                confidence=item.get("confidence", 1.0)
            ))
        
        return atomics


class LLMAlignerAgent(BaseAgent):
    """LLM agent that aligns and scores atomic interactions between prediction and ground truth"""
    
    def __init__(self, model: str = "gpt-4o-mini"):
        super().__init__(
            title="Interaction Aligner", 
            expertise="Semantic matching and biological interaction comparison",
            goal="Find optimal alignments between predicted and ground truth interactions",
            role="Semantic Matcher",
            model=model,
        )
        self.system_prompt = (
            "You are evaluating predicted biological interactions against ground truth. "
            "Your role is to find semantic matches that may use different words but describe "
            "the same biological phenomenon. Consider entity synonyms, relation equivalence, "
            "and partial coverage."
        )

    def align_interactions(
        self, 
        truth_atomics: List[AtomicInteraction], 
        pred_atomics: List[AtomicInteraction],
        context: str = ""
    ) -> Dict[str, Any]:
        """Find best semantic matches between truth and prediction atomics"""
        
        prompt = f"""
{self.construct_prompt("Align predicted interactions with ground truth")}

Context: {context if context else "General biological evaluation"}

GROUND TRUTH interactions:
{json.dumps([a.to_dict() for a in truth_atomics], indent=2)}

PREDICTED interactions:
{json.dumps([a.to_dict() for a in pred_atomics], indent=2)}

For each ground truth interaction, find the best matching prediction (if any).
Consider semantic equivalence, not just lexical matching:
- "CD8 T cell" ≈ "cytotoxic T lymphocyte" ≈ "CTL"
- "activates" ≈ "stimulates" ≈ "induces" 
- "via IFN-γ" ≈ "through interferon gamma"

Return JSON with these sections:

{{
  "matches": [
    {{
      "truth_idx": 0,
      "pred_idx": 2,
      "similarity": 0.95,
      "explanation": "Both describe CD8 T cell cytotoxicity against infected cells",
      "partial": false,
      "evidence_comparison": {{
        "truth": "exact truth evidence text",
        "pred": "exact pred evidence text"
      }}
    }}
  ],
  "unmatched_truth": [
    {{
      "idx": 3,
      "interaction": "truth interaction description", 
      "evidence": "truth evidence",
      "reason": "No prediction covers antibody-mediated neutralization"
    }}
  ],
  "unmatched_pred": [
    {{
      "idx": 5,
      "interaction": "pred interaction description",
      "evidence": "pred evidence", 
      "reason": "Introduces cell type not in ground truth"
    }}
  ],
  "coverage_analysis": {{
    "well_covered": ["T cell cytotoxicity", "macrophage phagocytosis"],
    "partially_covered": ["cytokine signaling - missing IL-12 pathway"],
    "missing": ["B cell responses", "NK cell ADCC"],
    "extra": ["fibroblast activation"]
  }}
}}
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
            return json.loads(response.choices[0].message.content)
        except json.JSONDecodeError:
            return {"matches": [], "unmatched_truth": [], "unmatched_pred": [], "coverage_analysis": {}}


class LLMSemanticEvaluator:
    """Main evaluator coordinating decomposition, alignment, and scoring using LLM agents"""
    
    def __init__(self, model: str = "gpt-4o-mini", output_dir: str = None):
        self.decomposer = LLMDecomposerAgent(model)
        self.aligner = LLMAlignerAgent(model)
        self.model = model
        
        if output_dir:
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)
        else:
            # Default to eval subfolder
            eval_dir = Path(__file__).parent
            self.output_dir = eval_dir / "llm_eval_outputs"
            self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def evaluate(
        self,
        pred_file: str,
        truth_file: str,
        context: str = "",
        save_intermediate: bool = True
    ) -> Dict[str, Any]:
        """Full evaluation pipeline using LLM agents"""
        
        # Load files
        pred_data = json.loads(Path(pred_file).read_text())
        truth_data = json.loads(Path(truth_file).read_text())
        
        # Extract interaction texts
        truth_texts = self._extract_truth_texts(truth_data)
        pred_texts = self._extract_pred_texts(pred_data)
        
        # Step 1: Decompose into atomics using LLM agents
        print("Decomposing ground truth interactions...")
        truth_atomics = self.decomposer.decompose_interactions(truth_texts, context)
        
        print("Decomposing predicted interactions...")
        pred_atomics = self.decomposer.decompose_interactions(pred_texts, context)
        
        if save_intermediate:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            decomp_file = self.output_dir / f"decomposition_{timestamp}.json"
            decomp_file.write_text(json.dumps({
                "truth_atomics": [a.to_dict() for a in truth_atomics],
                "pred_atomics": [a.to_dict() for a in pred_atomics],
                "truth_count": len(truth_atomics),
                "pred_count": len(pred_atomics),
                "decomposer_agent": self.decomposer.report(),
                "aligner_agent": self.aligner.report()
            }, indent=2))
            print(f"Saved decomposition to {decomp_file}")
        
        # Step 2: Align and match using LLM agent
        print("Aligning interactions...")
        alignment = self.aligner.align_interactions(truth_atomics, pred_atomics, context)
        
        # Step 3: Compute metrics
        metrics = self._compute_metrics(alignment, len(truth_atomics), len(pred_atomics))
        
        # Step 4: Generate detailed report
        report = self._generate_report(
            truth_atomics, 
            pred_atomics, 
            alignment, 
            metrics,
            pred_file,
            truth_file
        )
        
        if save_intermediate:
            report_file = self.output_dir / f"evaluation_{timestamp}.json"
            report_file.write_text(json.dumps(report, indent=2))
            print(f"Saved evaluation report to {report_file}")
            
            # Also save a markdown summary
            summary_file = self.output_dir / f"summary_{timestamp}.md"
            summary_file.write_text(self._generate_markdown_summary(report))
            print(f"Saved summary to {summary_file}")
        
        return report
    
    def _extract_truth_texts(self, truth_data: Dict) -> List[str]:
        """Extract interaction texts from ground truth"""
        texts = []
        
        # From explicit interactions
        if "interactions" in truth_data:
            texts.extend([str(x) for x in truth_data["interactions"]])
        
        # From biological processes
        for proc in truth_data.get("BiologicalProcesses", []):
            if proc.get("name"):
                texts.append(proc["name"])
            for step in proc.get("steps", []):
                if isinstance(step, str):
                    texts.append(step)
                elif isinstance(step, dict):
                    if step.get("step"):
                        texts.append(f"{step['step']} {step.get('detail', '')}".strip())
        
        return texts
    
    def _extract_pred_texts(self, pred_data: Dict) -> List[str]:
        """Extract interaction texts from predictions"""
        texts = []
        
        # From PI selected
        pi_selected = pred_data.get("pi", {}).get("selected", {})
        texts.extend([str(x) for x in pi_selected.get("interactions", [])])
        
        # From computationalist
        comp = pred_data.get("computationalist", {})
        texts.extend([str(x) for x in comp.get("interactions", [])])
        
        # From biologist keep
        bio = pred_data.get("biologist", {})
        texts.extend([str(x) for x in bio.get("keep_interactions", [])])
        
        return list(set(texts))  # dedupe
    
    def _compute_metrics(self, alignment: Dict, n_truth: int, n_pred: int) -> Dict:
        """Compute precision, recall, F1"""
        matches = alignment.get("matches", [])
        
        # Only count high-confidence matches
        tp = len([m for m in matches if m.get("similarity", 0) >= 0.7])
        fp = n_pred - tp
        fn = n_truth - tp
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "total_truth": n_truth,
            "total_pred": n_pred
        }
    
    def _generate_report(
        self,
        truth_atomics: List[AtomicInteraction],
        pred_atomics: List[AtomicInteraction],
        alignment: Dict,
        metrics: Dict,
        pred_file: str,
        truth_file: str
    ) -> Dict:
        """Generate comprehensive evaluation report"""
        
        return {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "model": self.model,
                "pred_file": pred_file,
                "truth_file": truth_file,
                "decomposer_agent": self.decomposer.report(),
                "aligner_agent": self.aligner.report()
            },
            "metrics": metrics,
            "decomposition": {
                "truth_atomic_count": len(truth_atomics),
                "pred_atomic_count": len(pred_atomics)
            },
            "alignment": alignment,
            "coverage": alignment.get("coverage_analysis", {}),
            "matched_pairs": [
                {
                    "truth": truth_atomics[m["truth_idx"]].to_dict() if m["truth_idx"] < len(truth_atomics) else {},
                    "pred": pred_atomics[m["pred_idx"]].to_dict() if m["pred_idx"] < len(pred_atomics) else {},
                    "similarity": m.get("similarity", 0),
                    "explanation": m.get("explanation", ""),
                    "evidence_comparison": m.get("evidence_comparison", {})
                }
                for m in alignment.get("matches", [])[:20]  # Top 20 matches
            ]
        }
    
    def _generate_markdown_summary(self, report: Dict) -> str:
        """Generate human-readable markdown summary"""
        
        md = f"""# LLM Semantic Evaluation Report

Generated: {report['metadata']['timestamp']}
Model: {report['metadata']['model']}

## Agent Information
- **Decomposer**: {report['metadata']['decomposer_agent']['title']} - {report['metadata']['decomposer_agent']['expertise']}
- **Aligner**: {report['metadata']['aligner_agent']['title']} - {report['metadata']['aligner_agent']['expertise']}

## Metrics
- **Precision**: {report['metrics']['precision']}
- **Recall**: {report['metrics']['recall']}
- **F1 Score**: {report['metrics']['f1']}
- **True Positives**: {report['metrics']['tp']}
- **False Positives**: {report['metrics']['fp']}
- **False Negatives**: {report['metrics']['fn']}

## Decomposition Results
- **Truth Atomic Interactions**: {report['decomposition']['truth_atomic_count']}
- **Predicted Atomic Interactions**: {report['decomposition']['pred_atomic_count']}

## Coverage Analysis

### Well Covered
{chr(10).join(f"- {x}" for x in report.get('coverage', {}).get('well_covered', []))}

### Partially Covered
{chr(10).join(f"- {x}" for x in report.get('coverage', {}).get('partially_covered', []))}

### Missing from Predictions
{chr(10).join(f"- {x}" for x in report.get('coverage', {}).get('missing', []))}

### Extra in Predictions
{chr(10).join(f"- {x}" for x in report.get('coverage', {}).get('extra', []))}

## Top Matches

"""
        for i, match in enumerate(report.get('matched_pairs', [])[:10], 1):
            md += f"""
### Match {i} (Similarity: {match.get('similarity', 0)})

**Truth**: {match['truth'].get('source')} → {match['truth'].get('action')} → {match['truth'].get('target')}
Evidence: "{match['truth'].get('evidence', '')}"

**Prediction**: {match['pred'].get('source')} → {match['pred'].get('action')} → {match['pred'].get('target')}
Evidence: "{match['pred'].get('evidence', '')}"

**Explanation**: {match.get('explanation', '')}

---
"""
        
        return md


# Convenience function for CLI usage
def run_llm_evaluation(
    pred_file: str,
    truth_file: str,
    context: str = "",
    model: str = "gpt-4o-mini",
    output_dir: str = None
) -> Dict[str, Any]:
    """Run full LLM evaluation pipeline using agent-based approach"""
    
    evaluator = LLMSemanticEvaluator(model=model, output_dir=output_dir)
    return evaluator.evaluate(pred_file, truth_file, context)
