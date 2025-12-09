"""
Comparative evaluator that runs both lexical and LLM-based evaluation
and provides insights into their differences.
"""

import json
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

from evaluator import evaluate_v2
from llm_evaluator import run_llm_evaluation

class ComparativeEvaluator:
    """Runs both evaluation methods and compares results"""
    
    def __init__(self, output_dir: str = None):
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            # Default to eval subfolder
            eval_dir = Path(__file__).parent
            self.output_dir = eval_dir / "comparative_eval_outputs"
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def compare(
        self,
        pred_file: str,
        truth_file: str,
        context: str = "",
        threshold: float = 0.6,
        model: str = "gpt-4o-mini"
    ) -> Dict[str, Any]:
        """Run both evaluations and compare"""
        
        print("Running lexical evaluation...")
        pred_data = json.loads(Path(pred_file).read_text())
        truth_data = json.loads(Path(truth_file).read_text())
        
        lexical_result = evaluate_v2(
            pred_data, 
            truth_data, 
            threshold=threshold,
            truth_mode="steps"
        )
        
        print("Running LLM semantic evaluation...")
        llm_result = run_llm_evaluation(
            pred_file,
            truth_file,
            context=context,
            model=model,
            output_dir=str(self.output_dir / "llm_intermediate")
        )
        
        # Analyze differences
        comparison = self._analyze_differences(lexical_result, llm_result)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report = {
            "metadata": {
                "timestamp": timestamp,
                "pred_file": pred_file,
                "truth_file": truth_file,
                "context": context,
                "lexical_threshold": threshold,
                "llm_model": model
            },
            "lexical": {
                "metrics": lexical_result.get("metrics", {}),
                "counts": lexical_result.get("counts", {}),
                "matched": len(lexical_result.get("matches", []))
            },
            "llm": {
                "metrics": llm_result.get("metrics", {}),
                "coverage": llm_result.get("coverage", {}),
                "matched": llm_result.get("metrics", {}).get("tp", 0),
                "agent_metadata": llm_result.get("metadata", {})
            },
            "comparison": comparison
        }
        
        report_file = self.output_dir / f"comparison_{timestamp}.json"
        report_file.write_text(json.dumps(report, indent=2))
        print(f"Saved comparison to {report_file}")
        
        # Generate insights
        insights_file = self.output_dir / f"insights_{timestamp}.md"
        insights_file.write_text(self._generate_insights(report))
        print(f"Saved insights to {insights_file}")
        
        return report
    
    def _analyze_differences(self, lexical: Dict, llm: Dict) -> Dict:
        """Analyze differences between evaluation methods"""
        
        lex_f1 = lexical.get("metrics", {}).get("f1", 0)
        llm_f1 = llm.get("metrics", {}).get("f1", 0)
        
        return {
            "f1_improvement": round(llm_f1 - lex_f1, 3),
            "precision_improvement": round(
                llm.get("metrics", {}).get("precision", 0) - 
                lexical.get("metrics", {}).get("precision", 0), 3
            ),
            "recall_improvement": round(
                llm.get("metrics", {}).get("recall", 0) - 
                lexical.get("metrics", {}).get("recall", 0), 3
            ),
            "additional_matches_found": llm.get("metrics", {}).get("tp", 0) - len(lexical.get("matches", [])),
            "semantic_gaps_identified": len(llm.get("coverage", {}).get("missing", [])),
            "extra_concepts_identified": len(llm.get("coverage", {}).get("extra", [])),
            "recommendation": self._get_recommendation(lex_f1, llm_f1)
        }
    
    def _get_recommendation(self, lex_f1: float, llm_f1: float) -> str:
        """Generate recommendation based on results"""
        
        if llm_f1 - lex_f1 > 0.2:
            return "Strong semantic gaps exist. Consider using LLM evaluation for this dataset."
        elif llm_f1 > 0.7:
            return "Good coverage detected. Minor prompt adjustments could improve alignment."
        elif llm_f1 < 0.3:
            return "Low coverage suggests fundamental mismatch. Review requirements or expand interaction set."
        else:
            return "Moderate coverage. Consider targeted improvements to key interactions."
    
    def _generate_insights(self, report: Dict) -> str:
        """Generate human-readable insights"""
        
        comp = report["comparison"]
        
        return f"""# Comparative Evaluation Insights

Generated: {report['metadata']['timestamp']}

## Performance Comparison

| Method | Precision | Recall | F1 Score | Matches |
|--------|-----------|--------|----------|---------|
| Lexical | {report['lexical']['metrics'].get('precision', 0)} | {report['lexical']['metrics'].get('recall', 0)} | {report['lexical']['metrics'].get('f1', 0)} | {report['lexical']['matched']} |
| LLM Semantic | {report['llm']['metrics'].get('precision', 0)} | {report['llm']['metrics'].get('recall', 0)} | {report['llm']['metrics'].get('f1', 0)} | {report['llm']['matched']} |

## Key Findings

- **F1 Improvement**: {comp['f1_improvement']} ({'+' if comp['f1_improvement'] > 0 else ''}{comp['f1_improvement']*100:.1f}%)
- **Additional Matches Found**: {comp['additional_matches_found']}
- **Semantic Gaps Identified**: {comp['semantic_gaps_identified']}
- **Extra Concepts in Predictions**: {comp['extra_concepts_identified']}

## LLM Agent Performance

- **Decomposer Agent**: {report['llm']['agent_metadata'].get('decomposer_agent', {}).get('title', 'N/A')}
- **Aligner Agent**: {report['llm']['agent_metadata'].get('aligner_agent', {}).get('title', 'N/A')}
- **Model Used**: {report['metadata']['llm_model']}

## Coverage Analysis

### Well Covered Topics
{chr(10).join(f"- {x}" for x in report['llm'].get('coverage', {}).get('well_covered', [])[:5])}

### Missing Topics  
{chr(10).join(f"- {x}" for x in report['llm'].get('coverage', {}).get('missing', [])[:5])}

### Extra Topics in Predictions
{chr(10).join(f"- {x}" for x in report['llm'].get('coverage', {}).get('extra', [])[:5])}

## Recommendation
{comp['recommendation']}

## Next Steps
1. Review missing interactions identified by LLM agent
2. Consider adding interactions for gaps in coverage
3. Validate matched pairs for biological accuracy
4. Adjust agent prompts if systematic misalignment detected
5. Use LLM evaluation for datasets with high semantic complexity

## Configuration Used
- **Lexical Threshold**: {report['metadata']['lexical_threshold']}
- **LLM Model**: {report['metadata']['llm_model']}
- **Context**: {report['metadata']['context'] if report['metadata']['context'] else 'None provided'}
"""
