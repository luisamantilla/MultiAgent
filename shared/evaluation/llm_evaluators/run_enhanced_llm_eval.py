#!/usr/bin/env python3
"""
CLI runner for Enhanced LLM Evaluation with step-level analysis
Usage:
  python src/eval/run_enhanced_llm_eval.py --pred /path/to/result.json --truth /path/to/truth.json --mode lenient
"""

import argparse
import json
from pathlib import Path
import sys
import os

# Add parent directory to path for local imports
sys.path.insert(0, os.path.dirname(__file__))

from enhanced_llm_evaluator import run_enhanced_evaluation

def main():
    parser = argparse.ArgumentParser(description="Run enhanced LLM evaluation with step-level analysis")
    parser.add_argument("--pred", required=True, help="Path to prediction JSON")
    parser.add_argument("--truth", required=True, help="Path to ground truth JSON")
    parser.add_argument("--mode", choices=["strict", "lenient", "process_aware"], default="lenient", 
                       help="Evaluation mode (default: lenient)")
    parser.add_argument("--context", default="", help="Biological context for evaluation")
    parser.add_argument("--model", default="gpt-4o", help="OpenAI model to use (default: gpt-4o)")
    parser.add_argument("--output-dir", default=None, help="Output directory")
    
    args = parser.parse_args()
    
    # Verify files exist
    if not Path(args.pred).exists():
        print(f"Error: Prediction file not found: {args.pred}")
        return 1
    if not Path(args.truth).exists():
        print(f"Error: Ground truth file not found: {args.truth}")
        return 1
    
    print("Running Enhanced LLM Evaluation...")
    print(f"Model: {args.model}")
    print(f"Mode: {args.mode}")
    print(f"Context: {args.context if args.context else 'None provided'}")
    print(f"Prediction file: {Path(args.pred).name}")
    print(f"Ground truth file: {Path(args.truth).name}")
    print()
    
    try:
        result = run_enhanced_evaluation(
            pred_file=args.pred,
            truth_file=args.truth,
            evaluation_mode=args.mode,
            context=args.context,
            model=args.model,
            output_dir=args.output_dir
        )
        
        print("=== EVALUATION RESULTS ===")
        print()
        
        # Step-level metrics
        step_metrics = result['metrics']['step_level']
        print("📊 Step-Level Analysis:")
        print(f"   Total Steps: {step_metrics['total_steps']}")
        print(f"   Matched Steps: {step_metrics['matched_steps']}")
        print(f"   Precision: {step_metrics['precision']}")
        print(f"   Recall: {step_metrics['recall']}")
        print(f"   F1 Score: {step_metrics['f1']}")
        print()
        
        # Entity coverage
        entity_metrics = result['metrics']['entity_coverage']
        print("🎯 Entity Coverage:")
        print(f"   Overall Coverage: {entity_metrics['coverage_rate']:.1%} ({entity_metrics['found_entities']}/{entity_metrics['total_entities']})")
        
        for entity_type, stats in entity_metrics['by_type'].items():
            coverage_pct = stats['found'] / stats['total'] if stats['total'] > 0 else 0
            print(f"   {entity_type.replace('_', ' ').title()}: {coverage_pct:.1%} ({stats['found']}/{stats['total']})")
        print()
        
        # Process coverage highlights
        print("🔬 Process Coverage Highlights:")
        for process in result['process_coverage'][:5]:  # Show top 5
            coverage_pct = process['matched_steps'] / process['total_steps'] if process['total_steps'] > 0 else 0
            print(f"   {process['process_name']}: {coverage_pct:.1%} ({process['matched_steps']}/{process['total_steps']} steps)")
        print()
        
        # Key insights
        print("💡 Key Insights:")
        
        # Well-represented entities
        well_represented = [e for e in result['entity_coverage'] if e['found_in_predictions']]
        if well_represented:
            print(f"   ✅ Found entities: {', '.join([e['entity_name'] for e in well_represented[:3]])}")
        
        # Missing entities
        missing = [e for e in result['entity_coverage'] if not e['found_in_predictions']]
        if missing:
            print(f"   ❌ Missing entities: {', '.join([e['entity_name'] for e in missing[:3]])}")
        
        # Unmatched predictions
        if result['unmatched_predictions']:
            print(f"   🔄 Extra predictions: {len(result['unmatched_predictions'])} interactions not in ground truth")
        
        print()
        print("📁 Detailed results saved to enhanced_llm_eval_outputs/")
        
        return 0
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
