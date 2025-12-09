#!/usr/bin/env python3
"""
Batch LLM evaluation runner for multiple crosstalk results.
Usage:
  python src/eval/batch_llm_eval.py --pred-dir /path/to/results --truth /path/to/truth.json --context "influenza infection"
"""

import argparse
import json
from pathlib import Path
import sys
import os
import csv
from datetime import datetime

# Add parent directory to path for local imports
sys.path.insert(0, os.path.dirname(__file__))

from llm_evaluator import run_llm_evaluation
from comparative_evaluator import ComparativeEvaluator

def main():
    parser = argparse.ArgumentParser(description="Batch LLM evaluation of multiple crosstalk results")
    parser.add_argument("--pred-dir", required=True, help="Directory containing crosstalk_result_*.json files")
    parser.add_argument("--truth", required=True, help="Path to ground truth JSON")
    parser.add_argument("--context", default="", help="Biological context for evaluation")
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model to use")
    parser.add_argument("--compare", action="store_true", help="Run comparative evaluation for each file")
    parser.add_argument("--output-dir", required=True, help="Output directory for results")
    parser.add_argument("--threshold", type=float, default=0.6, help="Threshold for lexical evaluation when comparing")
    
    args = parser.parse_args()
    
    pred_dir = Path(args.pred_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Verify truth file exists
    if not Path(args.truth).exists():
        print(f"Error: Ground truth file not found: {args.truth}")
        return 1
    
    # Find all crosstalk result files
    pred_files = list(pred_dir.glob("crosstalk_result_*.json"))
    if not pred_files:
        print(f"No crosstalk_result_*.json files found in {pred_dir}")
        return 1
    
    print(f"Found {len(pred_files)} files to evaluate")
    print(f"Using model: {args.model}")
    print(f"Context: {args.context if args.context else 'None provided'}")
    print(f"Output directory: {out_dir}")
    
    results = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for i, pred_path in enumerate(sorted(pred_files), 1):
        print(f"\n[{i}/{len(pred_files)}] Processing {pred_path.name}...")
        
        try:
            if args.compare:
                # Run comparative evaluation
                file_out_dir = out_dir / f"comparative_{pred_path.stem.split('crosstalk_result_')[-1]}"
                evaluator = ComparativeEvaluator(output_dir=str(file_out_dir))
                result = evaluator.compare(
                    str(pred_path),
                    args.truth,
                    context=args.context,
                    model=args.model,
                    threshold=args.threshold
                )
                
                # Extract metrics for CSV
                row = {
                    "file": pred_path.name,
                    "lexical_precision": result['lexical']['metrics'].get('precision', 0),
                    "lexical_recall": result['lexical']['metrics'].get('recall', 0),
                    "lexical_f1": result['lexical']['metrics'].get('f1', 0),
                    "lexical_matches": result['lexical']['matched'],
                    "llm_precision": result['llm']['metrics'].get('precision', 0),
                    "llm_recall": result['llm']['metrics'].get('recall', 0),
                    "llm_f1": result['llm']['metrics'].get('f1', 0),
                    "llm_matches": result['llm']['matched'],
                    "f1_improvement": result['comparison']['f1_improvement'],
                    "additional_matches": result['comparison']['additional_matches_found'],
                    "semantic_gaps": result['comparison']['semantic_gaps_identified'],
                    "recommendation": result['comparison']['recommendation']
                }
            else:
                # Run LLM evaluation only
                file_out_dir = out_dir / f"llm_{pred_path.stem.split('crosstalk_result_')[-1]}"
                result = run_llm_evaluation(
                    str(pred_path),
                    args.truth,
                    context=args.context,
                    model=args.model,
                    output_dir=str(file_out_dir)
                )
                
                # Extract metrics for CSV
                coverage = result.get('coverage', {})
                row = {
                    "file": pred_path.name,
                    "precision": result['metrics']['precision'],
                    "recall": result['metrics']['recall'],
                    "f1": result['metrics']['f1'],
                    "tp": result['metrics']['tp'],
                    "fp": result['metrics']['fp'],
                    "fn": result['metrics']['fn'],
                    "well_covered_count": len(coverage.get('well_covered', [])),
                    "partially_covered_count": len(coverage.get('partially_covered', [])),
                    "missing_count": len(coverage.get('missing', [])),
                    "extra_count": len(coverage.get('extra', []))
                }
            
            results.append(row)
            print(f"  ✓ F1: {row.get('llm_f1', row.get('f1', 0)):.3f}")
            
        except Exception as e:
            print(f"  ✗ Error processing {pred_path.name}: {e}")
            continue
    
    # Write summary CSV
    if results:
        csv_path = out_dir / f"batch_summary_{timestamp}.csv"
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader()
            for row in results:
                writer.writerow(row)
        
        print(f"\n=== Batch Summary ===")
        print(f"Processed: {len(results)}/{len(pred_files)} files")
        print(f"Summary CSV: {csv_path}")
        
        if args.compare:
            avg_lex_f1 = sum(r['lexical_f1'] for r in results) / len(results)
            avg_llm_f1 = sum(r['llm_f1'] for r in results) / len(results)
            avg_improvement = sum(r['f1_improvement'] for r in results) / len(results)
            print(f"Average lexical F1: {avg_lex_f1:.3f}")
            print(f"Average LLM F1: {avg_llm_f1:.3f}")
            print(f"Average improvement: {avg_improvement:.3f} ({avg_improvement*100:+.1f}%)")
        else:
            avg_f1 = sum(r['f1'] for r in results) / len(results)
            avg_coverage = sum(r['well_covered_count'] for r in results) / len(results)
            avg_missing = sum(r['missing_count'] for r in results) / len(results)
            print(f"Average F1: {avg_f1:.3f}")
            print(f"Average well-covered topics: {avg_coverage:.1f}")
            print(f"Average missing topics: {avg_missing:.1f}")
    
    return 0

if __name__ == "__main__":
    exit(main())
