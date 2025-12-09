#!/usr/bin/env python3
"""
CLI runner for LLM-based semantic evaluation.
Usage:
  python src/eval/run_llm_eval.py --pred /path/to/result.json --truth /path/to/truth.json --context "influenza infection" --compare
"""

import argparse
import json
from pathlib import Path
import sys
import os

# Add parent directory to path for local imports
sys.path.insert(0, os.path.dirname(__file__))

from llm_evaluator import run_llm_evaluation
from comparative_evaluator import ComparativeEvaluator

def main():
    parser = argparse.ArgumentParser(description="Run LLM-based semantic evaluation using agent architecture")
    parser.add_argument("--pred", required=True, help="Path to prediction JSON")
    parser.add_argument("--truth", required=True, help="Path to ground truth JSON")
    parser.add_argument("--context", default="", help="Biological context for evaluation (e.g., 'influenza infection in lung tissue')")
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model to use")
    parser.add_argument("--compare", action="store_true", help="Run comparative evaluation (both lexical and LLM)")
    parser.add_argument("--output-dir", default=None, help="Output directory (default: eval/llm_eval_outputs)")
    parser.add_argument("--threshold", type=float, default=0.6, help="Threshold for lexical evaluation when comparing")
    
    args = parser.parse_args()
    
    # Verify files exist
    if not Path(args.pred).exists():
        print(f"Error: Prediction file not found: {args.pred}")
        return 1
    if not Path(args.truth).exists():
        print(f"Error: Ground truth file not found: {args.truth}")
        return 1
    
    if args.compare:
        print("Running comparative evaluation (lexical + LLM)...")
        print(f"Using model: {args.model}")
        print(f"Context: {args.context if args.context else 'None provided'}")
        
        evaluator = ComparativeEvaluator(output_dir=args.output_dir)
        result = evaluator.compare(
            args.pred,
            args.truth,
            context=args.context,
            model=args.model,
            threshold=args.threshold
        )
        
        print(f"\n=== Results Summary ===")
        print(f"Lexical F1: {result['lexical']['metrics']['f1']}")
        print(f"LLM F1: {result['llm']['metrics']['f1']}")
        print(f"Improvement: {result['comparison']['f1_improvement']} ({result['comparison']['f1_improvement']*100:+.1f}%)")
        print(f"Additional matches found: {result['comparison']['additional_matches_found']}")
        print(f"Recommendation: {result['comparison']['recommendation']}")
        
        if result['llm']['coverage']:
            coverage = result['llm']['coverage']
            print(f"\n=== Coverage Analysis ===")
            print(f"Well covered: {len(coverage.get('well_covered', []))} topics")
            print(f"Partially covered: {len(coverage.get('partially_covered', []))} topics")
            print(f"Missing: {len(coverage.get('missing', []))} topics")
            print(f"Extra: {len(coverage.get('extra', []))} topics")
    else:
        print("Running LLM semantic evaluation...")
        print(f"Using model: {args.model}")
        print(f"Context: {args.context if args.context else 'None provided'}")
        
        result = run_llm_evaluation(
            args.pred,
            args.truth,
            context=args.context,
            model=args.model,
            output_dir=args.output_dir
        )
        
        print(f"\n=== Results ===")
        print(f"Precision: {result['metrics']['precision']}")
        print(f"Recall: {result['metrics']['recall']}")
        print(f"F1: {result['metrics']['f1']}")
        print(f"True Positives: {result['metrics']['tp']}")
        print(f"False Positives: {result['metrics']['fp']}")
        print(f"False Negatives: {result['metrics']['fn']}")
        
        if result.get('coverage'):
            coverage = result['coverage']
            print(f"\n=== Coverage Analysis ===")
            print(f"Well covered: {len(coverage.get('well_covered', []))} topics")
            if coverage.get('well_covered'):
                for topic in coverage['well_covered'][:3]:
                    print(f"  - {topic}")
            
            print(f"Missing: {len(coverage.get('missing', []))} topics")
            if coverage.get('missing'):
                for topic in coverage['missing'][:3]:
                    print(f"  - {topic}")
                    
            print(f"Extra: {len(coverage.get('extra', []))} topics")
            if coverage.get('extra'):
                for topic in coverage['extra'][:3]:
                    print(f"  - {topic}")

    return 0

if __name__ == "__main__":
    exit(main())
