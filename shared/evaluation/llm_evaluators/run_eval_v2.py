#!/usr/bin/env python3
"""
Quick runner to evaluate a saved crosstalk result JSON against ground truth using evaluate_v2.
Usage:
  python -m eval.run_eval_v2 --pred /path/to/crosstalk_result.json \
      --truth /path/to/influenza_processes.json \
      --threshold 0.6 --truth-mode auto --out /tmp/eval_v2.json
"""
import argparse
import json
import os
import sys
from pathlib import Path

# Ensure local eval module is importable when run as a script
HERE = os.path.dirname(__file__)
if HERE not in sys.path:
    sys.path.insert(0, HERE)
from evaluator import evaluate_v2


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pred", required=True)
    p.add_argument("--truth", required=True)
    p.add_argument("--threshold", type=float, default=0.6)
    p.add_argument("--truth-mode", choices=["auto","steps","names","explicit"], default="auto")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    pred = json.loads(Path(args.pred).read_text())
    truth = json.loads(Path(args.truth).read_text())

    report = evaluate_v2(pred, truth, threshold=args.threshold, truth_mode=args.truth_mode)

    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2))
        print(f"Wrote {args.out}")
    else:
        json.dump(report, sys.stdout, indent=2)
        print()

if __name__ == "__main__":
    main()
