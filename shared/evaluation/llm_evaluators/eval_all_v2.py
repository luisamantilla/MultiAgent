#!/usr/bin/env python3
"""
Batch evaluate all crosstalk_result_*.json files in a folder using evaluate_v2.
Writes per-file reports and a CSV summary.
"""
import argparse
import json
import sys
from pathlib import Path
import csv

# Local import path
import os
HERE = os.path.dirname(__file__)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from evaluator import evaluate_v2


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pred-dir", required=True, help="Directory containing crosstalk_result_*.json")
    p.add_argument("--truth", required=True)
    p.add_argument("--threshold", type=float, default=0.6)
    p.add_argument("--truth-mode", choices=["auto","steps","names","explicit"], default="auto")
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()

    pred_dir = Path(args.pred_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    truth = json.loads(Path(args.truth).read_text())

    rows = []
    for pred_path in sorted(pred_dir.glob("crosstalk_result_*.json")):
        try:
            pred = json.loads(pred_path.read_text())
        except Exception as e:
            print(f"Skipping {pred_path.name}: {e}")
            continue
        report = evaluate_v2(pred, truth, threshold=args.threshold, truth_mode=args.truth_mode)
        out_json = out_dir / f"eval_v2_{pred_path.stem.split('crosstalk_result_')[-1]}_{args.truth_mode}.json"
        out_json.write_text(json.dumps(report, indent=2))
        rows.append({
            "file": pred_path.name,
            "precision": report["metrics"]["precision"],
            "recall": report["metrics"]["recall"],
            "f1": report["metrics"]["f1"],
            "truth_interactions": report["counts"]["truth_interactions"],
            "pred_interactions": report["counts"]["pred_interactions"],
            "matched": report["counts"]["matched"],
            "threshold": report["threshold"],
        })

    csv_path = out_dir / "summary.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["file","precision","recall","f1","truth_interactions","pred_interactions","matched","threshold"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Wrote {csv_path} and {len(rows)} JSON reports to {out_dir}")


if __name__ == "__main__":
    main()
