#!/usr/bin/env python3
"""
merge_extracted_parameters.py

Combine all per-page JSON results into a single structured dataset
ready for embedding or further analysis.

Usage:
    python parameter_extraction/merge_extracted_parameters.py
"""

import os
import json
from pathlib import Path

# Directory where per-page JSONs are saved
results_dir = Path("parameter_extraction/results")

# Verify that directory exists
if not results_dir.exists():
    raise FileNotFoundError(f"Results folder not found: {results_dir}")

# Initialize combined data structure
merged_data = {
    "parameters": {},     # flattened parameter list
    "summaries": [],      # all summaries from pages
}

# Iterate through all JSON files in results
for file in sorted(results_dir.glob("*.json")):
    with open(file, "r") as f:
        try:
            page_data = json.load(f)
        except json.JSONDecodeError:
            print(f"⚠️ Skipping {file.name}: invalid JSON format")
            continue

    # Each file is expected to have page-level content
    if "tables" in page_data:
        # Add structured parameters
        for section in ["tables", "figures", "text", "equations"]:
            for param in page_data.get(section, []):
                key = param.get("parameter")
                if not key:
                    continue
                merged_data["parameters"][key] = {
                    "value": param.get("value"),
                    "units": param.get("units"),
                    "context": param.get("context"),
                    "source": file.name,
                    "section": section
                }

        # Add summaries
        if "summary" in page_data and page_data["summary"]:
            merged_data["summaries"].append({
                "page": file.name,
                "summary": page_data["summary"]
            })

# Save combined file
out_path = results_dir / "merged_parameters.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(merged_data, f, indent=2, ensure_ascii=False)

print(f"\n✅ Combined data saved to: {out_path}")
print(f"📦 Total parameters merged: {len(merged_data['parameters'])}")
print(f"📝 Total page summaries: {len(merged_data['summaries'])}")
