from pathlib import Path
import json
#updated this file on 10-08-25 so that each time this fucntion is called, it adds to growing merged database
results_dir = Path("parameter_extraction/results")
merged_file = results_dir / "merged_parameters.json"

# Load existing merged dataset if it exists
if merged_file.exists():
    with open(merged_file, "r") as f:
        merged_data = json.load(f)
else:
    merged_data = {"parameters": {}, "summaries": []}

# Collect all new JSONs
all_jsons = sorted(results_dir.rglob("*.json"))
print(f"🔍 Found {len(all_jsons)} files to check for new parameters")

new_added = 0

for file in all_jsons:
    if file.name == "merged_parameters.json":
        continue
    with open(file, "r") as f:
        page_data = json.load(f)

    for section in ["tables", "figures", "text", "equations"]:
        for param in page_data.get(section, []):
            if isinstance(param, dict):
                name = param.get("parameter")
                if name and name not in merged_data["parameters"]:
                    merged_data["parameters"][name] = {
                        "value": param.get("value"),
                        "units": param.get("units"),
                        "context": param.get("context"),
                        "source": file.name,
                        "section": section
                    }
                    new_added += 1

    if "summary" in page_data:
        merged_data["summaries"].append(page_data["summary"])

# Save back to file
with open(merged_file, "w") as f:
    json.dump(merged_data, f, indent=2, ensure_ascii=False)

print(f"✅ Added {new_added} new parameters")
print(f"📦 Total database size: {len(merged_data['parameters'])} parameters")
