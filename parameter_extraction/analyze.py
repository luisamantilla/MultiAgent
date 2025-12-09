import json

data = json.load(open("parameter_extraction/results/merged_parameters.json"))

for name, info in list(data["parameters"].items())[:50]:
    print(name, "→", info["context"])
