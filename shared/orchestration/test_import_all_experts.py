#!/usr/bin/env python
"""
Import & instantiation test for:
  - crosstalk.biological_expert_agent
  - crosstalk.experimental_expert_agent
  - crosstalk.computational_expert_agent (re-used)
  - crosstalk.pi_agent (PIAgent)

Run (from project root containing src/):
  PYTHONPATH=./src python src/tests/test_import_all_experts.py
Exit codes:
  0 = success
  1 = import failure
  2 = no classes found
  3 = instantiation failures
"""

import sys, types, inspect, importlib, traceback
from pathlib import Path
from typing import Dict, List

# --- Path setup ---
THIS_FILE = Path(__file__).resolve()
SRC_DIR = THIS_FILE.parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# --- OpenAI stub (optional) ---
if 'openai' not in sys.modules:
    stub = types.ModuleType('openai')
    class OpenAI:
        def __init__(self, api_key=None): pass
    stub.OpenAI = OpenAI
    sys.modules['openai'] = stub

MODULES = {
    "agents.biological_modeling_agents.crosstalk.biological_expert_agent": {"class_filter_suffix": "ExpertAgent"},
    "agents.biological_modeling_agents.crosstalk.experimental_expert_agent": {"class_filter_suffix": "ExpertAgent"},
    "agents.biological_modeling_agents.crosstalk.computational_expert_agent": {"class_filter_suffix": "ExpertAgent"},
    "agents.biological_modeling_agents.crosstalk.pi_agent": {"explicit": ["PIAgent"]},
}

instantiation_overrides = {
    # class_name: dict(kwargs) if extra required args
    "CustomComputationalExpertAgent": {"specialty": "test_specialty"},
}

def is_candidate(name: str, obj, module: str, suffix: str) -> bool:
    return (
        name.endswith(suffix)
        and inspect.isclass(obj)
        and obj.__module__ == module
    )

all_classes: Dict[str, List[str]] = {}
failed_imports = []
inst_failures = []

print("[INFO] Starting expert & PI module import tests\n")

# --- Import modules & collect classes ---
for mod_name, rules in MODULES.items():
    print(f"[INFO] Importing {mod_name}")
    try:
        mod = importlib.import_module(mod_name)
    except Exception as e:
        print(f"[FAIL] Import error: {mod_name}: {e}")
        traceback.print_exc(limit=1)
        failed_imports.append((mod_name, str(e)))
        continue

    classes = []
    if "explicit" in rules:
        for cname in rules["explicit"]:
            if hasattr(mod, cname):
                classes.append(cname)
            else:
                print(f"[WARN] Expected class {cname} not found in {mod_name}")
    else:
        suffix = rules.get("class_filter_suffix", "")
        for cname, obj in vars(mod).items():
            if is_candidate(cname, obj, mod_name, suffix):
                classes.append(cname)

    if not classes:
        print(f"[WARN] No matching classes found in {mod_name}")
    else:
        print(f"[OK] Found {len(classes)} classes: {', '.join(sorted(classes))}")
    all_classes[mod_name] = classes
    print()

if failed_imports:
    print("\n[SUMMARY] Import failures present -> aborting further tests.")
    for m, err in failed_imports:
        print(f"  - {m}: {err}")
    sys.exit(1)

total_classes = sum(len(v) for v in all_classes.values())
if total_classes == 0:
    print("[FAIL] No classes collected.")
    sys.exit(2)

# --- Instantiate classes ---
print("[INFO] Instantiating discovered classes\n")
success_count = 0

for mod_name, class_names in all_classes.items():
    if not class_names:
        continue
    mod = sys.modules.get(mod_name)
    for cname in class_names:
        cls = getattr(mod, cname, None)
        if cls is None or not inspect.isclass(cls):
            inst_failures.append((cname, "Not a class"))
            continue

        kwargs = {"model": "test-model"}
        # Add overrides if required
        if cname in instantiation_overrides:
            kwargs.update(instantiation_overrides[cname])

        # If signature has required positional params (besides self) without defaults, try to supply simple placeholders
        sig = inspect.signature(cls.__init__)
        for pname, param in list(sig.parameters.items())[1:]:
            if pname in kwargs:
                continue
            if (param.default is inspect._empty
                and param.kind in (param.POSITIONAL_OR_KEYWORD, param.KEYWORD_ONLY)):
                # Provide generic placeholder if not model/openai_api_key
                if pname not in ("model", "openai_api_key"):
                    kwargs[pname] = f"test_{pname}"

        try:
            _instance = cls(**kwargs)
            success_count += 1
            print(f"[OK] {cname} instantiated")
        except Exception as e:
            inst_failures.append((cname, str(e)))
            print(f"[FAIL] {cname} instantiation: {e}")

print("\n[RESULT]")
print(f"  Classes discovered: {total_classes}")
print(f"  Instantiated OK:    {success_count}")
print(f"  Failed:             {len(inst_failures)}")

if inst_failures:
    print("\n[FAILURES]")
    for cname, err in inst_failures:
        print(f"  - {cname}: {err}")
    sys.exit(3)


print("\n[SUCCESS] All expert & PI agent imports and instantiations passed.")
sys.exit(0)