#!/usr/bin/env python
"""
Isolated import test for computational expert agents.

Usage:
  PYTHONPATH=../src python tests/test_import_computational_experts.py
  (Run from agent-testing-framework directory)
"""
import sys, types, inspect
from pathlib import Path

# Resolve project src (this file is in src/tests/)
THIS_FILE = Path(__file__).resolve()
SRC_DIR = THIS_FILE.parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Optional: provide lightweight openai stub if library not installed
if 'openai' not in sys.modules:
    stub = types.ModuleType('openai')
    class OpenAI:
        def __init__(self, api_key=None): pass
        class chat:
            class completions:
                @staticmethod
                def create(**kwargs):
                    class _Msg: content = '{"implementation_feasibility":0.5,"computational_complexity":0.5,"scalability":0.5,"data_requirements":0.5,"confidence":0.5,"concerns":[],"suggestions":["stub"],"estimated_runtime":"n/a","memory_requirements":"n/a"}'
                    class _Choice: message = _Msg()
                    class _Resp: choices = [_Choice()]
                    return _Resp()
    stub.OpenAI = OpenAI
    sys.modules['openai'] = stub

MOD_NAME = "agents.biological_modeling_agents.crosstalk.computational_expert_agent"

print(f"[INFO] Importing {MOD_NAME}")
try:
    mod = __import__(MOD_NAME, fromlist=["*"])
except Exception as e:
    print(f"[FAIL] Import error: {e}")
    sys.exit(1)

# Collect candidate expert classes
expert_classes = {}
for name, obj in vars(mod).items():
    if (
        name.endswith("ExpertAgent")
        and inspect.isclass(obj)
        and obj.__module__ == MOD_NAME
    ):
        expert_classes[name] = obj

if not expert_classes:
    print("[FAIL] No *ExpertAgent classes discovered.")
    sys.exit(2)

print(f"[OK] Discovered {len(expert_classes)} expert classes:")
for n in sorted(expert_classes):
    print(f"  - {n}")

# Try instantiation (no API key needed due to stub)
instantiated = []
failed = []
for n, cls in expert_classes.items():
    try:
        inst = cls(model="test-model")
        instantiated.append(n)
    except Exception as e:
        failed.append((n, str(e)))

print(f"\n[RESULT] Instantiated: {len(instantiated)}/{len(expert_classes)}")
if failed:
    print("[WARN] Failed instantiations:")
    for n, err in failed:
        print(f"  - {n}: {err}")
    sys.exit(3)

print("[SUCCESS] All computational expert agent imports & instantiations passed.")
sys.exit(0)