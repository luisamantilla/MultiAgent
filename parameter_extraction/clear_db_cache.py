#!/usr/bin/env python3
"""
Force clear ChromaDB cache and rebuild the vector store.
Run this if you're getting database corruption errors.
"""
import shutil
import sys
from pathlib import Path

# Add parameter_extraction to path
sys.path.insert(0, str(Path(__file__).parent))

from utils.vector_store import MEMORY_DIR

def clear_and_rebuild():
    print(f"🗑️  Removing database at {MEMORY_DIR}...")
    if MEMORY_DIR.exists():
        shutil.rmtree(MEMORY_DIR)
        print("   ✓ Removed")
    else:
        print("   ℹ️  Directory doesn't exist")

    print("\n🔨 Rebuilding database...")
    print("   Run: python parameter_extraction/main.py --mode embed --persist parameter_extraction/memory/params_db")
    print("\nAfter running the rebuild command above, your database will be fresh and ready to use.")

if __name__ == "__main__":
    clear_and_rebuild()
