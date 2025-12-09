#!/usr/bin/env python3
"""
run_direct_test.py - Quick runner for the direct parameter extraction test

This script runs the direct parameter extraction test for bouchnita_2017.pdf
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def main():
    """Run the direct parameter extraction test."""
    print("Running Direct Parameter Extraction Test")
    print("Target: IFN-γ production rate for PD1-negative T cells")
    print("Paper: bouchnita_2017.pdf")
    print("Expected: 4.5 molecules/cell/second")
    print("=" * 60)

    try:
        from direct_parameter_test import run_direct_parameter_extraction
        success = run_direct_parameter_extraction()

        if success:
            print("\n✅ Direct parameter extraction test completed successfully!")
        else:
            print("\n❌ Direct parameter extraction test failed!")

        return success

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure direct_parameter_test.py is in the same directory")
        return False
    except Exception as e:
        print(f"❌ Error running test: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)