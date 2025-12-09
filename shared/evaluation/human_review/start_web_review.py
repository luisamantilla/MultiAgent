#!/usr/bin/env python3
"""
Quick starter script for the human review web interface
Usage: python start_web_review.py [evaluation_file.json]
"""

import sys
import os
from pathlib import Path

def main():
    # Get evaluation file from command line or use default
    if len(sys.argv) > 1:
        eval_file = sys.argv[1]
    else:
        # Look for most recent evaluation file in outputs directory
        outputs_dir = Path(__file__).parent / "enhanced_llm_eval_outputs"
        if outputs_dir.exists():
            eval_files = list(outputs_dir.glob("enhanced_evaluation_*.json"))
            if eval_files:
                eval_file = str(max(eval_files, key=os.path.getctime))
                print(f"Using most recent evaluation file: {eval_file}")
            else:
                print("No evaluation files found in enhanced_llm_eval_outputs/")
                print("Usage: python start_web_review.py <evaluation_file.json>")
                sys.exit(1)
        else:
            print("No enhanced_llm_eval_outputs/ directory found")
            print("Usage: python start_web_review.py <evaluation_file.json>")
            sys.exit(1)
    
    if not os.path.exists(eval_file):
        print(f"Error: File '{eval_file}' not found")
        sys.exit(1)
    
    print("🧬 Starting Biological Process Review Web Interface")
    print(f"📁 Evaluation file: {eval_file}")
    print("🌐 Web interface will start at: http://localhost:5000")
    print("⚡ Use Ctrl+C to stop the server")
    print("📱 Interface works on mobile devices too!")
    print("\n" + "="*60)
    
    # Import and start web interface
    try:
        from human_review_web import main as web_main
        sys.argv = ["human_review_web.py", eval_file]
        web_main()
    except ImportError as e:
        print(f"Error importing web interface: {e}")
        print("Make sure Flask is installed: pip install flask")
        sys.exit(1)

if __name__ == "__main__":
    main()
