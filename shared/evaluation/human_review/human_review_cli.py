#!/usr/bin/env python3
"""
Human Review Interface for LLM Evaluation Results (CLI Version)

This script provides an interactive command-line interface for human reviewers
to evaluate LLM-generated step matches from biological process evaluations.
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Any
import argparse
from pathlib import Path


class HumanReviewCLI:
    """Command-line interface for human review of LLM evaluation results"""
    
    def __init__(self, evaluation_file: str):
        self.evaluation_file = evaluation_file
        self.data = self.load_evaluation_data()
        self.reviewable_steps = self.get_reviewable_steps()
        self.current_index = 0
        self.review_session = {
            "start_time": datetime.now().isoformat(),
            "reviewer": None,
            "total_reviews": 0,
            "good_matches": 0,
            "bad_matches": 0,
            "skipped": 0
        }
    
    def load_evaluation_data(self) -> Dict[str, Any]:
        """Load the LLM evaluation results"""
        try:
            with open(self.evaluation_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Error: Evaluation file '{self.evaluation_file}' not found.")
            exit(1)
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON in '{self.evaluation_file}'.")
            exit(1)
    
    def get_reviewable_steps(self) -> List[Dict[str, Any]]:
        """Get all steps that have LLM matches and need human review"""
        reviewable = []
        for step in self.data.get("step_level_results", []):
            if step.get("matched_prediction") is not None:
                reviewable.append(step)
        return reviewable
    
    def print_header(self):
        """Print session header"""
        print("\n" + "="*80)
        print("🧬 BIOLOGICAL PROCESS EVALUATION - HUMAN REVIEW INTERFACE")
        print("="*80)
        print(f"Evaluation File: {os.path.basename(self.evaluation_file)}")
        print(f"Total Matches to Review: {len(self.reviewable_steps)}")
        print(f"LLM Model: {self.data.get('model', 'unknown')}")
        print(f"Evaluation Mode: {self.data.get('evaluation_mode', 'unknown')}")
        print("="*80)
        
        # Get reviewer name
        reviewer = input("Enter your name (for tracking): ").strip()
        if reviewer:
            self.review_session["reviewer"] = reviewer
        print()
    
    def display_step_comparison(self, step: Dict[str, Any], index: int) -> None:
        """Display a single step comparison for review"""
        print(f"\n{'='*60}")
        print(f"REVIEW {index + 1} of {len(self.reviewable_steps)}")
        print(f"{'='*60}")
        print(f"📋 Process: {step.get('process_name', 'Unknown')}")
        print(f"🔬 Context: {self.data.get('context', 'Not specified')}")
        print("-" * 60)
        
        print("🎯 GROUND TRUTH STEP:")
        print(f"   {step.get('step_detail', 'No detail available')}")
        print()
        
        print("🤖 LLM MATCHED PREDICTION:")
        print(f"   {step.get('matched_prediction', 'No match')}")
        print()
        
        if step.get('biological_reasoning'):
            print("🧠 LLM REASONING:")
            print(f"   {step['biological_reasoning']}")
            print()
        
        # Show current human judgment if exists
        current_judgment = step.get('human_judgment', 'pending')
        if current_judgment != 'pending':
            print(f"⚖️  CURRENT JUDGMENT: {current_judgment.upper()}")
            print()
    
    def get_human_judgment(self) -> str:
        """Get human judgment for current step"""
        print("⚖️  EVALUATION OPTIONS:")
        print("   [g] GOOD MATCH - LLM correctly matched this step")
        print("   [b] BAD MATCH - LLM incorrectly matched this step")
        print("   [s] SKIP - Need more context or uncertain")
        print("   [q] QUIT - Save progress and exit")
        print("   [p] PREVIOUS - Go back to previous step")
        print("   [c] CONTEXT - Show more biological context")
        
        while True:
            choice = input("\nYour judgment [g/b/s/q/p/c]: ").lower().strip()
            
            if choice in ['g', 'good']:
                return 'good'
            elif choice in ['b', 'bad']:
                return 'bad'
            elif choice in ['s', 'skip']:
                return 'skip'
            elif choice in ['q', 'quit']:
                return 'quit'
            elif choice in ['p', 'prev', 'previous']:
                return 'previous'
            elif choice in ['c', 'context']:
                return 'context'
            else:
                print("   Invalid choice. Please enter g, b, s, q, p, or c")
    
    def show_biological_context(self, step: Dict[str, Any]) -> None:
        """Show additional biological context for the step"""
        print("\n" + "📚 BIOLOGICAL CONTEXT" + "="*44)
        
        # Find the original process this step belongs to
        process_name = step.get('process_name', '')
        
        # Look for the full process in original data
        ground_truth = self.data.get('ground_truth_summary', {})
        for process in ground_truth.get('processes', []):
            if process.get('name') == process_name:
                print(f"Full Process: {process.get('name', 'Unknown')}")
                if process.get('description'):
                    print(f"Description: {process['description']}")
                if process.get('steps'):
                    print(f"Related Steps in Process:")
                    for i, related_step in enumerate(process['steps'][:5], 1):
                        marker = "👉" if related_step == step.get('step_detail') else "  "
                        print(f"   {marker} {i}. {related_step}")
                break
        
        # Show entities involved
        entities = self.data.get('entity_coverage_results', {})
        print(f"\nKey Entities in Evaluation:")
        for entity_type in ['cell_types', 'molecules', 'other_entities']:
            type_entities = entities.get(entity_type, [])
            if type_entities:
                found = [e['entity_name'] for e in type_entities if e.get('found_in_predictions')]
                missing = [e['entity_name'] for e in type_entities if not e.get('found_in_predictions')]
                if found:
                    print(f"   ✅ {entity_type.title()}: {', '.join(found[:3])}")
                if missing:
                    print(f"   ❌ Missing {entity_type.title()}: {', '.join(missing[:3])}")
        
        print("="*60)
        input("Press Enter to continue...")
    
    def update_step_judgment(self, step: Dict[str, Any], judgment: str) -> None:
        """Update step with human judgment"""
        step['human_judgment'] = judgment
        step['human_review_timestamp'] = datetime.now().isoformat()
        if self.review_session["reviewer"]:
            step['human_reviewer'] = self.review_session["reviewer"]
        
        # Update session stats
        self.review_session["total_reviews"] += 1
        if judgment == 'good':
            self.review_session["good_matches"] += 1
        elif judgment == 'bad':
            self.review_session["bad_matches"] += 1
        elif judgment == 'skip':
            self.review_session["skipped"] += 1
    
    def save_progress(self) -> str:
        """Save current progress to file"""
        # Update session info in data
        self.review_session["end_time"] = datetime.now().isoformat()
        self.data["human_review_session"] = self.review_session
        
        # Create output filename
        base_name = os.path.splitext(self.evaluation_file)[0]
        output_file = f"{base_name}_human_reviewed.json"
        
        # Save updated data
        with open(output_file, 'w') as f:
            json.dump(self.data, f, indent=2)
        
        return output_file
    
    def print_session_summary(self) -> None:
        """Print summary of review session"""
        total = self.review_session["total_reviews"]
        good = self.review_session["good_matches"]
        bad = self.review_session["bad_matches"]
        skipped = self.review_session["skipped"]
        
        print("\n" + "📊 REVIEW SESSION SUMMARY" + "="*32)
        print(f"Total Reviews Completed: {total}/{len(self.reviewable_steps)}")
        print(f"Good Matches: {good} ({good/total*100:.1f}%)" if total > 0 else "Good Matches: 0")
        print(f"Bad Matches: {bad} ({bad/total*100:.1f}%)" if total > 0 else "Bad Matches: 0")
        print(f"Skipped: {skipped} ({skipped/total*100:.1f}%)" if total > 0 else "Skipped: 0")
        
        if self.review_session["reviewer"]:
            print(f"Reviewer: {self.review_session['reviewer']}")
        
        if total > 0:
            # Calculate human-corrected metrics
            human_good = good
            total_matches = len(self.reviewable_steps)
            human_precision = human_good / total_matches if total_matches > 0 else 0
            print(f"Human-Validated Precision: {human_precision:.3f}")
        
        print("="*60)
    
    def run_review_session(self) -> None:
        """Run the interactive review session"""
        self.print_header()
        
        if not self.reviewable_steps:
            print("No matches found to review. All steps were unmatched by the LLM.")
            return
        
        print(f"🚀 Starting review of {len(self.reviewable_steps)} LLM matches...")
        print("   Use 'c' for context, 'p' for previous, 'q' to quit and save progress")
        
        while self.current_index < len(self.reviewable_steps):
            step = self.reviewable_steps[self.current_index]
            
            # Display the comparison
            self.display_step_comparison(step, self.current_index)
            
            # Get human judgment
            judgment = self.get_human_judgment()
            
            if judgment == 'quit':
                break
            elif judgment == 'previous':
                if self.current_index > 0:
                    self.current_index -= 1
                else:
                    print("Already at the first step.")
                continue
            elif judgment == 'context':
                self.show_biological_context(step)
                continue
            else:
                # Valid judgment, update and move on
                self.update_step_judgment(step, judgment)
                self.current_index += 1
                
                # Show progress
                remaining = len(self.reviewable_steps) - self.current_index
                print(f"✅ Recorded as '{judgment.upper()}'. {remaining} steps remaining.")
        
        # Save results and show summary
        output_file = self.save_progress()
        print(f"\n💾 Progress saved to: {output_file}")
        
        self.print_session_summary()
        
        if self.current_index >= len(self.reviewable_steps):
            print("\n🎉 All matches reviewed! Great work!")
        else:
            print(f"\n⏸️  Session paused. Resume with the same file to continue from step {self.current_index + 1}.")


def main():
    parser = argparse.ArgumentParser(
        description="Human review interface for LLM evaluation results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python human_review_cli.py enhanced_evaluation_20250909_204258.json
  python human_review_cli.py /path/to/evaluation.json
        """
    )
    
    parser.add_argument(
        "evaluation_file",
        help="Path to the enhanced LLM evaluation JSON file"
    )
    
    args = parser.parse_args()
    
    # Validate file exists
    if not os.path.exists(args.evaluation_file):
        print(f"Error: File '{args.evaluation_file}' not found.")
        exit(1)
    
    # Create and run review interface
    reviewer = HumanReviewCLI(args.evaluation_file)
    
    try:
        reviewer.run_review_session()
    except KeyboardInterrupt:
        print("\n\n⚠️  Review interrupted. Saving progress...")
        output_file = reviewer.save_progress()
        print(f"💾 Progress saved to: {output_file}")
        reviewer.print_session_summary()


if __name__ == "__main__":
    main()
