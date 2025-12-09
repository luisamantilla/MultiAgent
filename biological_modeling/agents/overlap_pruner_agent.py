import json
import sys
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Literal

# Import dependencies with fallback for notebook usage
try:
    from agents.base_agent import BaseAgent
    from agents.biological_modeling_agents.pathway_expander_agent import ExpandedInteraction, DetailedStep
except ImportError:
    # Fallback for notebook execution
    from base_agent import BaseAgent
    from pathway_expander_agent import ExpandedInteraction, DetailedStep

# --- Agent Definition ---

class OverlapPrunerAgent(BaseAgent):
    """
    Agent responsible for detecting and removing duplicate/overlapping steps 
    across multiple expanded interactions.
    """

    def __init__(self, model: str = "gpt-4.1"):
        super().__init__(
            title="Overlap Pruner Agent",
            expertise="Detecting redundancy and overlap in biological pathways, ensuring clean and non-repetitive mechanistic representations.",
            goal="To identify and eliminate duplicate mechanistic steps across multiple interactions while preserving unique biological insights.",
            role="Redundancy Elimination Specialist",
            model=model
        )

    def run(self, 
            expanded_interactions: List[ExpandedInteraction],
            similarity_threshold: float = 0.9) -> Dict[str, Any]:
        """
        Detects and removes overlapping steps across multiple expanded interactions.
        
        Args:
            expanded_interactions: List of ExpandedInteraction objects from PathwayExpanderAgent
            similarity_threshold: Threshold for determining step similarity (0.0-1.0)
            
        Returns:
            Dictionary containing deduplicated interactions and overlap analysis report
        """
        print(f"\n Analyzing overlaps across {len(expanded_interactions)} expanded interactions...")
        print(f"Using similarity threshold: {similarity_threshold}")
        
        # Collect all steps with their source interaction
        all_steps_with_source = []
        for expanded in expanded_interactions:
            for step in expanded.detailed_steps:
                all_steps_with_source.append({
                    'step': step,
                    'source_interaction': expanded.interaction_id,
                    'normalized_desc': self._normalize_description(step.description)
                })
        
        # Find duplicates and similar steps
        duplicates, unique_steps = self._find_duplicates_and_similarities(
            all_steps_with_source, similarity_threshold
        )
        
        # Generate overlap report
        overlap_report = {
            'total_steps': len(all_steps_with_source),
            'unique_steps': len(unique_steps),
            'duplicate_count': len(duplicates),
            'similarity_threshold': similarity_threshold,
            'duplicates': duplicates,
            'redundancy_rate': len(duplicates) / len(all_steps_with_source) * 100 if all_steps_with_source else 0
        }
        
        # Rebuild deduplicated interactions
        deduplicated_interactions = self._rebuild_deduplicated_interactions(
            expanded_interactions, unique_steps
        )
        
        print(f"Found {len(duplicates)} duplicate/similar steps out of {len(all_steps_with_source)} total steps")
        print(f"Redundancy rate: {overlap_report['redundancy_rate']:.1f}%")
        
        return {
            'deduplicated_interactions': deduplicated_interactions,
            'overlap_report': overlap_report,
            'original_interactions': expanded_interactions
        }
    
    def _normalize_description(self, description: str) -> str:
        """
        Normalizes a step description for comparison.
        """
        # Remove punctuation, convert to lowercase, strip whitespace
        import re
        normalized = re.sub(r'[^\w\s]', '', description.lower().strip())
        # Remove extra whitespace
        normalized = ' '.join(normalized.split())
        return normalized
    
    def _find_duplicates_and_similarities(self, 
                                        all_steps: List[Dict],
                                        threshold: float) -> tuple:
        """
        Finds duplicate and similar steps based on normalized descriptions.
        """
        seen_descriptions = {}
        duplicates = []
        unique_steps = []
        
        for item in all_steps:
            norm_desc = item['normalized_desc']
            
            # Check for exact matches first
            if norm_desc in seen_descriptions:
                # This is an exact duplicate
                existing_item = seen_descriptions[norm_desc]
                duplicates.append({
                    'description': item['step'].description,
                    'sources': [existing_item['source_interaction'], item['source_interaction']],
                    'step_details': item['step'],
                    'similarity_type': 'exact_match',
                    'similarity_score': 1.0
                })
                continue
            
            # Check for similarity if threshold < 1.0
            if threshold < 1.0:
                similar_found = False
                for existing_desc, existing_item in seen_descriptions.items():
                    similarity = self._calculate_similarity(norm_desc, existing_desc)
                    if similarity >= threshold:
                        duplicates.append({
                            'description': item['step'].description,
                            'sources': [existing_item['source_interaction'], item['source_interaction']],
                            'step_details': item['step'],
                            'similarity_type': 'similar',
                            'similarity_score': similarity,
                            'similar_to': existing_item['step'].description
                        })
                        similar_found = True
                        break
                
                if similar_found:
                    continue
            
            # This is a unique step
            seen_descriptions[norm_desc] = item
            unique_steps.append(item)
        
        return duplicates, unique_steps
    
    def _calculate_similarity(self, desc1: str, desc2: str) -> float:
        """
        Calculates similarity between two normalized descriptions using simple word overlap.
        For more sophisticated similarity, could use sentence embeddings or edit distance.
        """
        words1 = set(desc1.split())
        words2 = set(desc2.split())
        
        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def _rebuild_deduplicated_interactions(self, 
                                         original_interactions: List[ExpandedInteraction],
                                         unique_steps: List[Dict]) -> List[ExpandedInteraction]:
        """
        Rebuilds the expanded interactions with deduplicated steps.
        """
        interaction_steps = {}
        
        # Group unique steps by their source interaction
        for item in unique_steps:
            source = item['source_interaction']
            if source not in interaction_steps:
                interaction_steps[source] = []
            interaction_steps[source].append(item['step'])
        
        # Rebuild interactions, preserving original order
        deduplicated = []
        for original in original_interactions:
            deduplicated_steps = interaction_steps.get(original.interaction_id, [])
            # Re-number steps to maintain sequential order
            for i, step in enumerate(deduplicated_steps, 1):
                step.step = i
            
            deduplicated.append(ExpandedInteraction(
                interaction_id=original.interaction_id,
                detailed_steps=deduplicated_steps
            ))
        
        return deduplicated
    
    def print_overlap_report(self, overlap_report: Dict) -> None:
        """
        Prints a formatted report of overlapping steps.
        """
        print("\n" + "="*70)
        print("OVERLAP ANALYSIS REPORT")
        print("="*70)
        print(f"Total steps analyzed: {overlap_report['total_steps']}")
        print(f"Unique steps: {overlap_report['unique_steps']}")
        print(f"Duplicate/similar steps: {overlap_report['duplicate_count']}")
        print(f"Redundancy rate: {overlap_report['redundancy_rate']:.1f}%")
        print(f"Similarity threshold: {overlap_report['similarity_threshold']}")
        
        if overlap_report['duplicates']:
            print(f"\nDUPLICATE/SIMILAR STEPS (showing all of {len(overlap_report['duplicates'])}):")
            
            for i, dup in enumerate(overlap_report['duplicates'][:]):
                similarity_info = ""
                if dup['similarity_type'] == 'similar':
                    similarity_info = f" (similarity: {dup['similarity_score']:.2f})"
                
                print(f"\n{i+1}. '{dup['description'][:]}...'")
                print(f"   Found in interactions: {', '.join(dup['sources'])}{similarity_info}")
                
                if dup['similarity_type'] == 'similar' and 'similar_to' in dup:
                    print(f"   Similar to: '{dup['similar_to'][:]}...'")
            
        else:
            print("\nNo duplicate or similar steps found!")
    
    def export_deduplication_summary(self, 
                                   overlap_report: Dict, 
                                   output_path: str = "deduplication_summary.json") -> None:
        """
        Exports a detailed deduplication summary to a JSON file.
        """
        summary = {
            'analysis_metadata': {
                'total_steps_analyzed': overlap_report['total_steps'],
                'unique_steps_retained': overlap_report['unique_steps'],
                'duplicates_removed': overlap_report['duplicate_count'],
                'redundancy_rate_percent': overlap_report['redundancy_rate'],
                'similarity_threshold': overlap_report['similarity_threshold']
            },
            'duplicate_details': overlap_report['duplicates']
        }
        
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Deduplication summary exported to: {output_path}")