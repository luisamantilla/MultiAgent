import json
import yaml
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
import sys

# Add the shared directory to Python path to import BaseAgent
_PROJECT_ROOT = Path(__file__).resolve().parents[2]  # Go up to MultiAgent/
_SHARED_DIR = _PROJECT_ROOT / "shared"
if str(_SHARED_DIR) not in sys.path:
    sys.path.insert(0, str(_SHARED_DIR))

from base.base_agent import BaseAgent

class ParameterMapperAgent(BaseAgent):
    """
    Agent responsible for mapping biological interactions to mathematical equations
    and parameter needs for computational modeling.
    """
    
    def __init__(self, model: str = "gpt-4.1", base_output_dir: str = None):
        super().__init__(
            title="Parameter Mapper Agent",
            expertise="Mapping biological interactions to mathematical formulations and parameter requirements",
            goal="Create comprehensive parameter specifications for all model interactions",
            role="Mathematical Biology Translator",
            model=model
        )
        
        # Set up output directory
        if base_output_dir is None:
            base_output_dir = "/home/labuser/Desktop/lab_member_projects/Bobby_Ni/tumor-tcell-exp/influenza_modeling_agents_design"
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(base_output_dir) / f"parameter_mapper_run_{timestamp}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"📁 Output directory: {self.output_dir}")
    
    def map_parameters(self, 
                      requirements_path: str = None,
                      requirements_dict: Dict = None,
                      filter_essential_only: bool = False) -> Tuple[pd.DataFrame, Dict]:
        """
        Map biological interactions to equations and parameters.
        
        Args:
            requirements_path: Path to requirements JSON from BiologistAgent
            requirements_dict: Direct requirements dictionary (if not loading from file)
            filter_essential_only: Whether to only process essential interactions
            
        Returns:
            Tuple of (DataFrame with parameter mapping, Dict with full mapping details)
        """
        
        # Load requirements
        if requirements_dict:
            requirements = requirements_dict
        elif requirements_path:
            with open(requirements_path, 'r') as f:
                requirements = json.load(f)
        else:
            raise ValueError("Must provide either requirements_path or requirements_dict")
        
        # Save input
        self._save_intermediate("input_requirements", requirements)
        
        print("🔬 Analyzing biological interactions...")
        interactions = requirements.get('interactions', [])
        
        # Process each interaction
        all_mappings = []
        detailed_mappings = {}
        
        for i, interaction in enumerate(interactions):
            print(f"📊 Processing interaction {i+1}/{len(interactions)}: {interaction.get('name', 'Unknown')}")
            
            mapping = self._map_single_interaction(interaction, requirements)
            all_mappings.append(mapping)
            detailed_mappings[interaction.get('id', f'interaction_{i}')] = mapping
            
            # Save intermediate progress every 10 interactions
            if (i + 1) % 10 == 0:
                self._save_intermediate(f"progress_{i+1}", {"processed": all_mappings})
        
        # Create DataFrame
        df = self._create_parameter_dataframe(all_mappings)
        
        # Filter if requested
        if filter_essential_only:
            df = df[df['is_essential'] == True]
            print(f"✂️ Filtered to {len(df)} essential interactions")
        
        # Add parameter source analysis
        print("🔍 Analyzing parameter sources...")
        df = self._analyze_parameter_sources(df)
        
        # Save outputs
        self._save_outputs(df, detailed_mappings, requirements)
        
        return df, detailed_mappings
    
    def _map_single_interaction(self, interaction: Dict, requirements: Dict) -> Dict:
        """Map a single interaction to equations and parameters using LLM"""
        
        # Extract relevant context
        source_cells = interaction.get('participants', {}).get('sources', [])
        target_cells = interaction.get('participants', {}).get('targets', [])
        interaction_type = interaction.get('type', 'unknown')
        
        prompt = f"""
You are a mathematical biologist translating biological interactions into computational model specifications.

Interaction Details:
{json.dumps(interaction, indent=2)}

For this {interaction_type} interaction, provide:

1. MATHEMATICAL EQUATION(S):
   - Write the specific differential equations or rules
   - Use standard notation (dX/dt, probability functions, etc.)
   - Include all state transitions

2. PARAMETERS NEEDED:
   For each parameter, specify:
   - Parameter name (use descriptive names like 'viral_infection_rate')
   - Units (e.g., cells/hour, 1/hour, molecules/cell/hour)
   - Typical range from literature
   - Source category: "literature_available", "needs_estimation", "needs_calibration", or "from_previous_models"

3. DEPENDENCIES:
   - Which other cell types/molecules are involved
   - Environmental factors (spatial constraints, diffusion, etc.)

Output as JSON:
{{
    "is_essential": true/false,  // Is this essential for answering the key model questions?
    "equations": [
        {{
            "type": "ODE/stochastic/rule-based",
            "equation": "mathematical expression",
            "description": "what this equation represents"
        }}
    ],
    "parameters": [
        {{
            "name": "parameter_name",
            "symbol": "mathematical symbol",
            "units": "units",
            "typical_range": "min-max values",
            "source": "literature_available/needs_estimation/needs_calibration/from_previous_models",
            "references": ["paper references if known"]
        }}
    ],
    "dependencies": {{
        "cell_types": ["list of involved cell types"],
        "molecules": ["list of involved molecules"],
        "spatial_factors": ["contact_required", "diffusion", etc.],
        "other_interactions": ["IDs of dependent interactions"]
    }},
    "implementation_notes": "Any special considerations for coding this interaction"
}}
"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,  # Low temperature for consistency
                response_format={"type": "json_object"}
            )
            
            mapping = json.loads(response.choices[0].message.content)
            
            # Add interaction metadata
            mapping['interaction_id'] = interaction.get('id', 'unknown')
            mapping['interaction_name'] = interaction.get('name', 'unknown')
            mapping['interaction_type'] = interaction_type
            mapping['source_cells'] = [s['cell_type'] for s in source_cells]
            mapping['target_cells'] = [t['cell_type'] for t in target_cells]
            
            return mapping
            
        except Exception as e:
            print(f"Error mapping interaction {interaction.get('name')}: {e}")
            return self._get_fallback_mapping(interaction)
    
    def _create_parameter_dataframe(self, mappings: List[Dict]) -> pd.DataFrame:
        """Convert mappings to a structured DataFrame"""
        
        rows = []
        for mapping in mappings:
            # Base row for the interaction
            base_row = {
                'interaction_id': mapping.get('interaction_id'),
                'interaction_name': mapping.get('interaction_name'),
                'interaction_type': mapping.get('interaction_type'),
                'is_essential': mapping.get('is_essential', False),
                'source_cells': ', '.join(mapping.get('source_cells', [])),
                'target_cells': ', '.join(mapping.get('target_cells', [])),
                'involved_cells': ', '.join(mapping.get('dependencies', {}).get('cell_types', [])),
                'involved_molecules': ', '.join(mapping.get('dependencies', {}).get('molecules', [])),
                'spatial_factors': ', '.join(mapping.get('dependencies', {}).get('spatial_factors', [])),
                'implementation_notes': mapping.get('implementation_notes', '')
            }
            
            # Create a row for each parameter
            parameters = mapping.get('parameters', [])
            if parameters:
                for param in parameters:
                    param_row = base_row.copy()
                    param_row.update({
                        'parameter_name': param.get('name'),
                        'parameter_symbol': param.get('symbol'),
                        'units': param.get('units'),
                        'typical_range': param.get('typical_range'),
                        'source': param.get('source'),
                        'references': ', '.join(param.get('references', []))
                    })
                    
                    # Add equation info
                    equations = mapping.get('equations', [])
                    if equations:
                        param_row['equation_type'] = equations[0].get('type')
                        param_row['equation'] = equations[0].get('equation')
                    
                    rows.append(param_row)
            else:
                # Add row even if no parameters (for tracking)
                equations = mapping.get('equations', [])
                if equations:
                    base_row['equation_type'] = equations[0].get('type')
                    base_row['equation'] = equations[0].get('equation')
                rows.append(base_row)
        
        return pd.DataFrame(rows)
    
    def _analyze_parameter_sources(self, df: pd.DataFrame) -> pd.DataFrame:
        """Analyze parameter sources and add summary statistics"""
        
        # Add priority score based on essentiality and source availability
        def calculate_priority(row):
            score = 0
            if row['is_essential']:
                score += 10
            
            source = row.get('source', '')
            if source == 'literature_available':
                score += 1
            elif source == 'from_previous_models':
                score += 2
            elif source == 'needs_estimation':
                score += 5
            elif source == 'needs_calibration':
                score += 8
            
            return score
        
        df['priority_score'] = df.apply(calculate_priority, axis=1)
        
        # Sort by priority
        df = df.sort_values(['priority_score', 'is_essential'], ascending=[False, False])
        
        return df
    
    def _save_outputs(self, df: pd.DataFrame, detailed_mappings: Dict, requirements: Dict):
        """Save all outputs in multiple formats"""
        
        # Save DataFrame as CSV
        csv_path = self.output_dir / "parameter_mapping_table.csv"
        df.to_csv(csv_path, index=False)
        print(f"📊 Saved parameter table: {csv_path}")
        
        # Save as Excel with formatting
        excel_path = self.output_dir / "parameter_mapping_table.xlsx"
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Parameter Mapping', index=False)
            
            # Add summary sheet
            summary_df = self._create_summary_stats(df)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
        print(f"📊 Saved Excel file: {excel_path}")
        
        # Save detailed mappings as JSON
        json_path = self.output_dir / "detailed_parameter_mappings.json"
        with open(json_path, 'w') as f:
            json.dump(detailed_mappings, f, indent=2)
        
        # Create parameter wishlist
        wishlist = self._create_parameter_wishlist(df)
        wishlist_path = self.output_dir / "parameter_wishlist.json"
        with open(wishlist_path, 'w') as f:
            json.dump(wishlist, f, indent=2)
        
        # Create human-readable report
        report = self._generate_report(df, requirements)
        report_path = self.output_dir / "parameter_mapping_report.md"
        with open(report_path, 'w') as f:
            f.write(report)
        print(f"📝 Saved report: {report_path}")
    
    def _create_summary_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create summary statistics"""
        
        # Remove duplicate parameters for counting
        unique_params = df.dropna(subset=['parameter_name']).drop_duplicates(subset=['parameter_name'])
        
        summary = {
            'Metric': [
                'Total Interactions',
                'Essential Interactions',
                'Total Unique Parameters',
                'Parameters from Literature',
                'Parameters from Previous Models',
                'Parameters Needing Estimation',
                'Parameters Needing Calibration',
                'Interactions with Spatial Dependencies',
                'Multi-cell Interactions'
            ],
            'Count': [
                df['interaction_id'].nunique(),
                df[df['is_essential'] == True]['interaction_id'].nunique(),
                len(unique_params),
                len(unique_params[unique_params['source'] == 'literature_available']),
                len(unique_params[unique_params['source'] == 'from_previous_models']),
                len(unique_params[unique_params['source'] == 'needs_estimation']),
                len(unique_params[unique_params['source'] == 'needs_calibration']),
                len(df[df['spatial_factors'].str.len() > 0]),
                len(df[df['involved_cells'].str.contains(',')])
            ]
        }
        
        return pd.DataFrame(summary)
    
    def _create_parameter_wishlist(self, df: pd.DataFrame) -> Dict:
        """Create organized parameter wishlist"""
        
        wishlist = {
            'high_priority': [],
            'medium_priority': [],
            'low_priority': [],
            'by_source': {
                'literature_available': [],
                'from_previous_models': [],
                'needs_estimation': [],
                'needs_calibration': []
            }
        }
        
        # Remove duplicates
        unique_params = df.dropna(subset=['parameter_name']).drop_duplicates(subset=['parameter_name'])
        
        for _, row in unique_params.iterrows():
            param_info = {
                'name': row['parameter_name'],
                'symbol': row.get('parameter_symbol', ''),
                'units': row.get('units', ''),
                'typical_range': row.get('typical_range', ''),
                'used_in': row['interaction_name'],
                'source': row.get('source', ''),
                'priority_score': row.get('priority_score', 0)
            }
            
            # Categorize by priority
            if row['priority_score'] >= 15:
                wishlist['high_priority'].append(param_info)
            elif row['priority_score'] >= 10:
                wishlist['medium_priority'].append(param_info)
            else:
                wishlist['low_priority'].append(param_info)
            
            # Categorize by source
            source = row.get('source', 'unknown')
            if source in wishlist['by_source']:
                wishlist['by_source'][source].append(param_info)
        
        return wishlist
    
    def _generate_report(self, df: pd.DataFrame, requirements: Dict) -> str:
        """Generate human-readable markdown report"""
        
        report = f"""# Parameter Mapping Report

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Project Overview

**Goals**: {requirements.get('metadata', {}).get('model_scope', 'Not specified')}

## Summary Statistics

{self._create_summary_stats(df).to_markdown(index=False)}

## Essential Interactions

The following interactions have been identified as essential for the model:

"""
        
        essential_df = df[df['is_essential'] == True].drop_duplicates(subset=['interaction_id'])
        for _, row in essential_df.iterrows():
            report += f"\n### {row['interaction_name']}\n"
            report += f"- **Type**: {row['interaction_type']}\n"
            report += f"- **Source cells**: {row['source_cells']}\n"
            report += f"- **Target cells**: {row['target_cells']}\n"
            if pd.notna(row.get('equation')):
                report += f"- **Equation**: `{row['equation']}`\n"
            report += "\n"
        
        report += """
## Parameter Wishlist

### High Priority Parameters (Essential interactions needing calibration/estimation)

"""
        
        high_priority = df[(df['is_essential'] == True) & 
                          (df['source'].isin(['needs_calibration', 'needs_estimation']))].dropna(subset=['parameter_name'])
        
        for _, row in high_priority.iterrows():
            report += f"- **{row['parameter_name']}** ({row['units']})\n"
            report += f"  - Used in: {row['interaction_name']}\n"
            report += f"  - Typical range: {row['typical_range']}\n"
            report += f"  - Status: {row['source']}\n\n"
        
        return report
    
    def _save_intermediate(self, name: str, data: Dict):
        """Save intermediate results"""
        filepath = self.output_dir / f"{name}.json"
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    def _get_fallback_mapping(self, interaction: Dict) -> Dict:
        """Fallback mapping if LLM fails"""
        return {
            'interaction_id': interaction.get('id', 'unknown'),
            'interaction_name': interaction.get('name', 'unknown'),
            'interaction_type': interaction.get('type', 'unknown'),
            'is_essential': True,  # Conservative: assume essential
            'source_cells': [s['cell_type'] for s in interaction.get('participants', {}).get('sources', [])],
            'target_cells': [t['cell_type'] for t in interaction.get('participants', {}).get('targets', [])],
            'equations': [{
                'type': 'unknown',
                'equation': 'Needs manual specification',
                'description': 'Fallback - requires manual input'
            }],
            'parameters': [{
                'name': f"{interaction.get('type', 'unknown')}_rate",
                'units': 'unknown',
                'source': 'needs_estimation'
            }],
            'dependencies': {
                'cell_types': [],
                'molecules': []
            }
        }


# Example usage
if __name__ == "__main__":
    # Initialize mapper
    mapper = ParameterMapperAgent(model="gpt-4")
    
    # Option 1: Load from a BiologistAgent output file
    requirements_file = "/home/labuser/Desktop/lab_member_projects/Bobby_Ni/tumor-tcell-exp/influenza_modeling_agents_design/biologist_run_20240609_150000/requirements_v1.json"
    
    # Generate parameter mapping
    df, detailed_mappings = mapper.map_parameters(
        requirements_path=requirements_file,
        filter_essential_only=False  # Set to True to only process essential interactions
    )
    
    print("\n✅ Parameter mapping complete!")
    print(f"📊 Processed {len(df)} parameter specifications")
    print(f"🔍 Found {df['parameter_name'].nunique()} unique parameters")
    print(f"{len(df[df['source'] == 'needs_calibration'])} parameters need calibration (needs calibration)")
    print(f"🔬 {len(df[df['source'] == 'needs_estimation'])} parameters need estimation")