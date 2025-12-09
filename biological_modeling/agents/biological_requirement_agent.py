import json
import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union
import sys

# Add the shared directory to Python path to import BaseAgent
_PROJECT_ROOT = Path(__file__).resolve().parents[2]  # Go up to MultiAgent/
_SHARED_DIR = _PROJECT_ROOT / "shared"
if str(_SHARED_DIR) not in sys.path:
    sys.path.insert(0, str(_SHARED_DIR))

from base.base_agent import BaseAgent

class BiologistAgent(BaseAgent):
    """
    An agent that generates comprehensive biological requirements
    with flexible interactions (multiple sources/targets) and organized output management.
    Tissue-agnostic - can generate requirements for any biological system.
    """
    
    def __init__(self, model: str = "gpt-4.1", base_output_dir: str = None):
        super().__init__(
            title="Biologist Agent",
            expertise="Generating comprehensive biological requirements for any tissue/biological system models",
            goal="Create detailed biological interaction models with multi-participant interactions for any biological context",
            role="Universal Biological Requirements Generator",
            model=model
        )
        
        # Set up output directory
        if base_output_dir is None:
            base_output_dir = "/home/labuser/Desktop/lab_member_projects/Bobby_Ni/tumor-tcell-exp/biological_modeling_agents_design"
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(base_output_dir) / f"biologist_run_{timestamp}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.intermediate_dir = self.output_dir / "intermediate"
        self.intermediate_dir.mkdir(exist_ok=True)
        
        print(f"Output directory: {self.output_dir}")
    
    def generate_requirements(self, 
                            project_goals: str, 
                            key_questions: List[str], 
                            biological_context: str,
                            essential_cell_types: List[str] = None,
                            key_molecules: List[str] = None,
                            key_interactions: List[str] = None,
                            additional_context: str = "",
                            iteration: int = 1) -> Dict:
        """
        Multi-step generation of comprehensive biological requirements using LLM.
        
        Args:
            project_goals: High-level goals for the biological model
            key_questions: Specific questions the model should answer
            biological_context: The biological system/tissue being modeled (e.g., "lung tissue", "tumor microenvironment", "liver")
            essential_cell_types: List of cell types that must be included (optional)
            key_molecules: List of specific molecules/signals that must be captured (optional)
            key_interactions: List of specific interactions that must be included (optional)
            additional_context: Any additional constraints or focus areas
            iteration: Version number for outputs
        """
        self._save_intermediate("input_parameters", {
            "project_goals": project_goals,
            "key_questions": key_questions,
            "biological_context": biological_context,
            "essential_cell_types": essential_cell_types,
            "key_molecules": key_molecules,
            "key_interactions": key_interactions,
            "additional_context": additional_context,
            "iteration": iteration,
            "timestamp": datetime.now().isoformat()
        })

        print("Step 1: Generating metadata, cell_types, and interactions...")
        step1 = self._generate_metadata_cells_interactions(project_goals, key_questions, biological_context, essential_cell_types, key_interactions, additional_context)
        self._save_intermediate("step1_metadata_cells_interactions", step1)

        print("Step 2: Generating molecules...")
        step2 = self._generate_molecules(step1, project_goals, key_questions, biological_context, key_molecules, additional_context)
        self._save_intermediate("step2_molecules", step2)

        print("Step 3: Generating spatial_structures...")
        step3 = self._generate_spatial_structures(step2, project_goals, key_questions, biological_context, additional_context)
        self._save_intermediate("step3_spatial_structures", step3)

        # Validate completeness
        print("Validating biological completeness...")
        validation_results = self._validate_requirements(step3, biological_context, essential_cell_types, key_molecules, key_interactions)
        step3['validation'] = validation_results

        self._save_final_output(step3, iteration)
        return step3

    def _generate_metadata_cells_interactions(self, project_goals, key_questions, biological_context, essential_cell_types, key_interactions, additional_context):
        """Step 1: Generate metadata, cell_types, and interactions only."""
        
        # Build essential cell types section
        essential_cells_text = ""
        if essential_cell_types:
            essential_cells_text = f"""
You MUST include the following essential cell types specified by the user:
{', '.join(essential_cell_types)}

Additionally, include any other relevant cell types for the {biological_context} system.
"""
        else:
            essential_cells_text = f"""
Based on the biological context ({biological_context}), identify and include ALL relevant cell types.
Consider resident cells, recruited cells, structural cells, and any pathogenic entities if applicable.
"""

        # Build key interactions section
        key_interactions_text = ""
        if key_interactions:
            key_interactions_text = f"""

PRIORITY INTERACTIONS - You MUST include these specific interactions requested by the user:
{chr(10).join(f"- {interaction}" for interaction in key_interactions)}

Make sure these interactions are modeled with full detail including all participants, mediators, and outcomes.
Additionally, include other relevant interactions for the {biological_context} system.
"""
        else:
            key_interactions_text = """

Focus on identifying ALL biologically relevant interactions for this system.
"""

        prompt = f"""
You are an expert biologist tasked with defining COMPREHENSIVE biological requirements for a {biological_context} model.

Project Goals: {project_goals}

Key Questions:
{chr(10).join(f"- {q}" for q in key_questions)}

Biological Context: {biological_context}

Additional Context: {additional_context if additional_context else "Please focus on the most relevant biological processes for this system."}

{essential_cells_text}

{key_interactions_text}

For each cell type, provide an initial_count that is biologically plausible and reflects the relative abundance of each cell type in the {biological_context} system. The numbers do not need to be exact, but should be within a reasonable ballpark and allow for relative comparison.

Enumerate ALL biologically plausible interactions between the cell types, including:
- Direct cell-cell, cell-molecule, and molecule-molecule interactions
- Receptor-ligand binding events (specify receptor names and their biological roles)
- Competitive, cooperative, and cascade effects
- Rare or minor interactions if biologically justified
- ESPECIALLY the priority interactions specified above

For each interaction and molecule, specify the receptors involved (e.g., cell surface receptors, entry receptors, cytokine/chemokine receptors). Include receptor names and their biological roles wherever possible.

In the JSON structure, for each target in an interaction, include a 'receptors' field listing the relevant receptors.

Generate a MAXIMALLY COMPREHENSIVE model specification that captures ALL relevant biological interactions for the {biological_context} system. Be exhaustive - include every biologically plausible interaction, even if minor.

Output as JSON with this structure:

{{
    "metadata": {{
        "model_scope": "spatial and temporal scope description",
        "key_assumptions": ["list all biological assumptions"],
        "excluded_elements": ["what we're explicitly not modeling"]
    }},
    
    "cell_types": [
        {{
            "name": "CellTypeName",
            "category": "epithelial/immune/structural/pathogen",
            "description": "Detailed biological description",
            "states": ["comprehensive list of all possible states"],
            "subtypes": ["if applicable, e.g., M1/M2 for macrophages"],
            "initial_count": 1000,  // Use plausible, relatively scaled numbers
            "spatial_distribution": "description of where found",
            "key_markers": ["CD markers or other identifiers"],
            "lifespan": "typical lifespan in hours/days"
        }}
    ],
    
    "interactions": [
        {{
            "id": "unique_identifier",
            "name": "Descriptive interaction name",
            "type": "infection/phagocytosis/cytotoxicity/activation/inhibition/secretion/adhesion/migration",
            "participants": {{
                "sources": [  // Can have multiple sources
                    {{
                        "cell_type": "CellType1",
                        "required_states": ["state1", "state2"],
                        "min_count": 1  // minimum number needed
                    }}
                ],
                "targets": [  // Can have multiple targets
                    {{
                        "cell_type": "CellType2",
                        "required_states": ["state3"],
                        "max_count": 10,  // maximum that can be affected
                        "receptors": ["Receptor1", "Receptor2"]  // List all relevant receptors for this target
                    }}
                ],
                "mediators": [  // Optional molecules that mediate interaction
                    "MoleculeName1", "MoleculeName2"
                ]
            }},
            "parameters": {{
                "rate": "value with units",
                "threshold": "if applicable",
                "duration": "time course",
                "probability": "if stochastic"
            }},
            "spatial_requirements": {{
                "contact_required": true,
                "interaction_radius": "in micrometers",
                "directionality": "if applicable"
            }},
            "temporal_dynamics": {{
                "initiation_time": "when it can start",
                "duration": "how long it lasts",
                "frequency": "how often it can occur"
            }},
            "outcomes": [
                {{
                    "type": "state_change/molecule_release/cell_death/proliferation",
                    "target": "what is affected",
                    "description": "what happens"
                }}
            ],
            "biological_rationale": "Why this interaction is important",
            "references": ["key papers if known"]
        }}
    ],
    
    "molecules": [
        {{
            "name": "MoleculeName",
            "type": "cytokine/chemokine/antibody/enzyme/damage_signal",
            "description": "Biological function",
            "producers": [
                {{
                    "cell_type": "CellType1",
                    "states": ["producing_state"],
                    "rate": "molecules/cell/hour"
                }}
            ],
            "targets": [
                {{
                    "cell_type": "CellType2",
                    "receptor": "ReceptorName",
                    "effect": "activation/inhibition/migration"
                }}
            ],
            "kinetics": {{
                "diffusion_coefficient": "um^2/s",
                "decay_rate": "per hour",
                "binding_affinity": "Kd in nM"
            }}
        }}
    ],
    
    "spatial_structures": [
        {{
            "name": "spatial_structure_name",
            "description": "Spatial organization description relevant to {biological_context}",
            "components": ["CellType1", "CellType2"],
            "properties": {{
                "key_property1": "value",
                "key_property2": true
            }}
        }}
    ]
}}

Be EXHAUSTIVE. Include:
1. ALL cell types involved in the {biological_context} system
2. ALL their possible states and transitions
3. EVERY interaction between cells (even indirect ones)
4. Complex multi-participant interactions
5. ALL secreted molecules and their effects
6. Spatial organization and constraints relevant to {biological_context}
7. Explicit receptor-ligand interactions and receptor details for each relevant interaction or molecule

Remember: It's better to be overly comprehensive than to miss important biology relevant to {biological_context}.
"""
        self._save_intermediate("step1_prompt", {"prompt": prompt})
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                response_format={"type": "json_object"}
            )
            result = json.loads(response.choices[0].message.content)
            return result
        except Exception as e:
            print(f"Error in step 1: {e}")
            return {"metadata": {}, "cell_types": [], "interactions": []}

    def _generate_molecules(self, prev, project_goals, key_questions, biological_context, key_molecules, additional_context):
        """Step 2: Generate molecules, using previous output as context."""
        
        # Build key molecules section
        key_molecules_text = ""
        if key_molecules:
            key_molecules_text = f"""

PRIORITY MOLECULES - You MUST include these specific molecules requested by the user:
{chr(10).join(f"- {molecule}" for molecule in key_molecules)}

For each priority molecule, provide detailed information about:
- Producers (which cells produce it and under what conditions)
- Targets (which cells respond to it and via which receptors)
- Kinetics (diffusion, decay, binding properties)
- Biological function and effects

Additionally, include other relevant signaling molecules for the {biological_context} system.
"""
        else:
            key_molecules_text = f"""

Focus on identifying ALL relevant signaling molecules for the {biological_context} system.
"""
        
        prompt = f"""
You are continuing a comprehensive biological requirements document for a {biological_context} model.

Here is the current requirements (metadata, cell_types, interactions):
{json.dumps(prev, indent=2)}

Project Goals: {project_goals}
Key Questions:\n{chr(10).join(f"- {q}" for q in key_questions)}
Biological Context: {biological_context}
Additional Context: {additional_context if additional_context else "Please focus on the most relevant molecules for this system."}

{key_molecules_text}

Now, generate ONLY the 'molecules' section as a JSON array, following the detailed field structure from the full model specification. Include ALL relevant signaling molecules, cytokines, chemokines, enzymes, and other molecular mediators important for the {biological_context} system. Output a JSON object with a single key 'molecules'.
"""
        self._save_intermediate("step2_prompt", {"prompt": prompt})
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                response_format={"type": "json_object"}
            )
            molecules = json.loads(response.choices[0].message.content).get("molecules", [])
            merged = dict(prev)
            merged["molecules"] = molecules
            return merged
        except Exception as e:
            print(f"Error in step 2: {e}")
            merged = dict(prev)
            merged["molecules"] = []
            return merged

    def _generate_spatial_structures(self, prev, project_goals, key_questions, biological_context, additional_context):
        """Step 3: Generate spatial_structures, using all previous output as context."""
        prompt = f"""
You are finalizing a comprehensive biological requirements document for a {biological_context} model.

Here is the current requirements (metadata, cell_types, interactions, molecules):
{json.dumps(prev, indent=2)}

Project Goals: {project_goals}
Key Questions:\n{chr(10).join(f"- {q}" for q in key_questions)}
Biological Context: {biological_context}
Additional Context: {additional_context if additional_context else "Please focus on the most relevant spatial structures for this system."}

Now, generate ONLY the 'spatial_structures' section as a JSON array, following the detailed field structure from the full model specification. Include all relevant anatomical structures, tissue organization, and spatial constraints important for the {biological_context} system. Output a JSON object with a single key 'spatial_structures'.
"""
        self._save_intermediate("step3_prompt", {"prompt": prompt})
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                response_format={"type": "json_object"}
            )
            spatial_structures = json.loads(response.choices[0].message.content).get("spatial_structures", [])
            merged = dict(prev)
            merged["spatial_structures"] = spatial_structures
            return merged
        except Exception as e:
            print(f"Error in step 3: {e}")
            merged = dict(prev)
            merged["spatial_structures"] = []
            return merged

    def _enhance_with_complex_interactions(self, requirements: Dict, biological_context: str) -> Dict:
        """Enhance requirements by identifying complex multi-participant interactions"""
        
        prompt = f"""
Given these biological requirements for a {biological_context} model:
{json.dumps(requirements, indent=2)}

Identify ADDITIONAL complex interactions that involve:
1. Multiple source cells cooperating
2. Multiple targets affected simultaneously
3. Cascade effects
4. Competitive interactions (e.g., multiple cells competing for same resource)
5. Synergistic effects
6. Context-specific interactions relevant to {biological_context}

Add these as new interactions with multiple sources/targets as appropriate.

Return the enhanced requirements as a JSON object with the same structure, preserving all original content and adding new complex interactions.
"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
                response_format={"type": "json_object"}
            )
            
            enhanced = json.loads(response.choices[0].message.content)
            self._save_intermediate("enhanced_requirements", enhanced)
            
            return enhanced
            
        except Exception as e:
            print(f"Could not enhance with complex interactions: {e}")
            return requirements

    def _validate_requirements(self, requirements: Dict, biological_context: str, essential_cell_types: List[str] = None, key_molecules: List[str] = None, key_interactions: List[str] = None) -> Dict:
        """Validate the biological completeness and consistency"""
        
        validation_results = {
            "is_complete": True,
            "warnings": [],
            "suggestions": [],
            "statistics": {},
            "biological_context": biological_context
        }
        
        # Filter out non-dict items from interactions
        interactions = [i for i in requirements.get("interactions", []) if isinstance(i, dict)]
        
        # Count statistics
        validation_results["statistics"] = {
            "total_cell_types": len(requirements.get("cell_types", [])),
            "total_interactions": len(interactions),
            "total_molecules": len(requirements.get("molecules", [])),
            "multi_source_interactions": sum(1 for i in interactions 
                                           if len(i.get("participants", {}).get("sources", [])) > 1),
            "multi_target_interactions": sum(1 for i in interactions 
                                           if len(i.get("participants", {}).get("targets", [])) > 1)
        }
        
        # Check for essential components if specified
        if essential_cell_types:
            cell_names = {ct["name"] for ct in requirements.get("cell_types", [])}
            missing_essential = [c for c in essential_cell_types if not any(c in name for name in cell_names)]
            if missing_essential:
                validation_results["warnings"].append(
                    f"Missing user-specified essential cells: {', '.join(missing_essential)}"
                )
                validation_results["is_complete"] = False
        
        # Check for key molecules if specified
        if key_molecules:
            molecule_names = {mol["name"] for mol in requirements.get("molecules", []) if isinstance(mol, dict)}
            missing_molecules = [m for m in key_molecules if not any(m.lower() in name.lower() for name in molecule_names)]
            if missing_molecules:
                validation_results["warnings"].append(
                    f"Missing user-specified key molecules: {', '.join(missing_molecules)}"
                )
                validation_results["is_complete"] = False
        
        # Check for key interactions if specified
        if key_interactions:
            interaction_names = {i["name"] for i in interactions if "name" in i}
            interaction_descriptions = {i.get("biological_rationale", "") for i in interactions}
            all_interaction_text = " ".join(interaction_names | interaction_descriptions).lower()
            
            missing_interactions = []
            for key_int in key_interactions:
                if not any(key_int.lower() in text for text in [all_interaction_text]):
                    missing_interactions.append(key_int)
            
            if missing_interactions:
                validation_results["warnings"].append(
                    f"Potentially missing user-specified key interactions: {', '.join(missing_interactions)}"
                )
        
        # Check for basic interaction types (context-agnostic)
        interaction_types = {i["type"] for i in interactions}
        basic_interactions = ["activation", "inhibition", "secretion"]
        
        missing_interactions = set(basic_interactions) - interaction_types
        if missing_interactions:
            validation_results["warnings"].append(
                f"Missing basic interaction types: {', '.join(missing_interactions)}"
            )
        
        # Suggestions for improvement
        if validation_results["statistics"]["multi_source_interactions"] < 5:
            validation_results["suggestions"].append(
                "Consider adding more multi-source interactions (e.g., T cell help)"
            )
        
        if validation_results["statistics"]["multi_target_interactions"] < 3:
            validation_results["suggestions"].append(
                "Consider adding more multi-target interactions (e.g., cytokine affecting multiple cells)"
            )
        
        self._save_intermediate("validation_results", validation_results)
        
        return validation_results
    
    def _save_intermediate(self, name: str, data: Union[Dict, List], format: str = "json"):
        """Save intermediate results"""
        timestamp = datetime.now().strftime("%H%M%S")
        filename = f"{name}_{timestamp}"
        
        if format == "json":
            filepath = self.intermediate_dir / f"{filename}.json"
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        elif format == "yaml":
            filepath = self.intermediate_dir / f"{filename}.yaml"
            with open(filepath, 'w') as f:
                yaml.dump(data, f, default_flow_style=False)
        
        print(f"Saved intermediate: {filepath.name}")
    
    def _save_final_output(self, requirements: Dict, iteration: int):
        """Save final requirements in multiple formats"""
        
        # JSON format
        json_path = self.output_dir / f"requirements_v{iteration}.json"
        with open(json_path, 'w') as f:
            json.dump(requirements, f, indent=2, default=str)
        
        # YAML format for readability
        yaml_path = self.output_dir / f"requirements_v{iteration}.yaml"
        with open(yaml_path, 'w') as f:
            yaml.dump(requirements, f, default_flow_style=False, sort_keys=False)
        
        # Summary report
        summary = {
            "timestamp": datetime.now().isoformat(),
            "iteration": iteration,
            "statistics": requirements.get("validation", {}).get("statistics", {}),
            "output_files": {
                "json": str(json_path),
                "yaml": str(yaml_path)
            }
        }
        
        summary_path = self.output_dir / "summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nFinal outputs saved:")
        print(f"  JSON: {json_path}")
        print(f"  YAML: {yaml_path}")
        print(f"  Summary: {summary_path}")

    def _get_fallback_requirements(self, biological_context: str = "generic tissue") -> Dict:
        """Minimal fallback requirements if LLM fails"""
        return {
            "metadata": {
                "model_scope": f"Fallback minimal model for {biological_context}",
                "key_assumptions": ["LLM generation failed - using minimal defaults"],
                "biological_context": biological_context
            },
            "cell_types": [
                {
                    "name": "CellType1",
                    "category": "primary",
                    "states": ["healthy", "activated", "damaged"],
                    "initial_count": 1000
                },
                {
                    "name": "CellType2", 
                    "category": "secondary",
                    "states": ["resting", "active"],
                    "initial_count": 100
                }
            ],
            "interactions": [
                {
                    "id": "basic_interaction",
                    "name": "Basic cell interaction",
                    "type": "activation",
                    "participants": {
                        "sources": [{"cell_type": "CellType1", "required_states": ["activated"]}],
                        "targets": [{"cell_type": "CellType2", "required_states": ["resting"]}]
                    }
                }
            ],
            "molecules": [],
            "spatial_structures": []
        }