#!/usr/bin/env python3
"""
add_ground_truth_simple.py - Add ground truth column to Parameter_Extraction_Results.csv
"""

import pandas as pd
import numpy as np

def create_ground_truth_mapping():
    """Create ground truth mapping."""
    return {
        # T-Cell Parameters
        ("time_step", "t-cell"): 60,
        ("t_cell_diameter", "t-cell"): 7.5,
        ("diameter", "t-cell"): 7.5,
        ("t_cell_mass", "t-cell"): 2,
        ("mass", "t-cell"): 2,
        ("initial_pd1_negative_prob", "t-cell"): 0.8,
        ("initial_PD1n", "t-cell"): 0.8,
        ("refractory_threshold", "t-cell"): 3,
        ("refractory_count_threshold", "t-cell"): 3,
        ("activation_duration", "t-cell"): 21600,
        ("activation_time", "t-cell"): 21600,
        ("refractory_duration", "t-cell"): 43200,
        ("activation_refractory_time", "t-cell"): 43200,
        ("tcr_downregulated", "t-cell"): 0,
        ("TCR_downregulated", "t-cell"): 0,
        ("tcr_upregulated", "t-cell"): 50000,
        ("TCR_upregulated", "t-cell"): 50000,
        ("pdl1_death_threshold", "t-cell"): 10000,
        ("PDL1_critical_number", "t-cell"): 10000,
        ("death_prob_pd1_pos", "t-cell"): 0.35,
        ("death_prob_PD1_pos", "t-cell"): 0.35,
        ("death_PD1p_14hr", "t-cell"): 0.35,
        ("death_prob_pd1_neg", "t-cell"): 0.1,
        ("death_prob_PD1_neg", "t-cell"): 0.1,
        ("death_PD1n_14hr", "t-cell"): 0.1,
        ("death_prob_pd1_pos_near_pdl1", "t-cell"): 0.475,
        ("death_prob_PD1_pos_near_PDL1_pos", "t-cell"): 0.475,
        ("death_PD1p_next_to_PDL1p_14hr", "t-cell"): 0.475,
        ("ifng_prod_pd1_neg", "t-cell"): 4.5,
        ("ifng_production_PD1_neg", "t-cell"): 4.5,
        ("PD1n_IFNg_production", "t-cell"): 4.5,
        ("ifng_prod_pd1_pos", "t-cell"): 0.45,
        ("ifng_production_PD1_pos", "t-cell"): 0.45,
        ("PD1p_IFNg_production", "t-cell"): 0.45,
        ("pd1_equilibrium_expression", "t-cell"): 50000,
        ("PD1p_PD1_equilibrium", "t-cell"): 50000,
        ("recognition_threshold", "t-cell"): 10000,
        ("ligand_threshold", "t-cell"): 10000,
        ("growth_prob_pd1_neg", "t-cell"): 0.9,
        ("PD1n_growth_28hr", "t-cell"): 0.9,
        ("growth_prob_pd1_pos", "t-cell"): 0.2,
        ("PD1p_growth_28hr", "t-cell"): 0.2,
        ("division_threshold", "t-cell"): 5,
        ("PD1n_divide_threshold", "t-cell"): 5,
        ("delay_growth_time", "t-cell"): 32400,
        ("LymphNode_delay_growth", "t-cell"): 32400,
        ("migration_velocity_pd1_neg", "t-cell"): 10.0,
        ("PD1n_migration", "t-cell"): 10.0,
        ("dwell_velocity_mhci_pos", "t-cell"): 0.0,
        ("migration_MHCIp_tumor_dwell_velocity", "t-cell"): 0.0,
        ("dwell_time_pd1_neg", "t-cell"): 1500,
        ("PD1n_migration_MHCIp_tumor_dwell_time", "t-cell"): 1500,
        ("migration_velocity_pd1_pos", "t-cell"): 5.0,
        ("PD1p_migration", "t-cell"): 5.0,
        ("dwell_time_pd1_pos", "t-cell"): 600,
        ("PD1p_migration_MHCIp_tumor_dwell_time", "t-cell"): 600,
        ("migration_refractory_pd1_neg", "t-cell"): 2100,
        ("PD1n_migration_refractory_time", "t-cell"): 2100,
        ("migration_refractory_pd1_pos", "t-cell"): 1200,
        ("PD1p_migration_refractory_time", "t-cell"): 1200,
        ("cytotoxic_prod_rate", "t-cell"): 0.667,
        ("cytotoxic_packet_production", "t-cell"): 0.667,
        ("cytotoxic_capacity_pd1_neg", "t-cell"): 10000,
        ("PD1n_cytotoxic_packets_max", "t-cell"): 10000,
        ("cytotoxic_capacity_pd1_pos", "t-cell"): 1000,
        ("PD1p_cytotoxic_packets_max", "t-cell"): 1000,
        ("cytotoxic_fold_reduction_mhci_neg", "t-cell"): 400,
        ("MHCIn_reduction_production", "t-cell"): 400,
        ("cytotoxic_transfer_rate", "t-cell"): 400,
        
        # Tumor Cell Parameters
        ("time_step", "tumor"): 60,
        ("tumor_diameter", "tumor"): 15,
        ("diameter", "tumor"): 15,
        ("tumor_mass", "tumor"): 8,
        ("mass", "tumor"): 8,
        ("initial_pdl1_negative_prob", "tumor"): 0.9,
        ("initial_PDL1n", "tumor"): 0.9,
        ("apoptosis_prob", "tumor"): 0.5,
        ("death_apoptosis", "tumor"): 0.5,
        ("cytotoxic_kill_threshold", "tumor"): 12800,
        ("cytotoxic_packet_threshold", "tumor"): 12800,
        ("growth_prob_pdl1_neg", "tumor"): 0.6,
        ("PDL1n_growth", "tumor"): 0.6,
        ("ifng_internalization_max", "tumor"): 0.517,
        ("Max_IFNg_internalization", "tumor"): 0.517,
        ("ifng_internalization_fold_reduction", "tumor"): 2,
        ("reduction_IFNg_internalization", "tumor"): 2,
        ("ifng_transition_threshold", "tumor"): 15000,
        ("IFNg_threshold", "tumor"): 15000,
        ("pdl1_equilibrium_expression", "tumor"): 50000,
        ("PDL1p_PDL1_equilibrium", "tumor"): 50000,
        ("mhci_equilibrium_expression", "tumor"): 50000,
        ("PDL1p_MHCI_equilibrium", "tumor"): 50000,
        ("tumor_debris_release", "tumor"): 1.4e15,
        ("tumor_debris_amount", "tumor"): 1.4e15,
        ("ifng_molecular_weight", "tumor"): 17000,
        ("IFNg_MW", "tumor"): 17000,
        
        # Dendritic Cell Parameters
        ("time_step", "dendritic"): 60,
        ("dendritic_diameter", "dendritic"): 10,
        ("diameter", "dendritic"): 10,
        ("dendritic_mass", "dendritic"): 2,
        ("mass", "dendritic"): 2,
        ("migration_velocity", "dendritic"): 3.0,
        ("velocity", "dendritic"): 3.0,
        ("apoptosis_prob", "dendritic"): 0.5,
        ("death_apoptosis", "dendritic"): 0.5,
        ("death_time", "dendritic"): 345600,
        ("division_prob", "dendritic"): 0.6,
        ("divide_prob", "dendritic"): 0.6,
        ("division_time", "dendritic"): 432000,
        ("divide_time", "dendritic"): 432000,
        ("activation_debris_threshold", "dendritic"): 415000,
        ("internal_tumor_debris_threshold", "dendritic"): 415000,
        ("pdl1_equilibrium_expression", "dendritic"): 50000,
        ("PDL1p_PDL1_equilibrium", "dendritic"): 50000,
        ("mhci_equilibrium_expression", "dendritic"): 50000,
        ("PDL1p_MHCI_equilibrium", "dendritic"): 50000,
        ("uptake_rate", "dendritic"): 5.0,
        ("tumor_debris_uptake", "dendritic"): 5.0,
        ("debris_molecular_weight", "dendritic"): 29000,
        ("tumor_debris_MW", "dendritic"): 29000,
        
        # Lymph Node Parameters
        ("time_step", "lymph_node"): 60,
        ("dc_find_tcell_time", "lymph_node"): 2.0,
        ("time_dendritic_finds_tcell", "lymph_node"): 2.0,
        ("tcell_population", "lymph_node"): 3,
        ("n_tcells_in_lymph_node", "lymph_node"): 3,
        ("dc_tcell_interaction_prob", "lymph_node"): 0.95,
        ("tcell_find_dendritic_time", "lymph_node"): 0.95,
        ("dc_transit_time", "lymph_node"): 28800,
        ("expected_dendritic_transit_time", "lymph_node"): 28800,
        ("tcell_transit_time", "lymph_node"): 3600,
        ("expected_tcell_transit_time", "lymph_node"): 3600,
        ("tcell_dendritic_interaction_duration", "lymph_node"): 28800,
        ("expected_interaction_duration", "lymph_node"): 28800,
        ("migration_delay", "lymph_node"): 43200,
        ("expected_delay_before_migration", "lymph_node"): 43200
    }

def add_ground_truth_column():
    """Add ground truth column to the CSV."""
    
    # Read CSV
    df = pd.read_csv("Parameter_Extraction_Results.csv")
    
    # Create ground truth mapping
    gt_mapping = create_ground_truth_mapping()
    
    # Add ground truth column
    df['ground_truth_value'] = df.apply(
        lambda row: gt_mapping.get((row['parameter_name'], row['cell_type']), np.nan), 
        axis=1
    )
    
    # Save updated CSV
    df.to_csv("Parameter_Extraction_Results_with_ground_truth.csv", index=False)
    
    print(f"Added ground truth values to {len(df)} rows")
    print(f"Matched {df['ground_truth_value'].notna().sum()} parameters")
    print("Saved as: Parameter_Extraction_Results_with_ground_truth.csv")

if __name__ == "__main__":
    add_ground_truth_column()