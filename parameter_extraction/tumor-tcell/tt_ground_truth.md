# T Cell Process Parameters Tables

## Default Parameters Table

| cell type | cell behavior | parameter type | parameter name | parameter numerical value | unit | reference |
|-----------|---------------|----------------|----------------|---------------------------|------|-----------|
| t-cell | cell properties | physical properties | time_step | 60 | seconds | |
| t-cell | cell properties | physical properties | diameter | 7.5 | μm | |
| t-cell | cell properties | physical properties | mass | 2 | ng | |
| t-cell | cell properties | physical properties | initial_PD1n | 0.8 | probability | |
| t-cell | cell activation | activation timing | refractory_count_threshold | 3 | count | (Zhao 2020) |
| t-cell | cell activation | activation timing | activation_time | 21600 | seconds | (Salerno 2017, Gallegos 2016) |
| t-cell | cell activation | activation timing | activation_refractory_time | 43200 | seconds | (Salerno 2017, Gallegos 2016) |
| t-cell | cell activation | activation timing | TCR_downregulated | 0 | molecules/cell | |
| t-cell | cell activation | activation timing | TCR_upregulated | 5.00E+04 | molecules/cell | |
| t-cell | cell death | death rates | PDL1_critical_number | 1.00E+04 | molecules/cell | |
| t-cell | cell death | death rates | death_PD1p_14hr | 0.35 | probability/14hr | (Petrovas 2007) |
| t-cell | cell death | death rates | death_PD1n_14hr | 0.1 | probability/14hr | (Petrovas 2007) |
| t-cell | cell death | death rates | death_PD1p_next_to_PDL1p_14hr | 0.475 | probability/14hr | (Dong 2002, Tang 2015) |
| t-cell | cell production | production rates | PD1n_IFNg_production | 4.5 | molecules/cell/second | (Bouchnita 2017) |
| t-cell | cell production | production rates | PD1p_IFNg_production | 0.45 | molecules/cell/second | (Zelinskyy 2005) |
| t-cell | cell production | production rates | PD1p_PD1_equilibrium | 5.00E+04 | molecules/cell | |
| t-cell | cell recognition | recognition thresholds | ligand_threshold | 1.00E+04 | molecules/cell | |
| t-cell | cell division | division rates | PD1n_growth_28hr | 0.9 | probability/28hr | (Petrovas 2007, Vodnala 2019) |
| t-cell | cell division | division rates | PD1p_growth_28hr | 0.2 | probability/28hr | (Petrovas 2007, Vodnala 2019) |
| t-cell | cell division | division rates | PD1n_divide_threshold | 5 | count | (Zhao 2020) |
| t-cell | cell division | division rates | LymphNode_delay_growth | 32400 | seconds | (Mempel 2004, Bousso 2008) |
| t-cell | cell migration | migration rates | PD1n_migration | 10.0 | μm/minute | (Boissonnas 2007) |
| t-cell | cell migration | migration rates | migration_MHCIp_tumor_dwell_velocity | 0.0 | μm/minute | (Thibaut 2020) |
| t-cell | cell migration | migration rates | PD1n_migration_MHCIp_tumor_dwell_time | 1500 | seconds | (Thibaut 2020) |
| t-cell | cell migration | migration rates | PD1p_migration | 5.0 | μm/minute | (Boissonnas 2007) |
| t-cell | cell migration | migration rates | PD1p_migration_MHCIp_tumor_dwell_time | 600 | seconds | (Thibaut 2020) |
| t-cell | cell migration | migration rates | PD1n_migration_refractory_time | 2100 | seconds | (Thibaut 2020) |
| t-cell | cell migration | migration rates | PD1p_migration_refractory_time | 1200 | seconds | (Thibaut 2020) |
| t-cell | cell killing | killing rates | cytotoxic_packet_production | 0.667 | packets/second | (Betts 2004, Zhang 2006) |
| t-cell | cell killing | killing rates | PD1n_cytotoxic_packets_max | 1.00E+04 | packets | |
| t-cell | cell killing | killing rates | PD1p_cytotoxic_packets_max | 1.00E+03 | packets | (Zelinskyy 2005) |
| t-cell | cell killing | killing rates | MHCIn_reduction_production | 400 | fold reduction | (Bohm 1998, Merritt 2003) |
| t-cell | cell killing | killing rates | cytotoxic_transfer_rate | 400 | packets/minute | (Betts 2004, Zhang 2006) |

## Update Parameters Table

| cell type | cell behavior | parameter type | parameter name | update type | update value | condition |
|-----------|---------------|----------------|----------------|-------------|--------------|-----------|
| t-cell | cell activation | TCR expression | TCR | set | 0 | when TCR_timer > activation_time |
| t-cell | cell activation | TCR expression | TCR | set | 50000 | when TCR_timer > activation_refractory_time |
| t-cell | cell state | state transition | cell_state | set | 'PD1p' | when refractory_count > threshold OR PD1n_divide_count > threshold |
| t-cell | cell production | PD1 expression | PD1 | set | 50000 | when cell_state == 'PD1p' |
| t-cell | cell migration | velocity | velocity | set | 0.0 | when dwelling at tumor with MHCI |
| t-cell | cell migration | velocity | velocity | set | 10.0 or 5.0 | when moving (PD1n or PD1p respectively) |
| t-cell | cell production | cytotoxic packets | total_cytotoxic_packets | reset | 0 | after refractory period |
| t-cell | cell timing | velocity timer | velocity_timer | reset | 0 | after refractory period |
| t-cell | cell timing | TCR timer | TCR_timer | decrement | -43200 | after refractory period |


