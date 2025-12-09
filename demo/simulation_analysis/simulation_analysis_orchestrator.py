import pandas as pd
from .agents.simulation_evaluation_agent import simulation_evaluation_agent
from .fetch_rules import fetch_rules

from .queries import calculate_cell_count, calculate_cell_division, calculate_pd1_plus_average_movement, calculate_pd1_plus_ifng, calculate_pdl1_state_count

df = pd.read_csv("/Users/kevl0215/Documents/MultiAgent/demo/simulation_analysis/tcell_plot.csv")
tumor_df = pd.read_csv("/Users/kevl0215/Documents/MultiAgent/demo/simulation_analysis/tumor_plot.csv")

pd1_plus_average_movement = calculate_pd1_plus_average_movement(df)
pd1_plus_ifng = calculate_pd1_plus_ifng(df)
simulation_data = {
    "average_ifng_by_time_pd1_plus": pd1_plus_ifng,
    "pd1_plus_average_movement": pd1_plus_average_movement
}

# cell_count = calculate_cell_count(df)
# state_ratio = calculate_pdl1_state_count(df)
# cell_division_count = calculate_cell_division(df)

# cell_count_tumor= calculate_cell_count(tumor_df)
# state_ratio_tumor = calculate_pdl1_state_count(tumor_df)
# cell_division_count_tumor = calculate_cell_division(tumor_df)

# simulation_data = {
#     "cell_count": cell_count,
#     "state_ratio": state_ratio,
#     "cell_division_count": cell_division_count,
#     "cell_count_tumor": cell_count_tumor,
#     "state_ratio_tumor": state_ratio_tumor,
#     "cell_division_count_tumor": cell_division_count_tumor
# }

system_environment = "T-cells are interacting with tumor cells."
rules = fetch_rules(["T-Cell", "Tumor Cell"])

print(simulation_evaluation_agent(simulation_data, system_environment, rules))
