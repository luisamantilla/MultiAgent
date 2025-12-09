import os
import pickle
import argparse  # Added argparse
from vivarium.library.units import units, remove_units
from tumor_tcell.experiments.main import tumor_tcell_abm
from tumor_tcell.library.individual_analysis import individual_analysis
from datetime import datetime

# --- Argument Parsing ---
parser = argparse.ArgumentParser(
    description="Run ABM simulation with specified output directory."
)
parser.add_argument(
    "--output_base_dir",
    type=str,
    default="./data_modeling/ABM_outputs",  # Default if not provided
    help="Base directory for all simulation outputs.",
)
args = parser.parse_args()

# --- End Argument Parsing ---

# Get current date
date_str = datetime.now().strftime("%Y%m%d")

# Descriptive experiment name
experiment_name = f"{date_str}_abm_simulation"

# Encode key parameters in the experiment ID
exp_id = f"/60000s_120tumors_12tcells"

# Set up directories and experiment names
all_dir = args.output_base_dir  # Use the provided or default base directory

# Make a directory to save simulation data
exper_dir = os.path.join(all_dir, experiment_name)  # Use os.path.join for robustness
if not os.path.exists(exper_dir):
    os.makedirs(exper_dir)

outdir = os.path.join(exper_dir, exp_id.lstrip("/"))  # Use os.path.join and strip leading / from exp_id

######################################
##### SETTINGS FOR FULL EXPERIMENT #####
FULL_BOUNDS = [1200 * units.um, 1200 * units.um]  # size of area for simulation
TIMESTEP = 60  # 60 second intervals
NBINS = [120, 120]  # how to segment the area for diffusion, etc.

data = tumor_tcell_abm(
    halt_threshold=5000,  # stop simulation at this number of cells/agents
    # Cell set up
    n_tumors=120,  # number of tumors in simulation
    tumors_state_PDL1n=0.5,  # percent of tumors that start PDL1-
    n_tcells=12,  # number of T cells in simulation
    tcells_total_PD1n=9,  # number of T cells out of total that are PD1-
    tcells_state_PD1n=None,  # Set absolute amount above (could set percent but increase variation with replicates)
    # Lymph node options
    lymph_nodes=False,  # whether or not you want to use LN process
    dendritic_state_active=0.5,  # Assume half are in LN (active means they go to LN)
    n_dendritic=0,  # number of dendritic cells in tumor
    n_tcells_lymph_node=0,  # number of T cells that are in the LN
    # field
    bounds=FULL_BOUNDS,  # size of area for simulation
    n_bins=NBINS,  # how to segment the area for diffusion, etc.
    depth=15,  # um for calculating volumes for diffusion
    field_molecules=["IFNg"],  # molecules that we use, if using LN then include 'tumor_debris'
    # Historic not needed
    tumors=None,  # can import CODEX data
    tcells=None,  # can import CODEX data
    dendritic_cells=None,  # can import CODEX data
    # time
    total_time=60000,  # total time of simulation in seconds
    time_step=TIMESTEP,  # timestep intervals for updating simulation
    sim_step=100 * TIMESTEP,  # simulation increments at which halt_threshold is checked
    emit_step=10 * TIMESTEP,  # simulation increments at which data is emitted for analysis
    emitter="timeseries",  # how to save the data
    parallel=False,  # whether you could do parallel processing
    # Placement of cells
    tumors_distance=260 * units.um,  # sqrt(n_tumors)*15(diameter)/2 can be used to determine based on number of tumors
    tcells_distance=220 * units.um,  # in (less than tumors_distance) or out (None) of the tumor
    tumors_excluded_distance=None,  # if tumors need to be restricted radially
    tcells_excluded_distance=None,  # for creating a ring around tumor to say they are not enriched within tumors
    tumors_center=None,  #
    tcell_center=None,
)

############ Set parameters for plotting/saving the data #############
data = remove_units(data)
bounds = data[0.0]["tumor_environment"]["dimensions"]["bounds"]

if not os.path.exists(outdir):
    os.makedirs(outdir)
data_export_path = os.path.join(outdir, "data_export.pkl")  # Use os.path.join
data_export = open(data_export_path, "wb")
pickle.dump(data, data_export)
data_export.close()

individual_analysis(
    analysis_dir=exper_dir, experiment_id=exp_id, bounds=bounds, tcells=True, lymph_nodes=False
)
