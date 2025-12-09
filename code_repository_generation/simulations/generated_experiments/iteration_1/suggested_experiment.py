import os
import sys
import pickle
import argparse
from datetime import datetime
from vivarium.library.units import units, remove_units
from tumor_tcell.experiments.main import tumor_tcell_abm
from tumor_tcell.library.individual_analysis import individual_analysis

def main():
    ###########
    # ARGPARSE
    ###########
    parser = argparse.ArgumentParser(
        description="Parameter sweep for T cell number and PD1-negativity in tumor-T cell ABM."
    )
    parser.add_argument(
        "--output_base_dir",
        type=str,
        required=True,
        help="Base directory for simulation outputs.",
    )
    args = parser.parse_args()
    output_base_dir = args.output_base_dir

    ##############
    # PARAMETERS TO SWEEP
    # - n_tcells: 6, 12, 24 (T cell number)
    # - tcells_total_PD1n: 0.25, 0.5, 1.0 fraction of T cells that are PD1-negative
    # (PD1-negative = highly cytotoxic, checkpoint-inhibitor insensitive)
    ##############
    n_tcells_list = [6, 12, 24]
    pd1_neg_frac_list = [0.25, 0.5, 1.0]

    # Other experiment settings (fixed)
    n_tumors = 120
    tumors_state_PDL1n = 0.5
    total_time = 60000
    TIMESTEP = 60
    NBINS = [120, 120]
    FULL_BOUNDS = [1200 * units.um, 1200 * units.um]
    halt_threshold = 5000

    # Get current date for experiment name
    date_str = datetime.now().strftime("%Y%m%d")

    # For each (n_tcells, pd1_neg_frac) combination:
    for n_tcells in n_tcells_list:
        for pd1_neg_frac in pd1_neg_frac_list:

            # Compute integer number of PD1- T cells; must not exceed n_tcells
            tcells_total_PD1n = int(round(n_tcells * pd1_neg_frac))
            tcells_total_PD1n = min(tcells_total_PD1n, n_tcells)

            # Format experiment name and id
            experiment_name = (f"{date_str}_abm_sweep_nT{n_tcells}_PD1nFrac{pd1_neg_frac:.2f}")
            exp_id = f"60000s_120tumors_{n_tcells}tcells_{tcells_total_PD1n}PD1n"

            # Construct unique directories: output_base_dir/experiment_name/exp_id/
            exper_dir = os.path.join(output_base_dir, experiment_name)
            outdir = os.path.join(exper_dir, exp_id)

            print(f"\n========== Starting simulation: n_tcells={n_tcells}, PD1-neg frac={pd1_neg_frac:.2f} "
                  f"({tcells_total_PD1n} PD1- of {n_tcells}) ==========")
            print(f"Results will be saved to: {outdir}")

            # Make directories
            os.makedirs(outdir, exist_ok=True)

            # Run simulation
            data = tumor_tcell_abm(
                halt_threshold=halt_threshold,
                n_tumors=n_tumors,
                tumors_state_PDL1n=tumors_state_PDL1n,
                n_tcells=n_tcells,
                tcells_total_PD1n=tcells_total_PD1n,
                tcells_state_PD1n=None,
                lymph_nodes=False,
                dendritic_state_active=0.5,
                n_dendritic=0,
                n_tcells_lymph_node=0,
                bounds=FULL_BOUNDS,
                n_bins=NBINS,
                depth=15,
                field_molecules=["IFNg"],
                tumors=None,
                tcells=None,
                dendritic_cells=None,
                total_time=total_time,
                time_step=TIMESTEP,
                sim_step=100 * TIMESTEP,
                emit_step=10 * TIMESTEP,
                emitter="timeseries",
                parallel=False,
                tumors_distance=260 * units.um,
                tcells_distance=220 * units.um,
                tumors_excluded_distance=None,
                tcells_excluded_distance=None,
                tumors_center=None,
                tcell_center=None,
            )

            # Remove units for pickling
            data_nounits = remove_units(data)
            bounds = data_nounits[0.0]["tumor_environment"]["dimensions"]["bounds"]

            # Save raw simulation data via pickle
            data_export_path = os.path.join(outdir, "data_export.pkl")
            with open(data_export_path, "wb") as data_export_file:
                pickle.dump(data_nounits, data_export_file)

            # Call analysis; analysis_dir is the experiment directory, experiment_id is exp_id,
            # outputs (figs/csvs) saved within exper_dir (so subdirs per sweep value)
            individual_analysis(
                analysis_dir=exper_dir,
                experiment_id="/" + exp_id,  # Note: original code prepends '/'
                bounds=bounds,
                tcells=True,
                lymph_nodes=False
            )

            print(f"+++ Finished simulation: n_tcells={n_tcells}, PD1-neg frac={pd1_neg_frac:.2f}. "
                  f"Data and analysis in: {outdir}\n")

    print("All parameter-sweep simulations and analyses complete.")

if __name__ == "__main__":
    main()