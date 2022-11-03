# SPDX-FileCopyrightText: 2022 Koen van Greevenbroek & Aleksander Grochowicz
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Optimising a PyPSA network with respect to its total system costs."""

import logging
from pathlib import Path

import pandas as pd
import pypsa
from _helpers import configure_logging
from pypsa.linopf import ilopf, network_lopf
from utilities import get_basis_values

if __name__ == "__main__":
    # Set up logging so that everything is written to the right log file.
    configure_logging(snakemake)

    # Load the network and solving options.
    n = pypsa.Network(snakemake.input.network)
    solving_options = snakemake.config["pypsa-eur"]["solving"]["options"]
    solver_options = snakemake.config["pypsa-eur"]["solving"]["solver"].copy()
    solver_name = solver_options.pop("name")
    tmpdir = snakemake.config["pypsa-eur"]["solving"].get("tmpdir", None)
    if tmpdir is not None:
        Path(tmpdir).mkdir(parents=True, exist_ok=True)

    # Solve the network for the cost optimum and then get its
    # coordinates in the basis.
    logging.info("Compute initial, optimal solution.")
    if solving_options.get("skip_iterations", False):
        status, _ = network_lopf(
            n,
            solver_name=solver_name,
            solver_options=solver_options,
            solver_dir=tmpdir,
        )
    else:
        ilopf(
            n,
            solver_name=solver_name,
            solver_options=solver_options,
            solver_dir=tmpdir,
            track_iterations=solving_options.get("track_iterations", False),
            min_iterations=solving_options.get("min_iterations", 1),
            max_iterations=solving_options.get("max_iterations", 6),
        )
        # `ilopf` doesn't give us any optimisation status or
        # termination condition, and simply crashes if any
        # optimisation fails.
        status = "ok"

    # Check if the optimisation succeeded; if not we don't output
    # anything in order to make snakemake fail. Not checking for this
    # would result in an invalid (non-optimal) network being output.
    if status == "ok":
        # Write the result to the given output files. Save the objective
        # value for further processing.
        n.export_to_netcdf(snakemake.output.optimum)
        opt_point = get_basis_values(n, snakemake.config["projection"])
        pd.DataFrame(opt_point, index=[0]).to_csv(snakemake.output.optimal_point)
        with open(snakemake.output.obj, "w") as f:
            f.write(str(n.objective))
