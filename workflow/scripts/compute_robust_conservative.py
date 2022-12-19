# SPDX-FileCopyrightText: 2022 Koen van Greevenbroek & Aleksander Grochowicz
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Compute a robust solution based on the most expensive year considered.

This script takes the robust investment costs from the intersection of
near-optimal spaces and uses them as constraints on the network with
the highest optimal costs. Then the 1-year-long network is solved and
this is a heuristic solution for all the years with the interpretation
that one should brace for the worst (=most expensive) year.

"""

import pandas as pd
import pypsa
from pypsa.components import component_attrs, components
from _helpers import configure_logging
from compute_robust_exact import compute_robust
from utilities import apply_caps, set_nom_to_opt, override_component_attrs
import os.path

if __name__ == "__main__":
    # Set up logging so that everything is written to the right log file.
    configure_logging(snakemake)

    # Load the networks and solving options.
    overrides = override_component_attrs(snakemake.input.overrides)
    n = pypsa.Network(snakemake.input.network, override_component_attrs=overrides)
    n_exp = pypsa.Network(snakemake.input.most_expensive_network, override_component_attrs=overrides)
    
    n_exp.config = snakemake.config["pypsa-eur"]

    # Load the projected centre (robust) solution coordinates.
    coordinates = pd.read_csv(snakemake.input.centre, index_col=0)

    # Optimise `n_exp` with capacities from `coordinates` to get a
    # "conservative" robust solution.
    compute_robust(n_exp, basis=snakemake.config["projection"], coordinates=coordinates)

    # Set the nominal capacities (*_nom) to the optimal capacities
    # (*_nom_opt) which were just computed.
    set_nom_to_opt(n_exp)

    # Apply the capacities of `n_exp` to the network `n`. Note that
    # this only applies extendable capacities, which is fine.
    apply_caps(n_exp, n)

    n.export_to_netcdf(snakemake.output.network)
