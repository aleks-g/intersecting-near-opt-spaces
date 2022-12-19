# SPDX-FileCopyrightText: 2022 Koen van Greevenbroek & Aleksander Grochowicz
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Compute a robust solution based on the most expensive year considered.

We add investment costs uniformly on all technologies (on the optimal
solution of the most expensive year) to match investment costs in the
exact solution. This serves as a baseline for comparing the allocation
heuristics.

"""
import os.path

import pypsa
from pypsa.components import component_attrs, components
from _helpers import configure_logging
from utilities import annual_investment_cost, apply_caps, scale_caps, set_nom_to_opt, override_component_attrs



if __name__ == "__main__":
    # Set up logging so that everything is written to the right log file.
    configure_logging(snakemake)

    # Load the network and solving options.
    overrides = override_component_attrs(snakemake.input.overrides)
    n = pypsa.Network(snakemake.input.network, override_component_attrs=overrides)
    n_exp_year = pypsa.Network(snakemake.input.most_expensive_network, override_component_attrs=overrides)
    n_robust_exact = pypsa.Network(snakemake.input.robust_exact, override_component_attrs=overrides)    

    # Get network capacities and the investment cost of extendable
    # technologies. Since `n_exp_year` just comes from a network
    # optimisation, we need to use the `*_nom_opt` capacities.
    exp_year_costs = annual_investment_cost(
        n_exp_year, only_extendable=True, use_opt=True
    )

    # Get the corresponding investment costs for the exact robust
    # network. In this network, the capacities are just stored in
    # `*_nom`.
    robust_exact_costs = annual_investment_cost(n_robust_exact, only_extendable=True)

    # Now we can scale all extendable technologies in `n_exp_year` up
    # by such a factor that their total investment cost equals the
    # corresponding investment cost in the exact robust network.
    scaling_factor = robust_exact_costs / exp_year_costs
    n_scaled_exp_year = scale_caps(n_exp_year, scaling_factor, only_extendable=True)
    apply_caps(n_scaled_exp_year, n)
    set_nom_to_opt(n)

    # Export the network.
    n.export_to_netcdf(snakemake.output.network)
