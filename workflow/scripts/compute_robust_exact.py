# SPDX-FileCopyrightText: 2022 Koen van Greevenbroek & Aleksander Grochowicz
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Compute robust networks exactly given a robust set of total investments.

Provides functionality to compute the robust solution based on the
near-optimal space. It starts with the coordinates/values of the
projected decision variables coming from the Chebyshev centre and adds
them as fixed constraints whereupon the network is solved.

"""

import copy
import logging
from collections import OrderedDict
from pathlib import Path

import pandas as pd
import pypsa
from _helpers import configure_logging
from pypsa.linopf import network_lopf
from pypsa.linopt import define_constraints, linexpr
from solve_operations import set_extendable_false
from utilities import apply_caps, get_basis_variables, set_nom_to_opt


def compute_robust(
    n: pypsa.Network,
    basis: OrderedDict,
    coordinates: dict,
) -> None:
    """Solve `n` with fixed coordinates in the given basis.

    That is, the network `n` is solved with additional constraints
    forcing the projection of `n` in the given `basis` to be equal to
    `coordinates`. When the `coordinates` lie in the intersection of
    near-optimal feasible spaces, the result is a robust network.

    Modifies the argument `n`.

    Parameters
    ----------
    n : pypsa.Network
    basis : OrderedDict
    coordinates: dict

    """
    # Retrieve solver options from n.
    solver_options = n.config["solving"]["solver"].copy()
    solver_name = solver_options.pop("name")
    tmpdir = n.config["solving"].get("tmpdir", None)
    if tmpdir is not None:
        Path(tmpdir).mkdir(parents=True, exist_ok=True)

    def set_coordinates(n, snapshots):
        """Extra functionality to set total capacities on pypsa-eur network."""
        basis_variables = get_basis_variables(n, basis)
        # Add constraint to make the solution near-optimal.
        for key in basis_variables:
            x = basis_variables[key]
            b = coordinates[key]
            # Use a scaling factor to avoid too large coefficients and
            # potential resulting numerical issues. Use `float` in
            # case `coordinates` is a dataframe.
            scaling_factor = float(100 / b)
            # Define lower and upper constraints in the given basis
            # dimension, fixing the coordinates.
            define_constraints(
                n,
                linexpr((scaling_factor * x.coeffs, x.vars)).sum(),
                ">=",
                scaling_factor * b,
                "Investment_costs_technology_min",
                key,
            )
            define_constraints(
                n,
                linexpr((scaling_factor * x.coeffs, x.vars)).sum(),
                "<=",
                scaling_factor * 1.001 * b,
                "Investment_costs_technology_max",
                key,
            )

    # Solve the network.
    logging.info(
        f"Spreading the robust technology investment costs {coordinates} optimally."
    )
    network_lopf(
        n,
        solver_name=solver_name,
        solver_options=solver_options,
        extra_functionality=set_coordinates,
        solver_dir=tmpdir,
    )


if __name__ == "__main__":
    # Set up logging so that everything is written to the right log file.
    configure_logging(snakemake)

    # Load the network.
    n = pypsa.Network(snakemake.input.network)
    n.config = snakemake.config["pypsa-eur"]

    # Load the projected centre (robust) solution coordinates.
    coordinates = pd.read_csv(snakemake.input.centre, index_col=0)
    coordinates = dict(coordinates.iloc[0, :])

    # Compute the robust allocation. We do this on a copy of n and
    # apply the resulting capacities back to n; this is in order to
    # avoid solving n, which can lead to issues down the road. For
    # example, later adding load shedding to an already-solved network
    # does no work as expected.
    r = copy.deepcopy(n)
    compute_robust(
        r,
        basis=snakemake.config["projection"],
        coordinates=coordinates,
    )
    apply_caps(r, n)

    # Set the nominal capacities of the network to the optimal
    # capacities which were just computed.
    set_nom_to_opt(n)

    # Export the network.
    n.export_to_netcdf(snakemake.output.network)

    # Also export the "operated" network `r`, so we avoid repeating
    # work down the road. We set everything to non-extendable to make
    # it look like the results of the operations rule. However, we
    # do not add load shedding. Also note that the objective value of
    # this network will not be comparable to outputs from the
    # `solve_operations` rule since it includes investment costs.
    set_extendable_false(r)
    r.export_to_netcdf(snakemake.output.operated_network)

    # Export costs: note that these differ from the variables, as they
    # include all technologies (and not only the projected ones) as
    # well as all variable costs (which are not considered in the
    # basis variables).
    with open(snakemake.output.obj, "w") as f:
        f.write(str(r.objective))
