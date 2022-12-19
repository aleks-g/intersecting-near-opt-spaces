# SPDX-FileCopyrightText: 2022 Koen van Greevenbroek & Aleksander Grochowicz
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Functionality to solve the operations of a given network.

For this given network, all components are set to be non-extendable
and load shedding is added to make the optimisation problem feasible.
The costs for load shedding are set high enough to avoid its usage as
much as possible.

If the option `-single-years` in the configuration is set as the
operations method, then this script solves the operations year-by-year.
In that case, the CO2 budget is split up into annual budgets for each
operation.

If the `op_bound_network` input is given, there is a cap on the
variable costs the network is allowed to use before it has to use load
shedding instead. In that situation, load shedding and overuse of
variable costs is indistinguishable. If additionally single year
operations are used, then both the CO2 and the variable cost budgets
are split into annual budgets.

"""

import logging
from pathlib import Path
import os.path

import pypsa
from _helpers import configure_logging, parse_year_wildcard
from pypsa.components import component_attrs, components
from pypsa.descriptors import nominal_attrs
from pypsa.linopf import network_lopf
from pypsa.linopt import define_constraints, get_var, linexpr
from pypsa.pf import get_switchable_as_dense as get_as_dense
from utilities import set_nom_to_opt
from workflow_utilities import parse_net_spec



marginal_attr = {"Generator": "p", "Link": "p", "Store": "p", "StorageUnit": "p"}


def set_extendable_false(n: pypsa.Network) -> None:
    """Set all technologies in `n` to non-extendable.

    Modifies the argument `n`.

    """
    for c, attr in nominal_attrs.items():
        n.df(c)[attr + "_extendable"] = False


def add_load_shedding(n: pypsa.Network) -> None:
    """Add load shedding generator at every AC bus of `n`.

    Modifies the argument `n`.

    """
    n.add("Carrier", "load-shedding")
    buses_i = n.buses.query("carrier == 'AC'").index
    n.madd(
        "Generator",
        buses_i,
        " load shedding",
        bus=buses_i,
        carrier="load-shedding",
        # Marginal cost same as in highRES (Price and Zeyringer, 2022)
        marginal_cost=7.3e3,  # EUR/MWh
        p_nom=1e6,
    )


if __name__ == "__main__":
    # Set up logging so that everything is written to the right log
    # file.
    if "snakemake" not in globals():
        configure_logging(snakemake)

    # Load the network and solving options.
    overrides = override_component_attrs(snakemake.input.overrides)
    n = pypsa.Network(snakemake.input.network, override_component_attrs=overrides)

    # Set nominal to optimal capacities.
    set_nom_to_opt(n)

    # Set all components to be non-extenable.
    set_extendable_false(n)

    # Add load shedding to the network.
    add_load_shedding(n)

    # As a dirty fix (since we have had trouble with undefined values),
    # replaces all NaN-values by 0 in all nominal capacities.
    for c, attr in nominal_attrs.items():
        n.df(c)[attr] = n.df(c)[attr].fillna(0)

    # At this point, the network `n` should have:
    # - Defined capacities (`p_nom`, `s_nom`, etc.) for all included
    #   technologies,
    # - No extendable technologies,
    # - Load shedding installed at all nodes,
    # - Possibly a single global constraint on CO2 emissions.

    # Optionally (depending on whether the "op_bound_network" input
    # file is present), add a bound on the operational costs of the
    # network.
    if "op_bound_network" in snakemake.input.keys():
        # Load the network from which we retrieve the budget.
        op_bound_n = pypsa.Network(snakemake.input.op_bound_network)
        # Calculate the total operational costs in `op_bound_n`.
        budget = (
            op_bound_n.generators_t["p"]
            .multiply(op_bound_n.snapshot_weightings.objective, axis="index")
            .multiply(op_bound_n.generators.marginal_cost, axis="columns")
            .sum()
            .sum()
        )

        logging.info(f"Solving operations with operational budget of {budget}.")

        # Sanity check: `op_bound_n` should actually be equal to `budget`.
        logging.debug(f"(Objective of op_bound_n was f{op_bound_n.objective}.)")

        # Define an `extra_functionality` function to add the bound to
        # the network in the solving stage.

        def extra_functionality(m, snapshots):
            """Add operational budget to `n`."""
            # Figure out which fraction of the budget to give
            # depending on the number of snapshots we are optimising
            # over.
            fraction = (
                m.snapshot_weightings.objective.loc[snapshots].sum()
                / m.snapshot_weightings.objective.sum()
            )
            # Build a linear expression summing up all operational
            # costs in the network; adapted from pypsa/linopf.py:define_objective
            total_op_cost = ""
            for c, attr in marginal_attr.items():
                cost = (
                    get_as_dense(m, c, "marginal_cost", snapshots)
                    .loc[:, lambda ds: (ds != 0).all()]
                    .mul(m.snapshot_weightings.objective[snapshots], axis=0)
                )
                # Exclude load shedding from the budget.
                generators = [c for c in cost.columns if "load shedding" not in c]
                cost = cost.loc[:, generators]
                if cost.empty:
                    continue
                total_op_cost += (
                    linexpr((cost, get_var(m, c, attr).loc[snapshots, cost.columns]))
                    .sum()
                    .sum()
                )
            define_constraints(
                m,
                total_op_cost,
                "<=",
                budget * fraction,
                "Generator",
                "operational_budget",
            )

    else:

        def extra_functionality(m, snapshots):
            """Do nothing (dummy function)."""
            pass

    # Prepare solving options.
    solver_options = snakemake.config["pypsa-eur"]["solving"]["solver"]
    solver_name = solver_options.pop("name")
    tmpdir = snakemake.config["pypsa-eur"]["solving"].get("tmpdir", None)
    if tmpdir is not None:
        Path(tmpdir).mkdir(parents=True, exist_ok=True)

    # Solve the network.
    op_method = snakemake.wildcards.operation_method
    if op_method == "":
        # The default; run the operations for the entire network in one go.
        status, condition = network_lopf(
            n,
            solver_name=solver_name,
            solver_options=solver_options,
            solver_dir=tmpdir,
            extra_functionality=extra_functionality,
        )

        if status != "ok":
            raise RuntimeError(f"Solving for operations failed: {condition}.")
    elif op_method == "-single-years":
        # Run the operations for all the years separately.
        all_years = parse_year_wildcard(
            parse_net_spec(snakemake.wildcards.spec)["year"]
        )
        # Temporarily divide the network CO2 limit equally between the years.
        if "CO2Limit" in n.global_constraints.index:
            n.global_constraints.loc["CO2Limit", "constant"] /= len(all_years)
        # Optimise the operations of network year by year. Note that
        # the budget was divided into annual budgets (`fraction`) in
        # the extra_functionality before already.
        for y in all_years:
            status, condition = network_lopf(
                n,
                snapshots=n.snapshots[n.snapshots.slice_indexer(str(y), str(y))],
                solver_name=solver_name,
                solver_options=solver_options,
                solver_dir=tmpdir,
                extra_functionality=extra_functionality,
            )
            if status != "ok":
                raise RuntimeError(f"Solving for operations failed: {condition}.")
        # Reset the CO2 limit to the original value for future reference.
        if "CO2Limit" in n.global_constraints.index:
            n.global_constraints.loc["CO2Limit", "constant"] *= len(all_years)
    else:
        raise ValueError(f"Validation method {op_method} not recognised.")

    # Export the result.
    n.export_to_netcdf(snakemake.output.network)
