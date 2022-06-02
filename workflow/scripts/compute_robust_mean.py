# SPDX-FileCopyrightText: 2022 Koen van Greevenbroek & Aleksander Grochowicz
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Compute the "mean" robust heuristic.

Similar to `compute_robust_exact.py`, but instead of solving the
network outright (potentially leading to an intractably large
optimisation problem), this heuristic averages the capacities of
several optimisations with single weather years.

"""

from functools import reduce
from typing import Collection

import pypsa
from _helpers import configure_logging
from utilities import add_caps, apply_caps, scale_caps


def compute_robust_mean(
    n: pypsa.Network,
    exact_robusts: Collection[pypsa.Network],
) -> pypsa.Network:
    """Heuristically combine several networks into a single average.

    Combine the networks in `exact_robusts` into a single network like
    `n`, averaging capacities for each node, line and link.

    Note: we assume that the capacities we are interested in are
    stored in *_nom (as opposed to *_nom_opt), which they are in
    `exact_robusts`.

    Parameters
    ----------
    n : pypsa.Network
        Base (unsolved) network to generate capacities for.
    exact_robusts : Collection[pypsa.Network]
        Networks whose capacities to combine.

    Returns
    -------
    pypsa.Network

    """
    # Compute the node-wise average of the capacities, and apply the
    # resulting capacities to `n`.
    average = scale_caps(reduce(add_caps, exact_robusts), 1 / len(exact_robusts))
    apply_caps(average, n)

    return n


if __name__ == "__main__":
    # Set up logging so that everything is written to the right log file.
    configure_logging(snakemake)

    # Load the network.
    n = pypsa.Network(snakemake.input.network)

    # Load the solved robust networks for individual years.
    exact_robusts = [pypsa.Network(r) for r in snakemake.input.exact_robusts]

    # Compute the robust allocation.
    r = compute_robust_mean(n, exact_robusts)

    # Export the network.
    r.export_to_netcdf(snakemake.output.network)
