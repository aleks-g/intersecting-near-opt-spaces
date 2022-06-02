# SPDX-FileCopyrightText: 2022 Koen van Greevenbroek & Aleksander Grochowicz
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Summarise feasibility of the given networks.

We use curtailment/load shedding as a measure of (in-)feasibility of a
network.

"""

import pandas as pd
import pypsa
from _helpers import configure_logging
from utilities import compute_feasibility_criteria

if __name__ == "__main__":
    # Set up logging so that everything is written to the right log file.
    configure_logging(snakemake)

    # We want to compute the following values for each network to be validated:
    # 1. The total curtailment in MWh (however due to numerical
    #    instabilities, we only consider curtailment if it is above 1 MW
    #    systemwide).
    # 2. The ratio of load curtailed (again with the threshold of 1 MW
    #    to avoid numerical instabilities) to total load.
    networks_valid = {name: pypsa.Network(name) for name in snakemake.input}

    results = []
    for name, n in networks_valid.items():
        results.append(compute_feasibility_criteria(n, name))
    validated_feasibility = pd.concat(results)
    validated_feasibility.to_csv(snakemake.output.summary)
