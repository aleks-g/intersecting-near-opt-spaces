# SPDX-FileCopyrightText: 2022 Koen van Greevenbroek & Aleksander Grochowicz
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Find the most expensive among a set of networks.

This script finds the year with the highest optimal costs among the
years considered. It then outputs the investment costs in that year
addionally.

"""

import numpy as np
import pypsa
from utilities import annual_investment_cost

if __name__ == "__main__":
    # Load all the input networks.
    nets = [pypsa.Network(f) for f in snakemake.input]

    # Compute the investment costs of all the given networks.
    investment_costs = [annual_investment_cost(n, use_opt=True) for n in nets]

    # Get the most expensive network.
    most_exp_i = np.argmax(investment_costs)
    most_exp_n = nets[most_exp_i]
    most_exp_cost = max(investment_costs)

    # Write the output.
    most_exp_n.export_to_netcdf(snakemake.output.most_expensive_network)
    with open(snakemake.output.investment_cost, "w") as f:
        f.write(str(most_exp_cost))
