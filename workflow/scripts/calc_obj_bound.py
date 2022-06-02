# SPDX-FileCopyrightText: 2022 Koen van Greevenbroek & Aleksander Grochowicz
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Output the maximum objective of all input objectives, plus margin."""

if __name__ == "__main__":
    # Open all input objectives.
    objs = []
    for i in snakemake.input:
        with open(i, "r") as f:
            objs.append(float(f.read()))

    # Extract epsilon.
    eps = float(snakemake.wildcards.epsf)

    # Calculate the objective bound.
    obj_bound = (1 + eps) * max(objs)

    # Output the maximum allowed objective.
    with open(snakemake.output[0], "w") as f:
        f.write(str(obj_bound))
