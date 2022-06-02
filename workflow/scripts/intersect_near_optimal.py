# SPDX-FileCopyrightText: 2022 Koen van Greevenbroek & Aleksander Grochowicz
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Computes intersections of near-optimal feasible spaces."""

import logging

import pandas as pd
from _helpers import configure_logging
from geometry import ch_centre, intersection
from scipy.spatial import ConvexHull

if __name__ == "__main__":
    # Set up logging so that everything is written to the right log file.
    configure_logging(snakemake)

    # Read the convex hulls we are intersecting.
    hull_files = snakemake.input.hulls
    points = [pd.read_csv(file, index_col=0) for file in hull_files]
    list_hulls = [ConvexHull(i) for i in points]

    # Intersecting all convex hulls.
    logging.info("Trying to intersect all spaces.")
    intersected_points = intersection(hulls=list_hulls)
    if intersected_points is None:
        raise RuntimeError(
            "No intersection was possible. Consider working with"
            " robust solutions for single years."
        )

    # Write intersected space to the given output.
    pd.DataFrame(intersected_points, columns=points[0].columns).to_csv(
        snakemake.output.intersection
    )

    # Compute the centre point of the intersection space, and output that too.
    intersected_hull = ConvexHull(intersected_points)
    intersected_centre, intersected_radius, _ = ch_centre(intersected_hull)
    intersected_centre_df = pd.DataFrame(intersected_centre).T
    intersected_centre_df.columns = points[0].columns
    intersected_centre_df.to_csv(snakemake.output.centre)
    with open(snakemake.output.radius, "w") as f:
        f.write(str(intersected_radius))
