# SPDX-FileCopyrightText: 2022 Koen van Greevenbroek & Aleksander Grochowicz
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Prepare validation networks with shifted capacities.

This script defines the tests we run to validate the heuristics. It is
based on the coordinates of the Chebyshev centre and the shape of the
near-optimal intersection. It outputs the reductions in investment
costs we want to consider (for each cardinal direction).

"""

import itertools
import os
from collections import OrderedDict, namedtuple
from pathlib import Path
from typing import Collection, List

import numpy as np
import pandas as pd
import pypsa
import scipy.linalg as linalg
from _helpers import configure_logging
from geometry import init_polytope, probe_polytope
from scipy.spatial import ConvexHull
from utilities import BasisCapacities

# A `ValidationPoint` is just a tuple containing in the first place
# some coordinates (and np.array), and in the second place a
# description (a string).
ValidationPoint = namedtuple("ValidationPoint", ["coordinates", "description"])


def cardinal_direction_validation_points(
    hull: ConvexHull,
    centre: np.array,
    radius: float,
    basis: OrderedDict,
) -> List[ValidationPoint]:
    """Compute validation points in cardinal directions from given centre.

    Given a point `centre` inside the convex `hull`, return a list of
    points (given as ValidationPoints, which each consists of
    coordinates and a description) at certain distances away from
    `centre` in the negative cardinal directions. In particular, in
    each cardinal direction we compute points that are a half-radius, a
    radius away from `centre`, the boundary point on `hull`
    encountered when going from `centre` in the negative cardinal
    direction, and point between the last two mentioned.

    """
    # First, define the directions (in the projected space) in which
    # we perturb n.
    directions = list(-np.eye(len(basis)))

    # Now, for each direction d, compute the points (in the projected
    # space) which lay:
    # - A distance of `radius` away from `coordinates` in direction d.
    # - At the intersection of the boundary of the convex hull
    #   `intersection`, and the ray from `coordinates` in direction d.
    boundary_points = cardinal_boundary_points(hull, centre, directions)
    radius_points = [(centre + radius * dir) for dir in directions]

    # From the above two sets of points, we obtain a further two sets
    # of points: those between `coordinates` and the "radius" points,
    # and those between the "radius" and "boundary" points.
    half_radius_points = [(centre + r) / 2 for r in radius_points]
    half_boundary_points = [(r + b) / 2 for r, b in zip(radius_points, boundary_points)]

    # Collect all the validation points, and give them nice
    # descriptions.
    point_types = ["half_radius", "radius", "half_boundary", "boundary"]
    descriptions = [
        "reduction_" + k + "_" + d
        for d, k in itertools.product(point_types, basis.keys())
    ]
    points = half_radius_points + radius_points + half_boundary_points + boundary_points
    return [ValidationPoint(c, d) for c, d in zip(points, descriptions)]


def cardinal_boundary_points(
    hull: ConvexHull,
    centre: np.array,
    directions: Collection[np.array],
):
    """Compute boundary points of convex hull going from given point.

    In particular, return for each direction d in `directions` the
    point on the boundary of `hull` which lays on the line through
    `coordinates` parallel to d.

    """
    boundary_points = []
    c = centre.T
    for dir in directions:
        # Use a QR-decomposition to find an orthonormal basis q for
        # R^n (where n==len(coordinates)) with `dir` as the first
        # (column) vector in the basis (up to a scaling factor).
        a = np.hstack([np.matrix(dir).T, np.eye(len(dir))])
        q, _ = linalg.qr(a, mode="economic")
        # The rest of the columns in q form a basis of the normal
        # space to `dir`.
        orth_to_dir = q[:, 1:]
        # Now create a Gurobi model whose feasible space is `hull`.
        m, x = init_polytope(hull.equations)
        # Add constraints to the model which limit the feasible space
        # to a line through `coordinates` parallel to `dir`.
        eps = 1e-8 * linalg.norm(c)
        for column in orth_to_dir.T:
            m.addConstr(column @ x <= np.dot(column, c) + eps)
            m.addConstr(column @ x >= np.dot(column, c) - eps)
        # Now solve the model in the given direction to get the
        # desired boundary point.
        boundary_points.append(probe_polytope(m, dir))

    return boundary_points


if __name__ == "__main__":
    # Set up logging so that everything is written to the right log file.
    configure_logging(snakemake)

    # Load the network to apply capacities to, as well as geometric
    # data (convex hull, centre, radius).
    n = pypsa.Network(snakemake.input.network)

    intersection_points = pd.read_csv(snakemake.input.intersection, index_col=0)
    intersection = ConvexHull(intersection_points)

    centre = pd.read_csv(snakemake.input.centre, index_col=0).iloc[0].values

    with open(snakemake.input.radius, "r") as f:
        radius = float(f.read())

    basis = snakemake.config["projection"]

    # Create the output directory of networks if it does not already exist.
    Path(snakemake.output[0]).mkdir(exist_ok=True)

    # Get a list of validation points.
    ps = cardinal_direction_validation_points(intersection, centre, radius, basis)

    # For each validation point, determine the difference between it
    # and the centre point, and apply that difference to `n`.
    caps = BasisCapacities(basis, init=n)
    for p in ps:
        diff = p.coordinates - centre
        diff_dict = {k: diff[i] for i, k in enumerate(basis.keys())}
        shifted_caps = caps.shift(diff_dict)
        shifted_caps.apply_to_network(n, set_opt=True)
        n.export_to_netcdf(os.path.join(snakemake.output[0], p.description + ".nc"))
