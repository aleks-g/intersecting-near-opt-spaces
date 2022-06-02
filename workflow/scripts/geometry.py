# SPDX-FileCopyrightText: 2022 Koen van Greevenbroek & Aleksander Grochowicz
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Common geometric functions for use in the scripts and notebooks.

This includes methods for working with convex hulls (taking
intersections, finding the Chebyshev centre, checking containment and
non-emptyness, optimising over a convex hull) and methods for
generating and filtering directions in various ways (random sampling,
based on facet normals, etc.)

"""

import itertools
import logging
import math
from typing import Collection, Iterable

import gurobipy as gp
import numpy as np
import scipy.linalg as linalg
from gurobipy import GRB
from scipy.spatial import ConvexHull
from scipy.stats.qmc import LatinHypercube


def intersection(
    hulls: Collection[ConvexHull], max_iterations: int = 2000
) -> Collection[float]:
    """Compute an approximate intersection of a collection of convex hulls.

    Specifically, we find a large number of points whose convex hull
    approximates the intersection of the inputs. This is due to high
    computational efforts to intersect many convex hulls
    deterministically.

    Parameters
    ----------
    hulls: Collection[ConvexHull]
        Convex hulls to be intersected.
    max_iterations: int = 2000
        Maximal number of directions to be probed to approximate the intersection.

    Returns
    -------
    points: Collection[float]
        Approximation of the intersection of the given near-optimal feasible spaces.

    """
    # Gather the defining constraints of all the hulls.
    constraints = np.concatenate([h.equations for h in hulls])
    dims = constraints.shape[1] - 1

    # Check that enough iterations are allowed.
    if max_iterations < 2 * dims:
        raise ValueError("Not enough iterations allowed.")

    # Check if the intersection is even nonempty.
    if not is_nonempty(constraints):
        logging.warning("The intersection is empty!")
        return None

    # Start by finding points in the cardinal directions.
    m, _ = init_polytope(constraints)
    card_points = []
    card_directions = list(np.eye(dims)) + list(-np.eye(dims))
    for dir in card_directions:
        card_points.append(probe_polytope(m, dir))

    # Now we scale the axes to make the width of intersection equal to
    # 1 in each dimension.
    maxs = np.array([max(p[i] for p in card_points) for i in range(dims)])
    mins = np.array([min(p[i] for p in card_points) for i in range(dims)])
    widths = maxs - mins
    # Scale the columns of the constraint matrix by multiplying on the right.
    scaled_constraints = np.matmul(constraints, np.diag(np.append(widths, 1)))
    # Scale each point.
    scaled_points = [p / widths for p in card_points]

    # Initialise the (scaled) convex hull of the intersection.
    scaled_m, _ = init_polytope(scaled_constraints)

    # Initiate a random direction sampler.
    sampler = uniform_random_hypersphere_sampler(dims)
    directions = filter_vectors_auto(
        sampler, init_angle=45, initial_vectors=card_directions, max_retries=10
    )

    # Sample a given number of directions for the direction generator.
    for dir in itertools.islice(directions, max_iterations):
        scaled_points.append(probe_polytope(scaled_m, dir))

    # Take the convex hull of the scaled points.
    scaled_hull = ConvexHull(
        scaled_points, incremental=True, qhull_options="Qx C-0.0001"
    )
    scaled_points = scaled_hull.points[scaled_hull.vertices]

    # Scaled the points back and return them.
    points = np.matmul(scaled_points, np.diag(widths))
    return points


def ch_centre(hull: ConvexHull) -> (np.array, float, np.array):
    r"""Compute the Chebyshev centre and its radius from a given convex hull.

    Writes a linear program that outputs the point with the maximal
    radius inside the convex hull. This corresponds to writing the
    problem as follows:

    max R s.t.
        a_i * x + R * np.linalg.norm(a_i) \leq b_i for i = 1,...,num_eqns
    (cf. Boyd and Vandenberghe, Ch. 8.5)

    where (a_i * x \leq b_i)_i are the equations defining the convex
    hull.

    Note that when using qhull we use `-b_i`, as normal points are
    defined to be pointing outward, i.e. the convex hull satisfies `Ax
    <= -b` (cf. http://www.qhull.org/html/qh-opto.htm#n). In our case
    the vectors a_i are already normalised, so we just have:

    max R s.t.
        a_i * x + R \leq -b_i for i = 1,...,num_eqns

    Using a matrix formulation:

    max (0,...,0, 1) \cdot (x, R) s.t.
        (a_i, 1) \cdot (x, R) \leq -b for i = 1,...,num_eqns.

    In addition to the actual Chebyshev centre and the radius of the
    Chebyshev ball, this function also returns the tight constraints
    of the above problem (in order of tightness) which are the facets
    touched by the Chebyshev ball.

    Parameters
    ----------
    hull : scipy.spatial.ConvexHull

    Returns
    -------
    centre : np.array of shape (num_dims,),
    radius : float
    tight_constraints : np.array

    """
    num_eqn = hull.equations.shape[0]
    dims = hull.equations.shape[1] - 1

    # Prepare the objective function, which just has a single
    # coefficient for the radius.
    objective = np.array(([0] * dims) + [1])

    # Get the constraints of the form (a_i**T, norm(a_i)) * (x, R) <=
    # b_i. ConvexHull should normalise the vectors, so the norm is
    # always 1.
    A = np.hstack((hull.equations[:, :-1], np.ones(shape=(num_eqn, 1))))
    b = -hull.equations[:, -1]  # note the sign coming from a qhull equation

    # Prepare variable lower bounds: we want coordinates to be
    # unbounded and the radius to be nonnegative. The upper bounds
    # are positive infinity by default, so we do not need to set them.
    lb = [[-GRB.INFINITY] * dims + [0]]

    # Finally, solve the linear program.
    m = gp.Model()
    m.Params.OutputFlag = 0  # Do not log this optimisation.
    x = m.addMVar(shape=dims + 1, lb=lb)
    m.setObjective(objective @ x, GRB.MAXIMIZE)
    m.addConstr(A @ x <= b)
    m.optimize()

    # Check of the optimisation was successful.
    good_codes = [gp.GRB.OPTIMAL, gp.GRB.SUBOPTIMAL]
    if m.status not in good_codes:
        logging.warning(
            "ch_centre could not find centre point. Gurobi failed at"
            f" optimisation with status code {m.status}."
        )
        return None, None, None

    centre = x.X[:-1]
    radius = x.X[-1]

    # Exctract the tight constraints, which are those whose
    # corresponding slack values are non-zero.
    duals = [(i, c.pi) for i, c in enumerate(m.getConstrs())]
    duals.sort(reverse=True, key=lambda x: x[1])
    non_zero_dual_i = [i for i, d in duals if d != 0]
    tight_constraints = A[non_zero_dual_i, :-1]

    # Return the results
    return (centre, radius, tight_constraints)


def contains(hull: ConvexHull, point: np.array) -> bool:
    """Check if a convex hull contains a given point.

    Parameters
    ----------
    hull: scipy.spatial.ConvexHull
    point: np.array
        Must be of the same dimension as the points consituting `hull`.

    Returns
    -------
    Bool

    """
    num_eqn = hull.equations.shape[0]
    dims = hull.equations.shape[1] - 1
    if dims != len(point):
        raise ValueError("Dimension of hull and point do not match.")

    # In order to check if the point is in the convex hull, we simply
    # check that it satisfies every equation defining the hull.
    for i in range(num_eqn):
        eq = hull.equations[i, :-1]
        b = hull.equations[i, -1]
        if np.dot(eq, point) > -b:
            # The equation was violated!
            return False

    # If none of the equations were violated, then the point is
    # contained in the hull.
    return True


def is_nonempty(constraints: np.array) -> bool:
    """Check if a polytope is nonempty.

    Each row of `constraints` consists of the coefficients of an
    equation c_1 x_1 + c_2 x_2 + ... + c_n x_n <= -b. Return True if
    there exists a solution to all given constraints, or equivalently,
    if the polytope defined by the equations is non-empty.
    """
    A = constraints[:, :-1]
    b = -constraints[:, -1]
    c = np.array([1] * A.shape[1])  # The objective function is arbitrary.
    lb = [[-GRB.INFINITY] * A.shape[1]]
    m = gp.Model()
    m.Params.OutputFlag = 0  # Do not log this optimisation.
    x = m.addMVar(shape=len(c), lb=lb)
    m.setObjective(c @ x, GRB.MAXIMIZE)
    m.addConstr(A @ x <= b)
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        return True
    else:
        return False


def init_polytope(constraints: np.array) -> (gp.Model, gp.MVar):
    """Return a Gurobi model with a feasible space given by `constraints`."""
    A = constraints[:, :-1]
    b = -constraints[:, -1]
    lb = [[-GRB.INFINITY] * A.shape[1]]
    m = gp.Model()
    m.Params.OutputFlag = 0  # Do not log anything related to this model.
    x = m.addMVar(shape=A.shape[1], lb=lb)
    m.addConstr(A @ x <= b)
    return m, x


def probe_polytope(m: gp.Model, direction: np.array) -> np.array:
    """Return a point in `direction` inside the space defined by `constraints`."""
    m.setMObjective(None, direction, 0.0, None, None, None, GRB.MAXIMIZE)
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        return np.array([x.X for x in m.getVars()])
    else:
        raise RuntimeError("Gurobi could not optimise over the given polytope.")


def facet_normals(convex_hull: ConvexHull) -> np.array:
    """Return the facet normals of a convex hull, sorted by facet size."""
    # Extract all facets of the convex hull by points and compute
    # their volume.
    facets = []
    for s, e in zip(convex_hull.simplices, convex_hull.equations[:, :-1]):
        # Get the points of the facet, and compute edge vectors
        # spanning the facet (which is a simplex).
        vertices = [convex_hull.points[p] for p in s]
        edges = [vertices[0] - v for v in vertices[1:]]
        # To compute the volume, we compute the QR decomposition of
        # the matrix whose column vectors are the simplex edges. This
        # gives those edges in an orthonormal basis for the simplex.
        # Then we take the product of the diagonal of these new
        # coordinates (R), which is the determinant since R is upper
        # triangular. This actually gives n factorial times the volume
        # (where n is the dimension), but we do not care since it is
        # just a uniform scaling factor.
        A = np.array(edges).T
        _, R = linalg.qr(A, mode="economic")
        volume = linalg.det(R)
        facets.append((e, volume))

    # Sort the facets by volume (negation to get decreasing order).
    facets.sort(key=lambda t: -t[1])
    return [f[0] for f in facets]


def uniform_random_hypersphere_sampler(n: int):
    """Generate points on the `n`-dimensional hypersphere at random.

    The points are normalised and following the uniform distribution
    on the hypersphere.
    """
    while True:
        # Transform from unit cube to cube around origin.
        p = 2 * np.random.random_sample((n,)) - 1
        if np.linalg.norm(p) <= 1:
            # Transform to lie on the unit hypersphere.
            yield p / np.linalg.norm(p)


def lhc_random_hypersphere_sampler(n: int):
    """Generate points on the `n`-dimensional hypersphere at random.

    The points are generated using Latin hypercube sampling and
    normalised. As in
    https://en.wikipedia.org/wiki/Latin_hypercube_sampling.

    The difference with `uniform_random_hypersphere_sampler` is that
    the points generated by this sampler do not follow the uniform
    distribution on the hypersphere. Instead, coordinates generated by
    LHS are more evenly distributed. This leads to a distribution on
    the hypersphere which is less dense around the axes.

    """
    sampler = LatinHypercube(d=n)
    while True:
        lhc = sampler.random(n)
        for p in lhc:
            q = 2 * p - 1  # Transform from unit cube to cube around origin.
            yield q / np.linalg.norm(q)  # Transform to lie on the unit hypersphere.


def angle_threshold(candidate: np.array, previous: np.array, angle: float):
    """Filter a candidate angle on previous angles.

    Return False if the vector `candidate` is within an angle of
    `angle` (in degrees) of any vector in `previous`, True otherwise.

    """
    for p in previous:
        p_norm = p / np.linalg.norm(p)
        c_norm = candidate / np.linalg.norm(candidate)
        dot = min(1, np.dot(p_norm, c_norm))  # Avoid getting 1.00000001
        t = np.degrees(np.arccos(dot))
        if t < angle:
            return False
    return True


def filter_vectors(
    vecs: Iterable,
    angle: float = 10,
    initial_vectors: Collection = None,
    max_retries: int = 1000,
):
    """Run a vector generator and filter similar vectors out.

    In particular, this generator keeps track of previously seen
    vectors and filters away any vector closer than an `angle`
    degrees to previous ones.

    Parameters
    ----------
    vecs : Iterable,
        The vectors to filter by angle.
    angle : float
        Initial threshold angle below which new vectors are discarded.
    initial_vectors : Collection
        Initial set of vectors with which new vectors from `vecs` are
        compared.
    max_retries : int
        Number of consecutive vectors from `vecs` that can be
        discarded for being too close to previously seen vectors,
        before the threshold angle is decreased or the generator
        terminates.

    """
    # Copy collection of previous vectors if any.
    if initial_vectors is not None:
        previous_vecs = initial_vectors[:]
    else:
        previous_vecs = []

    num_retries = 0
    for vec in vecs:
        # Check if we have reached the maximum number of retries, in
        # which case we give up.
        if num_retries >= max_retries:
            return
        # Check if the current vector is far enough away from all
        # previous ones.
        if not angle_threshold(vec, previous_vecs, angle):
            num_retries += 1
            continue
        previous_vecs = np.vstack((previous_vecs, vec))
        # Since we found a vector, reset the `num_retries` to 0.
        num_retries = 0
        yield vec


def filter_vectors_auto(
    vecs: Iterable,
    init_angle: float = 10.0,
    initial_vectors: Collection = None,
    max_retries: int = 100,
    min_angle_tolerance: float = 0.1,
):
    """Run a vector generator and filter similar vectors out.

    This generator yields vectors from `vecs` in order, but filters
    out any vectors too close to previously seen vectors. By
    "previously seen", we meet all the vectors previously yielded,
    together with the vectors in `initial_vectors`. By "too close", we
    mean within a certain angle theta. The angle theta is initially
    set to `init_angle`. However, every time `max_retries` consecutive
    vectors have been discarded, theta is decreased by 20%. This
    continues until theta drops below `min_angle_tolerance`, at
    which point the generator stops after `max_retries` consecutive
    discarded vectors.

    This method works best when `vecs` yields an indefinite number of
    vectors, such as by independent random generatation.

    Parameters
    ----------
    vecs : Iterable,
        The vectors to filter by angle.
    init_angle : float
        Initial threshold angle below which new vectors are discarded.
    initial_vectors : Collection
        Initial set of vectors with which new vectors from `vecs` are
        compared.
    max_retries : int
        Number of consecutive vectors from `vecs` that can be
        discarded for being too close to previously seen vectors,
        before the threshold angle is decreased or the generator
        terminates.
    min_angle_tolerance : float
        The minimum threshold angle allowed. If the threshold angle is
        decreased below this, the generated is terminated.

    """
    a = init_angle

    # Copy collection of previous vectors if any.
    if initial_vectors is not None:
        previous_vecs = initial_vectors[:]
    else:
        previous_vecs = []

    num_retries = 0
    for vec in vecs:
        # If we tried too many times to generate a new direction but
        # failed, decrease the threshold angle.
        if num_retries >= max_retries:
            a *= 0.8
            num_retries = 0
            if a < min_angle_tolerance:
                # At this point we have really run out of directions.
                return
            logging.info(f"Decreased angle threshold to {a}.")

        # Check if the current vector is far enough away from all
        # previous ones.
        if not angle_threshold(vec, previous_vecs, a):
            num_retries += 1
            continue
        previous_vecs = np.vstack((previous_vecs, vec))
        # Since we found a vector, reset the `num_retries` to 0.
        num_retries = 0
        yield vec


def hypersphere_packing_bound(dim: int, theta: float):
    """Lower bound on number of points `theta` degrees apart on unit hypersphere.

    Return a lower bound on the number of points which can be fitted
    on the (`dim`-1)-sphere (i.e. points in `dim`-dimensional
    Euclidean space at distance 1 from the origin) such that each pair
    of points is at least `theta` degrees apart.

    """
    # We only support dimensions `dim` for which the packing density
    # in dimension `dim`-1 is known. (Except dimension 24, which we
    # do not bother to support, for which the packing density is in
    # fact known.)
    if dim < 3:
        return ValueError(f"Dimension {dim} too low.")
    if dim > 9:
        return ValueError(f"Dimension {dim} too high.")

    # Hypersphere packing densities, see
    # https://mathworld.wolfram.com/HyperspherePacking.html.
    densities = {
        2: 0.90689968,
        3: 0.74048052,
        4: 0.61685029,
        5: 0.46525763,
        6: 0.37294756,
        7: 0.29529789,
        8: 0.25366952,
    }
    dim_const = (
        dim * math.gamma((dim + 1) / 2) * math.sqrt(math.pi) / math.gamma(dim / 2 + 1)
    )
    theta_rad = math.pi * theta / 180
    return densities[dim - 1] * dim_const / math.pow(theta_rad / 2, dim - 1)
