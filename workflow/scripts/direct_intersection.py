# SPDX-FileCopyrightText: 2022 Koen van Greevenbroek & Aleksander Grochowicz
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Computes an approximation of the intersection of near-optimal spaces directly.

In particular, this algorithm stars with crude approximations of
near-optimal spaces F_1, ..., F_n, and approximates the intersection
F_1 ∩ ... ∩ F_n without approximating each space F_1, ..., F_n
individually in detail.

"""

import copy
import logging
import multiprocessing
import os
import time
from collections import OrderedDict, namedtuple
from multiprocessing import Manager, Queue, get_context
from pathlib import Path
from typing import Collection, List

import numpy as np
import pandas as pd
import pypsa
from _helpers import configure_logging
from compute_near_opt import (
    large_facet_directions,
    maximal_centre_then_facets,
    touching_ball_directions,
)
from geometry import (
    ch_centre,
    filter_vectors_auto,
    intersection,
    probe_intersection,
    uniform_random_hypersphere_sampler,
)
from scipy.spatial import ConvexHull
from utilities import get_basis_values, solve_network_in_direction

# Define a simple named tuple datastructure to store pairs of space
# networks and corresponding near-optimal spaces (or approximations
# thereof.)
NearOpt = namedtuple("NearOpt", ["net", "space"])


def compute_intersection_direct(
    networks: List[pypsa.Network],
    mga_spaces: List[pd.DataFrame],
    obj_bound: float,
    basis: OrderedDict,
    conv_method: str,
    direction_method: str,
    direction_angle_sep: float,
    conv_eps: float,
    conv_iter: int,
    max_iter: int,
    debug_dir: str,
    cache_dir: str,
    num_parallel_solvers: int,
    qhull_options: str = None,
    angle_tolerance: float = 0.1,
) -> Collection[np.array]:
    """Approximate an intersection of near-optimal space directly.

    In particular, this algorithm stars with crude approximations of
    near-optimal spaces F_1, ..., F_n, (the `mga_spaces`) and
    approximates the intersection F_1 ∩ ... ∩ F_n without
    approximating each space F_1, ..., F_n individually in detail.

    Parameters
    ----------
    networks : List[pypsa.Network]
        The networks whose near-optimal feasible spaces to intersect.
    mga_spaces : List[pd.DataFrame]
        The "MGA" spaces for the given networks, treated as crude
        first-order approximations of near-optimal spaces.
    obj_bound : float
        Upper bound for costs in near-optimality constraint.
    basis : OrderedDict
        A basis on which to project the feasible space of `n`. The
        keys being the decision variables with values given in the
        format produced by `pypsa.linopt.linexpr`.
    conv_method : str
        Convergence method. Can be 'volume' or 'centre'.
    direction method : str
        Method for choosing directions. Can be one of 'facets',
        'random-uniform', 'maximal-centre' or
        'maximal-centre-then-facets'.
    direction_angle_sep : float
        Initial angle threshold for direction generation. Depending on
        `direction` in the configuration, this threshold can decrease
        automatically down to `angle_tolerance`.
    conv_eps : float
        Convergence threshold in percent.
    conv_iter : int
        Number of iterations for which the convergence criterion must
        be below `conv_eps` for the algorithm to terminate.
    max_iter : int
        The maximum number of iterations after which the algorithm is
        termined regardless of convergence.
    debug_dir : str
        Directory where the debug files (networks, volume, centre,
        radius, probed directions) should be saved.
    cache_dir : str
        Directory where the cache files (config, points, probed
        directions) should be saved. If points and probed directions
        from previous runs with the same configuration exist, these
        are being used.
    num_parallel_solvers : int
        The number of parallel processes to use.
    qhull_options : str
        Options for qhull, e.g. for numerical stability or to avoid
        certain degeneracies.
    angle_tolerance : float
        Minimal angle threshold for filtering.

    Returns
    -------
    Collection[np.array]
        A collection of points in the basis of `basis`, whose
        convex hull is an approximation of the near-optimal feasible
        space of the given model.

    """
    # Start by validating some of the arguments.
    # Check that the conv_method is defined.
    if conv_method not in ["volume", "centre"]:
        raise ValueError("'conv_method' argument should be 'volume' or 'centre'.")
    # Check that not too many parallel processes are called for.
    if num_parallel_solvers > max_iter:
        raise ValueError("Argument `num_parallel_solvers` is larger than `max_iter`")

    # Make sure that the debug- and cache directories exist.
    Path(debug_dir).mkdir(parents=True, exist_ok=True)
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    # Create list of pairs of networks and corresponding near-optimal spaces.
    spaces = [
        NearOpt(n, ConvexHull(s, incremental=True))
        for n, s in zip(networks, mga_spaces)
    ]

    # Compute initial intersection of mga spaces
    A = ConvexHull(intersection([F.space for F in spaces]))

    # Initialise the dataframe of points found by optimisations. Each
    # point is accompanied by the index of the network for which is
    # was computed.
    points = pd.DataFrame(columns=["net_index", *mga_spaces[0].columns])

    # Initialise proved directions. Initially empty.
    # probed_directions = pd.DataFrame(colunms=points.columns)
    probed_directions = []

    # Initialise a DataFrame for debug data for each iteration.
    iteration_data = pd.DataFrame(columns=[*points.columns, "radius", "volume"])

    num_iters = 0

    # If there are results in the cache from previous runs with the
    # same configuration, load those.
    # TODO: we should actually track all the approximations of
    # individual spaces that have been built up.
    # (
    #     previous_points,
    #     previous_directions,
    #     previous_iteration_data,
    #     num_iters,
    # ) = reuse_results(cache_dir, debug_dir, basis)

    # If we found any previous iterations (when `num_iters` is greater
    # than 0), we can just pick up where we left off.
    # if num_iters > 0:
    #     logging.info(f"Found {num_iters} previous iterations.")
    #     points = previous_points
    #     probed_directions = previous_directions
    #     iteration_data = previous_iteration_data

    # TODO: consider scaling the spaces.
    # scaling_ranges = mga_space.max() - mga_space.min()
    # # Check that we do not have any degenerate ranges. The value of 1.0
    # # is somewhat arbitrary.
    # if not (scaling_ranges > 1.0).all():
    #     raise RuntimeError("After MGA, the near-optimal space is degenerate.")
    # scaled_points = points / scaling_ranges
    # scaled_hull = ConvexHull(
    #     scaled_points.to_numpy(), incremental=True, qhull_options=qhull_options
    # )

    # If no previous iterations were found, initialise the first row
    # of the `iteration_data` DataFrame.
    if num_iters == 0:
        centre, radius, _ = ch_centre(A)
        iteration_data.loc[-1] = np.hstack((-1, centre, radius, A.volume))

    # Prepare the generator of directions to probe.

    # TODO: need to adapt _all_ methods we use here to take index into
    # account! (Optimisations in the same direction are fine as long
    # as they are in different networks.) This could actually increase
    # the complexity of generating new direction. The general strategy
    # should be something like: (in a loop)
    # 1. Generate an initial unfiltered candidate direction d,
    # 2. Find the index i of network corresponding to d,
    # 3. Filter d against probed directions _in the ith network_.
    #   3(a) Potentially try some direction d against other networks
    #        if they also have active constraints at vertex in
    #        direction d.
    # 4. Repeat from the top until direction has been found.
    if direction_method == "random-uniform":
        # Uniformly random directions.
        sampler = uniform_random_hypersphere_sampler(len(basis))
        dir_gen = filter_vectors_auto(
            sampler,
            init_angle=direction_angle_sep,
            initial_vectors=probed_directions,
            min_angle_tolerance=angle_tolerance,
        )
    elif direction_method == "facets":
        # Directions are normal vectors to the largest facets.
        dir_gen = large_facet_directions(
            A,
            probed_directions,
            direction_angle_sep,
            autodecrease=True,
            min_angle_tolerance=angle_tolerance,
        )
    elif direction_method == "maximal-centre":
        # Directions are normals to hyperplanes touched by largest centre ball.
        dir_gen = touching_ball_directions(A, probed_directions, angle_tolerance)
    elif direction_method == "maximal-centre-then-facets":
        # Directions are first picked by "maximal-centre", then
        # "facets".
        dir_gen = maximal_centre_then_facets(A, probed_directions, direction_angle_sep)
    else:
        raise ValueError("No mode of choosing directions defined.")

    # Set up a queue that can be shared among different processes.
    manager = Manager()
    queue = manager.Queue()
    results = []

    # Start a pool of worker processes. Set the child process start
    # method to spawn (as opposed to the Linux default "fork") in
    # order to avoid occasional queue deadlocks. This is also
    # supported by all platforms.
    with get_context("spawn").Pool(num_parallel_solvers) as pool:
        # Generate initial directions, and start solving in those directions.
        directions = []
        for i in range(num_parallel_solvers):
            try:
                d = next(dir_gen)
                if d is not None:
                    directions.append(d)
                    probed_directions.append(d)
                else:
                    logging.warning(
                        f"Could only generate {len(directions)} directions for"
                        f" {num_parallel_solvers} workers. Will try to generate more"
                        " after first iteration."
                    )
                    break
            except StopIteration:
                logging.warning(
                    "Ran out of directions to probe! Could only generate"
                    f" {len(directions)} directions for {num_parallel_solvers} parallel"
                    " solvers."
                )
                break

        logging.info("Finished generating initial directions.")

        for d in directions:
            # Solve in the generated directions. First figure out
            # which network we should optimise over. We use the
            # `probe_intersection` function, which:
            # 1. "Probes" the intersection in the given direction to see which
            #    constraints are binding at the vertex furthest in that
            #    direction.
            # 2. Determines which of the spaces those binding constraints
            #    belong to, and returns its index.
            # Finally, we optimise the chosen network in the given direction.
            idx = probe_intersection([F.space for F in spaces], d)
            args = (queue, spaces[idx].net, idx, d, basis, obj_bound)
            results.append(pool.apply_async(solve_worker, args))

        # Process solver results as they are put into the queue.
        while True:
            result = queue.get()
            if result is None:
                # In this case the last optimisation was unsuccessful;
                # this can happen sporadically due to, for example,
                # numerical issues.
                logging.info(
                    f"Iteration {num_iters} unsuccessful: ignoring results and repeating."
                )
            else:
                # This block is only executed if the last optimisation
                # result was successful.
                logging.info(f"Finished iteration {num_iters}.")
                num_iters += 1

                # Unpack the result: it is a point and and index for
                # which network this point was obtained.
                i, p = result

                # TODO: do we do scaling?
                # sp = list(p.values()) / scaling_ranges

                # Add the point to the convex hull of the 'i'th network.
                spaces[i].space.add_points([list(p.values())])

                # Recompute the intersection of all the spaces.
                A = ConvexHull(intersection([F.space for F in spaces]))

                # Compute new Chebyshev centre and radius. Include a sanity
                # check that the radius does not decrease!
                centre, radius, _ = ch_centre(A)
                old_radius = iteration_data.radius.iloc[-1]
                if (old_radius - radius) / old_radius > 0.001:
                    logging.info("Radius decreased. Check what is going on?")

                # Log the newly found point (together with the direction
                # that generated it), both in cache and debug directories.
                # These are non-scaled values. Note that the direction is
                # already added at this point.
                points.loc[len(points)] = [i, *p]
                for d in [cache_dir, debug_dir]:
                    points.to_csv(os.path.join(d, "points.csv"))
                    # TODO: right now "probed_directions" is a list
                    # but it should probably be a dataframe or
                    # something else keeping track of network indices
                    # too.
                    pd.DataFrame(probed_directions, columns=points.columns[1:]).to_csv(
                        os.path.join(d, "probed_directions.csv")
                    )

                # Also write additional information about the new centre
                # point, radius and volume to the debug directory. These
                # come from the scaled near-optimal space.
                iteration_data.loc[len(iteration_data)] = np.hstack(
                    (i, centre, radius, A.volume)
                )
                iteration_data.to_csv(os.path.join(debug_dir, "debug.csv"))

                # Evaluate convergence criteria. We need at least 2
                # iterations to do this.
                if num_iters >= 2:
                    if conv_method == "volume":
                        # Get array of volumes.
                        volumes = iteration_data.volume.values
                        # Compute percentage differences between iterations.
                        conv_deltas_percent = (
                            100 * (volumes[1:] - volumes[:-1]) / volumes[:-1]
                        )
                    elif conv_method == "centre":
                        # Compute a list of distances between the centres
                        # from successive iterations.
                        centres = list(iteration_data[list(points.columns)].values)
                        centre_shifts = np.array(
                            [dist(x, y) for x, y in zip(centres[1:], centres[:-1])]
                        )
                        # Calculate the percentage these distances make up
                        # of the magnitude of the centre at each
                        # iteration.
                        norms = np.array([np.linalg.norm(c) for c in centres])
                        conv_deltas_percent = 100 * centre_shifts / norms[:-1]

                    # Log the latest delta percentage.
                    logging.info(
                        "Latest convergence criteria (percent):"
                        f" {conv_deltas_percent[-1]:.3}"
                    )

                else:
                    conv_deltas_percent = []

                # If needed, we pad the `conv_deltas_percent` list in
                # order to contain at least `conv_iter` elements.
                conv_deltas_percent = np.concatenate(
                    [
                        (conv_iter - len(conv_deltas_percent)) * [np.inf],
                        conv_deltas_percent,
                    ]
                )

                # End the approximation algorithm if we converge or reach
                # the maximum number of iterations.
                conv_crit = all(
                    [d < conv_eps for d in conv_deltas_percent[-conv_iter:]]
                )
                if conv_crit or (num_iters >= max_iter):
                    logging.info("Terminating pool.")
                    pool.terminate()
                    break

            # The following is executed regardless of whether the last
            # optimisation was successful or not.

            # Add additional jobs to the queue. Most of the time we
            # should only need to start _one_ additional worker (since
            # we just finished processing the results of another
            # worker), but it is possible that the pool has more than
            # one idle workers when it was not possible to generate
            # enough directions at an earlier stage (for example when
            # our space does not have enough facets yet right in the
            # beginning).
            num_idle_workers = num_parallel_solvers - len(
                [r for r in results if not r.ready()]
            )
            if num_idle_workers > 1:
                logging.info(
                    f"Trying to generate new directions for {num_idle_workers} idle"
                    " workers."
                )
            for i in range(num_idle_workers):
                # Try generating a new direction and give it to a
                # worker.
                try:
                    dir = next(dir_gen)
                    if dir is not None:
                        probed_directions.append(dir)
                        idx = probe_intersection([F.space for F in spaces], dir)
                        args = (queue, spaces[idx].net, idx, dir, basis, obj_bound)
                        results.append(pool.apply_async(solve_worker, args))
                    else:
                        # We've (possibly temporarily) ran out of
                        # directions. Can try again after the next
                        # iteration.
                        break
                except StopIteration:
                    # In this case we really cannot generate any more
                    # directions.
                    break

            # If the worker pool is now completely idle (i.e. all the
            # results are ready), that means we have completely run
            # out of directions to probe. In that case, we end the
            # approximation.
            if all([r.ready() for r in results]):
                # All processes are done.
                pool.close()
                break

    # Log the algorithm termination.
    logging.info(f"\nApproximation ended after {num_iters} iterations.\n")
    deltas_str = np.array_str(
        np.array(conv_deltas_percent[-conv_iter:]), precision=3, suppress_small=True
    )
    logging.info(f"Last {conv_iter} deltas (percent): " + deltas_str)
    if conv_crit:
        logging.info("Conclusion: converged.")
    elif num_iters >= max_iter:
        logging.info("Conclusion: reached maximum number of iterations.")
    else:
        logging.info("Conclusion: ran out of directions to probe.")

    # Now reverse the scaling to obtain non-scaled "real" values.
    # TODO: use something like this if we decide to do scaling.
    # scaled_points = points / scaling_ranges
    # scaled_hull = ConvexHull(scaled_points.to_numpy(), qhull_options=qhull_options)
    # scaled_vertices = scaled_hull.points[scaled_hull.vertices, :]
    # vertices = scaled_vertices * scaling_ranges.values
    return spaces, A


def solve_worker(
    queue: Queue,
    n: pypsa.Network,
    idx: int,
    dir: np.array,
    basis: OrderedDict,
    obj_bound: float,
) -> None:
    """Solve a network in a given direction and put the results in a queue.

    This function returns an extreme point of the reduce near-optimal
    feasible space of `n` in the direction `dir`.

    Note that the `idx` argument isn't actually used in this function,
    but put with the extreme point on the results queue as a label.

    """
    # Log the start of this iteration. Note that we cannot really use
    # the `logging` package here since it is not process-safe, so we
    # just use a print statement. At least it lets the user know
    # what is going on.
    worker_name = multiprocessing.current_process().name
    dir_str = np.array_str(dir, precision=3, suppress_small=True)
    print(f"{worker_name}: Optimising over {idx}th space in direction {dir_str}")

    # Do the optimisation in the given direction.
    t = time.time()
    r = copy.deepcopy(n)
    status, _ = solve_network_in_direction(r, dir, basis, obj_bound)
    solve_time = round(time.time() - t)
    print(f"{worker_name}: Finishing optimisation in {solve_time} seconds.")

    # Put the result in the result queue if the optimisation was
    # successful. If unsuccessful, put a None in the queue. This may
    # happen sporadically due to, for example, numerical issues. Note
    # that we do have to put _something_ in the queue, otherwise (and
    # if there is only one parallel process) the main program loop
    # will get stuck waiting for a result.
    if status == "ok":
        queue.put((idx, get_basis_values(r, basis)))
    else:
        queue.put(None)


def reuse_results(
    cache_dir: str, debug_dir: str, basis: OrderedDict
) -> (pd.DataFrame, Collection[np.array], pd.DataFrame, int):
    """Reuse results from cache directory, if any.

    This is based on hashing the configuration excluding iterations,
    conv_epsilon, conv_iterations from near_opt_approx.

    Parameters
    ----------
    cache_dir : str
        Directory where the cache files (config, points, probed
        directions) should be loaded from.
    debug_dir : str
        Directory where the debug files (networks, volume, centre,
        radius, probed directions) should be loaded from.
    basis : OrderedDict
        A basis on which to project the feasible space of the network.
        The keys being the decision variables with values given in the
        format produced by `pypsa.linopt.linexpr`.

    Returns
    -------
    pd.DataFrame
        A collection of points in the basis of `basis` that were
        previously generated with this configuration.
    Collection[np.array]
        Previously probed directions.
    pd.DataFrame
        Previous iteration data (centre, radius, volume).
    int
        Number of previous iterations with this configuration.
    """
    try:
        # Read points and directions files.
        points = pd.read_csv(os.path.join(cache_dir, "points.csv"), index_col=0)
        probed_directions = pd.read_csv(
            os.path.join(cache_dir, "probed_directions.csv"), index_col=0
        )
        previous_debug = pd.read_csv(os.path.join(debug_dir, "debug.csv"), index_col=0)
        probed_directions = list(probed_directions.values)
        # Calculate the number of iterations that were performed. Note
        # that `probed_directions` (and `points`) contain data from
        # MGA iterations, so we subtract the number of MGA iterations
        # (which is 2*`len(basis)`).
        previous_iters = max(0, len(probed_directions) - 2 * len(basis))
    except OSError:
        logging.info(
            "No previous runs with this configuration found. Start with num_iters = 0."
        )
        return None, None, None, 0

    try:
        # Also try to read previous debug information if possible.
        previous_debug = pd.read_csv(os.path.join(debug_dir, "debug.csv"), index_col=0)
    except (OSError, TypeError):
        logging.info("Could not find debug data, use only cached data.")
        previous_debug = None
    return points, probed_directions, previous_debug, previous_iters


if __name__ == "__main__":
    # Set up logging so that everything is written to the right log file.
    configure_logging(snakemake)

    # Disable logging from pypsa; it mostly just distracts for this script.
    pypsa_logger = logging.getLogger("pypsa")
    pypsa_logger.setLevel(logging.WARNING)

    # Load the networks.
    networks = [pypsa.Network(fn) for fn in snakemake.input.networks]

    # Attach solving configuration to the network.
    for n in networks:
        n.config = snakemake.config["pypsa-eur"]

    # Load the "MGA" approximations for the near-optimal feasible
    # spaces of above networks. Note: network and mga space inputs
    # lists are assumed to be given in matching orders.
    mga_spaces = [pd.read_csv(fn, index_col=0) for fn in snakemake.input.mga_spaces]

    # Depending on the 'eps' wildcard, determine the cutoff of the
    # near-optimal feasible space.
    with open(snakemake.input.obj_bound, "r") as f:
        obj_bound = float(f.read())

    # Compute the near-optimal feasible space.
    spaces, A = compute_intersection_direct(
        networks,
        mga_spaces,
        obj_bound=obj_bound,
        basis=snakemake.config["projection"],
        conv_method=snakemake.config["near_opt_approx"]["conv_method"],
        direction_method=snakemake.config["near_opt_approx"]["directions"],
        direction_angle_sep=snakemake.config["near_opt_approx"][
            "directions_angle_separation"
        ],
        conv_eps=float(snakemake.config["near_opt_approx"]["conv_epsilon"]),
        conv_iter=int(snakemake.config["near_opt_approx"]["conv_iterations"]),
        max_iter=int(
            snakemake.config["near_opt_approx"].get(
                "direct_intersection_iterations",
                snakemake.config["near_opt_approx"]["iterations"],
            )
        ),
        debug_dir=snakemake.log.iterations,
        cache_dir=snakemake.log.cache,
        num_parallel_solvers=snakemake.config["near_opt_approx"].get(
            "direct_intersection_parallel_solvers",
            snakemake.config["near_opt_approx"]["num_parallel_solvers"],
        ),
        qhull_options=snakemake.config["near_opt_approx"].get("qhull_options", None),
        angle_tolerance=snakemake.config["near_opt_approx"].get("angle_tolerance", 0.1),
    )

    # Put the vertices of `intersection` (which is given as a
    # ConvexHull) in a DataFrame with properly labelled columns.
    intersection_df = pd.DataFrame(A.points, columns=mga_spaces[0].columns)

    # Write the points defining the intersection to the given output
    # file.
    intersection_df.to_csv(snakemake.output.intersection)

    # Also compute the centre point and radius and output those.
    centre, radius, _ = ch_centre(A)
    centre_df = pd.DataFrame(centre).T
    centre_df.columns = mga_spaces[0].columns
    centre_df.to_csv(snakemake.output.centre)
    with open(snakemake.output.radius, "w") as f:
        f.write(str(radius))
