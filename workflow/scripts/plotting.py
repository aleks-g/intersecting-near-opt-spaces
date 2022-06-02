# SPDX-FileCopyrightText: 2022 Koen van Greevenbroek & Aleksander Grochowicz
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Plotting functions for convex hulls."""

from typing import Collection

import matplotlib as mpl
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as a3
import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
from sympy import Plane, Point3D


def prepare_3d_plot(
    box_dims: Collection[float],
    labels: Collection[str],
    equal_aspect: bool = False,
    rotation: Collection[float] = [45, 20],
):
    """Return a figure with 3D axis.

    Arguments
    ---------
    box_dims : Collection[float]
        Determines the size of the 3d plot, e.g. through a max method on points.
    labels : Collection[str]
        Labels the axes.
    equal_aspect : bool (default = False)
        Activates equal aspect of 3d plot with equal distances on axes.
    rotation : Collection[float]
        Allows rotations for different perspectives of the plot. The
        first entry ("azimuth") corresponds to rotation around the
        vertical z-axis, the second ("elevation") is the angle to the
        x-y-plane.

    """
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.dist = 10
    ax.azim = rotation[0]
    ax.elev = rotation[1]
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_zlabel(labels[2])
    ax.set_xlim([0, box_dims[0]])
    ax.set_ylim([0, box_dims[1]])
    ax.set_zlim([0, box_dims[2]])
    if equal_aspect:
        ax.set_box_aspect(aspect=tuple(box_dims))
    return fig, ax


def plot_conv_hull_proj_simple(
    points: pd.DataFrame, columns: Collection[str], ax, **kwargs
):
    """Plot a convex hull, simple but performant."""
    hull = ConvexHull(points.loc[:, columns].to_numpy())
    xs = hull.points[:, 0]
    ys = hull.points[:, 1]
    zs = hull.points[:, 2]
    tri = mpl.tri.Triangulation(xs, ys, triangles=hull.simplices)
    ax.plot_trisurf(tri, zs, **kwargs)


# The following functions follow https://stackoverflow.com/a/49115448.
# It works, but suffers of performance issues.


def plot_conv_hull_proj(
    points: pd.DataFrame,
    columns: Collection[str],
    ax,
    optimum=None,
    centre=None,
):
    """Create a 3D plot of a convex hull within an enclosing box.

    Follows https://stackoverflow.com/a/49115448. Works, but suffers
    of performance issues.

    Arguments
    ---------
    points: pd.DataFrame
        The points whose convex hull is to be plotted.
    columns: Collection[str]
        The columns of `points` to select. len(columns) must be exactly 3.
    ax: matplotlib axes
        Axes to plot the convex hull to.
    optimum: np.array = None
        An optional addition point to plot and mark as the optimum.
    centre: np.array = None
        An optional addition point to plot and mark as the centre.

    """
    # Project the points down to the given components.
    if len(columns) != 3:
        raise ValueError(f"Passed {len(columns)} columns to project onto, not 3.")
    verts = points.loc[:, columns].to_numpy()

    # Construct the convex hull.
    hull = ConvexHull(verts)
    faces = hull.simplices

    # May or may not want to plot the optimum
    if optimum is not None:
        ax.plot3D(
            xs=optimum[0],
            ys=optimum[1],
            zs=optimum[2],
            marker="x",
            color="black",
            label="optimal point",
        )

        ax.plot3D(
            xs=[0, optimum[0]],
            ys=[optimum[1], optimum[1]],
            zs=[optimum[2], optimum[2]],
            ls=":",
            color="grey",
        )
        ax.plot3D(
            xs=[optimum[0], optimum[0]],
            ys=[0, optimum[1]],
            zs=[optimum[2], optimum[2]],
            ls=":",
            color="grey",
        )
        ax.plot3D(
            xs=[optimum[0], optimum[0]],
            ys=[optimum[1], optimum[1]],
            zs=[0, optimum[2]],
            ls=":",
            color="grey",
        )

    # Likewise, optionally plot the centre point.
    if centre:
        ax.plot3D(
            xs=centre[0],
            ys=centre[1],
            zs=centre[2],
            marker="o",
            color="red",
            label="central point",
        )
        ax.plot3D(
            xs=[0, centre[0]],
            ys=[centre[1], centre[1]],
            zs=[centre[2], centre[2]],
            ls="-.",
            color="red",
        )
        ax.plot3D(
            xs=[centre[0], centre[0]],
            ys=[0, centre[1]],
            zs=[centre[2], centre[2]],
            ls="-.",
            color="red",
        )
        ax.plot3D(
            xs=[centre[0], centre[0]],
            ys=[centre[1], centre[1]],
            zs=[0, centre[2]],
            ls="-.",
            color="red",
        )

    triangles = []
    for s in faces:
        sq = [
            (verts[s[0], 0], verts[s[0], 1], verts[s[0], 2]),
            (verts[s[1], 0], verts[s[1], 1], verts[s[1], 2]),
            (verts[s[2], 0], verts[s[2], 1], verts[s[2], 2]),
        ]
        triangles.append(sq)

    new_faces = simplify(triangles)
    for sq in new_faces:
        f = a3.art3d.Poly3DCollection([list(sq)])
        f.set_color(colors.rgb2hex(np.random.rand(3)))
        f.set_edgecolor("grey")
        f.set_alpha(0.1)
        ax.add_collection3d(f)


def reorder(vertices):
    """Reorder nodes into a "hull" of the input nodes.

    Note:
    -----
    Not tested on edge cases, and likely to break.
    Probably only works for convex shapes.

    """
    if len(vertices) <= 3:
        # Just a triangle.
        return vertices
    else:
        # Take random vertex (here simply the first).
        reordered = [vertices.pop()]
        # Get next closest vertex that is not yet reordered.
        # Repeat until only one vertex remains in original list.
        vertices = list(vertices)
        while len(vertices) > 1:
            idx = np.argmin(get_distance(reordered[-1], vertices))
            v = vertices.pop(idx)
            reordered.append(v)
        # Add remaining vertex to output.
        reordered += vertices
        return reordered


def is_adjacent(a, b):
    """Test if two triangles, given as points, are adjacent."""
    # Triangles sharing 2 points and hence a side are adjacent.
    return len(set(a) & set(b)) == 2


def is_coplanar(a, b, tolerance_in_radians=0):
    """Test if two triangles are coplanar within a given tolerance."""
    a1, a2, a3 = a
    b1, b2, b3 = b
    plane_a = Plane(Point3D(a1), Point3D(a2), Point3D(a3))
    plane_b = Plane(Point3D(b1), Point3D(b2), Point3D(b3))
    # If we only accept exact results:
    if not tolerance_in_radians:
        return plane_a.is_coplanar(plane_b)
    else:
        angle = plane_a.angle_between(plane_b).evalf()
        # Make sure that angle is between 0 and np.pi
        angle %= np.pi
        return (angle - tolerance_in_radians <= 0.0) or (
            (np.pi - angle) - tolerance_in_radians <= 0.0
        )


def flatten(ls):
    """Flatten a list of lists."""
    return [item for sublist in ls for item in sublist]


def get_distance(v1, v2):
    """Return Euclidean distance between two points."""
    v2 = np.array(list(v2))
    difference = v2 - v1
    ssd = np.sum(difference**2, axis=1)
    return np.sqrt(ssd)


def simplify(triangles):
    """Simplify an iterable of triangles, joining faces.

    Each pair of triangles which is adjacent and coplanar is
    simplified to a single face. Each triangle is a set of 3 points in
    3D space.

    """
    # Create a graph in which nodes represent triangles; nodes are
    # connected if the corresponding triangles are adjacent and
    # coplanar.
    G = nx.Graph()
    G.add_nodes_from(range(len(triangles)))
    for ii, a in enumerate(triangles):
        for jj, b in enumerate(triangles):
            # Test relationships only in one way as adjacency and
            # co-planarity are commutative.
            if ii < jj:
                if is_adjacent(a, b):
                    if is_coplanar(a, b, np.pi / 180.0):
                        G.add_edge(ii, jj)

    # triangles that belong to a connected component can be combined
    components = list(nx.connected_components(G))
    simplified = [
        set(flatten(triangles[index] for index in component))
        for component in components
    ]

    # need to reorder nodes so that patches are plotted correctly
    reordered = [reorder(face) for face in simplified]

    return reordered
