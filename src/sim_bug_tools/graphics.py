"""
Visualization tools
"""
import itertools
import warnings

import matplotlib.axes
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import networkx as nx
import numpy as np
import pandas as pd
import scipy.spatial
import sklearn.decomposition
from matplotlib.path import Path
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib import patches
from mpl_toolkits.mplot3d import Axes3D
from numpy import ndarray

import sim_bug_tools.structs as structs
import sim_bug_tools.utils as utils
import sim_bug_tools.constants as constants
from sim_bug_tools.structs import Domain, Point
from treelib import Tree, Node
from sim_bug_tools.exploration.brrt_std.brrt import DATA_LOCATION
from itertools import combinations, product


class Grapher:
    def __init__(
        self,
        is3d=False,
        domain: Domain = None,
        axes_titles: list[str] = None,
        style="standard",
    ):
        self._fig = plt.figure(**constants.DEFAULT_FIG_CONFIG)
        self._ax: Axes = (
            self._fig.add_subplot(111, projection="3d")
            if is3d
            else self._fig.add_subplot()
        )
        if domain is not None:
            self._ax.set_xlim(domain[0])
            self._ax.set_ylim(domain[1])
            if is3d:
                self._ax.set_zlim(domain[2])

        if axes_titles is not None:
            self._ax.set_xlabel(axes_titles[0])
            self._ax.set_ylabel(axes_titles[1])
            if is3d:
                self._ax.set_zlabel(axes_titles[2])

        print(self._ax)
        self._is3d = is3d

        if style == "standard":
            plt.subplots_adjust(**constants.DEFAULT_PLOT_CONFIG)

    @property
    def ax(self):
        return self._ax

    def set_title(self, name: str):
        self._ax.set_title(name)

    def fig(self):
        return self._fig

    def add_all_arrows(self, locs: list[Point], directions: list[ndarray], **kwargs):
        "Uses matplotlib.pyplot.quiver. Kwargs will be passed to ax.quiver"
        arrows = zip(*map(lambda l, d: np.append(l.array, d), locs, directions))
        return self._ax.quiver(*arrows, **kwargs)

    def add_arrow(self, loc: Point, direction: ndarray, **kwargs):
        return self._ax.quiver(*loc, *direction, **kwargs)

    def plot_point(self, loc: Point, **kwargs):
        return self._ax.scatter(*loc, **kwargs)

    def plot_all_points(self, locs: list[Point], **kwargs):
        return self._ax.scatter(*np.array(locs).T, **kwargs)

    def draw_sphere(self, loc: Point, radius: float, **kwargs):
        return (
            self._draw_3d_sphere(loc, radius, **kwargs)
            if self._is3d
            else self._draw_2d_circle(loc, radius, **kwargs)
        )

    def _draw_3d_sphere(self, loc: Point, radius: float, **kwargs):
        u, v = np.mgrid[0 : 2 * np.pi : 20j, 0 : np.pi : 10j]
        x = loc[0] + radius * np.cos(u) * np.sin(v)
        y = loc[1] + radius * np.sin(u) * np.sin(v)
        z = loc[2] + radius * np.cos(v)
        return self._ax.plot_wireframe(x, y, z, **kwargs)

    def _draw_2d_circle(self, loc: Point, radius: float, **kwargs):
        circle = patches.Circle(loc, radius=radius, **kwargs)
        return self._ax.add_patch(circle)

    def draw_cube(self, domain: Domain):
        r = [0, 1]
        for s, e in combinations(np.array(list(product(r, r, r))), 2):
            if np.sum(np.abs(s - e)) == r[1] - r[0]:
                p1 = Domain.translate_point_domains(
                    Point(s), Domain.normalized(3), domain
                )
                p2 = Domain.translate_point_domains(
                    Point(e), Domain.normalized(3), domain
                )

                self.ax.plot3D(*zip(p1.array, p2.array), color="b")

    def draw_tree(self, tree: Tree, **kwargs):
        stack: list[Node] = [(tree.get_node(0), tree.children(0))]

        while len(stack) > 0:
            parent, children = stack.pop()
            parent_point = parent.data[DATA_LOCATION]

            for child in children:
                self._ax.plot(
                    *zip(parent_point, child.data[DATA_LOCATION]), "bo", linestyle="-"
                )
                stack.append((child, tree.children(child.identifier)))

    def draw_path(self, vertices: list[Point], typ="--", **kwargs):
        # codes = [Path.MOVETO] + [Path.LINETO] * (len(vertices) - 1)
        # path = Path(vertices, codes)
        # patch = patches.PathPatch(path, facecolor='none', lw=2)

        # self._ax.add_patch(patch)

        self._ax.plot(*np.array(vertices).T, typ, **kwargs)

    def set_yformat(self, fmtstr: str):
        current_yval = self.ax.get_yticks()
        self.ax.set_yticklabels([fmtstr.format(x) for x in current_yval])

    def save(self, path: str):
        self._fig.savefig(path)


class Voronoi(scipy.spatial.Voronoi):
    def __init__(self, points: list[structs.Point], bugs: list[bool]):
        """
        A version of the scipy voronoi plot with bug clusters.
        Parent class documentation:
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.Voronoi.html

        -- Parameters --
        points : list[Point]
            Iterable of points
        bugs : list[bool]
            Iterable of bugs that correspect to the points.
        """
        assert len(points) == len(bugs)
        super().__init__(np.array([p.array for p in points]))

        self._dimension = np.int32(len(points[0]))
        self._bugs = bugs
        self._bug_indices = list(itertools.compress(range(len(bugs)), bugs))
        self._bug_graph = self._init_bug_graph()

        # self.test()

    @property
    def bugs(self) -> list[bool]:
        return self._bugs

    @property
    def bug_indices(self) -> list[int]:
        return self._bug_indices

    @property
    def bug_graph(self) -> nx.Graph:
        return self._bug_graph

    @property
    def dimension(self) -> np.int32:
        return self._dimension

    def _filter_bug_edges(self) -> list[int]:
        """
        Selects and returns the edges around bug regions.

        -- Return --
        list[int]
            Edge indices which border bug regions
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # Get the vertices of the bug regions
            bug_regions = [
                self.regions[bpr]
                for bpr in [self.point_region[bri] for bri in self.bug_indices]
            ]

            # Remove bug clusters that are on the outside.
            temp = []
            for br in bug_regions:
                if not np.any(np.isin(-1, br)):
                    temp.append(np.array(br))
            bug_regions = np.array(temp)

            # Get edges of the bug regions
            bug_edges = np.concatenate(
                np.array(
                    list(
                        map(
                            lambda br: [
                                (br[i], br[(i + 1) % len(br)]) for i in range(len(br))
                            ],
                            bug_regions,
                        )
                    )
                )
            )

            # Filter uniqe edges
            bug_edges = utils.filter_unique(bug_edges)

        return bug_edges

    def _init_bug_graph(self) -> nx.Graph:
        """
        Intializes the bug graph.

        -- Return --
        nx.Graph
            Bug envelope in graph representation.
        """
        bug_edges = self._filter_bug_edges()

        # Create a graph
        graph = nx.Graph()

        # Add nodes
        graph.add_nodes_from(
            [(node, {"point": self.vertices[node]}) for node in np.unique(bug_edges)]
        )

        # Add edges
        graph.add_edges_from(bug_edges)

        return graph


def top2pca(points: list[structs.Point]) -> list[structs.Point]:
    """
    Reduces the dimension of a list of point by representing the data
    with the first two principle components.

    -- Parameters --
    points : list[structs.Point]
        Data Points of n-dimensions.

    -- Return --
    list[structs.Point]
        Dimensionally reduced versions of input points.
    """
    pca = sklearn.decomposition.PCA()
    X = pca.fit_transform(np.array([p.array for p in points]))
    score = X[:, 0:2]
    return [structs.Point(s) for s in score]


def new_axes() -> matplotlib.axes.Axes:
    """
    Creates a new plot and axes

    -- Return --
    matplotlib.axes
    """
    plt.figure(figsize=(5, 5))
    return plt.axes()


def apply_pc2_labels_and_limits(ax: matplotlib.axes.Axes):
    """
    Sets the limits of the axes to [-1,1] and gives the lables of PC1 and PC2.

    -- Arguments --
    ax : matplotlib.axes.Axes
        Axes to apply on
    """
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2", labelpad=-3)
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    return


def plot_bugs(points: list[structs.Point], bugs: list[bool]) -> matplotlib.axes:
    """
    Plot bugs using the first two principle components.

    -- Arguments --
    points : list[structs.Point]
        List of points of 2+ dimensions.
    bugs : list[bool]
        List of bug classification

    -- Return --
    Axis of plot
    """
    points = top2pca(points)  # Reduce to 2 dimensions.

    ax = new_axes()

    colors = ["black", "red"]
    for i, is_bug in enumerate(bugs):
        ax.plot(points[i][0], points[i][1], color=colors[int(is_bug)], marker=".")

    black_patch = mpatches.Patch(color=colors[0], label="Not a bug.")
    red_patch = mpatches.Patch(color=colors[1], label="Bug")
    ax.legend(handles=[red_patch, black_patch])

    apply_pc2_labels_and_limits(ax)
    return ax


def plot_voronoi_only(voronoi: Voronoi) -> matplotlib.axes:
    """
    Plot a voronoi diagram with the lines ones to an axis.

    -- Parameters --
    voronoi : Voronoi
        Voronoi object of 2 dimensions.

    -- Return --
    Plot axis.
    """
    assert voronoi.dimension == 2
    ax = new_axes()

    scipy.spatial.voronoi_plot_2d(
        voronoi, ax, show_vertices=False, show_points=False, line_alpha=1
    )

    apply_pc2_labels_and_limits(ax)
    return ax


def plot_voronoi_bug_envelope(voronoi: Voronoi) -> matplotlib.axes:
    """
    Plots the voronoi diagram.
    """
    assert voronoi.dimension == 2
    ax = new_axes()

    scipy.spatial.voronoi_plot_2d(
        voronoi, ax, show_vertices=False, show_points=False, line_alpha=0.7
    )

    # print( len(voronoi.points) )

    edges = voronoi.bug_graph.edges()
    for edge in edges:
        line = [voronoi.vertices[node] for node in edge]
        x = [xy[0] for xy in line]
        y = [xy[1] for xy in line]
        ax.plot(x, y, color="red")
        continue

    red_patch = mpatches.Patch(color="red", label="Bug Envelope")
    ax.legend(handles=[red_patch])

    apply_pc2_labels_and_limits(ax)
    return ax


def redudancy_table(points: list[structs.Point], bugs: list[bool]) -> pd.DataFrame:
    """
    Generates a bug redundancy table to count unique points

    -- Parameters --
    points : list[structs.Point]
        List of points
    bugs : list[bool]
        List of bugs. Must be the same length as points

    -- Return --
    pf.DataFrame
        Redudancy table
    """
    assert len(points) == len(bugs)

    bugs_amt = 0
    n_bugs = []
    repeated_hits_tracker = dict()
    repeated_hits = []
    for i in range(len(points)):
        key = str(points[i])
        try:
            repeated_hits_tracker[key] += 1
        except KeyError:
            repeated_hits_tracker[key] = 1
        rh = repeated_hits_tracker[key]
        repeated_hits.append(rh)
        if bugs[i] and rh == 1:
            bugs_amt += 1
        n_bugs.append(bugs_amt)

    data = {
        "point": points,
        "is_bug": bugs,
        "n_repeated_hits": repeated_hits,
        "n_bugs": n_bugs,
    }

    return pd.DataFrame(data)


class Signal:
    def __init__(
        self, time: list[float], amplitude: list[float], on_after: float = 0.5
    ):
        assert len(time) == len(amplitude)
        self._time = time
        self._amplitude = amplitude
        self._on_after = self._on_after
        self._is_on = [amp >= on_after for amp in amplitude]
        return

    @property
    def time(self) -> list[float]:
        return self._time

    @property
    def amplitude(self) -> list[float]:
        return self._amplitude

    @property
    def on_after(self) -> float:
        return self._on_after

    @property
    def is_on(self) -> list[bool]:
        return self._is_on


# def plot


def new_signal_axes() -> matplotlib.axes.Axes:
    """
    Creates a new plot and axes

    -- Return --
    matplotlib.axes
    """
    plt.figure(figsize=(5, 1))
    return plt.axes()


def plot_signal(time: list[float], is_bug: list[bool]) -> matplotlib.axes.Axes:
    assert len(time) == len(is_bug)
    ax = new_signal_axes()
    time = np.array(time, dtype=int)
    is_bug = np.array(is_bug, dtype=int)
    ax.plot(time, is_bug, color="black")
    return ax


if __name__ == "__main__":
    g = Grapher(domain=Domain.normalized(2), axes_titles=["First", "Second"])
    g.set_title("Test Title")

    plt.show()
