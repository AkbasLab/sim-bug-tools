from time import time
from typing import Callable

import numpy as np
from numpy import ndarray
from rtree import index
from treelib import Node, Tree

from sim_bug_tools.exploration.boundary_core.adherer import AdherenceFactory
from sim_bug_tools.exploration.boundary_core.explorer import Explorer
from sim_bug_tools.structs import Domain, Point, Spheroid

from .adherer import BoundaryAdherenceFactory

DATA_LOCATION = "location"
DATA_NORMAL = "normal"


class BoundaryRRT(Explorer):
    """
    The Boundary RRT (BRRT) Strategy provides a means of finding a boundary and
    following that boundary for a given number of desired boundary samples.
    The time complexity of this strategy is $O(n)$ where $n$ is the number of
    desired estimated boundary points.
    """

    def __init__(self, b0: Point, n0: ndarray, adhererF: AdherenceFactory):
        """
        Args:
            classifier (Callable[[Point], bool]): The function that determines whether
                or not a sampled point is a target value or not.
            b0 (Point): The root boundary point to begin exploration from.
            n0 (ndarray): The root boundary point's orthonormal surface vector.
            adhererF (AdherenceFactory): A factory for the desired adherence
                strategy.
        """
        super().__init__(b0, n0, adhererF)

        self._ndims = len(b0)

        self._tree = Tree()
        self._root = Node(identifier=0, data=self._create_data(*self.prev))
        self._next_id = 1

        p = index.Property()
        p.set_dimension(self._ndims)
        self._index = index.Index(properties=p)

        self._index.insert(0, b0)
        self._tree.add_node(self._root)

        self._prev_dir: ndarray = None

    @property
    def previous_node(self) -> Node:
        return self._tree.get_node(self._next_id - 1)

    @property
    def previous_direction(self) -> ndarray:
        return self._prev_dir

    def _select_parent(self) -> tuple[Point, ndarray]:
        self._r = self._random_point()
        self._parent = self._find_nearest(self._r)
        self._p = self._parent.data[DATA_LOCATION]
        return self._parent.data[DATA_LOCATION], self._parent.data[DATA_NORMAL]

    def _pick_direction(self) -> ndarray:
        return (self._r - self._p).array

    def _add_child(self, bk: Point, nk: ndarray):
        self._add_node(bk, nk, self._parent.identifier)

    def _add_node(self, p: Point, n: ndarray, parentID: int):
        node = Node(identifier=self._next_id, data=self._create_data(p, n))
        self._tree.add_node(node, parentID)
        self._index.insert(self._next_id, p)
        self._next_id += 1

    def _random_point(self) -> Point:
        return Point(np.random.rand(self._ndims) * 2)

    def _find_nearest(self, p: Point) -> Node:
        node = self._tree.get_node(next(self._index.nearest(p)))

        return node

    @staticmethod
    def _create_data(location, normal) -> dict:
        return {DATA_LOCATION: location, DATA_NORMAL: normal}


def measure_time(f: Callable, *args, **kwargs) -> float:
    t0 = time()
    r = f(*args, **kwargs)
    t1 = time()
    return r, t1 - t0


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes

    from sim_bug_tools.exploration.boundary_core.surfacer import find_surface
    from sim_bug_tools.graphics import Grapher
    

    ndims = 3

    # Jump Distance and Change in Angle
    d = 0.03
    theta = 5 * np.pi / 180

    # The spherical target envelope
    loc = Point([0.5 for x in range(ndims)])
    radius = 0.25
    classifier = lambda p: p.distance_to(loc) <= radius
    domain = Domain.normalized(ndims)

    print("Building brrt...")
    bpair, interm = find_surface(classifier, loc, d)
    sphere_scaler = Spheroid(d)
    adhF = BoundaryAdherenceFactory(classifier, domain, sphere_scaler, theta)
    brrt = BoundaryRRT(*bpair, adhF)

    # The series of points that were sampled to reach the surface

    g = Grapher(True)
    g.ax.set_xlim([0, 1])
    g.ax.set_ylim([0, 1])
    g.ax.set_zlim([0, 1])

    g.plot_all_points(interm)
    plt.pause(0.01)
    # Setup
    num_iterations = 1000
    batch = 100
    num_points = 0
    i = 0
    points = []

    print("Launching...")

    boundary_points = []
    other_point_count = 0
    osvas = []
    errs = 0

    while i < num_iterations:
        pk, nk = brrt.expand()
        # if i % 50 == 0:
        #     print(f"Iteration #{i} started...")
        boundary_points.append(pk)
        osvas.append(nk)

        i += 1

    g.plot_all_points(boundary_points)
    plt.show()
