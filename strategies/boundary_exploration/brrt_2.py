"""
Uses the BoundaryAdherer from adherer_2
"""
from copy import copy
from time import time
from typing import Callable

import numpy as np
from numpy import ndarray
from rtree import index
from sim_bug_tools.rng.lds.sequences import RandomSequence, Sequence
from sim_bug_tools.structs import Domain, Point
from treelib import Node, Tree

from adherer_2 import BoundaryAdherer

DATA_LOCATION = "location"
DATA_NORMAL = "normal"


class BoundaryRRT:
    """
    The Boundary RRT (BRRT) Strategy provides a means of finding a boundary and
    following that boundary for a given number of desired boundary samples.
    The time complexity of this strategy is $O(n)$ where $n$ is the number of
    desired estimated boundary points.
    """

    def __init__(
        self,
        classifier: Callable[[Point], bool],
        t0: Point,
        d: float,
        delta_theta: float,
        rate: float,
        num: int,
    ):
        """
        Args:
            classifier (Callable[[Point], bool]): The function that determines whether
                or not a sampled point is a target value or not.
            t0 (Point): An initial target value within the target envelop whose surface
                is to be explored.
            d (float): The jump distance between estimated boundary points.
            theta (float): How much to rotate by for crossing the boundary.
        """
        self._classifier = classifier
        self._d = d
        self._theta = delta_theta
        self._rate = rate
        self._num = num
        self._ndims = len(t0)

        p0, n0 = self._surface(t0)

        self._tree = Tree()
        self._root = Node(identifier=0, data=self._create_data(p0, n0))
        self._next_id = 1

        p = index.Property()
        p.set_dimension(self._ndims)
        self._index = index.Index(properties=p)

        self._index.insert(0, p0)
        self._tree.add_node(self._root)

    def grow(self) -> tuple[Point, ndarray]:
        """
        Grows the RRT; i.e., Finds a new boundary point to add to the tree.
        If the boundary is lost, it will attempt again infinitely many
        times (Warning: not guaranteed to halt.)

        Returns:
            tuple[Point, ndarray]: The newly added point on the surface
                and its estimated orthogonal surface vector

        Throws:
            BoundaryLostException: if the boundary was not reacquired by
                the boundary adherence algorithm.
        """
        r = self._random_point()
        parent = self._find_nearest(r)

        return self.growFrom(parent, r)

    def growFrom(self, node: Node, r: Point):
        """
        Grows off a specified node within the tree.

        Args:
            node (Node): The node to grow from
            r (Point): The vector to move towards

        Throws:
            BoundaryLostException: if the boundary was not reacquired by
                the boundary adherence algorithm.
        """
        p, n = node.data[DATA_LOCATION], node.data[DATA_NORMAL]

        ba = BoundaryAdherer(
            self._classifier,
            p,
            n,
            (r - p).array,
            self._d,
            self._theta,
            self._rate,
            self._num,
        )

        ba.find_boundary()
        pk, nk = ba.bn
        self._add_node(pk, nk, node.identifier)

        return pk, nk

    def growBy(self, num: int) -> list[tuple[Point, ndarray]]:
        """
        Grows the RRT by the provided number of samples.

        Args:
            num (int, optional): The number of boundary points to find and
                expand the RRT by.

        Returns:
            list[tuple[Point, ndarray]]: A list of boundary points and their
                estimated orthogonal surface vectors.
        """
        data = []
        i = 0
        while i < num:
            try:
                data += [self.grow()]
                i += 1
            except:
                pass
        return data

    def _add_node(self, p: Point, n: ndarray, parentID: int):
        node = Node(identifier=self._next_id, data=self._create_data(p, n))
        self._tree.add_node(node, parentID)
        self._index.insert(self._next_id, p)
        self._next_id += 1

    def _random_point(self) -> Point:
        return Point(np.random.rand(self._ndims))

    def _find_nearest(self, p: Point) -> Node:
        node = self._tree.get_node(next(self._index.nearest(p)))

        return node

    def _surface(self, t0):
        v = np.random.rand(len(t0))

        s = v * self._d

        self._interm = [t0]

        prev = None
        cur = t0

        while self._classifier(cur):
            prev = cur
            self._interm += [prev]
            cur = prev + Point(s)

        while not self._classifier(cur):
            print("Getting closer...")
            s *= 0.5
            cur = prev + Point(s)

        self._interm += [prev]

        return prev, v

    @staticmethod
    def _create_data(location, normal) -> dict:
        return {DATA_LOCATION: location, DATA_NORMAL: normal}


def measure_time(f: Callable, *args, **kwargs) -> float:
    t0 = time()
    r = f(*args, **kwargs)
    t1 = time()
    return r, t1 - t0


if __name__ == "__main__":
    # A simple test for showing the strategy in action
    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes

    ndims = 3

    # Jump Distance and Change in Angle
    d = 0.015
    theta = 5 * np.pi / 180

    # The spherical target envelope
    loc = Point([0.5 for x in range(ndims)])
    radius = 0.25
    classifier = lambda p: p.distance_to(loc) <= radius

    print("Building brrt...")
    brrt = BoundaryRRT(classifier, loc, d, theta, 2, 5)

    # The series of points that were sampled to reach the surface
    path_points = np.array(brrt._interm)

    fig3d = plt.figure()

    # 3D View
    ax3d: Axes = fig3d.add_subplot(111, projection="3d")
    ax3d.set_xlim([0, 1])
    ax3d.set_ylim([0, 1])
    ax3d.set_zlim([0, 1])
    ax3d.scatter(*path_points.T)
    plt.pause(0.01)

    # Setup
    batch = 10
    num_points = 0
    i = 0
    points = []

    print("Launching...")
    while True:
        # nodes, t = measure_time(brrt.growBy, batch)
        data = brrt.growBy(100)

        points += [p for p, n in data]

        # num_points += batch
        ax3d.scatter(*np.array(points).squeeze().T, c="red")
        points = []
        plt.pause(0.01)

        input("Press Enter...")
        i += 1
