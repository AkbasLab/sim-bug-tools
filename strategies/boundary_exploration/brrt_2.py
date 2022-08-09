"""
Uses the BoundaryAdherer from adherer_2
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from copy import copy
from time import time
from typing import Callable

import numpy as np
from numpy import ndarray
from rtree import index
from sim_bug_tools.rng.lds.sequences import RandomSequence, Sequence
from sim_bug_tools.structs import Domain, Point
from tools.grapher import Grapher
from treelib import Node, Tree

from adherer_2 import BoundaryAdherer

DATA_LOCATION = "location"
DATA_NORMAL = "normal"

g: Grapher = Grapher(True, Domain.normalized(3))

import matplotlib.pyplot as plt
from matplotlib.axes import Axes

ndims = 3

# Jump Distance and Change in Angle
d = 0.015
theta = 90 * np.pi / 180

# The spherical target envelope
loc = Point([0.5 for x in range(ndims)])
radius = 0.25
classifier = lambda p: p.distance_to(loc) <= radius



# Setup
batch = 10
num_points = 0
i = 0

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
        
        self._debug = None

    def grow(self, getAllPoints: bool = False) -> tuple[Point, ndarray]:
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

        return self.growFrom(parent, r, getAllPoints)

    def growFrom(self, node: Node, r: Point, getAllPoints: bool = False):
        """
        Grows off a specified node within the tree.

        Args:
            node (Node): The node to grow from
            r (Point): The vector to move towards

        Throws:
            BoundaryLostException: if the boundary was not reacquired by
                the boundary adherence algorithm.
        """
        all_points = []
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

        b, all_points = ba.find_boundary(True)
        pk, nk = ba.bn       
        if getAllPoints:
            results = (pk, nk, all_points)
        else:
            results = (pk, nk)
        
        self._add_node(pk, nk, node.identifier)


        return results

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
    print("Building brrt...")
    r = 10
    N = 1
    brrt = BoundaryRRT(classifier, loc, d, theta, r, N)
    path_points = np.array(brrt._interm)
    g.plot_all_points(path_points)
    g.create_sphere(loc, radius)
    plt.pause(0.01)    

    print("Launching...")
    # while True:
    #     # nodes, t = measure_time(brrt.growBy, batch)
    #     data = brrt.growBy(100)
    #     points = [p for p, n in data]
    #     # points = [data[0]]
    #     norms = [n for p, n in data]
    #     # norms = [data[1]]

    #     # num_points += batch
    #     g.plot_all_points(points)
    #     # arrows = g.add_all_arrows(points, norms)
        
    #     plt.pause(0.01)

    #     input("Press Enter...")
    #     # arrows.remove()
    #     i += 1
    #     # try:
    #     # except:
    #     #     print("lost")
    
    # getting exception count, error, and efficiency
    i = 0

    num_iterations = 1000
    boundary_points = []
    other_point_count = 0
    osvas = []
    errs = 0
    while i < num_iterations:
        # if i % 50 == 0:
        #     print(f"Iteration #{i} started...")
        
        # pk, nk, points = brrt.grow(getAllPoints=True)
        pk, nk, points = brrt.grow(getAllPoints=True)
        boundary_points.append(pk)
        osvas.append(nk)
        other_point_count += len(points) - 1
        i += 1              
        # try:
        # except:
        #     errs += 1
            
        
    
    average_error = sum(map(lambda b: radius - loc.distance_to(b), boundary_points)) / len(boundary_points)
    
    print("Number of BLEs:", errs)
    print("Average Boundary Error:", average_error)
    print("Non-Boundary Points Sampled:", other_point_count)
    print("Efficiency: ", len(boundary_points) / other_point_count * 100, "%", sep="")
    
    
    
    g.plot_all_points(boundary_points)
    plt.show()
    

