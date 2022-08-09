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

from adherer import BoundaryAdherer

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
        self, classifier: Callable[[Point], bool], t0: Point, d: float, theta: float
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
        self._theta = theta
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
        
        self._prev_dir: ndarray = None
        
    @property 
    def previous_node(self) -> Node:
        return self._tree.get_node(self._next_id-1)
    
    @property
    def previous_direction(self) -> ndarray:
        return self._prev_dir

    def grow(self, getAllPoints: bool = False) -> tuple[Point, ndarray]:
        """
        Grows the RRT; i.e., Finds a new boundary point to add to the tree.

        Returns:
            tuple[Point, ndarray]: The newly added point on the surface
                and its estimated orthogonal surface vector
                
        Throws:
            BoundaryLostException: Thrown if the boundary is lost
                
        """
        
        r = self._random_point()
        parent = self._find_nearest(r)

        return self.growFrom(parent, r, getAllPoints)
    
    def growFrom(self, node: Node, r: Point, getAllPoints: bool = False):
        p: Point = node.data[DATA_LOCATION]
        n: ndarray = node.data[DATA_NORMAL]
        direction = (r - p).array
        self._prev_dir = direction

        ba = BoundaryAdherer(
            self._classifier, p, n, direction, self._d, self._theta
        )
        b, all_points = ba.find_boundary(True)
            
        bn = ba.bn
        if getAllPoints:
            results = bn[0], bn[1], all_points
        else:
            results = bn

        self._add_node(*bn, node.identifier)
        
        return results

    def growBy(self, num: int) -> list[tuple[Point, ndarray]]:
        """
        Grows the RRT by the provided number of samples. If the boundary is lost,
        it will attempt again until it succeeds (WARNING: not guarnateed to halt.)

        Args:
            num (int, optional): The number of boundary points to find and
                expand the RRT by.

        Returns:
            list[tuple[Point, ndarray]]: A list of boundary points and their
                estimated orthogonal surface vectors.
        """
        data = []
        for i in range(num):
            failed = True
            while failed:
                
                try:
                    data += [self.grow()]
                    failed = False
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
        cur_class = None
        
        # First, reach within d distance from surface
        while self._classifier(cur):
            prev = cur
            self._interm += [prev]
            cur = prev + Point(s)
        
        s *= 0.5
        ps = Point(s)
        cur = prev + Point(s)      
        
        # Get closer until within d/2 distance from surface
        while self._classifier(cur):
            prev = cur
            cur = prev + ps



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
    d = 0.03
    theta = 5 * np.pi / 180

    # The spherical target envelope
    loc = Point([0.5 for x in range(ndims)])
    radius = 0.25
    classifier = lambda p: p.distance_to(loc) <= radius

    print("Building brrt...")
    brrt = BoundaryRRT(classifier, loc, d, theta)

    # The series of points that were sampled to reach the surface
    path_points = np.array(brrt._interm)

    # fig3d = plt.figure()
    
    # 3D View
    # ax3d: Axes = fig3d.add_subplot(111, projection="3d")
    # ax3d.set_xlim([0, 1])
    # ax3d.set_ylim([0, 1])
    # ax3d.set_zlim([0, 1])
    # ax3d.scatter(*path_points.T)
    # plt.pause(0.01)
    g = Grapher(True)
    g.ax.set_xlim([0, 1])
    g.ax.set_ylim([0, 1])
    g.ax.set_zlim([0, 1])
    path_points = np.array(brrt._interm)
    g.plot_all_points(path_points)
    plt.pause(0.01)
    # Setup
    num_iterations = 1000
    batch = 100
    num_points = 0
    i = 0
    points = []
    

    print("Launching...")
    # while True:
    #     nodes, t = measure_time(brrt.growBy, batch)
    #     print(nodes[0])

    #     points = [p for p, n in nodes]
    #     normals = [n*0.1 for p, n in nodes]
        
    #     num_points += batch
    #     # ax3d.scatter(*np.array(points).squeeze().T, c="red")
    #     g.plot_all_points(points)
    #     g.add_all_arrows(points, normals)
    #     plt.pause(0.01)

    #     input("Press Enter...")
    #     i += 1
    boundary_points = []
    other_point_count = 0
    osvas = []
    errs = 0
    
    while i < num_iterations:
        try:
            pk, nk, points = brrt.grow(getAllPoints=True)
            # if i % 50 == 0:
            #     print(f"Iteration #{i} started...")
            boundary_points.append(pk)
            osvas.append(nk)
            other_point_count += len(points)
            i += 1
        except:
            errs += 1
            
    average_error = sum(map(lambda b: radius - loc.distance_to(b), boundary_points)) / len(boundary_points)
    
    print("Number of BLEs:", errs)
    print("Average Boundary Error:", average_error)
    print("Efficiency: ", len(boundary_points) / other_point_count * 100, "%", sep="")
    
    g.plot_all_points(boundary_points)
    plt.show()

    
    
    
