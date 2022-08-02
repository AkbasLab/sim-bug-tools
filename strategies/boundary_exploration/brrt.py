from copy import copy
from time import time
from typing import Callable

import numpy as np
from numpy import ndarray
from rtree import index
from sim_bug_tools.rng.lds.sequences import RandomSequence, Sequence
from sim_bug_tools.structs import Domain, Point
from treelib import Node, Tree

from adherer import BoundaryAdherer

DATA_LOCATION = "location"
DATA_NORMAL = "normal"


class BoundaryRRT:
    def __init__(
        self, classifier: Callable[[Point], bool], t0: Point, d: float, theta: float
    ):
        self._classifier = classifier
        self._d = d
        self._theta = theta
        self._ndims = len(t0)

        self._tree = Tree()
        p = index.Property()
        p.set_dimension(self._ndims)
        self._index = index.Index(properties=p)

        p0, n0 = self._surface(t0)
        self._index.insert(0, p0)
        self._root = Node(identifier=0, data=self._create_data(p0, n0))
        self._next_id = 1

        self._tree.add_node(self._root)

    def grow(self) -> tuple[Point, ndarray]:
        failed = True

        while failed:
            r = self._random_point()
            parent = self._find_nearest(r)
            p, n = parent.data[DATA_LOCATION], parent.data[DATA_NORMAL]

            ba = BoundaryAdherer(
                self._classifier, p, n, (r - p).array, self._d, self._theta
            )
            try:
                ba.find_boundary()
                failed = False
            except:
                print("lost boundary")

        pk, nk = ba.bn

        self._add_node(pk, nk, parent.identifier)

        return pk, nk

    def growBy(self, num: int = 1) -> list[tuple[Point, ndarray]]:
        data = []
        for i in range(num):
            data += [self.grow()]
        return data

    def _add_node(self, p: Point, n: ndarray, parentID: int):
        node = Node(identifier=self._next_id, data=self._create_data(p, n))
        self._tree.add_node(node, parentID)
        self._index.insert(self._next_id, p)
        self._next_id += 1

    def _random_point(self) -> Point:
        return Point(np.random.rand(self._ndims))

    def _find_nearest(self, p: Point) -> Node:
        # closest = {
        #     "id": 0,
        #     "distance": self._tree.nodes[0].data[DATA_LOCATION].distance_to(p),
        # }

        # for id, node in self._tree.nodes.items():
        #     dist = node.data[DATA_LOCATION].distance_to(p)
        #     if dist < closest["distance"]:
        #         closest["id"] = id
        #         closest["distance"] = dist
        
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
    from sys import getsizeof

    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes


    ndims = 3
    num_iter = 10

    d = 0.015
    theta = 5 * np.pi / 180

    loc = Point([0.5 for x in range(ndims)])
    radius = 0.25

    n = np.array([0.5 for x in range(ndims - 1)] + [0.5 - radius])

    classifier = lambda p: p.distance_to(loc) <= radius

    print("Building brrt...")
    brrt = BoundaryRRT(classifier, loc, d, theta)

    path_points = np.array(brrt._interm)
    b = Point(path_points[-1])
    print(b, "e =", radius - b.distance_to(loc))



    fig3d = plt.figure()
    fig_time = plt.figure()
    
    # 3D
    ax3d: Axes = fig3d.add_subplot(111, projection="3d")
    ax3d.set_xlim([0, 1])
    ax3d.set_ylim([0, 1])
    ax3d.set_zlim([0, 1])
    ax3d.scatter(*path_points.T)
    plt.pause(0.01)

    # 2D
    ax_time: Axes = fig_time.add_subplot()
        
    target = 1000
    batch = 10
    num_points = 0
    i = 0
    times = []
    points = []
    

    print("Launching...")
    while True:
        nodes, t = measure_time(brrt.growBy, batch)

        points += [p for p, n in nodes]
        times += [t]
        
        num_points += batch
        ax3d.scatter(*np.array(points).squeeze().T, c="red")
        ax_time.scatter(range(0, batch * (i + 1), batch), times, c="blue")
        plt.pause(0.01)
        input("Press Enter...")
        i += 1
        # points = np.array([p for p, n in brrt.growBy(num_iter)])
    
    plt.show()
    
    
