from copy import copy
from dataclasses import dataclass
from this import d
from typing import Callable
from avtools.structs import Domain, Point
from avtools.rng.lds.sequences import Sequence, RandomSequence
from numpy import ndarray
from treelib import Node, Tree

import matplotlib.pyplot as plt
import numpy as np


DATA_LOCATION = "location"
DATA_NORMAL = "normal"

ANGLE_90 = np.pi / 2

# Graphing tools for testing purposes
class Display2D:
    def __init__(self, domain: Domain, title: str = "Test"):
        self._domain = domain
        # self._rrt = rrt
        self._fig, self._ax = plt.subplots(1)
        self._ax.set_xlim(xmin=domain[0][0], xmax=domain[0][1])
        self._ax.set_ylim(ymin=domain[1][0], ymax=domain[1][1])

        plt.pause(0.01)

    def draw_circle(self, loc: Point, radius: float):
        c = plt.Circle(loc.array, radius, color="b", fill=False)
        self._ax.add_patch(c)
        plt.pause(0.01)
        return c

    def plot_vector(self, loc: Point, v: ndarray):
        ele = self._ax.quiver(*loc, v[0], v[1])
        plt.pause(0.01)
        return ele

    def plot_point(self, p: Point, color="blue"):
        ele = self._ax.scatter(p[0], p[1], color=color)
        plt.pause(0.01)
        return ele


class Display3D:
    def __init__(self, domain: Domain = Domain.normalized(3)):
        self._fig = plt.figure()
        self._ax = self._fig.gca(projection="3d")

        plt.pause(0.01)

    def plot_point(self, p: Point, color="blue"):
        self._ax.scatter([p[0]], [p[1]], [p[2]], color=color)
        plt.pause(0.01)

    # def plot_vector(self, v: ndarray):

    def draw_sphere(self, location: Point, radius: float):

        u, v = np.mgrid[0 : 2 * np.pi : 20j, 0 : np.pi : 10j]
        x = location[0] + radius * np.cos(u) * np.sin(v)
        y = location[1] + radius * np.sin(u) * np.sin(v)
        z = location[2] + radius * np.cos(v)

        self._ax.plot_wireframe(x, y, z, color="grey")
        plt.pause(0.01)


# Boundary Exploration Classes


class PseudoRRT:
    """
    The "PseudoRRT" is a partially implemented RRT that is responsible over
    selecting the next node to expand from and in which direction.
    """

    def __init__(self, root_data: dict):
        self._tree = Tree()
        self._tree.create_node(identifier=0, data=root_data)

    def add_node(self, parent_id, child_data):
        "Adds data to the tree from the given parent."
        id = self.generate_id()
        self._tree.create_node(identifier=id, data=child_data, parent=parent_id)

        return id

    def generate_id(self):
        "Creates an ID for a node in the tree."
        if not hasattr(self, "_node_id_counter"):
            self._node_id_counter = 1

        id = self._node_id_counter
        self._node_id_counter += 1

        return id

    def select_next(self) -> tuple[Node, ndarray]:
        "Returns the ID and data of the next node to extend from and in which direction."
        raise NotImplementedError("The select_next method must be implemented.")


class NeivePRRT(PseudoRRT):
    """
    This implementation of a PseudoRRT simply selects a node by randomly sampling and
    selecting the node closest to that point and in the direction of the point rel-
    ative to that node.
    """

    def __init__(self, root_data: dict, selection_seq: Sequence):
        """
        Args:
            root_data (dict): The first point on the boundary's surface.
            selection_seq (Sequence): What sequence to use for selecting the random
                point.
        """
        super().__init__(root_data)
        self._seq = selection_seq

    def select_next(self) -> tuple[Node, ndarray]:
        p = self.sample_random()
        node_id = self.find_closest_node(p)

        # gd.plot_point(p, color="green")

        n: Node = self._tree.get_node(node_id)

        direction = (p - n.data[DATA_LOCATION]).array

        return (n, direction)

    def sample_random(self) -> Point:
        return self._seq.get_points(1)[0]

    def find_closest_node(self, p: Point):
        "Nodes will contain a normal vector and their coordinates as a Point object."
        closest = {
            "id": 0,
            "distance": self._tree.nodes[0].data[DATA_LOCATION].distance_to(p),
        }

        for id, node in self._tree.nodes.items():
            dist = node.data[DATA_LOCATION].distance_to(p)
            if dist < closest["distance"]:
                closest["id"] = id
                closest["distance"] = dist

        return closest["id"]


class BoundaryExplorer:
    """
    A blackbox solution to exploring the boundary of an N-Dimensional volume.
    """

    def __init__(
        self,
        distance: float,
        delta_angle: float,
        prrt: PseudoRRT,
        bug_detector: Callable[[Point], bool],
    ):
        "Dimensions must match between start, normal, bug in bugs, and seq domain"

        self._distance = distance
        self._delta = delta_angle
        self._prrt = prrt
        self.point_is_bug = bug_detector

    @property
    def bug_detector(self) -> Callable[[Point], bool]:
        return self.point_is_bug

    @bug_detector.setter
    def bug_detector(self, detector: Callable[[Point], bool]):
        self.point_is_bug = detector

    @staticmethod
    def orthogonalization(
        v1: np.ndarray, v2: np.ndarray
    ) -> list[np.ndarray, np.ndarray]:
        """
        Generates orthogonal vectors given two vectors @v1, @v2 which form a span.

        -- Parameters --
        v1, v2 : np.ndarray
            Two n-d vectors of the same length
        -- Return --
        (n1, n2)
            Orthogonal vectors for the plane defined by @v1, @v2
        """
        assert len(v1) == len(v2)

        n1 = v1 / np.linalg.norm(v1)
        v2 = v2 - np.dot(n1, v2.T) * n1
        n2 = v2 / np.linalg.norm(v2)

        if not (np.dot(n1, n2.T) < 1e-4):
            raise Exception("Vectors %s and %s are already orthogonal." % (n1, n2))

        return n1, n2

    @staticmethod
    def generate_rotation_matrix_function(
        v1: ndarray, v2: ndarray
    ) -> Callable[[float], ndarray]:
        """
        p: Point
            The point on the boundary that the next branch of the tree is stemming off from
        normal: ndarray
            The vector that is normal to the surface of the boundary. The direction of the
            vector points away from the bug's volume.
        other: Point
            Another point that the span passes through

        Creates a function that returns the rotation matrix for a given angle.
        """

        num_dimensions = len(v1)
        I = np.identity(num_dimensions)

        v1 = v1 if len(v1.shape) > 1 else v1[np.newaxis]
        v2 = v2 if len(v2.shape) > 1 else v2[np.newaxis]

        vector_u, vector_v = BoundaryExplorer.orthogonalization(v1, v2)

        coef_A = vector_v * vector_u.T - vector_u * vector_v.T
        coef_B = vector_u * vector_u.T + vector_v * vector_v.T

        return lambda angle: I + np.sin(angle) * coef_A + (np.cos(angle) - 1) * coef_B

    def normalize(self, v: ndarray) -> ndarray:
        return v / np.linalg.norm(v)

    def sample_next(self) -> Point:
        """Samples for the next point on the boundary.

        Args:
            previous (tuple[Point, bool], optional): The previous point and if it was a bug for this iteration.
                Defaults to None which signifies a new iteration.

        Returns:
            Point: The next sampled point
        """

        node, direction = self._prrt.select_next()

        normal: ndarray = node.data[DATA_NORMAL]
        p: Point = node.data[DATA_LOCATION]

        rotation_matrix = BoundaryExplorer.generate_rotation_matrix_function(
            normal, direction
        )

        s: ndarray = copy(normal) * self._distance
        s = np.dot(rotation_matrix(-ANGLE_90), s)

        cur: Point = p + Point(s)
        prev: Point = cur
        prev_was_bug = self.point_is_bug(prev)

        boundary_found = False

        while not boundary_found:
            if prev_was_bug:
                s = np.dot(rotation_matrix(self._delta), s)

            else:
                s = np.dot(rotation_matrix(-self._delta), s)

            cur = p + Point(s)
            cur_is_bug = self.point_is_bug(cur)

            if cur_is_bug != prev_was_bug:
                boundary_found = True
            else:
                prev = cur
                prev_was_bug = cur_is_bug

        # Ensures that all "boundary points" are within the bug volume
        if prev_was_bug:
            cur = prev

        new_normal = self.normalize(np.dot(rotation_matrix(ANGLE_90), s))
        self._prrt.add_node(
            node.identifier, {DATA_LOCATION: cur, DATA_NORMAL: new_normal}
        )

        return cur


## Tests for 2D and 3D


def threeDTest():
    sphere_radius = 0.25
    sphere_loc = Point(0.5, 0.5, 0.5)

    starting_point = Point(0.5, 0.5, 0.25)
    starting_normal = np.array([0, 0, -1])
    root_data = {DATA_LOCATION: starting_point, DATA_NORMAL: starting_normal}

    distance = 0.01
    delta = np.pi * 5 / 180
    seq = RandomSequence(Domain.normalized(3), ["x", "y", "z"])
    prrt = NeivePRRT(root_data, seq)

    def bug_detector(p: Point):
        return sphere_loc.distance_to(p) <= sphere_radius

    be = BoundaryExplorer(distance, delta, prrt, bug_detector)

    d3d = Display3D()
    d3d.draw_sphere(sphere_loc, sphere_radius)
    print("starting...")

    d3d.plot_point(starting_point)
    # input("Press enter")

    while True:
        p = be.sample_next()
        d3d.plot_point(p)
        input("press enter")


def twoDTest():

    root_data = {DATA_LOCATION: starting_point, DATA_NORMAL: starting_normal}

    distance = 0.05
    delta = np.pi * 15 / 180
    seq = RandomSequence(Domain.normalized(2), ["x", "y"])
    prrt = NeivePRRT(root_data, seq)
    # bug_detector = lambda p: circle_loc.distance_to(p) <= circle_radius

    def bug_detector(p: Point) -> bool:
        gd.plot_point(p)
        return circle_loc.distance_to(p) <= circle_radius

    be = BoundaryExplorer(distance, delta, prrt, bug_detector)

    n = 20

    v = np.array([0, 1])
    vr = np.array([1, 1])
    rot = BoundaryExplorer.generate_rotation_matrix_function(v, vr)

    for x in range(4):
        print(rot(x * np.pi * 15 / 180))

    while n > 0:
        p = be.sample_next()
        gd.plot_point(p)

        # _v = d.plot_vector(starting_point, v)

        input("Press enter")
        # _v.remove()

        # v = np.dot(rot(np.pi), v)

    plt.show()


if __name__ == "__main__":
    threeDTest()
