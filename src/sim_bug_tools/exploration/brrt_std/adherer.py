from copy import copy
from typing import Callable

import numpy as np
from numpy import ndarray

from sim_bug_tools.exploration.boundary_core.adherer import (
    AdherenceFactory,
    Adherer,
    BoundaryLostException,
    SampleOutOfBoundsException,
)
from sim_bug_tools.structs import Domain, Point, Scaler

import matplotlib.pyplot as plt

DATA_LOCATION = "location"
DATA_NORMAL = "normal"

ANGLE_90 = np.pi / 2

normalize = lambda v: v / np.linalg.norm(v)


def angle_between(u, v):
    u, v = normalize(u), normalize(v)
    return np.arccos(np.clip(np.dot(u, v), -1, 1.0))


class ConstantAdherer(Adherer):
    def __init__(
        self,
        classifier: Callable[[Point], bool],
        b: Point,
        n: ndarray,
        direction: ndarray,
        scaler: Scaler,
        delta_theta: float,
        domain: Domain = None,
        fail_out_of_bounds: bool = True,
    ):
        """
        Boundary error, e, is within the range: 0 <= e <= d * theta. Average error is d * theta / 2

        Args:
            classifier (Callable[[Point], bool]): The function that returns true or false depending
                on whether or not the provided Point lies within or outside of the target envelope.
            p (Point): Parent boundary point - used as a starting point for finding the neighboring
                boundary point.
            n (ndarray): The parent boundary point's estimated orthonormal surface vector.
            direction (ndarray): The general direction to travel in (MUST NOT BE PARALLEL WITH @n)
            d (float): How far to travel from @p
            theta (float): How far to rotate to find the boundary.
        """
        super().__init__(
            classifier,
            (b, n),
            direction,
            domain,
            fail_out_of_bounds,
        )
        self._scaler = scaler
        self._delta_theta = delta_theta

        self._rotater_function = self.generateRotationMatrix(n, direction)
        A = self._rotater_function(-ANGLE_90)

        # Get the direction we want to travel in
        self._v = copy(n.squeeze())
        self._v: ndarray = np.dot(A, self._v)

        if angle_between(self._v, n) * 180 / np.pi > 92:
            print("wonk occurred")

        # Scale the vector to get our displacement vector
        self._s: ndarray = copy(self._v)
        self._s = self._scaler * self._s

        self._prev: Point = None
        self._prev_class: bool = None

        self._cur: Point = b + Point(self._s)
        self._cur_class = None

        self._initialized = False

        self._iteration = 0
        self._max_iteration = int((np.pi) // delta_theta)

        # from sim_bug_tools.graphics import Grapher

        # self._g = Grapher(True, Domain.normalized(3))
        # self._g.draw_sphere(Point([0.5] * 3), 0.4)
        # self._tmp_b = b
        # # self._tmp_n = n
        # self._gb = self._g.plot_point(b, color="green")
        # self._gn = self._g.add_arrow(b, n, color="green")
        # self._gs = None
        # # plt.pause(0.01)

    @property
    def delta_theta(self):
        return self._delta_theta

    def _classify_cur(self):
        "Will only run the classifier IFF the sample is in domain"
        self._prev_class = self._cur_class
        self._cur_class = self._classify(self._cur)

    def _initialize_rotater(self):
        # self._cur_class = classifier(self._cur)
        self._classify_cur()

        if self._cur_class:
            self._rotate = self._rotater_function(self._delta_theta)
        else:
            self._rotate = self._rotater_function(-self._delta_theta)

        self._initialized = True

    def _rotate_displacement(self):
        self._prev = self._cur
        self._prev_v = self._v
        self._v: ndarray = np.dot(self._rotate, self._v)
        self._s = self._scaler * self._v
        self._cur = self._pivot + Point(self._s)

    def sample_next(self) -> tuple[Point, bool]:
        if not self._initialized:
            self._initialize_rotater()
        else:
            self._rotate_displacement()
            self._classify_cur()

        # if self._gs != None:
        #     self._gs.remove()
        # self._gs = self._g.add_arrow(self._tmp_b, self._v * 0.10)
        # plt.pause(0.01)

        if self._prev_class is not None and self._cur_class != self._prev_class:
            self._new_b = self._cur if self._cur_class else self._prev
            self._new_n = self.normalize(
                np.dot(self._rotater_function(ANGLE_90), self._s)
            )
            # self._gb.remove()
            # self._gn.remove()
            # if self._gs is not None:
            #     self._gs.remove()

            self.sample_next = lambda: None

        elif self._iteration > self._max_iteration:
            raise BoundaryLostException()

        self._iteration += 1
        return self._cur, self._cur_class

    @staticmethod
    def normalize(u: ndarray):
        return u / np.linalg.norm(u)

    @staticmethod
    def orthonormalize(u: ndarray, v: ndarray) -> tuple[ndarray, ndarray]:
        """
        Generates orthonormal vectors given two vectors @u, @v which form a span.

        -- Parameters --
        u, v : np.ndarray
            Two n-d vectors of the same length
        -- Return --
        (un, vn)
            Orthonormal vectors for the span defined by @u, @v
        """
        u = u.squeeze()
        v = v.squeeze()

        assert len(u) == len(v)

        u = u[np.newaxis]
        v = v[np.newaxis]

        un = ConstantAdherer.normalize(u)
        vn = ConstantAdherer.normalize(v)
        vn = v - np.dot(un, v.T) * un
        vn = ConstantAdherer.normalize(vn)
        print(un, vn)
        return (un.squeeze(), vn.squeeze())

    @classmethod
    def generateRotationMatrix(
        cls, u: ndarray, v: ndarray
    ) -> Callable[[float], ndarray]:
        """
        Creates a function that can construct a matrix that rotates by a given angle.

        Args:
            u, v : ndarray
                The two vectors that represent the span to rotate across.

        Raises:
            Exception: fails if @u and @v aren't vectors or if they have differing
                number of dimensions.

        Returns:
            Callable[[float], ndarray]: A function that returns a rotation matrix
                that rotates that number of degrees using the provided span.
        """
        u = u.squeeze()
        v = v.squeeze()

        if u.shape != v.shape:
            raise Exception("Dimension mismatch...")
        elif len(u.shape) != 1:
            raise Exception("Arguments u and v must be vectors...")

        u, v = (
            cls.orthonormalize(u, v)
            if abs(np.dot(u, v[np.newaxis].T)) > 1e-4
            else (u, v)
        )
        u = u[np.newaxis]
        v = v[np.newaxis]

        I = np.identity(len(u.T))

        coef_a = v * u.T - u * v.T
        coef_b = u * u.T + v * v.T

        return lambda theta: I + np.sin(theta) * coef_a + (np.cos(theta) - 1) * coef_b


class ConstantAdherenceFactory(AdherenceFactory[ConstantAdherer]):
    def __init__(
        self,
        classifier: Callable[[Point], bool],
        scaler: Scaler,
        delta_theta: float,
        domain: Domain = None,
        fail_out_of_bounds: bool = False,
    ):
        super().__init__(classifier, domain, fail_out_of_bounds)
        # self._d = d
        self._scaler = scaler
        self._delta_theta = delta_theta

    def adhere_from(self, b: Point, n: ndarray, direction: ndarray):
        return ConstantAdherer(
            self.classifier,
            b,
            n,
            direction,
            self._scaler,
            self._delta_theta,
            self.domain,
            self._fail_out_of_bounds,
        )


def test_rotation():
    from sim_bug_tools.graphics import Grapher
    import matplotlib.pyplot as plt

    theta = np.pi * 10 / 180

    ndims = 3
    domain = Domain.normalized(ndims)
    g = Grapher(ndims == 3, domain)

    v1 = np.array([1, 0, 0])
    v2 = np.array([1, 1, 1])
    print(v1, v2)
    _v1 = g.add_arrow(Point.zeros(ndims), v1, color="blue")
    _v2 = g.add_arrow(Point.zeros(ndims), v2, color="red")
    plt.pause(0.01)

    v1, v2 = ConstantAdherer.orthonormalize(v1, v2)
    print(v1, v2)
    _v1.remove()
    _v2.remove()
    _v1 = g.add_arrow(Point.zeros(ndims), v1, color="blue")
    _v2 = g.add_arrow(Point.zeros(ndims), v2, color="red")
    plt.pause(0.01)

    rotater = ConstantAdherer.generateRotationMatrix(v1, v2)
    s = copy(v2)
    _s = g.add_arrow(Point.zeros(ndims), s, color="green")
    plt.pause(0.01)

    for i in range(int(np.pi // theta)):
        s = np.dot(rotater(theta), s)
        _s.remove()
        _s = g.add_arrow(Point.zeros(ndims), s, color="green")
        _p = g.plot_point(Point(s), color="blue")
        plt.pause(0.01)
        print("next")
    print("done")


if __name__ == "__main__":
    test_rotation()
