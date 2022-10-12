from copy import copy
from typing import Callable

import numpy as np
from numpy import ndarray
from sim_bug_tools.exploration.boundary_core.adherer import (
    AdherenceFactory, Adherer, BoundaryLostException)
from sim_bug_tools.structs import Domain, Point

DATA_LOCATION = "location"
DATA_NORMAL = "normal"

ANGLE_90 = np.pi / 2


class BoundaryAdherer(Adherer):
    """
    The BoundaryAdherer provides the ability to identify a point that lies on the
    boundary of a N-D volume (target envelope). A "classifier" function describes whether or not
    a sampled Point is within or outside of the target envelope.
    """

    def __init__(
        self,
        classifier: Callable[[Point], bool],
        p: Point,
        n: ndarray,
        direction: ndarray,
        d: float,
        delta_theta: float,
        r: float,
        num: int,
    ):
        """
        Boundary error, e, is within the range: 0 <= e <= d * theta. Average error is d * theta / 2

        Args:
            classifier (Callable[[Point], bool]): The function that returns true or false depending
                on whether or not the provided Point lies within or outside of the target envelope.
            p (Point): Parent boundary point - used as a starting point for finding the neighboring
                boundary point.
            n (ndarray): The parent boundary point's estimated orthogonal surface vector.
            direction (ndarray): The general direction to travel in (MUST NOT BE PARALLEL WITH @n)
            d (float): How far to travel from @p
            delta_theta (float): The initial change in angle (90 degrees is a good start).
        """
        super().__init__(classifier)
        self._p = p

        n = BoundaryAdherer.normalize(n)

        self._rotater_function = self.generateRotationMatrix(n, direction)

        self._s: ndarray = copy(n) * d
        self._s = np.dot(self._rotater_function(-ANGLE_90), self._s)

        self._prev: Point = None
        self._prev_class = None

        self._cur: Point = p + Point(self._s)
        self._cur_class = classifier(self._cur)

        self._r = r
        self._num = num

        self._b = None
        self._n = None
        self._prev_b = None
        self._prev_s = None

        self._iteration = 0
        self._angle = self._next_angle(delta_theta)

    @property
    def b(self) -> Point:
        """The identified boundary point"""
        return self._b

    @property
    def n(self) -> Point:
        """The identified boundary point's estimated orthogonal surface vector"""
        return self._n

    @property
    def boundary(self) -> tuple[Point, ndarray]:
        """Boundary point and its surface vector"""
        return (self._b, self._n)

    def has_next(self) -> bool:
        """
        Returns:
            bool: True if the boundary has not been found and has remaining
                samples.
        """
        return self._b is None

    def sample_next(self) -> Point:
        """
        Takes the next sample to find the boundary. When the boundary is found,
        (property) b will be set to that point and sample_next will no longer
        return anything.

        Raises:
            BoundaryLostException: This exception is raised if the adherer
                fails to acquire the boundary.

        Returns:
            Point: The next sample
            None: If the boundary was acquired or lost
        """
        self._prev = self._cur

        self._s = np.dot(self._rotater_function(self._angle), self._s)
        self._cur = self._p + Point(self._s)

        self._prev_class = self._cur_class
        self._cur_class = self._classifier(self._cur)
        self._angle = self._next_angle(self._angle)

        if self._cur_class:
            self._prev_b = self._p + Point(self._s)
            self._prev_s = copy(self._s)

        if self._iteration > self._num and self._prev_b is not None:
            self._b = self._prev_b
            self._n = self.normalize(
                np.dot(self._rotater_function(ANGLE_90), self._prev_s)
            )
            self.sample_next = lambda: None

        elif self._iteration > self._num and self._prev_b is None:
            raise BoundaryLostException()

        self._iteration += 1
        return self._cur

    def find_boundary(self) -> tuple[Point, ndarray]:
        """
        Samples until the boundary point is found.

        Returns:
            Point: The estimated boundary point
        """
        all_points = []
        while self.has_next():
            all_points.append(self.sample_next())

        return self._b, self._n

    def _next_angle(self, angle: float):
        return (
            abs(angle / (self._r**self._iteration))
            if self._cur_class
            else -abs(angle / (self._r**self._iteration))
        )

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

        un = BoundaryAdherer.normalize(u)
        vn = v - np.dot(un, v.T) * un
        vn = BoundaryAdherer.normalize(vn)

        if not (np.dot(un, vn.T) < 1e-4):
            raise Exception("Vectors %s and %s are already orthogonal." % (un, vn))

        return un, vn

    @staticmethod
    def generateRotationMatrix(u: ndarray, v: ndarray) -> Callable[[float], ndarray]:
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

        u, v = BoundaryAdherer.orthonormalize(u, v)

        I = np.identity(len(u.T))

        coef_a = v * u.T - u * v.T
        coef_b = u * u.T + v * v.T

        return lambda theta: I + np.sin(theta) * coef_a + (np.cos(theta) - 1) * coef_b


class BoundaryAdherenceFactory(AdherenceFactory):
    def __init__(
        self,
        classifier: Callable[[Point], bool],
        d: float,
        delta_theta: float,
        r: float,
        num: int,
    ):
        super().__init__(classifier)
        self._d = d
        self._delta_theta = delta_theta
        self._r = r
        self._num = num

    def adhere_from(self, p: Point, n: ndarray, direction: ndarray) -> Adherer:
        return BoundaryAdherer(
            self.classifier,
            p,
            n,
            direction,
            self._d,
            self._delta_theta,
            self._r,
            self._num,
        )


if __name__ == "__main__":
    import os
    import sys

    sys.path.append(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        )
    )
    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes
    from tools.grapher import Grapher
    

    d = 0.005
    r = 2
    num = 4
    angle = 30 * np.pi / 180

    s_loc = Point(0.5, 0.5, 0.5)
    s_rad = 0.25

    p = Point(0.5, 0.5, 0.25)
    n = np.array([0, 0, -1])
    direction = np.array([0, 1, 0])

    classifier = lambda p: s_loc.distance_to(p) < s_rad

    g = Grapher(is3d=True, domain=Domain.normalized(3))
    adh_f = BoundaryAdherenceFactory(classifier, d, angle, r, num)
    adh = adh_f.adhere_from(p, n, direction)
    while adh.has_next():
        pk = adh.sample_next()
        g.plot_point(pk)
        plt.pause(0.01)
        input("Waiting...")
