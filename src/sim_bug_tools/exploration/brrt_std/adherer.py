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

DATA_LOCATION = "location"
DATA_NORMAL = "normal"

ANGLE_90 = np.pi / 2


class ConstantAdherer(Adherer):
    def __init__(
        self,
        classifier: Callable[[Point], bool],
        domain: Domain,
        p: Point,
        n: ndarray,
        direction: ndarray,
        scaler: Scaler,
        delta_theta: float,
        fail_out_of_bounds: bool = False,
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
            theta (float): How far to rotate to find the boundary.
        """
        super().__init__(classifier, domain)
        self._scaler = scaler
        self._p = p

        self._fail_out_of_bounds = fail_out_of_bounds
        n = ConstantAdherer.normalize(n)

        self._rotater_function = self.generateRotationMatrix(n, direction)
        A = self._rotater_function(-ANGLE_90)
        # Get the direction we want to travel in
        self._v = copy(n.squeeze())
        self._v = np.dot(A, self._v)

        # Scale the vector to get our displacement vector
        self._s: ndarray = self._v
        self._s = self._scaler * self._s

        self._prev: Point = None
        self._prev_class: bool = None

        self._cur: Point = p + Point(self._s)
        self._cur_class = None
        # self._cur_class = classifier(self._cur)
        self._classify_sample()

        if self._cur_class:
            self._rotate = self._rotater_function(delta_theta)
        else:
            self._rotate = self._rotater_function(-delta_theta)

        self._b: Point = None
        self._n: ndarray = None

        self._sub_samples = []

        self._iteration = 0
        self._max_iteration = (2 * np.pi) // delta_theta

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
        return (self._b, self._n)

    @property
    def sub_samples(self):
        return self._sub_samples

    def has_next(self) -> bool:
        return self._b is None

    def _classify_sample(self):
        "Will only run the classifier IFF the sample is in domain"
        self._prev_class = self._cur_class
        in_domain = self._cur in self._domain
        if self._fail_out_of_bounds and not in_domain:
            raise SampleOutOfBoundsException()

        self._cur_class = in_domain and self._classifier(self._cur)

    def sample_next(self) -> tuple[Point, bool]:
        self._prev = self._cur
        self._v = np.dot(self._rotate, self._v)
        self._s = self._scaler * self._v
        self._cur = self._p + Point(self._s)

        # self._cur_class = self._classifier(self._cur)
        self._classify_sample()

        if self._prev_class is not None and self._cur_class != self._prev_class:
            self._b = self._cur if self._cur_class else self._prev
            self._n = ConstantAdherer.normalize(
                np.dot(self._rotater_function(ANGLE_90), self._s)
            )
            self.sample_next = lambda: None

        elif self._iteration > self._max_iteration:
            raise BoundaryLostException()

        self._sub_samples.append(self._cur)

        self._iteration += 1
        return self._cur, self._cur_class

    def find_boundary(self) -> tuple[Point, ndarray]:
        all_points = []
        while self.has_next():
            all_points.append(self.sample_next())

        return self._b, self._n

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
        vn = v - np.dot(un, v.T) * un
        vn = ConstantAdherer.normalize(vn)

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

        u, v = ConstantAdherer.orthonormalize(u, v)

        I = np.identity(len(u.T))

        coef_a = v * u.T - u * v.T
        coef_b = u * u.T + v * v.T

        return lambda theta: I + np.sin(theta) * coef_a + (np.cos(theta) - 1) * coef_b


class ConstantAdherenceFactory(AdherenceFactory):
    def __init__(
        self,
        classifier: Callable[[Point], bool],
        domain: Domain,
        scaler: Scaler,
        delta_theta: float,
        fail_out_of_bounds: bool = False,
    ):
        super().__init__(classifier, domain)
        # self._d = d
        self._scaler = scaler
        self._theta = delta_theta
        self._fail_out_of_bounds = fail_out_of_bounds

    def adhere_from(self, p: Point, n: ndarray, direction: ndarray) -> Adherer:
        return ConstantAdherer(
            self.classifier,
            self.domain,
            p,
            n,
            direction,
            self._scaler,
            self._theta,
            self._fail_out_of_bounds,
        )


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes

    angle = 30 * np.pi / 180

    fig = plt.figure()
    ax: Axes = fig.add_subplot()
    ax.set_xlim([0, 3])
    ax.set_ylim([0, 3])

    vec = np.array([0, 1])

    u = np.array([0.5, 0.5])
    v = np.array([0.1, 0.9])
    # u, v = BoundaryAdherer.orthonormalize(u, v)

    ar_vec = ax.arrow(1.5, 1.5, vec[0], vec[1])
    plt.pause(0.05)

    input("Waiting...")
    rotate = ConstantAdherer.generateRotationMatrix(u, v)(angle)

    while True:
        ar_vec.remove()
        vec = np.dot(rotate, vec)
        ar_vec = ax.arrow(1.5, 1.5, vec[0], vec[1])
        plt.pause(0.05)
        input("Waiting...")
