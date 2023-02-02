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


class ExponentialAdherer(Adherer):
    """
    The BoundaryAdherer provides the ability to identify a point that lies on the
    boundary of a N-D volume (target envelope). A "classifier" function describes whether or not
    a sampled Point is within or outside of the target envelope.
    """

    def __init__(
        self,
        classifier: Callable[[Point], bool],
        b: Point,
        n: ndarray,
        direction: ndarray,
        scaler: Scaler,
        theta0: float,
        num: int,
        domain: Domain = None,
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
            delta_theta (float): The initial change in angle (90 degrees is a good start).
            r (float): Influences the rate of convergence
            init_class (bool): The initial classification of @p.
                When None, initial state is determined by @classifier(@p)
        """
        super().__init__(classifier, (b, n), direction, domain, fail_out_of_bounds)
        self._rotater_function = self.generateRotationMatrix(n, direction)

        self._initial_angle = abs(theta0)
        self._s: ndarray = scaler * copy(n)
        self._s = np.dot(self._rotater_function(-ANGLE_90), self._s)

        self._prev_b: Point = None
        self._prev_s: ndarray = None

        self._prev: Point = None
        self._prev_class = None

        self._cur: Point = b + Point(self._s)
        self._cur_class = None

        self._num = num

        self._iteration = 0
        self._found_nontarget = False

    # @property
    # def b(self) -> Point:
    #     """The identified boundary point"""
    #     return self._b

    # @property
    # def n(self) -> Point:
    #     """The identified boundary point's estimated orthogonal surface vector"""
    #     return self._n

    # @property
    # def bnode(self) -> tuple[Point, ndarray]:
    #     """Boundary point and its surface vector"""
    #     return (self._b, self._n)

    @property
    def boundary_found(self):
        """
        The boundary has been found when a target and nontarget
        sample has been acquired. NOTE that this does not mean
        that the algorithm has finished (which only occurs after
        max_iterations_exceeded occurs.) It only means that the
        boundary has been found between two samples and that it
        will not result in boundary lost exception.
        """
        return self._prev_b is not None and self._found_nontarget

    @property
    def max_iterations_exceeded(self):
        return self._iteration >= self._num

    def _classify_cur(self):
        self._cur_class = self._classify(self._cur)

        if self._cur_class:
            self._prev_b = self._pivot + Point(self._s)
            self._prev_s = copy(self._s)
        else:
            self._found_nontarget = True

    def _rotate_cur(self):
        """
        This executes an iteration of the exponentially decaying angle.Sets prev
        to next, rotates next by angle, updates angle with next_angle().
        """
        self._prev = self._cur
        self._prev_class = self._cur_class
        
        self._angle = self._next_angle(self._initial_angle)

        self._s: ndarray = np.dot(self._rotater_function(self._angle), self._s)
        self._cur = self._pivot + Point(self._s)
        self._cur_class = None

        self._iteration += 1

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
        self._classify_cur()
        self._rotate_cur()

        if not self.max_iterations_exceeded:
            pass
        elif self.boundary_found:
            self._new_b = self._prev_b
            self._new_n = self.normalize(
                np.dot(self._rotater_function(ANGLE_90), self._prev_s)
            )
            self.sample_next = lambda: None

        else:
            raise BoundaryLostException()

        return self._prev, self._prev_class

    def _next_angle(self, angle: float):
        return (
            abs(angle / (2**self._iteration))
            if self._cur_class
            else -abs(angle / (2**self._iteration))
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

        un = ExponentialAdherer.normalize(u)
        vn = v - np.dot(un, v.T) * un
        vn = ExponentialAdherer.normalize(vn)

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

        u, v = ExponentialAdherer.orthonormalize(u, v)

        I = np.identity(len(u.T))

        coef_a = v * u.T - u * v.T
        coef_b = u * u.T + v * v.T

        return lambda theta: I + np.sin(theta) * coef_a + (np.cos(theta) - 1) * coef_b


class ExponentialAdherenceFactory(AdherenceFactory):
    def __init__(
        self,
        classifier: Callable[[Point], bool],
        scaler: Scaler,
        theta0: float,
        num: int,
        domain: Domain = None,
        fail_out_of_bounds: bool = False,
    ):
        super().__init__(classifier, domain, fail_out_of_bounds)
        self._scaler = scaler
        self._theta0 = theta0
        self._num = num
        self._fail_out_of_bounds = fail_out_of_bounds

    def adhere_from(self, b: Point, n: ndarray, direction: ndarray) -> Adherer:
        return ExponentialAdherer(
            self.classifier,
            b,
            n,
            direction,
            self._scaler,
            self._theta0,
            self._num,
            self.domain,
            self._fail_out_of_bounds,
        )
