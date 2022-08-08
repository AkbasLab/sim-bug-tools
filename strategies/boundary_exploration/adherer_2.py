"""
The primary differences between adherer_2 and adherer are in the
improvements to efficiency and accuracey through converging
parameters. Also, we implemented improved orthonormal 
surface vector approximations (OSVA).
"""

from copy import copy
from typing import Callable

import numpy as np
from numpy import ndarray
from sim_bug_tools.structs import Point

DATA_LOCATION = "location"
DATA_NORMAL = "normal"

ANGLE_90 = np.pi / 2


class BoundaryLostException(Exception):
    def __init__(self, msg="Failed to locate boundary!"):
        self.msg = msg
        super().__init__(msg)

    def __str__(self):
        return f"<BoundaryLostException: Angle: {self.theta}, Jump Distance: {self.d}>"


class _AdhererIterator:
    def __init__(self, ba: "BoundaryAdherer"):
        self.ba = ba

    def __next__(self):
        if self.ba.has_next:
            return self.ba.sample_next()
        else:
            raise StopIteration


class BoundaryAdherer:
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
        self._classifier = classifier
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
    def bn(self) -> tuple[Point, ndarray]:
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
            self._n = BoundaryAdherer.normalize(
                np.dot(self._rotater_function(ANGLE_90), self._prev_s)
            )
            self.sample_next = lambda: None

        elif self._iteration > self._num and self._prev_b is None:
            raise BoundaryLostException()

        self._iteration += 1
        return self._cur

    def find_boundary(self) -> Point:
        """
        Samples until the boundary point is found.

        Returns:
            Point: The estimated boundary point
        """
        while self.has_next():
            self.sample_next()

        return self._b

    def _next_angle(self, angle: float):
        return (
            -angle / (self._r**self._iteration)
            if self._cur_class
            else angle / (self._r**self._iteration)
        )

    def __iter__(self):
        return _AdhererIterator(self)

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


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes

    ndims = 3

    # Jump Distance and Change in Angle
    d = 0.015
    theta = 90 * np.pi / 180
    num, r = 2, 5

    # The spherical target envelope
    loc = Point([0.5 for x in range(ndims)])
    radius = 0.25
    classifier = lambda p: p.distance_to(loc) <= radius

    p0 = Point([0.5 for x in range(ndims - 1)] + [0.5 - radius])
    n0 = np.array([0 for x in range(ndims - 1)] + [-1])

    direction = np.array([1, 1, 0.2])

    adh = BoundaryAdherer(classifier, p0, n0, direction, d, theta, r, num)

    fig3d = plt.figure()

    # 3D View
    ax3d: Axes = fig3d.add_subplot(111, projection="3d")
    ax3d.set_xlim([0, 1])
    ax3d.set_ylim([0, 1])
    ax3d.set_zlim([0, 1])

    print("Max error =", theta / r**num)

    while adh.has_next():
        p = adh.sample_next()
        ax3d.scatter(*p.array.T, color="b")
        # ar_vec = ax3d.quiver(1.5, 1.5, vec[0], vec[1])
        print(*p.array.T)
        plt.pause(0.05)
        input("Waiting...")
        # ar_vec.remove()

    print("Error =", radius - adh._b.distance_to(loc))

    input("Pres enter to finish...")
