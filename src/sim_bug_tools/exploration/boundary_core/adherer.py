from abc import ABC
from abc import abstractmethod as abstract
from copy import copy
from typing import Callable

import numpy as np
from numpy import ndarray
from sim_bug_tools.structs import Point, Domain


class BoundaryLostException(Exception):
    "When a boundary Adherer fails to find the boundary, this exception is thrown"

    def __init__(self, msg="Failed to locate boundary!"):
        self.msg = msg
        super().__init__(msg)

    def __str__(self):
        return f"<BoundaryLostException: Angle: {self.theta}, Jump Distance: {self.d}>"


class _AdhererIterator:
    def __init__(self, ba: "Adherer"):
        self.ba = ba

    def __next__(self):
        if self.ba.has_next():
            return self.ba.sample_next()
        else:
            raise StopIteration


class Adherer(ABC):
    """
    An Adherer provides the ability to identify a point that lies on the
    boundary of an N-D volume (i.e. target envelope). Furthermore, it allows
    for the incremental stepping through the process and the collection
    of intermediate samples. A "classifier" function describes whether
    or not a sampled Point is within or outside of the target envelope.
    """

    def __init__(self, classifier: Callable[[Point], bool], domain: Domain):
        self._classifier = classifier

    @property
    def classifier(self):
        return self._classifier

    @abstract
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
        pass

    @abstract
    def has_next(self) -> bool:
        """
        Returns:
            bool: True if the boundary has not been found and has remaining
                samples.
        """
        pass

    @abstract
    def find_boundary(self) -> tuple[Point, ndarray]:
        """
        Samples until the boundary point is found.

        Returns:
            Point: The estimated boundary point
            ndarray: The orthonormal surface vector
        """
        pass

    @property
    def boundary(self) -> tuple[Point, ndarray]:
        """
        Boundary point and its surface vector, returns None
        if the boundary hasn't been found yet.
        """
        pass

    @property
    def sub_samples(self):
        pass

    def __iter__(self):
        return _AdhererIterator(self)


class AdherenceFactory(ABC):
    """
    Different adherence strategies can require different initial parameters.
    Since the Explorer does not know what these parameters are, we must decouple
    the construction of the Adherer from the explorer, allowing for initial
    parameters to be defined prior to the execution of the exploration alg.
    """

    def __init__(self, classifier: Callable[[Point], bool], domain: Domain):
        self._classifier = classifier
        self._domain = domain

    @property
    def classifier(self):
        return self._classifier

    @property
    def domain(self):
        return self._domain

    @abstract
    def adhere_from(self, p: Point, n: ndarray, direction: ndarray) -> Adherer:
        """
        Find a boundary point that neighbors the given point, p, in the
        provided direction, given the provided surface vector, n.

        Args:
            p (Point): A boundary point to use as a pivot point
            n (ndarray): The orthonormal surface vector for boundary point p
            direction (ndarray): The direction to sample towards

        Returns:
            Adherer: The adherer object that will find the boundary given the
                above parameters.
        """
        pass
