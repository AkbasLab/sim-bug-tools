from abc import ABC
from abc import abstractmethod as abstract
from copy import copy
from typing import Callable

import numpy as np
from numpy import ndarray
from sim_bug_tools.structs import Point


class BoundaryLostException(Exception):
    def __init__(self, msg="Failed to locate boundary!"):
        self.msg = msg
        super().__init__(msg)

    def __str__(self):
        return f"<BoundaryLostException: Angle: {self.theta}, Jump Distance: {self.d}>"


class _AdhererIterator:
    def __init__(self, ba: "Adherer"):
        self.ba = ba

    def __next__(self):
        if self.ba.has_next:
            return self.ba.sample_next()
        else:
            raise StopIteration


class Adherer(ABC):
    """
    An Adherer provides the ability to identify a point that lies on the
    boundary of an N-D volume (target envelope). A "classifier" function
    describes whether or not a sampled Point is within or outside of the
    target envelope.
    """

    @abstract
    def sample_next(self) -> tuple[Point, ndarray]:
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
        """
        pass

    @abstract
    @property
    def boundary(self) -> tuple[Point, ndarray]:
        """
        Boundary point and its surface vector, returns None
        if the boundary hasn't been found yet.
        """
        pass

    @abstract
    @property
    def sub_samples(self):
        pass

    def __iter__(self):
        return _AdhererIterator(self)
