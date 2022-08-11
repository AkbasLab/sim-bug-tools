from abc import ABC, abstractmethod as abstract
from copy import copy
from typing import Callable

import numpy as np
from numpy import ndarray
from sim_bug_tools.structs import Point
from adherer import Adherer


class Explorer(ABC):
    def __init__(self, adherer: Adherer):
        self._adherer = adherer

    @property
    def adherer(self):
        return self._adherer

    @abstract
    @property
    def sub_samples(self) -> list[Point]:
        "All samples taken, including the boundary points."
        pass

    @abstract
    def expand(self) -> tuple[Point, ndarray]:
        """
        Take one step along the boundary; i.e., Finds a new
        boundary point.

        Returns:
            tuple[Point, ndarray]: The newly added point on the surface
                and its estimated orthogonal surface vector

        Throws:
            BoundaryLostException: Thrown if the boundary is lost
        """
        pass
