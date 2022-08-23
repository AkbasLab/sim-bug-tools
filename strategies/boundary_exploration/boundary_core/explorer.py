from abc import ABC, abstractmethod as abstract
from copy import copy
from typing import Callable

import numpy as np
from numpy import ndarray
from sim_bug_tools.structs import Point
from adherer import AdherenceFactory


class Explorer(ABC):
    def __init__(self, b0: Point, n0: ndarray, adhererF: AdherenceFactory):
        self._adhererF = adhererF
        self._boundary = [b0]
        self._all_points = []

        self._prev = (b0, n0)

    @property
    def prev(self):
        return self._prev

    @property
    def sub_samples(self) -> list[Point]:
        "All samples taken, including the boundary points."
        return self._all_points

    @property
    def boundary(self):
        return self._boundary

    @abstract
    def _select_parent(self) -> tuple[Point, ndarray]:
        pass

    @abstract
    def _pick_direction(self) -> ndarray:
        pass

    @abstract
    def _add_child(self, bk: Point, nk: ndarray):
        pass

    def expand(self) -> tuple[Point, ndarray]:
        """
        Take one step along the boundary; i.e., Finds a new
        boundary point.

        Returns:
            tuple[Point, ndarray]: The newly added point on the surface
                and its estimated orthonormal surface vector

        Throws:
            BoundaryLostException: Thrown if the boundary is lost
        """
        b, n = self._select_parent()
        direction = self._pick_direction()
        adherer = self._adhererF.adhere_from(b, n, direction)
        for b in adherer:
            self._all_points.append(b)

        self._add_child(*adherer.boundary)
        self._prev = adherer.boundary
        return adherer.boundary

    def expand_by(self, N: int) -> list[tuple[Point, ndarray]]:
        for i in range(N):
            yield self.expand()
