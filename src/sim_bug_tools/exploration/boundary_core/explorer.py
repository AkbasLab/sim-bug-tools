from abc import ABC
from abc import abstractmethod as abstract
from copy import copy
from typing import Callable

import numpy as np
from numpy import ndarray
from sim_bug_tools.structs import Point

from .adherer import (
    AdherenceFactory,
    Adherer,
    BoundaryLostException,
    SampleOutOfBoundsException,
)


class Explorer(ABC):
    """
    An abstract class that provides the skeleton for a Boundary Exploration
    strategy.
    """

    def __init__(self, b0: Point, n0: ndarray, adhererF: AdherenceFactory):
        self._adhererF = adhererF
        self._boundary = [(b0, n0)]

        self._prev = (b0, n0)

        self._adherer: Adherer = None
        self._step_count = 0

        self._test_n = None

    @property
    def prev(self):
        return self._prev

    @property
    def step_count(self):
        return self._step_count

    @property
    def sub_samples(self) -> list[tuple[Point, bool]]:
        "All samples taken, including the boundary points."
        return self._all_points

    @property
    def boundary(self):
        return self._boundary

    @abstract
    def _select_parent(self) -> tuple[Point, ndarray]:
        "Select which boundary point to explore from next."
        pass

    @abstract
    def _pick_direction(self) -> ndarray:
        "Select a direction to explore towards."
        pass

    @abstract
    def _add_child(self, bk: Point, nk: ndarray):
        "Add a newly found boundary point and its surface vector."
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
        self._test_n = n
        direction = self._pick_direction()
        self._adherer = self._adhererF.adhere_from(b, n, direction)
        for b in self._adherer:
            pass

        self._add_child(*self._adherer.boundary)
        self._prev = self._adherer.boundary
        self._step_count += 1
        return self._adherer.boundary

    def expand_by(self, N: int) -> list[tuple[Point, ndarray]]:
        for i in range(N):
            yield self.expand()

    def step(self):
        """
        Take a single sample. Will start a new boundary step if one
        is not already in progress, otherwise take another adherence
        step.

        A "boundary step" is the overall process of take a step along
        the boundary, and a "adherence step" is a single sample towards
        finding the boundary. There are two or more adherence steps per
        boundary step.

        Returns:
            tuple[Point, bool]: Returns the next sampled point and its
            target class.
        """
        p = None
        cls = None
        if self._adherer is None:
            # Start new step
            b, n = self._select_parent()
            direction = self._pick_direction()
            self._test_n = n
            self._test_dir = direction

            try:
                self._adherer = self._adhererF.adhere_from(b, n, direction)
                # NOTE: PRONE TO FAILURE! Not in the ABC, needs refactor
                p = self._adherer._cur
                cls = self._adherer._cur_class

            except SampleOutOfBoundsException as e:
                self._adherer = None
                raise e

        else:
            # Continue to look for boundary
            try:
                p, cls = self._adherer.sample_next()

            except BoundaryLostException as e:
                # If boundary lost, we need to reset adherer and rethrow exception
                self._adherer = None
                raise e

            except SampleOutOfBoundsException as e:
                self._adherer = None
                raise e

        if self._adherer is not None and not self._adherer.has_next():
            # Handle newly found boundary
            node = self._adherer.boundary
            self._boundary.append(node)
            self._add_child(*node)
            self._prev = self._adherer.boundary
            self._adherer = None
            self._step_count += 1

        return p, cls
