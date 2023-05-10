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
    T_BNODE,
)


class ExplorationCompletedException(Exception):
    "When an exploration algorithm has finished, this exception may be thrown."

    def __init__(self, msg="Exploration complete"):
        self.msg = msg
        super().__init__(msg)

    def __str__(self):
        return f"<ExplorationCompletedException: {self.msg}>"


class Explorer(ABC):
    """
    An abstract class that provides the skeleton for a Boundary Exploration
    strategy.
    """

    def __init__(self, b0: Point, n0: ndarray, adhererF: AdherenceFactory):
        self._adhererF = adhererF
        self._boundary = [(b0, n0)]
        self._ndims = len(b0)

        self._prev = (b0, n0)

        self._adherer: Adherer = None
        self._boundary_count = 0

    @property
    def ndims(self) -> int:
        return self._ndims

    @property
    def prev(self) -> T_BNODE:
        "Previous boundary node"
        return self._prev

    @property
    def boundary_count(self):
        "The number of boundary nodes found, not including root."
        return self._boundary_count

    @property
    def boundary(self):
        "Returns the set of boundary nodes, including root."
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

    def _explore_in_new_direction(self, parent: T_BNODE, direction: ndarray):
        # Start new step
        b, n = parent
        self._tmp_parent = parent
        direction = direction

        self._adherer = self._adhererF.adhere_from(b, n, direction)

    def _continue_boundary_search(self):
        # Continue to look for boundary
        try:
            p, cls = self._adherer.sample_next()
            self._s = self._adherer._s

        except BoundaryLostException as e:
            # If boundary lost, we need to reset adherer and rethrow exception
            self._adherer = None
            raise e

        except SampleOutOfBoundsException as e:
            self._adherer = None
            raise e

        return p, cls

    def step(self):
        """
        Take a single sample. Will start a new boundary step if one
        is not already in progress, otherwise take another adherence
        step.

        A "boundary step" is the overall process of take a step along
        the boundary, and a "adherence step" is a single sample towards
        finding the boundary. There are two or more adherence steps per
        boundary step.

        Sequence for constructing adherer:
            1. _select_parent
            2. _pick_direction
            3. _explore_in_new_direction

        Sequence after boundary found:
            1. fetch boundary from adherer
            2. append node to _boundary
            3. _add_child()
            4. set _prev
            5. reset _adherer
            6. update _step_count

        Returns:
            tuple[Point, bool]: Returns the next sampled point and its
            target class.
        """
        if self._adherer is None:
            self._explore_in_new_direction(
                self._select_parent(), self._pick_direction()
            )

        p, cls = self._continue_boundary_search()

        if self._adherer is not None and not self._adherer.has_next():
            # Handle newly found boundary
            node = self._adherer.bnode
            self._boundary.append(node)
            self._add_child(*node)
            self._prev = self._adherer.bnode
            self._adherer = None
            self._boundary_count += 1

        return p, cls
