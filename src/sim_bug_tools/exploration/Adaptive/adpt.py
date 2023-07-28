from copy import copy
from typing import Callable

import numpy as np
from numpy import ndarray

from sim_bug_tools.exploration.brrt_std import ConstantAdherenceFactory
from sim_bug_tools.exploration.brrt_v2 import ExponentialAdherenceFactory
from sim_bug_tools.exploration.boundary_core.adherer import (
    AdherenceFactory,
    Adherer,
    BoundaryLostException,
    SampleOutOfBoundsException,
)
from sim_bug_tools.structs import Domain, Point, Scaler


class AdaptiveAdherer(Adherer):
    MODE_IDLE = 0
    MODE_CONST = 1
    MODE_EXP = 2
    MODE_COMPLETE = 3
    MODE_FAIL = 4

    def __init__(
        self,
        classifier: Callable[[Point], bool],
        b: Point,
        n: ndarray,
        direction: ndarray,
        const_adh_f: ConstantAdherenceFactory,
        exp_adh_f: ExponentialAdherenceFactory,
        domain: Domain = None,
        fail_out_of_bounds: bool = True,
    ):
        super().__init__(classifier, (b, n), direction, domain, fail_out_of_bounds)
        self._const_f = const_adh_f
        self._exp_f = exp_adh_f

        self._adh = None
        self._mode = self.MODE_IDLE

    def sample_next(self) -> tuple[Point, bool]:
        if self._adh is None:
            self._adh = self._const_f.adhere_from(*self.parent_bnode, self.direction)
            self._mode = self.MODE_CONST

        try:
            r = self._adh.sample_next()
        except BoundaryLostException as e:
            if self._mode == self.MODE_CONST:
                self._adh = self._exp_f.adhere_from(*self.parent_bnode, self.direction)
                r = self._adh.sample_next()
                self._mode = self.MODE_EXP
            else:
                self._adh = None
                self._mode = self.MODE_FAIL
                raise e

        if self._adh is not None and not self._adh.has_next():
            self._new_b, self._new_n = self._adh.bnode
            self._mode = self.MODE_COMPLETE

        return r


class AdaptiveAdherenceFactory(AdherenceFactory[AdaptiveAdherer]):
    def __init__(
        self,
        classifier: Callable[[Point], bool],
        const_adh_f: ConstantAdherenceFactory,
        exp_adh_f: ExponentialAdherenceFactory,
        domain: Domain = None,
        fail_out_of_bounds: bool = True,
    ):
        super().__init__(classifier, domain, fail_out_of_bounds)
        self._const_f = const_adh_f
        self._exp_f = exp_adh_f

    def adhere_from(self, b: Point, n: ndarray, direction: ndarray) -> AdaptiveAdherer:
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
        return AdaptiveAdherer(
            self.classifier,
            b,
            n,
            direction,
            self._const_f,
            self._exp_f,
            self.domain,
            self._fail_out_of_bounds,
        )
