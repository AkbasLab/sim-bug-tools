from typing import Callable
from copy import copy
import numpy as np
import rtree
import treelib

import sim_bug_tools.structs as structs
import sim_bug_tools.exploration.brrt_std.brrt as brrt_std
import sim_bug_tools.exploration.brrt_std.adherer as adherer_std
import sim_bug_tools.exploration.boundary_core.adherer as adherer_core



class BoundaryRRT(brrt_std.BoundaryRRT):

    def __init__(self, 
        b0: structs.Point, 
        n0: np.ndarray, 
        adhererF: adherer_core.AdherenceFactory):
        """
        Args:
            classifier (Callable[[Point], bool]): The function that determines whether
                or not a sampled point is a target value or not.
            b0 (Point): The root boundary point to begin exploration from.
            n0 (ndarray): The root boundary point's orthonormal surface vector.
            adhererF (AdherenceFactory): A factory for the desired adherence
                strategy.
        """
        super().__init__(b0, n0, adhererF)
        return

    @property
    def tree(self) -> treelib.Tree:
        return self._tree

    @property
    def index(self) -> rtree.index.Index:
        return self._index

    # def expand(self) -> tuple[structs.Point, np.ndarray]:
    #     """
    #     Take one step along the boundary; i.e., Finds a new
    #     boundary point.

    #     Returns:
    #         tuple[Point, ndarray]: The newly added point on the surface
    #             and its estimated orthonormal surface vector

    #     Throws:
    #         BoundaryLostException: Thrown if the boundary is lost
    #     """
    #     b, n = self._select_parent()
    #     direction = self._pick_direction()
    #     adherer = self._adhererF.adhere_from(b, n, direction)
        
    #     for b in adherer:
    #         self._all_points.append(b)

    #     self._add_child(*adherer.boundary)
    #     self._prev = adherer.boundary
    #     return adherer.boundary


class BoundaryAdherenceFactory(adherer_core.AdherenceFactory):
    def __init__(
        self,
        classifier: Callable[[structs.Point], bool],
        d: float,
        theta: float,
        domain : structs.Domain
    ):
        super().__init__(classifier, domain)
        self._d = d
        self._theta = theta
        return

    def adhere_from(self, 
        p: structs.Point, 
        n: np.ndarray, 
        direction: np.ndarray
    ) -> adherer_core.Adherer:
        return adherer_std.BoundaryAdherer(
            self.classifier, 
            self.domain,
            p, 
            n, 
            direction, 
            self._d, 
            self._theta
        )



def round_to_limits(
        arr : np.ndarray, 
        min : np.ndarray, 
        max : np.ndarray
    ) -> np.ndarray:
    """
    Rounds each dimensions in @arr to limits within @min limits and @max limits.
    """
    is_lower = arr < min
    is_higher = arr > max
    for i in range(len(arr)):
        if is_lower[i]:
            arr[i] = min[i]
        elif is_higher[i]:
            arr[i] = max[i]
    return arr

def find_surface(
    classifier: Callable[[structs.Point], bool], 
    t0: structs.Point, 
    d: structs.Point,
    lim_min : structs.Point,
    lim_max : structs.Point
) -> tuple[tuple[structs.Point, np.ndarray], list[structs.Point]]:
    v = np.random.rand(len(t0))
    s = v * d
    interm = [t0]

    prev = None
    cur = t0


    # First, reach within d distance from surface
    while True:
        prev = cur
        interm += [prev]
        cur = structs.Point(round_to_limits(
            (prev + structs.Point(s)).array,
            lim_min.array,
            lim_max.array
        ))

        # At parameter boundary
        # print(cur-prev)
        if cur == prev:
            break
        
        if not classifier(cur):
            break
        continue

    s *= 0.5
    ps = structs.Point(s)
    cur = structs.Point(round_to_limits(
        (prev + structs.Point(s)).array,
        lim_min.array,
        lim_max.array
    ))

    # Get closer until within d/2 distance from surface
    while cur != prev and classifier(cur):
        prev = cur
        interm += [prev]
        cur = structs.Point(round_to_limits(
            (prev + ps).array,
            lim_min.array,
            lim_max.array
        ))
        
        # At parameter boundary
        if cur == prev:
            break
        continue
        

    return ((prev, v), interm)
