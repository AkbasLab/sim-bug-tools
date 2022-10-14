from typing import Callable
import sim_bug_tools.structs as structs
import numpy as np


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
