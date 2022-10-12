from typing import Callable
import sim_bug_tools.structs as structs
import numpy as np


def find_surface(
    classifier: Callable[[structs.Point], bool], 
    t0: structs.Point, 
    d: float
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
        cur = prev + structs.Point(s)

        if not classifier(cur):
            break
        continue

    s *= 0.5
    ps = structs.Point(s)
    cur = prev + structs.Point(s)

    # Get closer until within d/2 distance from surface
    while classifier(cur):
        prev = cur
        interm += [prev]
        cur = prev + ps

    return ((prev, v), interm)
