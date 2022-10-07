"""
Surfacing algorithms
"""

from typing import Callable
from sim_bug_tools.structs import Point
import numpy as np
from numpy import ndarray


def surface(
    classifier: Callable[[Point], bool], t0: Point, d: float
) -> tuple[tuple[Point, ndarray], list[Point]]:
    v = np.random.rand(len(t0))

    s = v * d

    interm = [t0]

    prev = None
    cur = t0

    # First, reach within d distance from surface
    while classifier(cur):
        prev = cur
        interm += [prev]
        cur = prev + Point(s)

    s *= 0.5
    ps = Point(s)
    cur = prev + Point(s)

    # Get closer until within d/2 distance from surface
    while classifier(cur):
        prev = cur
        interm += [prev]
        cur = prev + ps

    return ((prev, v), interm)
