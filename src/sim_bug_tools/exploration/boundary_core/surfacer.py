"""
Surfacing algorithms
"""

from typing import Callable

import numpy as np
from numpy import ndarray

from sim_bug_tools.structs import Domain, Point


def find_surface(
    classifier: Callable[[Point], bool], t0: Point, d: float, domain: Domain, 
    v: ndarray = None) -> tuple[tuple[Point, ndarray], list[Point]]:
    """
    Finds the surface given a target sample and a jump distance. The error, e, 
    between the estimated boundary point and the real boundary will be
    0 <= e <= d

    Args:
        classifier (Callable[[Point], bool]): classifies a point as target or non-target
        t0 (Point): A target sample
        d (float): The jump distance to take with each step,
        v (ndarray) [optional]: the direction to find the surface

    Raises:
        Exception: Thrown if the target sample is not classified as a target sample

    Returns:
        tuple[tuple[Point, ndarray], list[Point]]: ((b0, n0), [intermediate_samples])
    """
    if v is not None:
        assert len(np.squeeze(v)) == len(t0)
    else:
        v = np.random.rand(len(t0))

    s = v * d

    interm = [t0]

    prev = None
    cur = t0

    # First, reach within d distance from surface
    while cur in domain and classifier(cur):
        prev = cur
        interm += [prev]
        cur = prev + Point(s)
        
    if prev is None:
        raise Exception("t0 must be a target sample!")

    s *= 0.5
    ps = Point(s)
    cur = prev + Point(s)

    # Get closer until within d/2 distance from surface
    while classifier(cur):
        prev = cur
        interm += [prev]
        cur = prev + ps

    return ((prev, v), interm)
