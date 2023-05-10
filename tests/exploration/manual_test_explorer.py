import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt

import sys

print(sys.version)

from sim_bug_tools.exploration.boundary_core.adherer import (
    SampleOutOfBoundsException,
    BoundaryLostException,
)
from sim_bug_tools.exploration.brrt_std.adherer import ConstantAdherenceFactory
from sim_bug_tools.exploration.brrt_v2.adherer import ExponentialAdherenceFactory

from sim_bug_tools.exploration.brrt_std.brrt import BoundaryRRT

from sim_bug_tools.graphics import Grapher
from sim_bug_tools.structs import Point, Domain, Spheroid


class Sphere:
    def __init__(self, radius: float, loc: Point):
        self.radius = radius
        self.loc = loc

    def __contains__(self, p: Point):
        return (self.loc.distance_to(p)) <= self.radius


ADH_V = 1


ndims = 3
domain = Domain.normalized(ndims)
envelope = Sphere(0.35, Point([0.5 for x in range(ndims)]))
classifier = lambda p: p in envelope

b0 = Point([0.5 for x in range(ndims - 1)] + [0.5 + envelope.radius])
n0 = np.array([0 for x in range(ndims - 1)] + [1])

d = 0.05
scaler = Spheroid(d)

## ConstantAdh
delta_theta = np.pi * 5 / 180
const_adh_f = ConstantAdherenceFactory(classifier, scaler, delta_theta, domain, True)

## ExpAdh
theta0 = np.pi / 2  # 90 degrees
num = 6  # iterations
exp_adh_f = ExponentialAdherenceFactory(classifier, scaler, theta0, num, domain, True)

g = Grapher(ndims == 3, domain, "x y z".split()[:ndims])

if ADH_V == 1:
    adh_f = const_adh_f
elif ADH_V == 2:
    adh_f = exp_adh_f

brrt = BoundaryRRT(b0, n0, adh_f)

points = []
while brrt.boundary_ < 100:
    points.append(brrt.step())

boundary_set = set([tuple(b) for b, _ in brrt.boundary])
g.plot_all_points([p for p, _ in brrt.boundary], color="blue", marker="o")
if len(lst := [p for p, cls in points if cls and tuple(p) not in boundary_set]) > 0:
    g.plot_all_points(lst, color="red", marker="x")
if len(lst := [p for p, cls in points if not cls]) > 0:
    g.plot_all_points(lst, color="green", marker="x")
plt.show()
