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

from sim_bug_tools.graphics import Grapher
from sim_bug_tools.structs import Point, Domain, Spheroid


class Sphere:
    def __init__(self, radius: float, loc: Point):
        self.radius = radius
        self.loc = loc

    def __contains__(self, p: Point):
        return (self.loc.distance_to(p)) <= self.radius


ndims = 3
domain = Domain.normalized(ndims)
envelope = Sphere(0.35, Point([0.5 for x in range(ndims)]))
classifier = lambda p: p in envelope

# b0 is at the top of the sphere
b0 = Point([0.5 for x in range(ndims - 1)] + [0.5 + envelope.radius])
n0 = np.array([0 for x in range(ndims - 1)] + [1])
direction = np.array([1] + [0 for x in range(ndims - 1)])

d = 0.05
scaler = Spheroid(d)

## ConstantAdh
delta_theta = np.pi * 5 / 180
const_adh_f = ConstantAdherenceFactory(classifier, scaler, delta_theta, domain, True)

## ExpAdh
theta0 = np.pi / 2  # 90 degrees
num = 4  # iterations
exp_adh_f = ExponentialAdherenceFactory(classifier, scaler, theta0, num, domain, True)

_all_names = "x,y,z".split(",")

g = Grapher(ndims == 3, domain, _all_names[:ndims])

print("Constant algorithm...")
const_adh = const_adh_f.adhere_from(b0, n0, direction)
points = []

gb0 = g.plot_point(b0, color="green")
gn0 = g.add_arrow(b0, n0, color="grey")

gp = g.plot_point(const_adh._cur)
gs = g.add_arrow(b0, const_adh._s, color="blue")

g.draw_sphere(envelope.loc, envelope.radius, fill=False)

while const_adh.has_next():
    gp.remove()
    gs.remove()

    p, cls = const_adh.sample_next()
    points.append((p, cls))

    gp = g.plot_point(const_adh._cur)
    gs = g.add_arrow(b0, const_adh._s, color="blue")
    
    plt.pause(0.01)



exp_adh = exp_adh_f.adhere_from(b0, n0, direction)


while exp_adh.has_next():
    gp.remove()
    gs.remove()

    p, cls = exp_adh.sample_next()
    points.append((p, cls))

    gp = g.plot_point(exp_adh._prev)
    gs = g.add_arrow(b0, (exp_adh._prev - b0).array, color="blue")
    
    plt.pause(0.01)