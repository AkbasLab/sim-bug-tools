"""
Objectives:
    Visually illustrate pattern changes on sphere of exploration using
        Elliptical vector scaling (EVS)
        Linear vector scaling 
"""

import matplotlib.pyplot as plt
import numpy as np

from sim_bug_tools.exploration.boundary_core.surfacer import find_surface
from sim_bug_tools.exploration.brrt_std.adherer import BoundaryAdherenceFactory
from sim_bug_tools.exploration.brrt_std.brrt import BoundaryRRT
from sim_bug_tools.graphics import Grapher
from sim_bug_tools.structs import Cuboid, Domain, Ellipsoid, Point, Spheroid


# Envelope / BRRT setup
class Sphere:
    def __init__(self, loc: Point, radius: float):
        self.loc = loc
        self.radius = radius

    def __contains__(self, p: Point):
        return self.loc.distance_to(p) <= self.radius
d = 0.1
axis_names = ["1d", "0.5d", "0.75d"]
# How much we want to bias search for each axis:
axes = [1 * d, 0.5 * d, 0.75 * d]


ndims = 3
d = 0.05  # the distance we jump with each sample
theta = 5 * np.pi / 180  # 5 degree rotations for finding the boundary

envelope = Sphere(Point([0.5 for x in range(ndims)]), 0.25)
classifier = lambda p: p in envelope
domain = Domain.normalized(ndims)

is_target = False 
while not is_target:
    v = np.random.randn(ndims)
    v = v / np.linalg.norm(v)
    t0 = Point(np.array(envelope.loc) + v * (envelope.radius * 0.90))
    is_target = t0 in domain 
    if not is_target:
        print("Retrying...")

node0, _ = find_surface(classifier, t0, d, domain)
b0, n0 = node0  # initial boundary point and surface vector


grapher = Grapher(True, domain, axis_names)
grapher.ax.set_title("All")
num_iterations = 500

# The simplest scaling approach is purely to scale using a constant value.
# This is called a Spheroid scaler because it scales with spherical symmetry
scaler = Spheroid(d)
adhFactory = BoundaryAdherenceFactory(classifier, domain, scaler, theta)
brrt = BoundaryRRT(b0, n0, adhFactory)

# nodes = brrt.expand_by(num_iterations)
errs = 0
points = []
for i in range(num_iterations):
    try:
        points.append(brrt.expand()[0])
    except:
        errs += 1
        
print("num errors:", errs)
grapher.plot_all_points(points, color = "red")

sphere_g = Grapher(True, domain, axis_names)
sphere_g.ax.set_title("Sphere Scaler")
sphere_g.plot_all_points(points, color = "red")
plt.pause(0.01)
input("Press enter...")

# Or we can scale dimensions linearly without concern of its magnitude
scaler = Cuboid(axes)
adhFactory = BoundaryAdherenceFactory(classifier, domain, scaler, theta)
brrt = BoundaryRRT(b0, n0, adhFactory)


errs = 0
points = []
for i in range(num_iterations):
    try:
        points.append(brrt.expand()[0])
    except:
        errs += 1
print("num errors:", errs)
grapher.plot_all_points(points, color = "green")
cube_g = Grapher(True, domain, axis_names)
cube_g.plot_all_points(points, color = "green")
cube_g.ax.set_title("Cube Scaler")

plt.pause(0.01)
input("Press enter...")

# Finally e can disproportionately scale dimensions using an Ellipsoid. This
# ensures that the magnitude never exceeds the largest axis, while still biasing
# exploration for each axis.
scaler = Ellipsoid(axes)
adhFactory = BoundaryAdherenceFactory(classifier, domain, scaler, theta)
brrt = BoundaryRRT(b0, n0, adhFactory)

errs = 0
points = []
for i in range(num_iterations):
    try:
        points.append(brrt.expand()[0])
    except:
        errs += 1

print("num errors:", errs)
grapher.plot_all_points(points, color = "blue")
ell_g = Grapher(True, domain, axis_names)
ell_g.plot_all_points(points, color = "blue")
ell_g.ax.set_title("Ellipse Scaler")

plt.pause(0.01)
input("Press enter...")



