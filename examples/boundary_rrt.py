import numpy as np

"""
We can use the pre-defined boundary exploration strategy "BoundaryRRT" to 
explore the boundary of our envelope.

The "classifier" is the function that is executed to determine if a given point
lies within our outside of the envelope. Your classifier may wrap your function
under test (FUT) or even an API call to a simulated environment.

# For example:
def classifier(p: Point) -> bool:
    val = f(*p)
    return val > 10

The example above classifiers the function, f, by comparing its result to a 
constant. If the function results in a value greater than 10, then the sample,
p, is a target value, otherwise it is a non-target value.
"""

# Importing commonly used structures
from sim_bug_tools.structs import Domain, Point

# To illustrate this visually, we will stick to only 3 dimensions
ndims = 3

# We need to define some basic parameters for later:
d = 0.05  # the distance we jump with each sample
theta = np.pi / 180 * 5  # 5 degree rotations for finding the boundary

# We will be using a simple sphere as our envelope, for simplicity's sake.
class Sphere:
    def __init__(self, loc: Point, radius: float):
        self.loc = loc
        self.radius = radius

    def __contains__(self, p: Point):
        return self.loc.distance_to(p) <= self.radius


envelope = Sphere(Point([0.5 for x in range(ndims-1)] + [0]), 0.25)

# If the point lies WITHIN the envelope, it is classified as a target sample
classifier = lambda p: p in envelope
domain = Domain.normalized(ndims)

# We are using the point below as our initial sample...
# Generate an initial target sample:
is_target = False 
while not is_target:
    v = np.random.randn(ndims)
    v = v / np.linalg.norm(v)
    t0 = Point(np.array(envelope.loc) + v * (envelope.radius * 0.90))
    is_target = t0 in domain 
    if not is_target:
        print("Retrying...")


# Since we don't know if the target sample is on the boundary, we must surface:
from sim_bug_tools.exploration.boundary_core.surfacer import find_surface

node0, _ = find_surface(classifier, t0, d)
b0, n0 = node0  # initial boundary point and surface vector

# Now that we have our inital boundary point and its orthonormal surface vector,
# we now construct our surface explorer...

# First, we must select and build our adherence strategy's factory, we will use
# the predefined solution:
from sim_bug_tools.exploration.brrt_std.adherer import BoundaryAdherenceFactory

adhFactory = BoundaryAdherenceFactory(classifier, domain, d, theta)

# Now, we can create our explorer:
from sim_bug_tools.exploration.brrt_std.brrt import BoundaryRRT

brrt = BoundaryRRT(b0, n0, adhFactory)

# Finally, we can start using this explorer to sample from the boundary:
nsamples = 100
nbatches = 20

if ndims == 3:
    import os
    import sys

    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import matplotlib.pyplot as plt
    from tools.grapher import Grapher  # (used for plotting)

    grapher = Grapher(is3d=True, domain=Domain.normalized(ndims))


for i in range(nbatches):
    nodes = brrt.expand_by(nsamples)
    points = list(map(lambda s: s[0], nodes))

    if ndims == 3:
        grapher.plot_all_points(points)
        plt.pause(0.05)

    input("Press enter to continue...")
