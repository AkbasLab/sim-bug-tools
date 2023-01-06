import numpy as np
from numpy import ndarray, float64
from sim_bug_tools.exploration.boundary_core.adherer import (
    BoundaryLostException,
    SampleOutOfBoundsException,
)
import json

DATA_OUTPUT = "output.json"

ADHERER_VERSION = "v1"
ENVELOPE_TYPE = "sphere"
ndims = 30

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
from sim_bug_tools.structs import Domain, Point, Spheroid

# To illustrate this visually, use only 3 dimensions

# We need to define some basic parameters for later:
d = 0.05  # the distance we jump with each sample

# Constant Adherence Strategy
delta_theta = 15 * np.pi / 180  # 5 degree rotations for finding the boundary

# Exponential Adherence Strategy
theta0 = np.pi / 180 * 90  # Exponential Adherence Strategy: 90 degree start rotation
r = 2
N = 4

# We will be using a simple sphere as our envelope, for simplicity's sake.
class Sphere:
    def __init__(self, loc: Point, radius: float):
        self.loc = loc
        self.radius = radius

    def __contains__(self, p: Point):
        return self.loc.distance_to(p) <= self.radius

    def theoretical_normal(self, p: Point) -> ndarray:
        dir = (p - self.loc).array
        return dir / np.linalg.norm(dir)


class Cube:
    def __init__(self, center_loc: Point, length: float):
        self.loc = center_loc
        self.length = length

        half_length = length / 2
        _high = center_loc + Point([half_length for i in range(len(center_loc))])
        _low = center_loc - Point([half_length for i in range(len(center_loc))])
        self._domain = Domain.from_bounding_points(_low, _high)

    def __contains__(self, p: Point):
        return p in self._domain


# Setting up envelope:
center_loc = Point([0.5 for x in range(ndims)])

if ENVELOPE_TYPE == "sphere":
    envelope = Sphere(center_loc, 0.25)
elif ENVELOPE_TYPE == "cube":
    envelope = Cube(center_loc, 0.25)

# Setting up classifying function:
# If the point lies WITHIN the envelope, it is classified as a target sample
classifier = lambda p: p in envelope


domain = Domain.normalized(ndims)

# We are using the point below as our initial sample...
# Generate an initial target sample:
is_target = False
while not is_target:
    v = np.random.randn(ndims)
    v = v / np.linalg.norm(v)
    if ENVELOPE_TYPE == "sphere":
        t0 = Point(np.array(envelope.loc) + v * (envelope.radius * 0.90))
    elif ENVELOPE_TYPE == "cube":
        t0 = Point(np.array(envelope.loc) + v * (envelope.length / 2 * 0.90))
    is_target = t0 in envelope
    if not is_target:
        print("Retrying...")


# Since we don't know if the target sample is on the boundary, we must surface:
from sim_bug_tools.exploration.boundary_core.surfacer import find_surface

# By default, find_surface randomly picks a direction; however, a better
# solution is to find both a target and non-target sample, using their
# normalized displacement vector for the direction of travel (v).
tmp = find_surface(classifier, t0, d, domain)
if tmp is not None:
    node0, _, _ = tmp
else:
    print("none?")
b0, n0 = node0  # initial boundary point and surface vector

# Now that we have our inital boundary point and its orthonormal surface vector,
# we now construct our surface explorer...

# First, we must select and build our adherence strategy's factory, we will use
# the predefined solutions. BAF1 is Constant Adherer, and BAF2 is Exponential.
from sim_bug_tools.exploration.brrt_std.adherer import (
    ConstantAdherenceFactory as ConstAdherer,
)
from sim_bug_tools.exploration.brrt_v2.adherer import (
    ExponentialAdherenceFactory as ExpAdherer,
)

sphere_scaler = Spheroid(d)

if ADHERER_VERSION == "v1":
    adhFactory = ConstAdherer(classifier, domain, sphere_scaler, delta_theta, True)
elif ADHERER_VERSION == "v2":
    adhFactory = ExpAdherer(classifier, domain, d, theta0, r, N, True)

# Now, we can create our explorer, in our case we will use our default Boundary
# Rapidly Exploring Random Tree:
from sim_bug_tools.exploration.brrt_std.brrt import BoundaryRRT

if ndims == 3:
    import matplotlib.pyplot as plt
    from sim_bug_tools.graphics import Grapher  # (used for plotting)

    grapher = Grapher(is3d=True, domain=Domain.normalized(ndims))
    if ENVELOPE_TYPE == "sphere":
        grapher.draw_sphere(envelope.loc, envelope.radius)
    # a = grapher.add_arrow(b0, n0)
    # plt.pause(0.01)

# Finally, we can start using this explorer to sample from the boundary:
nsamples = 500

i = 0
bp_k = 1  # Back-prop depth, 0 for no backprop
batch = 1

backprop_enabled = bp_k > 0

brrt = BoundaryRRT(b0, n0, adhFactory)

i = 0
all_points: list[tuple[Point, bool]] = []
all_exp_steps = []
exp_step = {"b-node": None, "all-points": []}
errs = 0
out_of_bounds_count = 0

while brrt.step_count < (nsamples):
    try:
        tmp = brrt.step()
        all_points.append(tmp)
        exp_step["all-points"].append((tuple(tmp[0]), tmp[1]))

    except BoundaryLostException:
        # Occurs when the adherer fails to find the boundary
        errs += 1

    except SampleOutOfBoundsException as e:
        # Occurs (iff fail_out_of_bounds enabled) when the sample is outside the
        # set domain (e.g. outside the normalized domain.) If not enabled, then
        # it will simply adhere to the boundary between the domain edge and the
        # envelope.
        out_of_bounds_count += 1

    if brrt.step_count > i:
        if backprop_enabled:
            brrt.back_propegate_prev(bp_k)

        exp_step["b-node"] = [tuple(x) for x in brrt.boundary[-1]]
        all_exp_steps.append(exp_step)
        exp_step = {"b-node": None, "all-points": []}
        i = brrt.step_count

# How to determine the non-boundary points
non_boundary_points = [
    (tuple(p), bool(cls))
    for p, cls in all_points
    if p not in map(lambda node: node[0], brrt.boundary)
]

# A summary of the results
data = {
    "name": f"BRRT{ADHERER_VERSION} {ENVELOPE_TYPE} {ndims}d - {nsamples}",
    "backprop-enabled": backprop_enabled,
    "backprop-params": {"k": bp_k} if backprop_enabled else None,
    "brrt-type": ADHERER_VERSION,
    "brrt-params": {"d": d, "theta": delta_theta}
    if ADHERER_VERSION == "v1"
    else {"d": d, "delta-theta": theta0, "r": r, "N": N},
    "meta-data": {
        "err-count": errs,
        "out-of-bounds-count": out_of_bounds_count,
        "ratio": brrt.step_count / len(non_boundary_points),
        "b-count": brrt.step_count,
        "nonb-count": len(non_boundary_points),
    },
    "dimensions": [str(x) for x in range(ndims)],
    "steps": all_exp_steps,
    "b-nodes": [(tuple(b), tuple(n)) for b, n in brrt.boundary],
}

print("Saving results to json file...")
with open(
    DATA_OUTPUT,
    "w",
) as f:
    f.write(json.dumps(data, indent=4))
