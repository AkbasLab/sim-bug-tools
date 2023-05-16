# Import external modules
import numpy as np
import matplotlib.pyplot as plt

# Import internal modules

# Imports from external modules
from numpy import ndarray

# Imports from internal modules
from sim_bug_tools.exploration import (
    Adherer,
    Explorer,
    ConstantAdherenceFactory,
    ExponentialAdherenceFactory,
)
from sim_bug_tools.simulation.simulation_core import Scorable
from sim_bug_tools.graphics import Grapher
from sim_bug_tools.structs import Point, Domain, Spheroid


class Square(Scorable):
    def __init__(self, domain: Domain):
        self._domain = domain

    @property
    def domain(self):
        return self._domain

    def score(self, p: Point) -> ndarray:
        return np.array([1]) if p in self._domain else np.array([0])

    def classify_score(self, score: ndarray) -> bool:
        return score[0] == 1

    def get_input_dims(self):
        return len(self._domain)

    def get_score_dims(self):
        return 1

    def generate_random_target(self):
        raise NotImplementedError()

    def generate_random_nontarget(self):
        raise NotImplementedError()


ndims = 2
domain = Domain.normalized(ndims)
envelope = Square(Domain([(0, 0.5), (0, 0.5)]))

delta_theta = 10 * np.pi / 180
theta0 = 100 * np.pi / 180
N = 5
scaler = Spheroid(0.1)

boundary = [
    Point(0.05, 0.48),
    Point(0.15, 0.48),
    Point(0.25, 0.48),
    Point(0.35, 0.48),
    Point(0.45, 0.48),
]

assert all(
    map(envelope.domain.__contains__, boundary)
), "Not all boundary points are in envelope?"


g = Grapher(domain=domain)
# g.draw_cube(envelope.domain)
g.draw_path(
    [
        Point(0, 0.5),
        Point(0.5, 0.5),
        Point(0.5, 0),
    ],
    color="red",
)

g.plot_all_points(boundary, color="red")
g.draw_path(boundary, color="red")
plt.pause(0.01)

const_adh_f = ConstantAdherenceFactory(
    envelope.classify, scaler, delta_theta, domain, fail_out_of_bounds=True
)

exp_adh_f = ExponentialAdherenceFactory(
    envelope.classify, scaler, theta0, N, domain, True
)

b0 = boundary[-1]
n0 = np.array([0, 1])
direction = np.array([1, 0.1])
adh = const_adh_f.adhere_from(b0, n0, direction)
# adh = exp_adh_f.adhere_from(b0, n0, direction)

_s = None
_p = None
i = 0
while adh.has_next():
    p, c = adh.sample_next()
    i += 1

    s = (p - b0).array

    if _s is not None:
        _s.remove()
        _p.remove()
    _p = g.plot_point(p, color="black")
    _s = g.add_arrow(b0, s, color="black")
    plt.pause(0.4)

_s.remove()
_p.remove()

print(i)

g.plot_point(p, color="red")
g.draw_path([boundary[-1], p], color="red")
plt.pause(2)
