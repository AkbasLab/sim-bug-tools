import numpy as np
import matplotlib.pyplot as plt
from itertools import permutations

from sim_bug_tools.structs import Point, Domain 
from sim_bug_tools.graphics import Grapher
from sim_bug_tools.exploration.brrt_std.adherer import ConstantAdherer
# from sim_bug_tools.exploration.brrt_std.brrt import BoundaryRRT
def unit_vector(vector):
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

ndims = 3

unit_vectors = set(permutations([0] * (ndims - 1) + [1], ndims))
unit_vectors = list(map(np.array, unit_vectors))

v = [0.5, 0.5, 0.5]
v /= np.linalg.norm(v)

domain = Domain.normalized(ndims)

g = Grapher(ndims == 3)

g.add_arrow(Point.zeros(ndims), v)
g.add_all_arrows([Point.zeros(ndims)] * (ndims - 1), unit_vectors[1:], color="gray")
g.add_arrow(Point.zeros(ndims), unit_vectors[0], color="red")
rotater = ConstantAdherer.generateRotationMatrix(v, np.array(unit_vectors[0]))
theta = angle_between(v, unit_vectors[0])
A = rotater(theta)

# v = np.dot(A, v)
unit_vectors = [np.dot(A, uv) for uv in unit_vectors]

# After change
# g.add_arrow(Point.zeros(ndims), v, linestyle='--')
g.add_all_arrows([Point.zeros(ndims)] * ndims, [np.array(uv) for uv in unit_vectors], color="gray", linestyle='--')

print([np.linalg.norm(uv) for uv in unit_vectors])

plt.show()