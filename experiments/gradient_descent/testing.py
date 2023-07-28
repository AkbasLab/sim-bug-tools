import matplotlib.pyplot as plt
import numpy as np

from numpy import ndarray

from sim_bug_tools.graphics import Grapher
from sim_bug_tools import Point, Domain

from exp_ann import ProbilisticSphere, ProbilisticSphereCluster

# locs = np.array(
#     (
#         (0, 0),
#         (1, 1),
#         (0.5, 0),
#         (0.25, 0.25),
#     )
# )

# radii = np.array((0.25, 0.5, 0.5, 0.5))

# lmbda = np.array(
#     (
#         0.25,
#         0.25,
#         0.25,
#         0.25,
#     )
# )


# def all_error(p: ndarray):
#     print(p - locs)
#     return abs(np.linalg.norm(p - locs, axis=1) - radii)


# def lowest_error_index(p: ndarray):
#     print("Errors:\n", all_error(p).T)
#     return min(enumerate(all_error(p).T), key=lambda pair: pair[1])[0]


# p = np.array((1.0, 1.5))

# print(lowest_error_index(p))


ndim = 2
domain = Domain.normalized(ndim)
g = Grapher(ndim == 3, domain)

# r = 0.17
# n = 7
# k = 4
# # n = 3
# # k = 2

loc = Point(0.5, 0.5)

r = 0.17
n = 7
k = 4

clst = ProbilisticSphereCluster(
    n,
    k,
    r,
    loc,
    min_dist_b_perc=-0.05,
    max_dist_b_perc=0,
    min_rad_perc=0.5,
    max_rad_perc=0.01,
    seed=1,
    domain=domain,
)

index = 1670
print(
    "radii, lmbda, loc",
    radius := clst._sph_radii[index],
    lmbda := clst._sph_lmbda[index],
    loc := clst._sph_locs[index],
)
print("c", 1 / radius**2 * np.log(1 / lmbda))

p = Point(0.099, 0.324)
print(clst.boundary_err(p))
print("score:", clst.score(p))
print("score:", clst.classify(p))

print(sum(map(lambda sph: sph.score(p), clst.spheres)))
dist = 0
closest = None

scrs = (clst._base) ** np.linalg.norm(p.array - clst._sph_locs, axis=1) ** 2

print("locations:", p.array, clst._sph_locs[1670])
print("base, dif, distance, distance^2, score")
print(
    clst._base[1670],
    p.array - clst._sph_locs[1670],
    np.linalg.norm(p.array - clst._sph_locs[1670]),
    np.linalg.norm(p.array - clst._sph_locs[1670]) ** 2,
    scrs[1670],
)

for index, sph, clst_scr in zip(range(len(scrs)), clst.spheres, scrs):
    # if (sph_scr := sph.score(p)) > 1e-10:
    #     print("index:", index)
    #     print("sphere:", sph_scr)
    #     print("clst:", clst_scr)

    g.draw_sphere(sph.loc, sph.radius, facecolor="none", edgecolor="black")
    if closest is None:
        closest = sph
        dist = sph.loc.distance_to(p)
        c_ind = index
    elif sph.loc.distance_to(p) < dist:
        closest = sph
        dist = sph.loc.distance_to(p)
        c_ind = index

print("Closest =", closest)
g.draw_sphere(closest.loc, closest.radius, facecolor="none", edgecolor="red")
print("C-score", closest.score(p), closest.boundary_err(p))


g.plot_point(p, color="red")

print(closest.radius, closest.loc, closest.lmda)
print("index", c_ind)

plt.show()
print()
