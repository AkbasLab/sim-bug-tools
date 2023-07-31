import json

import numpy as np
import sim_bug_tools.exploration.brute_force as utils
import matplotlib.pyplot as plt

from copy import copy
from rtree.index import Index, Property
from time import time
from contextlib import contextmanager

from exp_ann import ProbilisticSphereCluster, ProbilisticSphere
from exp_expl import (
    MeshExplorer,
    BoundaryRRT,
    ProbabilisticCube,
    ConstantAdherenceFactory,
    ExponentialAdherenceFactory,
    ExplorationExperiment,
    ExplorationParams,
)
from sim_bug_tools import find_surface, Grid, Domain, Point, Spheroid
from sim_bug_tools.graphics import Grapher


def create_envelope(e_type: str, loc: Point, seed: int, domain: Domain):
    if e_type == "sphere":
        return ProbilisticSphere(loc, 0.35, 0.25)
    elif e_type == "sphere-cluster":
        r = 0.17
        n = 7
        k = 4

        return ProbilisticSphereCluster(
            n,
            k,
            r,
            loc,
            min_dist_b_perc=-0.05,
            max_dist_b_perc=0,
            min_rad_perc=0.5,
            max_rad_perc=0.01,
            seed=seed,
            domain=domain,
        )
    elif e_type == "cube":
        return ProbabilisticCube(loc, 0.6, 1, 0.25)


# seed = 1
e_type = "sphere-cluster"
adh_type = "const"
exp_type = "mesh"

ndims = [2, 3, 4, 5, 6]
d = 0.075
delta_theta = np.pi * 10 / 180
theta0 = np.pi / 2
scaler = Spheroid(d)


expl_exp = ExplorationExperiment()


@contextmanager
def stopwatch():
    times = {"t0": time(), "t1": None, "total": None}
    yield times
    times["t1"] = time()
    times["total"] = times["t1"] - times["t0"]


for nd in ndims:
    loc = Point([0.5] * nd)
    domain = Domain.normalized(nd)

    coverage = []

    t0, t1 = 0, 0

    for seed in range(20):
        with stopwatch() as setup_times:
            p = Property()
            p.set_dimension(nd)
            rtree_index = Index(properties=p)
            envelope = create_envelope(e_type, loc, seed, domain)

            (b0, n0), _, valid = find_surface(
                envelope.classify, loc, d, domain, fail_out_of_bounds=True
            )

            print(envelope.boundary_err(b0))
            if adh_type == "const":
                adh_f = ConstantAdherenceFactory(
                    envelope.classify,
                    scaler,
                    delta_theta,
                    Domain.normalized(nd),
                    True,
                )
            elif adh_type == "exp":
                adh_f = ExponentialAdherenceFactory(
                    envelope.classify, scaler, theta0, 4, domain, True
                )

            if exp_type == "mesh":
                expl = MeshExplorer(
                    b0, n0, adh_f, scaler, -0.02, back_prop_on_group=True
                )
            elif exp_type == "rrt":
                expl = BoundaryRRT(b0, n0, adh_f)

            expl_params = ExplorationParams(
                f"brrt-{e_type}-{adh_type}-{nd}d-{seed}", expl, None
            )

        with stopwatch() as expl_times:
            results = expl_exp.experiment(expl_params)

        # osv_errs = [
        #     MeshExplorer.angle_between(
        #         n, (b - loc).array / np.linalg.norm((b - loc).array)
        #     )
        #     # envelope.osv_err(b, n)
        #     for b, n in results.bpoints
        # ]

        # avg_err = sum(osv_errs) / len(osv_errs)

        with stopwatch() as ground_truth_times:
            resolution = [d] * nd
            grid = Grid(resolution)

            ## Ground truth
            ground_truth_scored_matrix = utils.brute_force_grid_search(
                envelope, domain, grid
            )
            ground_truth_class_matrix = copy(ground_truth_scored_matrix)
            for index, item in np.ndenumerate(ground_truth_scored_matrix):
                ground_truth_class_matrix[index] = envelope.classify_score(item)

            envelopes = utils.true_envelope_finding_alg(ground_truth_class_matrix, 5)
            boundary = utils.true_boundary_algorithm(
                ground_truth_class_matrix, envelopes[0]
            )

        # points = [b for b, n in results.bpoints]

        # points_indices = np.array(list(map(grid.calculate_point_index, points)))

        ## experiment
        # group = list(map(grid.convert_index_to_point, points_indices))
        # g.plot_all_points(group)

        # plt.show()

        ## ground truth
        # for index, value in np.ndenumerate(scored_matrix):
        #     p = grid.convert_index_to_point(index)
        #     g.plot_point(p, color="red" if envelope.classify(p) else "blue")

        with stopwatch() as mapping_truth_times:
            for id, true_bp_index in enumerate(boundary):
                rtree_index.add(id, grid.convert_index_to_point(true_bp_index))

            exp_boundary = np.array(
                [boundary[next(rtree_index.nearest(b))] for b, n in results.bpoints]
            )

            # ## tp, tn, fp, fn
            # acc = np.array([[1, 1, 0, 0]])
            # prec_d = np.array([[1, 0, 1, 0]])

            # colors = ("red", "blue", "green", "yellow", "orange", "cyan")

            # for boundary, color in zip(boundaries, colors):
            #     group = list(map(grid.convert_index_to_point, boundary))
            #     g.plot_all_points(group, color=color)

            # ground_truth = set(map(tuple, ground_truth_class_matrix.tolist()))
            # exp = set(map(tuple, points_indices))

            # ground_truth = np.zeros(ground_truth_class_matrix.shape)
            ground_truth = np.zeros(ground_truth_class_matrix.shape, "int32")
            ground_truth[*boundary.T] = 1

            exp_truth = np.zeros(ground_truth_class_matrix.shape, "int32")
            # exp_truth[*points_indices.T] = 1
            exp_truth[*exp_boundary.T] = 1

        # for index in points_indices:
        #     exp_truth[index] = 1

        # for index in boundary:
        #     ground_truth

        # exp = [index for index in points_indices if index in ground_truth_class_matrix]
        # print(boundary)
        # print(exp_truth)
        # print(ground_truth)
        # print("Correct boundary")
        # print(np.bitwise_and(exp_truth, ground_truth))
        # print("False Positive boundary")
        # print(np.bitwise_and(exp_truth, ~ground_truth))
        # print("False Negative boundary")
        # print(np.bitwise_and(~exp_truth, ground_truth))
        # # print(exp_truth and ground_truth_class_matrix)

        with stopwatch() as evaluation_times:
            correct_points = [
                grid.convert_index_to_point(index)
                for index in np.argwhere(np.bitwise_and(exp_truth, ground_truth))
            ]
            fp_points = [
                grid.convert_index_to_point(index)
                for index in np.argwhere(np.bitwise_and(exp_truth, ~ground_truth))
            ]
            fn_points = [
                grid.convert_index_to_point(index)
                for index in np.argwhere(np.bitwise_and(~exp_truth, ground_truth))
            ]

        # g = Grapher(nd == 3, domain)

        # g.plot_all_points([b for b, n in results.bpoints])
        # g.add_all_arrows(
        #     *zip(*[(b, n * 0.05) for b, n in results.bpoints]), color="orange"
        # )
        # plt.show()

        # g2 = Grapher(nd == 3, domain)
        # g3 = Grapher(nd == 3, domain)

        # g.plot_all_points(correct_points, color="green")
        # if len(fp_points) > 0:
        #     g.plot_all_points(fp_points, color="orange")
        # if len(fn_points) > 0:
        #     g.plot_all_points(fn_points, color="red")

        # g2.plot_all_points(
        #     [grid.convert_index_to_point(index) for index in np.argwhere(exp_truth)],
        #     color="blue",
        # )
        # g3.plot_all_points(
        #     [grid.convert_index_to_point(index) for index in np.argwhere(ground_truth)],
        #     color="green",
        # )

        # plt.pause(0.1)
        # input("press enter")

        # grid.convert_index_to_point

        # gtrue = Grapher(nd == 3, domain)
        # gtrue.plot_all_points(
        #     [grid.convert_index_to_point(index) for index in boundary], color="green"
        # )

        # gexp = Grapher(nd == 3, domain)
        # gexp.plot_all_points(
        #     [grid.convert_index_to_point(index) for index in points_indices], color="orange"
        # )

        print(
            f"Coverage: {(sum(np.bitwise_and(exp_truth, ground_truth).flatten() ) / sum(ground_truth.flatten())) * 100}%"
        )
        # plt.show()

        # input("catch")
        coverage.append(
            {
                "seed": seed,
                "coverage": sum(np.bitwise_and(exp_truth, ground_truth).flatten())
                / sum(ground_truth.flatten()),
                "true": int(sum(np.bitwise_and(exp_truth, ground_truth).flatten())),
                "fp": int(sum(np.bitwise_and(exp_truth, ~ground_truth).flatten())),
                "fn": int(sum(np.bitwise_and(~exp_truth, ground_truth).flatten())),
                "#samples": len(results.bpoints),
            }
        )

    with open(f".tmp/coverage/{nd}.json", "w") as f:
        f.write(json.dumps(coverage))
