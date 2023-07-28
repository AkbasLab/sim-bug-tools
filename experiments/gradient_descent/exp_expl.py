import matplotlib.pyplot as plt
import numpy as np
import random
import json


from numpy import ndarray


from sim_bug_tools.structs import Point, Domain, Spheroid
from sim_bug_tools.experiment import Experiment, ExperimentParams, ExperimentResults
from sim_bug_tools.exploration.boundary_core import (
    Explorer,
    BoundaryLostException,
    SampleOutOfBoundsException,
)
from sim_bug_tools.exploration.brrt_std import BoundaryRRT, ConstantAdherenceFactory
from sim_bug_tools.exploration.brrt_v2 import ExponentialAdherenceFactory
from sim_bug_tools.exploration.Mesh_Explorer.mesh import (
    MeshExplorer,
    ExplorationCompletedException,
)
from sim_bug_tools.exploration.Adaptive.adpt import AdaptiveAdherenceFactory

from sim_bug_tools.simulation.simulation_core import Scorable

from exp_ann import ProbilisticSphere, ProbilisticSphereCluster


# from sim_bug_tools.exploration.


class ExplorationParams(ExperimentParams):
    def __init__(
        self,
        name: str,
        # adherer: Adherer,
        explorer: Explorer,
        n_bpoints: int,
        bp: bool = True,
        desc: str = None,
    ):
        super().__init__(name, desc)
        self.explorer = explorer
        self.n_bpoints = n_bpoints
        self.bp = bp


class ExplorationResults(ExperimentResults[ExplorationParams]):
    def __init__(
        self,
        params: ExplorationParams,
        bpoints: list[tuple[Point, ndarray]],
        nonbpoints: list[tuple[Point, bool]],
        ble_count: int,
        out_of_bounds_count: int,
    ):
        super().__init__(params)
        self.bpoints = bpoints
        self.nonbpoints = nonbpoints
        self.ble_count = ble_count
        self.out_of_bounds_count = out_of_bounds_count

    @property
    def eff(self) -> float:
        return len(self.bpoints) / len(self.nonbpoints)


class ExplorationExperiment(Experiment[ExplorationParams, ExplorationResults]):
    def experiment(self, params: ExplorationParams) -> ExplorationResults:
        ble_count = 0
        oob_count = 0
        points = []

        is_complete = False
        i = 0
        sequence = []
        while params.explorer.boundary_count < params.n_bpoints and not is_complete:
            try:
                sequence.append(params.explorer.step())
            except BoundaryLostException:
                ble_count += 1
                if ble_count > 500:
                    raise BoundaryLostException("Got stuck?")
                sequence = []
            except SampleOutOfBoundsException:
                oob_count += 1
                sequence = []
            except ExplorationCompletedException:
                is_complete = True

            if i != params.explorer.boundary_count:
                if params.bp:
                    params.explorer.back_propegate_prev(1)
                points.extend(sequence)
                sequence = []
                i += 1

        nonbpoints = [
            (p, cls)
            for p, cls in points
            if tuple(p) not in set(map(lambda b: tuple(b[0]), params.explorer.boundary))
        ]

        return ExplorationResults(
            params, params.explorer.boundary, nonbpoints, ble_count, oob_count
        )


def test():
    import numpy as np

    expl_exp = ExplorationExperiment()

    ndims = 30
    loc = Point([0.5] * ndims)
    radius = 0.4

    theta = np.pi * 10 / 180
    d = 0.05
    scaler = Spheroid(d)

    # v = np.random.rand(ndims)
    v = np.ones(ndims)
    v /= np.linalg.norm(v)
    b0 = Point(loc.array + v * radius * (1 - (d * 0.1) / radius))
    n0 = v

    normalize = lambda v: v / np.linalg.norm(v)

    def angle_between(u, v):
        u, v = normalize(u), normalize(v)
        return np.arccos(np.clip(np.dot(u, v), -1, 1.0))

    true_osv_at = lambda b: normalize((b - loc).array)

    envelope = ProbilisticSphere(loc, radius, 0.25)
    adh_f = ConstantAdherenceFactory(
        envelope.classify, scaler, theta, Domain.normalized(ndims), True
    )
    expl = BoundaryRRT(b0, n0, adh_f)

    expl_params = ExplorationParams(f"brrt-const-{ndims}d", expl, 500)

    results = expl_exp.experiment(expl_params)
    osv_errs = [
        angle_between(osv, true_osv_at(b)) * 180 / np.pi for b, osv in results.bpoints
    ]
    b_errs = [loc.distance_to(b) - radius for b, osv in results.bpoints]

    avg_osv_err = sum(osv_errs) / len(osv_errs)
    avg_b_err = sum(b_errs) / len(b_errs)

    # from sim_bug_tools.graphics import Grapher

    # g = Grapher(True, Domain.normalized(3))
    # g.plot_all_points([p for p, n in results.bpoints])

    # plt.show()

    print(
        f"BLEs: {results.ble_count}, OOBs: {results.out_of_bounds_count}, eff: {results.eff}, avg-osv-err: {avg_osv_err}, avg-b-err: {avg_b_err}"
    )


def test_brrt_cluster():
    import numpy as np

    expl_exp = ExplorationExperiment()

    ndims = 3
    domain = Domain.normalized(ndims)
    loc = Point([0.5] * ndims)
    radius = 0.4

    theta = np.pi * 10 / 180
    d = 0.025
    scaler = Spheroid(d)

    # v = np.random.rand(ndims)
    v = np.ones(ndims)
    v /= np.linalg.norm(v)
    # b0 = Point(loc.array + v * radius * (1 - (d * 0.1) / radius))
    # n0 = v

    normalize = lambda v: v / np.linalg.norm(v)

    def angle_between(u, v):
        u, v = normalize(u), normalize(v)
        return np.arccos(np.clip(np.dot(u, v), -1, 1.0))

    true_osv_at = lambda b: normalize((b - loc).array)

    p0 = Point([0.5] * ndims)
    r0 = 0.15
    k = 4
    n = 5
    envelope = ProbilisticSphereCluster(
        n, k, r0, p0, min_dist_b_perc=0, min_rad_perc=0, max_rad_perc=0.01, seed=1
    )

    t0 = Point([0.5] * ndims)
    print(envelope.classify(t0))

    from sim_bug_tools.exploration.boundary_core.surfacer import find_surface
    from sim_bug_tools.graphics import Grapher

    g = Grapher(ndims == 3, Domain.normalized(ndims))

    ((b0, n0), interm, was_in_domain) = find_surface(envelope.classify, t0, d, domain)

    # g.draw_path(interm, color="red")
    # g.plot_all_points(interm, color="red")

    # envelope = ProbilisticSphere(loc, radius, 0.25)
    adh_f = ConstantAdherenceFactory(
        envelope.classify, scaler, theta, Domain.normalized(ndims), True
    )

    from sim_bug_tools.exploration.Mesh_Explorer.mesh import MeshExplorer

    n_samples = 500
    bp = True
    etype = "mesh"

    if etype == "mesh":
        expl = MeshExplorer(b0, n0, adh_f, scaler)
    elif etype == "brrt":
        expl = BoundaryRRT(b0, n0, adh_f)

    expl_params = ExplorationParams(
        f"{etype}-const-{ndims}d-{n_samples}-" + "bp" if bp else "nobp",
        expl,
        n_samples,
        bp,
    )

    results = expl_exp.experiment(expl_params)

    import pandas as pd

    err_df = pd.DataFrame(
        {
            "osv_errs": [envelope.osv_err(b, osv) for b, osv in results.bpoints],
            "b_errs": [envelope.boundary_err(b) for b, osv in results.bpoints],
        }
    )
    print(err_df.describe())
    # avg_osv_err = sum(osv_errs) / len(osv_errs)
    # avg_b_err = sum(b_errs) / len(b_errs)

    # from sim_bug_tools.graphics import Grapher

    # g.plot_all_points([p for p, n in results.bpoints])

    g.draw_tree(expl.tree)

    # g.plot_all_points([p for p, c in results.nonbpoints if c], color="red")
    # g.plot_all_points([p for p, c in results.nonbpoints if not c], color="green")

    print(
        f"BLEs: {results.ble_count}, OOBs: {results.out_of_bounds_count}, eff: {results.eff}"
        # f"osv err: {avg_osv_err}, b err: {avg_b_err}"
    )
    plt.show()


class ProbabilisticCube(Scorable):
    def __init__(self, loc: Point, width: float, max_val: float, boundary_val: float):
        "loc is the CENTER of the cube!"
        self._loc = loc
        self._width = width
        self._max_val = max_val
        self._boundary_val = boundary_val
        self._slope = 2 * (max_val - boundary_val) / width

    def _density_function(self, x: float):
        prod = self._slope * x
        return self._max_val - prod

    def score(self, p: Point) -> ndarray:
        dif = (p - self._loc).array
        abs_dif = abs(dif)
        max_dif = max(abs_dif)

        return np.array(self._density_function(max_dif))

    def classify_score(self, score: ndarray) -> bool:
        return score >= self._boundary_val

    def get_input_dims(self):
        return len(self._loc)

    def get_score_dims(self):
        return 1

    def generate_random_target(self):
        return Point(np.random.rand(self.get_input_dims()) * self._width) + Point(
            self._loc.array - self._width / 2
        )

    def generate_random_nontarget(self):
        raise NotImplementedError()

    def boundary_err(self, b: Point) -> float:
        """
        Rough estimate of distance from boundary by finding the distance between
        the nearest side and the point @b.
        Useful only if point @b near boundary or in envelope
        """
        return abs(max(abs((b - self._loc).array)) - self._width / 2)


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


def get_adh(adh_type):
    pass


def test_cube():
    domain = Domain.normalized(3)
    cube = create_envelope("cube", Point([0.5] * 3), 1, domain)

    points = [
        Point(0.5, 0, 0),
        Point(0.5, 1, 0),
        Point(0.5, 1, 1),
        Point(0.5, 0, 1),
        Point(0.5, 0, 0),
        Point(0.5, 0.2, 0),
        Point(0.5, 0.2, 0.2),
        Point(0.5, 0, 0.2),
        Point(0.5, 0.8, 0),
        Point(0.5, 0.8, 0.8),
        Point(0.5, 0, 0.8),
        Point(0.5, 0.5, 0.5),
        Point(0.5, 0.1, 0.1),
        Point(0.5, 0.14, 0),
        Point(0.5, 0.15, 0),
        Point(0.5, 0.16, 0),
    ]

    import matplotlib.pyplot as plt
    from sim_bug_tools.graphics import Grapher

    box = Domain.from_dimensions([0.6] * 3, Point([0.2] * 3))

    g = Grapher(True, domain)
    g.draw_cube(box)
    # g.plot_all_points([p for p in points if cube.classify(p)], color="red")
    # g.plot_all_points([p for p in points if not cube.classify(p)], color="blue")

    for p in points:
        g.plot_point(p, color="red" if cube.classify(p) else "blue")
        plt.pause(0.1)
        print("Point:", p)
        print("Score:", cube.score(p))
        print("Class:", cube.classify(p))

    print("done")


def test_dimensions():
    import numpy as np
    import matplotlib.pyplot as plt
    from sim_bug_tools.exploration.boundary_core.surfacer import find_surface
    from sim_bug_tools.graphics import Grapher

    expl_exp = ExplorationExperiment()

    ndims_exps = [3, 5, 10, 15, 20, 25, 30, 50, 75, 100]
    # e_type = "sphere-cluster"  # sphere, sphere-cluster, cube
    # adh_type = "const"  # const, exp, adpt

    pairs = (
        # ("sphere-cluster", "mesh"),
        # ("sphere-cluster", "rrt"),
        # ("sphere-cluster", "exp"),
        # ("sphere", "const", "mesh"),
        # ("sphere", "exp"),
        # ("cube", "const", "mesh"),
        # ("sphere", "const", "rrt"),
        # ("sphere", "exp", "rrt"),
        # ("cube", "const", "rrt"),
        ("cube", "exp", "rrt"),
        # ("cube", "exp"),
    )

    delta_theta = np.pi * 10 / 180
    theta0 = np.pi / 2
    d = 0.05
    scaler = Spheroid(d)
    num_runs = 20

    ## Some analysis functions
    normalize = lambda v: v / np.linalg.norm(v)

    def angle_between(u, v):
        u, v = normalize(u), normalize(v)
        return np.arccos(np.clip(np.dot(u, v), -1, 1.0))

    true_osv_at = lambda b: normalize((b - loc).array)
    ##
    final_results = {}

    np.seterr(divide="ignore")
    # v = np.random.rand(ndims)
    for e_type, adh_type, exp_type in pairs:
        pair_results = {}
        for ndims in ndims_exps:
            domain = Domain.normalized(ndims)

            cube = Domain.from_dimensions([0.6] * ndims, Point([0.2] * ndims))
            loc = Point([0.5] * ndims)

            sub_results = []
            i = 0
            seed = 0
            while seed < num_runs + i:
                seed += 1
                np.random.seed(seed)
                random.seed(seed)
                try:
                    envelope = create_envelope(e_type, loc, seed, domain)
                except Exception as e:
                    print(e)
                    i += 1
                    continue
                # ProbilisticSphere(Point([0.5] * ndims), 0.4, 0.25)
                # v = np.ones(ndims)
                # v /= np.linalg.norm(v)

                (b0, n0), _, valid = find_surface(
                    envelope.classify, loc, d, domain, fail_out_of_bounds=True
                )

                # print(valid)
                # print("b-err:", envelope.boundary_err(b0))
                # true_n0 = b0.array - envelope.loc.array
                # true_n0 /= np.linalg.norm(true_n0)
                # print("n-err:", angle_between(n0, true_n0))

                if adh_type == "const":
                    adh_f = ConstantAdherenceFactory(
                        envelope.classify,
                        scaler,
                        delta_theta,
                        Domain.normalized(ndims),
                        True,
                    )
                elif adh_type == "exp":
                    adh_f = ExponentialAdherenceFactory(
                        envelope.classify, scaler, theta0, 4, domain, True
                    )
                elif adh_type == "adpt":
                    const_f = ConstantAdherenceFactory(
                        envelope.classify,
                        scaler,
                        delta_theta,
                        Domain.normalized(ndims),
                        True,
                        max_samples=3,
                    )
                    exp_f = ExponentialAdherenceFactory(
                        envelope.classify, scaler, theta0, 4, domain, True
                    )

                    adh_f = AdaptiveAdherenceFactory(
                        envelope.classify, const_f, exp_f, domain, True
                    )

                if exp_type == "mesh":
                    expl = MeshExplorer(b0, n0, adh_f, scaler)
                elif exp_type == "rrt":
                    expl = BoundaryRRT(b0, n0, adh_f)

                expl_params = ExplorationParams(
                    f"brrt-{e_type}-{adh_type}-{ndims}d-{seed}", expl, 500
                )

                try:
                    results = expl_exp.experiment(expl_params)

                    b_errs = [envelope.boundary_err(b) for b, osv in results.bpoints]

                    avg_b_err = sum(b_errs) / len(b_errs)

                    sub_results.append(
                        {
                            # "e-type": e_type,
                            # "adh-type": adh_type,
                            # "exp-type": exp_type,
                            "avg-err": avg_b_err,
                            "eff": results.eff,
                            "BLEs": results.ble_count,
                            "OOBs": results.out_of_bounds_count,
                            "boundary": [tuple(p) for p, n in results.bpoints],
                        }
                    )

                    # g = Grapher(ndims == 3, domain)
                    # points = list(map(lambda n: n[0], results.bpoints))

                    # g.draw_cube(cube)

                    # g.plot_point(b0, color="red")
                    # g.add_arrow(b0, n0, color="red")

                    # g.plot_all_points(points)
                    # plt.pause(0.1)
                    # print("h")

                # except BoundaryLostException as e:
                #     # To catch BLEs and treat them as failures
                #     print("BLE,\n", e)
                #     sub_results.append(
                #         {
                #             "seed": seed,
                #             "failed": True,
                #             "avg-err": np.inf,
                #             "eff": 0,
                #             "BLEs": 500,
                #             "OOBs": 0,
                #         }
                #     )
                except Exception as e:
                    print(e)

                    i += 1  # got to figure out why rotation matrix is not rotating correctly for mesh?
                    continue

            pair_results[f"{ndims}d"] = sub_results

        # final_results[f"expl_results-{e_type}-{adh_type}.json"] = pair_results
        with open(
            f".tmp/results/expl_results-{e_type}-{exp_type}-{adh_type}.json", "w"
        ) as f:
            f.write(json.dumps(pair_results))

    print("Done")


if __name__ == "__main__":
    test_dimensions()
    # test_cube()
