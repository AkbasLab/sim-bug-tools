import matplotlib.pyplot as plt

from numpy import ndarray
import numpy as np

from sim_bug_tools.structs import Point, Domain, Spheroid
from sim_bug_tools.experiment import Experiment, ExperimentParams, ExperimentResults
from sim_bug_tools.exploration.boundary_core import (
    Explorer,
    BoundaryLostException,
    SampleOutOfBoundsException,
)
from sim_bug_tools.exploration.brrt_std import BoundaryRRT, ConstantAdherenceFactory

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

        i = 0
        sequence = []
        while params.explorer.boundary_count < params.n_bpoints:
            try:
                sequence.append(params.explorer.step())
            except BoundaryLostException:
                ble_count += 1
                sequence = []
            except SampleOutOfBoundsException:
                oob_count += 1
                sequence = []

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


if __name__ == "__main__":
    test_brrt_cluster()
