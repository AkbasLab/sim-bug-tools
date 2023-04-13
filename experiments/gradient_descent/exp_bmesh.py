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

from exp_ann import ProbilisticSphere


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
        from sim_bug_tools.graphics import Grapher

        g = Grapher(True, Domain.normalized(3))
        g.draw_sphere(Point([0.5] * 3), 0.4)
        ble_count = 0
        oob_count = 0
        points = []

        _prev = None

        # g = Grapher(True, Domain.normalized(3))
        # g.draw_sphere(Point([0.5] * 3), 0.4)
        # g.plot_point(params.explorer.boundary[-1][0])
        # g.add_arrow(*params.explorer.boundary[-1])
        # plt.pause(0.01)
        i = 0
        cntrl = False

        sequence = []
        while params.explorer.boundary_count < params.n_bpoints:
            try:
                p, cls = params.explorer.step()
                if cntrl:
                    g.plot_point(p, color="red" if cls else "blue")
                    plt.pause(0.01)
                sequence.append((p, cls))
            except BoundaryLostException:
                path = [p for p, cls in sequence]
                # g.plot_all_points(path)
                # g.plot_point(params.explorer._tmp_parent[0], color="red")
                # g.add_arrow(
                #     params.explorer._tmp_parent[0],
                #     params.explorer._tmp_parent[1] * 0.1,
                #     color="red",
                # )
                # g.draw_path(path, typ="-")
                # v1 = path[0].array - params.explorer._tmp_parent[0].array
                # v1 /= np.linalg.norm(v1)
                # v2 = path[-1].array - params.explorer._tmp_parent[0].array
                # v2 /= np.linalg.norm(v2)
                # plt.pause(0.01)
                ble_count += 1
                sequence = []
            except SampleOutOfBoundsException:
                oob_count += 1
                sequence = []

            if params.bp and i != params.explorer.boundary_count:
                params.explorer.back_propegate_prev(1)
                points.extend(sequence)
                sequence = []
                # if _prev is not None:
                #     _prev.remove()
                i += 1
            #     i += 1
            #     g.plot_point(params.explorer.boundary[-1][0])
            #     g.add_arrow(*params.explorer.boundary[-1])
            #     g.add_arrow(params.explorer.boundary[-2][0], params.explorer._s)

            #     plt.pause(0.01)

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

    ndims = 3
    loc = Point([0.5] * ndims)
    radius = 0.4

    theta = np.pi * 10 / 180
    d = 0.01
    scaler = Spheroid(d)

    # v = np.random.rand(ndims)
    v = np.array([1.0, 1.0, 1.0])
    v /= np.linalg.norm(v)
    b0 = Point(loc.array + radius * v * (1 - (d * 0.1) / radius))
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
    errs = [
        angle_between(osv, true_osv_at(b)) * 180 / np.pi for b, osv in results.bpoints
    ]
    avg_err = sum(errs) / len(errs)

    from sim_bug_tools.graphics import Grapher

    g = Grapher(True, Domain.normalized(3))
    g.plot_all_points([p for p, n in results.bpoints])
    for b, n in results.bpoints:
        g.add_arrow(b, n, color="blue")
        plt.pause(0.01)

    # g.add_all_arrows(*zip(*results.bpoints))
    plt.show()

    print(
        f"BLEs: {results.ble_count}, OOBs: {results.out_of_bounds_count}, eff: {results.eff}, avg-err: {avg_err}"
    )


if __name__ == "__main__":
    test()
