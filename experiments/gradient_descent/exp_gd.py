import tensorflow as tf
import numpy as np

from exp_ann import ANNExperiment, ANNParams, ANNResults, Scorable, ProbilisticSphere
from exp_expl import ExplorationExperiment, ExplorationParams, ExplorationResults

from sim_bug_tools.structs import Point, Domain, Spheroid
from sim_bug_tools.experiment import Experiment, ExperimentParams, ExperimentResults
from sim_bug_tools.rng.lds.sequences import SobolSequence, RandomSequence
from sim_bug_tools.exploration.brrt_std import ConstantAdherenceFactory, BoundaryRRT
from sim_bug_tools.exploration.boundary_core import (
    find_surface,
    SampleOutOfBoundsException,
)

from dataclasses import dataclass
from numpy import ndarray
from typing import Callable
from random import shuffle


@dataclass
class PointData:
    point: Point
    score: ndarray
    cls: bool
    gradient: ndarray = None


class GDExplorerParams(ExperimentParams):
    def __init__(
        self,
        name: str,
        ann_results: ANNResults,
        num_bpoints: int,
        h: Callable[[ndarray], ndarray],
        alpha=0.01,
        max_steps=1000,
        desc: str = None,
    ):
        """
        ann_exp_name: if None, will default to previously cached experiment
        (i.e. previously trained model).

        x_{i+1} = x_i + alpha*h(g(x_i})), where
            g(x) is the gradient and h is describes how to use the gradient to
            find the next sample location.
        """
        super().__init__(name, desc)
        self.num_bpoints = num_bpoints
        self.h = h
        self.alpha = alpha
        self.max_steps = max_steps
        self.ann_results = ann_results


class GDExplorerResults(ExperimentResults[GDExplorerParams]):
    def __init__(
        self,
        params: GDExplorerParams,
        boundary_paths: list[list[PointData]],
        nonboundary_paths: list[list[PointData]],
    ):
        """
        paths: Each path represents a sequence of points, starting from where
        the explorer began looking for the boundary and ending where it found
        the boundary.

        2 <= len(boundary_paths) <= max_steps
        1 <= len(nonboundary_paths) <= max_steps
        boundary = [path[-2] for path in boundary_paths]

        """
        super().__init__(params)
        self.boundary_paths = boundary_paths
        self.nonboundary_paths = nonboundary_paths

    @property
    def n_failures(self):
        return len(self.nonboundary_paths)

    @property
    def n_successes(self):
        return len(self.boundary_paths)

    @property
    def failure_percentage(self):
        return (
            self.n_failures / (self.n_failures + self.n_successes) * 100
            if (self.n_failures + self.n_successes) != 0
            else 0
        )

    @property
    def eff_posttrain_nofailure(self):
        # The number of bps per non bps, not including training data nor failure
        # paths
        n_nonbcount = sum(map(len, self.boundary_paths)) - len(self.boundary_paths)
        return len(self.boundary_paths) / n_nonbcount if n_nonbcount != 0 else 0

    @property
    def eff_posttrain(self):
        # The number of bps per non bps, not including training data
        # #bps / (#bps - #samples in boundary paths + #samples in nonboundary
        # paths). I.e., #bps / (non-bps from both success and failure paths.)
        n_nonbpoint = (
            sum(map(len, self.boundary_paths))
            - len(self.boundary_paths)
            + sum(map(len, self.nonboundary_paths))
        )
        return len(self.boundary_paths) / n_nonbpoint if n_nonbpoint != 0 else 0

    @property
    def eff(self):
        # The number of bps per non bps
        n_nonbpoint = (
            sum(map(len, self.boundary_paths))
            - len(self.boundary_paths)
            + sum(map(len, self.nonboundary_paths))
            + self.params.ann_results.params.training_size
        )
        return len(self.boundary_paths) / n_nonbpoint if n_nonbpoint != 0 else 0

    @property
    def boundary(self):
        return map(lambda path: path[-1], self.boundary_paths)

    @property
    def b_errs(self):
        calc_b_err = self.params.ann_results.params.envelope.boundary_err
        return list(map(lambda pd: abs(calc_b_err(pd.point)), self.boundary))

    @property
    def avg_b_err(self):
        b_errs = self.b_errs
        return sum(b_errs) / len(b_errs) if len(b_errs) != 0 else -1


class GDExplorerExperiment(Experiment[GDExplorerParams, GDExplorerResults]):
    def __init__(self):
        super().__init__()
        self._previous_result: tuple[Point, list[PointData]] = None

    def experiment(self, params: GDExplorerParams) -> GDExplorerResults:
        scored_data = params.ann_results.scored_data
        shuffle(scored_data)
        model = tf.keras.models.load_model(params.ann_results.model_path)
        envelope = params.ann_results.params.envelope

        self._domain = Domain.normalized(len(scored_data[0][0]))

        b_paths: list[list[PointData]] = []
        nonb_paths: list[list[PointData]] = []

        num_bpoints = 0
        i = 0
        # for init_p, init_score in scored_data[: params.num_bpoints]:
        while num_bpoints < params.num_bpoints and i < len(scored_data):
            init_p, init_score = scored_data[i]
            init_p = Point(init_p) if type(init_p) is not Point else init_p
            path = []

            try:
                for pd in self.find_boundary_from(
                    init_p, init_score, params, envelope, model
                ):
                    path.append(pd)

                b_paths.append(path)
                num_bpoints += 1
            except SampleOutOfBoundsException as e:
                nonb_paths.append(path)

            # if self._previous_was_bpath:
            # else:

            i += 1

        return GDExplorerResults(params, b_paths, nonb_paths)

    def predict_gradient(self, p: Point, model: tf.keras.Sequential):
        inp = tf.Variable(np.array([p]), dtype=tf.float32)

        with tf.GradientTape() as tape:
            preds = model(inp)

        return np.array(tape.gradient(preds, inp)).squeeze()

    def find_boundary_from(
        self,
        init_p: Point,
        init_score: ndarray,
        params: GDExplorerParams,
        envelope: Scorable,
        model: tf.keras.Sequential,
    ):
        i = 0
        prev = init_p
        prev_cls = envelope.classify_score(init_score)

        cur = init_p
        cur_score = init_score
        cur_cls = prev_cls
        s = np.zeros(cur.array.shape)

        # Loop if max_steps not exceeded and boundary has not been found
        while i < params.max_steps and cur_cls == prev_cls and cur in self._domain:
            prev = cur
            prev_score = cur_score
            prev_cls = cur_cls

            # Make a step according to gradient descent solution
            prev_g = self.predict_gradient(prev, model)
            s = params.h(prev_g, s)
            cur += params.alpha * (-s if prev_cls else s)

            if cur not in self._domain:
                raise SampleOutOfBoundsException()

            cur_score = envelope.score(cur)
            cur_cls = envelope.classify_score(cur_score)
            i += 1

            yield PointData(prev, prev_score, prev_cls, prev_g)

            if np.linalg.norm(s) < 1e-4:  # stop if we haven't moved much
                break

        cur_g = self.predict_gradient(cur, model)

        self._previous_was_bpath = cur_cls != prev_cls and cur in self._domain

        yield PointData(cur, cur_score, cur_cls, cur_g)

    @staticmethod
    def steepest_descent(g: ndarray, *args) -> ndarray:
        return g

    @staticmethod
    def gradient_descent_with_momentum(g: ndarray, prev_v: ndarray) -> ndarray:
        beta = 0.9
        v = prev_v * beta + (1 - beta) * g
        return v


def _gd_param_name(ann_name: str, gd_type: str, alpha: float) -> str:
    return f"{ann_name}-{gd_type}-{alpha}"


def _ann_param_name(n_samples: int, ndims: int, envelope_name: str):
    return f"psphere-{envelope_name}-{ndims}d-{n_samples}"


def _expl_param_name(
    envelope_type: str, expl_type: str, adh_type: str, ndims: int, bp_enabled: bool
) -> str:
    suffix = "-bp" if bp_enabled else ""
    return f"{envelope_type}-{expl_type}-{adh_type}-{ndims}d" + suffix


def test_samplesXgd():
    import matplotlib.pyplot as plt
    from sim_bug_tools.graphics import Grapher

    ndims = 10
    num_bpoints = 100
    domain = Domain.normalized(ndims)

    # g = Grapher(ndims == 3, domain)

    gd_exp = GDExplorerExperiment()
    ann_exp = ANNExperiment()
    envelope = ProbilisticSphere(Point([0.5 for i in range(ndims)]), 0.4, 0.25)
    # g.draw_sphere(envelope.loc, envelope.radius, color="grey")
    seq = SobolSequence(domain, [str(i) for i in range(ndims)])

    meta_data: list[dict] = []

    for ann_samples in [50, 100, 200, 400, 800, 1600, 3200, 6400]:
        ann_name = _ann_param_name(ann_samples, ndims)
        ann_params = ANNParams(
            ann_name, envelope, seq, ann_samples, ann_samples // 10, n_epochs=200
        )

        ann_results = ann_exp.lazily_run(ann_params)

        gd_params = GDExplorerParams(
            f"{ann_name}-{ndims}d-sd",
            ann_results,
            ann_samples,
            GDExplorerExperiment.steepest_descent,
        )

        gd_result = gd_exp.run(gd_params)

        boundary = [path[-2] for path in gd_result.boundary_paths]
        if len(boundary) > 0:
            boundary_err = list(
                map(lambda pd: envelope.boundary_err(pd.point), boundary)
            )
        else:
            boundary_err = [-1]

        avg = lambda lst: sum(lst) / len(lst)
        nonb_count = (
            len(sum(gd_result.boundary_paths, []))
            - len(boundary)
            + len(sum(gd_result.nonboundary_paths, []))
        )
        post_eff = len(boundary) / nonb_count
        full_eff = len(boundary) / (nonb_count + ann_samples)

        md = {
            "name": gd_params.name,
            "failure-count": len(gd_result.nonboundary_paths),
            "b-count": len(gd_result.boundary_paths),
            "nonb-cound": nonb_count,
            "avg-err": avg(boundary_err),
            "min-err": min(boundary_err),
            "max-err": max(boundary_err),
            "post-train-eff%": post_eff * 100
            if len(boundary) > 0
            else 0,  # bp/non-bp * 100
            "full-eff%": full_eff * 100,
            "train-size": len(ann_results.scored_data),
        }

        meta_data.append(md)

        # to_path = lambda pds: [pd.point for pd in pds]
        # for b_path, nonb_path in zip(
        #     gd_result.boundary_paths, gd_result.nonboundary_paths
        # ):
        #     print(len(b_path), len(nonb_path))
        #     b_path = to_path(b_path)
        #     nonb_path = to_path(nonb_path)
        #     g.draw_path(b_path, markersize=1, color="green")
        #     g.plot_all_points(b_path, markersize=1, color="green")
        #     g.draw_path(nonb_path, markersize=1, color="red")
        #     g.plot_all_points(nonb_path, markersize=1, color="red")
        #     plt.pause(0.01)

    print(meta_data)
    import json

    with open("tmp-sd.json", "w") as f:
        f.write(json.dumps(meta_data, indent=4))


def test_oddones():
    ndims = 3
    num_bpoints = 100
    domain = Domain.normalized(ndims)

    gd_exp = GDExplorerExperiment()
    ann_exp = ANNExperiment()
    envelope = ProbilisticSphere(Point([0.5 for i in range(ndims)]), 0.4, 0.25)
    seq = SobolSequence(domain, [str(i) for i in range(ndims)])

    meta_data: list[dict] = []

    odd_models = [100, 50, 400, 1600, 6400]
    # ann_samples = odd_models[0]
    for ann_samples in odd_models:
        ann_name = _ann_param_name(ann_samples)
        ann_params = ANNParams(
            ann_name, envelope, seq, ann_samples, int(ann_samples / 10)
        )

        ann_results = ann_exp.lazily_run(ann_params)

        model = tf.keras.models.load_model(ann_results.model_path)

        data = ann_results.scored_data

        truth_table = ANNExperiment.class_acc(model, data, envelope)
        tp, tn, fp, fn = truth_table

        print(truth_table)


def find_init_boundary_pair(
    seq: SobolSequence, envelope: Scorable
) -> tuple[Point, Point]:
    prev_target = None
    prev_nontarget = None
    sample_count = 0

    while prev_target is None or prev_nontarget is None:
        cur = seq.get_sample(1).points[0]
        cls = envelope.classify(cur)

        if cls and prev_target is None:
            prev_target = cur
        elif not cls and prev_nontarget is None:
            prev_nontarget = cur

        sample_count += 1

    return (
        prev_target,
        prev_nontarget,
        sample_count,
    )  # envelope.generate_random_target(), envelope.generate_random_nontarget()


def test_comparison():
    print("initializing")

    gd_exp = GDExplorerExperiment()
    ann_exp = ANNExperiment()
    expl_exp = ExplorationExperiment()

    ndims = 30
    c = (ndims / 4) ** 0.5
    c0 = (3 / 4) ** 0.5
    training_size = 5120

    domain = Domain.normalized(ndims)
    seq = SobolSequence(domain, [f"d{i}" for i in range(ndims)])

    loc = Point([0.5] * ndims)
    radius = c * 0.5 / c0  # Will be larger as ndims grows
    lmbd = 0.25
    envelope = ProbilisticSphere(loc, radius, lmbd)

    print("Training ANN")
    ann_name = _ann_param_name(training_size, ndims)
    ann_params = ANNParams(ann_name, envelope, seq, training_size)
    ann_results = ann_exp(ann_params)

    n_bpoints = 100  # must be < training_size for our purposes
    alpha = 0.1  # scales gradient descent step-size
    max_steps = 1000  # #steps prior to aborting gradient descent and trying again

    gd_type_name = "sd"
    if gd_type_name == "sd":
        gd_type = GDExplorerExperiment.steepest_descent
    elif gd_type_name == "gdm":
        gd_type = GDExplorerExperiment.gradient_descent_with_momentum
    else:
        gd_type = None  # err

    gd_params = GDExplorerParams(
        _gd_param_name(ann_name, gd_type_name, alpha),
        ann_results,
        n_bpoints,
        gd_type,
        alpha,
        max_steps,
    )
    print("Running GD")
    gd_results = gd_exp(gd_params)

    d = 0.05
    delta_theta = 10 * np.pi / 180  # 10 degrees
    bp_enabled = True
    scaler = Spheroid(d)
    adh_f = ConstantAdherenceFactory(
        envelope.classify, scaler, delta_theta, domain, True
    )

    t0, nont0, count = find_init_boundary_pair(seq, envelope)
    direction = (nont0 - t0).array
    direction /= np.linalg.norm(direction)
    (b0, n0), path, _ = find_surface(
        envelope.classify, t0, d, domain, direction, fail_out_of_bounds=True
    )

    count += len(path)

    explorer = BoundaryRRT(b0, n0, adh_f)

    expl_params = ExplorationParams(
        _expl_param_name("psphere", "brrt", "const", ndims, bp_enabled),
        explorer,
        n_bpoints,
        bp_enabled,
    )

    print("Running Exploration")
    expl_results = expl_exp(expl_params)

    # osv_errs = [
    #     angle_between(osv, true_osv_at(b)) * 180 / np.pi for b, osv in expl_params.bpoints
    # ]
    b_errs = [abs(loc.distance_to(b) - radius) for b, osv in expl_results.bpoints]

    # avg_osv_err = sum(osv_errs) / len(osv_errs)
    avg_b_err = sum(b_errs) / len(b_errs)

    # print("Saving")
    # with open(".tmp-comp-brrt.json", "w") as f:
    #     f.write(json.dumps(expl_results))
    # with open(".tmp-comp-gd.json", "w") as f:
    #     f.write(json.dumps(gd_results))

    rounded_percentage_effs = list(
        map(
            lambda x: round(x * 100, 2),
            [
                gd_results.eff_posttrain_nofailure,
                gd_results.eff_posttrain,
                gd_results.eff,
            ],
        )
    )

    print(
        "GD",
        f"failure rate: {round(gd_results.failure_percentage, 2)}%",
        f"Pretrain, posttrain, posttrain w failure efficiencies: \n{rounded_percentage_effs}",
        f"Avg err: {gd_results.avg_b_err}",
        sep="\n",
    )
    print(
        "BRRT",
        f"BLEs: {expl_results.ble_count}",
        f"OOBs: {expl_results.out_of_bounds_count}",
        f"Pre-t0 Eff: {expl_results.eff}",
        f"Post-t0 Eff: {len(expl_results.bpoints) / (count + len(expl_results.nonbpoints))}",
        f"Avg err: {avg_b_err}",
        sep="\n",
    )

    from sim_bug_tools.graphics import Grapher
    import matplotlib.pyplot as plt

    # g = Grapher(ndims == 3, domain)
    # g.draw_sphere(loc, radius)
    # for pd_path in gd_results.nonboundary_paths:
    #     path = list(map(lambda pd: pd.point, pd_path))

    #     g.draw_path(path, "-")
    #     g.plot_point(path[0], color="blue")
    #     g.plot_point(path[-1], color="red")
    #     plt.pause(0.01)

    # print(expl_results.__dict__)
    print("expl err", avg_b_err)
    # print(gd_results.__dict__)


def test_across_dimensions():
    from sim_bug_tools.rng.lds.sequences import RandomSequence

    calc_acc = lambda tab: sum(tab[:2]) / sum(tab)

    print("initializing")
    gd_exp = GDExplorerExperiment()
    ann_exp = ANNExperiment()

    exp_ndims = [5, 10, 15, 20, 25, 30, 50, 100]
    c0 = (3 / 4) ** 0.5
    training_size = 5120

    envelope_type = "dsphere_scaled"

    model_acc = []
    model_err = []

    # Order: eff_posttrain_nofailure, eff_posttrain, eff_pretrain,
    gd_eff: list[dict[str, float]] = []
    gd_err: list[float] = []

    # GD params
    gd_type_name = "sd"
    n_bpoints = 100  # must be < training_size for our purposes
    alpha = 0.1  # scales gradient descent step-size
    max_steps = 1000  # #steps prior to aborting gradient descent and trying again

    for ndims in exp_ndims:
        c = (ndims / 4) ** 0.5
        domain = Domain.normalized(ndims)
        loc = Point([0.5] * ndims)
        radius = (
            c * 0.5 / c0 if envelope_type == "dsphere_scaled" else 0.5
        )  # Will be larger as ndims grows
        lmbd = 0.25
        envelope = ProbilisticSphere(loc, radius, lmbd)
        seq = SobolSequence(domain, [f"d{i}" for i in range(ndims)])
        test_data = [
            (p, envelope.score(p))
            for p in (
                RandomSequence(domain, [f"x{i}" for i in range(ndims)])
                .get_sample(5120)
                .points
            )
        ]

        print("Training ANN")
        ann_name = _ann_param_name(training_size, ndims, envelope_type)
        ann_params = ANNParams(ann_name, envelope, seq, training_size)
        ann_results = ann_exp(ann_params)

        model = tf.keras.models.load_model(ann_results.model_path)
        test_truth_table = ANNExperiment.class_acc(model, test_data, envelope)
        model_acc.append(calc_acc(test_truth_table))
        model_err.append(ANNExperiment.calc_err(model, test_data, envelope))

        if gd_type_name == "sd":
            gd_type = GDExplorerExperiment.steepest_descent
        elif gd_type_name == "gdm":
            gd_type = GDExplorerExperiment.gradient_descent_with_momentum
        else:
            gd_type = None  # err

        gd_params = GDExplorerParams(
            _gd_param_name(ann_name, gd_type_name, alpha),
            ann_results,
            n_bpoints,
            gd_type,
            alpha,
            max_steps,
        )

        print("Running GD")
        gd_results = gd_exp(gd_params)

        gd_err.append(gd_results.avg_b_err)
        gd_eff.append(
            {
                "eff": gd_results.eff,
                "eff-posttrain": gd_results.eff_posttrain,
                "eff-posttrain_nofailure": gd_results.eff_posttrain_nofailure,
            }
        )

    from sim_bug_tools.graphics import Grapher
    import matplotlib.pyplot as plt

    gModelAccuracy = Grapher()
    gModelError = Grapher()

    model_err = list(map(float, model_err))

    gModelAccuracy.draw_path(list((map(Point, zip(exp_ndims, model_acc)))))
    gModelAccuracy.set_title("Model Accuracy")
    gModelAccuracy.plot_all_points(list((map(Point, zip(exp_ndims, model_acc)))))

    gModelError.set_title("Model Error")
    gModelError.draw_path(list((map(Point, zip(exp_ndims, map(float, model_err))))))
    gModelError.plot_all_points(list((map(Point, zip(exp_ndims, model_err)))))

    gGdEff = Grapher()
    gGdError = Grapher()

    eff, eff_pre, eff_pre_nofail = zip(
        *map(lambda d: [x for name, x in d.items()], gd_eff)
    )
    gGdEff.set_title("GD Efficiency")
    gGdEff.draw_path(list((map(Point, zip(exp_ndims, eff)))))
    gGdEff.plot_all_points(list((map(Point, zip(exp_ndims, eff)))))
    gGdError.set_title("GD Error")
    gGdError.draw_path(list((map(Point, zip(exp_ndims, gd_err)))))
    gGdError.plot_all_points(list((map(Point, zip(exp_ndims, gd_err)))))

    plt.show()
    print("Done")


from highway2.highway2 import HighwayTrafficParameterManager, HighwayTrafficTest


class HighwayScoreable(Scorable):
    def __init__(self):
        self.manager = HighwayTrafficParameterManager()
        self.test = HighwayTrafficTest()
        df = self.manager.params_df
        axes_names = df["feature"]
        ndims = len(axes_names)
        true_limits = [(df.iloc[i]["min"], df.iloc[i]["max"]) for i in range(ndims)]
        self.true_domain = Domain(true_limits, axes_names)
        self.norm_domain = Domain.normalized(ndims, axes_names)
        self.score_schema = ["e_brake", "collision"]

        self.test.__enter__()

    def score(self, p: Point) -> ndarray:
        concrete_params = self.manager.map_parameters(p)
        scores = self.test.run(concrete_params)
        return np.array([scores[self.score_schema[0]], scores[self.score_schema[1]]])

    def classify_score(self, score: ndarray) -> bool:
        return score[0] > 0 or score[1] > 0

    def classify(self, p: Point) -> bool:
        return self.classify_score(self.score(p))

    def get_input_dims(self):
        return len(self.norm_domain)

    def get_score_dims(self):
        return len(self.score_schema)

    def generate_random_target(self):
        raise NotImplementedError()

    def generate_random_nontarget(self):
        raise NotImplementedError()


def test_sim():
    print("initializing")

    gd_exp = GDExplorerExperiment()
    ann_exp = ANNExperiment()
    expl_exp = ExplorationExperiment()

    envelope = HighwayScoreable()
    ndims = envelope.get_input_dims()
    c = (ndims / 4) ** 0.5
    c0 = (3 / 4) ** 0.5
    training_size = 100

    domain = Domain.normalized(ndims)
    seq = RandomSequence(domain, [f"d{i}" for i in range(ndims)])
    # seq = SobolSequence(domain, [f"d{i}" for i in range(ndims)])

    loc = Point([0.5] * ndims)
    radius = c * 0.5 / c0  # Will be larger as ndims grows
    lmbd = 0.25
    # envelope = ProbilisticSphere(loc, radius, lmbd)

    print("Training ANN")
    ann_name = _ann_param_name(training_size, ndims, "hw2.1")
    ann_params = ANNParams(ann_name, envelope, seq, training_size)
    ann_results = ann_exp(ann_params)

    n_bpoints = 100  # must be < training_size for our purposes
    alpha = 0.1  # scales gradient descent step-size
    max_steps = 1000  # #steps prior to aborting gradient descent and trying again

    gd_type_name = "sd"
    if gd_type_name == "sd":
        gd_type = GDExplorerExperiment.steepest_descent
    elif gd_type_name == "gdm":
        gd_type = GDExplorerExperiment.gradient_descent_with_momentum
    else:
        gd_type = None  # err

    gd_params = GDExplorerParams(
        _gd_param_name(ann_name, gd_type_name, alpha),
        ann_results,
        n_bpoints,
        gd_type,
        alpha,
        max_steps,
    )
    print("Running GD")
    gd_results = gd_exp(gd_params)

    d = 0.05
    delta_theta = 10 * np.pi / 180  # 10 degrees
    bp_enabled = True
    scaler = Spheroid(d)
    adh_f = ConstantAdherenceFactory(
        envelope.classify, scaler, delta_theta, domain, True
    )

    t0, nont0, count = find_init_boundary_pair(seq, envelope)
    direction = (nont0 - t0).array
    direction /= np.linalg.norm(direction)
    (b0, n0), path, _ = find_surface(
        envelope.classify, t0, d, domain, direction, fail_out_of_bounds=True
    )

    count += len(path)

    explorer = BoundaryRRT(b0, n0, adh_f)

    expl_params = ExplorationParams(
        _expl_param_name("psphere", "brrt", "const", ndims, bp_enabled),
        explorer,
        n_bpoints,
        bp_enabled,
    )

    print("Running Exploration")
    expl_results = expl_exp(expl_params)

    # osv_errs = [
    #     angle_between(osv, true_osv_at(b)) * 180 / np.pi for b, osv in expl_params.bpoints
    # ]
    # b_errs = [abs(loc.distance_to(b) - radius) for b, osv in expl_results.bpoints]

    # avg_osv_err = sum(osv_errs) / len(osv_errs)
    # avg_b_err = sum(b_errs) / len(b_errs)

    # print("Saving")
    # with open(".tmp-comp-brrt.json", "w") as f:
    #     f.write(json.dumps(expl_results))
    # with open(".tmp-comp-gd.json", "w") as f:
    #     f.write(json.dumps(gd_results))

    rounded_percentage_effs = list(
        map(
            lambda x: round(x * 100, 2),
            [
                gd_results.eff_posttrain_nofailure,
                gd_results.eff_posttrain,
                gd_results.eff,
            ],
        )
    )

    print(
        "GD",
        f"failure rate: {round(gd_results.failure_percentage, 2)}%",
        f"Pretrain, posttrain, posttrain w failure efficiencies: \n{rounded_percentage_effs}",
        # f"Avg err: {gd_results.avg_b_err}",
        sep="\n",
    )
    print(
        "BRRT",
        f"BLEs: {expl_results.ble_count}",
        f"OOBs: {expl_results.out_of_bounds_count}",
        f"Pre-t0 Eff: {expl_results.eff}",
        f"Post-t0 Eff: {len(expl_results.bpoints) / (count + len(expl_results.nonbpoints))}",
        # f"Avg err: {avg_b_err}",
        sep="\n",
    )


if __name__ == "__main__":
    test_sim()
