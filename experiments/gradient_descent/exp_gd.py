import tensorflow as tf
import numpy as np

from exp_ann import ANNExperiment, ANNParams, ANNResults, Scorable, ProbilisticSphere

from sim_bug_tools.structs import Point, Domain
from sim_bug_tools.experiment import Experiment, ExperimentParams, ExperimentResults
from sim_bug_tools.rng.lds.sequences import SobolSequence

from dataclasses import dataclass
from numpy import ndarray
from typing import Callable


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


class GDExplorerExperiment(Experiment[GDExplorerParams, GDExplorerResults]):
    def __init__(self):
        super().__init__()
        self._previous_result: tuple[Point, list[PointData]] = None

    def experiment(self, params: GDExplorerParams) -> GDExplorerResults:
        scored_data = params.ann_results.scored_data
        model = tf.keras.models.load_model(params.ann_results.model_path)
        envelope = params.ann_results.params.envelope

        self._domain = Domain.normalized(len(scored_data[0][0]))

        b_paths: list[list[PointData]] = []
        nonb_paths: list[list[PointData]] = []

        for init_p, init_score in scored_data[: params.num_bpoints]:
            init_p = Point(init_p) if type(init_p) is not Point else init_p
            path = []
            for pd in self.find_boundary_from(
                init_p, init_score, params, envelope, model
            ):
                path.append(pd)

            if self._previous_was_bpath:
                b_paths.append(path)
            else:
                nonb_paths.append(path)

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
            cur += params.alpha * (s if prev_cls else -s)

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


def _ann_param_name(n_samples: int, ndims: int):
    return f"ANN-psphere-{ndims}d-{n_samples}"


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


if __name__ == "__main__":
    test_samplesXgd()
