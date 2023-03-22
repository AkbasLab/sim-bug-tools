import tensorflow as tf
import numpy as np

from exp_ann import ANNExperiment, ANNParams, ANNResults, Graded

from sim_bug_tools.structs import Point, Domain
from sim_bug_tools.experiment import Experiment, ExperimentParams, ExperimentResults
from sim_bug_tools.rng.lds.sequences import SobolSequence

from dataclasses import dataclass
from numpy import ndarray
from typing import Callable


class ProbilisticSphere(Graded):
    def __init__(self, loc: Point, radius: float, lmbda: float):
        """
        Probability density is formed from the base function f(x) = e^-(x^2),
        such that f(radius) = lmbda and is centered around the origin with a max
        of 1.

        Args:
            loc (Point): Where the sphere is located
            radius (float): The radius of the sphere
            lmbda (float): The density of the sphere at its radius
        """
        self.loc = loc
        self.radius = radius
        self.lmda = lmbda
        self.ndims = len(loc)

        self._c = 1 / radius**2 * np.log(1 / lmbda)

    def score(self, p: Point) -> ndarray:
        "Returns between 0 (far away) and 1 (center of) envelope"
        dist = self.loc.distance_to(p)

        return np.array(1 / np.e ** (self._c * dist**2))

    def classify_score(self, score: ndarray) -> bool:
        return np.linalg.norm(score) < self.lmda

    def gradient(self, p: Point) -> np.ndarray:
        s = p - self.loc
        s /= np.linalg.norm(s)

        return s * self._dscore(p)

    def get_input_dims(self):
        return len(self.loc)

    def get_score_dims(self):
        return 1

    def _dscore(self, p: Point) -> float:
        return -self._c * self.score(p) * self.loc.distance_to(p)


@dataclass
class PointData:
    point: Point
    score: ndarray
    cls: bool
    gradient: ndarray = None


class GDPerfParams(ExperimentParams):
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


class GDPerfResults(ExperimentResults[GDPerfParams]):
    def __init__(
        self,
        params: GDPerfParams,
        b_counts: list[int],
        b_errs: list[float],
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


class GDExplorerExperiment(Experiment[GDPerfParams, GDPerfResults]):
    def __init__(self):
        super().__init__()
        self._previous_result: tuple[Point, list[PointData]] = None

    def experiment(self, params: GDPerfParams) -> GDPerfResults:
        scored_data = self.ann_results.scored_data
        model = tf.keras.models.load_model(self.ann_results.model_path)
        envelope = self.ann_results.params.envelope

        b_paths: list[list[PointData]] = []
        nonb_paths: list[list[PointData]] = []

        for init_p, init_score in scored_data[: params.num_bpoints]:
            path = []
            for pd in self.find_boundary_from(
                init_p, init_score, params, envelope, model
            ):
                path.append(pd)

            if self._previous_was_bpath:
                b_paths.append(path)
            else:
                nonb_paths.append(path)

        return GDPerfResults(params, b_paths, nonb_paths)

    def predict_gradient(self, p: Point, model: tf.keras.Sequential):
        inp = tf.Variable(np.array([p]), dtype=tf.float32)

        with tf.GradientTape() as tape:
            preds = model(inp)

        return np.array(tape.gradient(preds, inp)).squeeze()

    def find_boundary_from(
        self,
        init_p: Point,
        init_score: ndarray,
        params: GDPerfParams,
        envelope: Scorable,
        model: tf.keras.Sequential,
    ):
        domain = Domain.normalized(len(init_p))
        i = 0
        prev = init_p
        prev_cls = envelope.classify_score(init_score)

        cur = init_p
        cur_score = init_score
        cur_cls = prev_cls

        # Loop if max_steps not exceeded and boundary has not been found
        while i < params.max_steps and cur_cls == prev_cls and cur in domain:
            prev = cur
            prev_score = cur_score
            prev_cls = cur_cls
            # Make a step according to gradient descent solution
            prev_g = self.predict_gradient(prev, model)
            cur += params.h(prev_g) if prev_cls else -params.h(prev_g)

            cur_score = envelope.score(cur)
            cur_cls = envelope.classify_score(cur_score)
            i += 1

            yield PointData(prev, prev_score, prev_cls, prev_g)

        cur_g = self.predict_gradient(cur, model)

        self._previous_was_bpath = cur_cls != prev_cls and cur in domain

        yield PointData(cur, cur_score, cur_cls, cur_g)

    @staticmethod
    def steepest_descent(self, g: ndarray) -> ndarray:
        pass

    @staticmethod
    def gradient_descent_with_momentum(self, g: ndarray) -> ndarray:
        pass


if __name__ == "__main__":
    ndims = 3
    num_bpoints = 100
    domain = Domain.normalized(ndims)

    gd_exp = GDExplorerExperiment()
    ann_exp = ANNExperiment()
    envelope = ProbilisticSphere(Point([0.5 for i in range(ndims)]), 0.4, 0.25)
    seq = SobolSequence(domain, [str(i) for i in range(ndims)])

    for ann_samples in [50, 100, 200, 400, 800, 1600, 3200, 6400]:
        ann_name = f"ANN-psphere-{ann_samples}"
        ann_params = ANNParams(
            ann_name, envelope, seq, ann_samples, int(ann_params / 10)
        )

        ann_results = ann_exp.lazily_run(ann_params)

        gd_params = GDPerfParams(
            f"{ann_name}-sd",
            ann_results,
            num_bpoints,
            GDExplorerExperiment.steepest_descent,
        )
        gd_result = gd_exp.lazily_run(gd_params)
