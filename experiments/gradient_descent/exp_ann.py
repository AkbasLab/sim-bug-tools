from sim_bug_tools.structs import Point, Domain
from sim_bug_tools.experiment import Experiment, ExperimentParams, ExperimentResults

from numpy import ndarray
from typing import Callable


class ANNParams(ExperimentParams):
    def __init__(
        self,
        name: str,
        classified_points: list[tuple[Point, bool]],
        batch_size=128,
        n_epochs=500,
        desc: str = None,
    ):
        super().__init__(name, desc)
        self.classified_points = classified_points


class ANNResults(ExperimentResults[ANNParams]):
    def __init__(self, exp_name: str, params: ANNParams, param_name: str = None):
        super().__init__(exp_name, params, param_name)


class ANNExperiment(Experiment[ANNParams, ANNResults]):
    def experiment(self, params: ANNParams) -> ANNResults:
        pass
