from sim_bug_tools.experiment import Experiment, ExperimentParams, ExperimentResults

from numpy import ndarray
from typing import Callable


class GDExplorerParams(ExperimentParams):
    def __init__(
        self,
        name: str,
        h: Callable[[ndarray], ndarray],
        alpha=0.01,
        max_samples=1000,
        desc: str = None,
    ):
        super().__init__(name, desc)
        self.h = h
        self.alpha = alpha
        self.max_samples = max_samples
        self.desc = desc


class GDExplorerResults(ExperimentResults[ExperimentParams]):
    def __init__(self, exp_name: str, params: ExperimentParams, param_name: str = None):
        super().__init__(exp_name, params, param_name)


class GDExplorerExperiment(Experiment[GDExplorerParams, GDExplorerResults]):
    def experiment(self, params: GDExplorerParams) -> GDExplorerResults:
        pass
