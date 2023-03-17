

from numpy import ndarray
from typing import Callable

from sim_bug_tools.structs import Point, Domain
from sim_bug_tools.experiment import Experiment, ExperimentParams, ExperimentResults


class Scorable:
    pass 

class Graded(Scorable):
    pass

class ANNParams(ExperimentParams):
    def __init__(
        self,
        name: str,
        classified_points: list[tuple[Point, bool]],
        envelope,
        batch_size=128,
        n_epochs=500,
        desc: str = None,
    ):
        super().__init__(name, desc)
        self.classified_points = classified_points
        
    


class ANNResults(ExperimentResults[ANNParams]):
    def __init__(self, params: ANNParams, model, labeled_data):
        super().__init__(params)


class ANNExperiment(Experiment[ANNParams, ANNResults]):
    def experiment(self, params: ANNParams) -> ANNResults:
        pass
    
    
