# 
import numpy as np

from numpy import ndarray

from exp_ann import ANNExperiment, ANNParams, ANNResults, ProbilisticSphere
from exp_gd import GDExplorerExperiment, GDExplorerParams, GDExplorerResults
from sim_bug_tools.simulation.simulation_core import Scorable, Graded
from sim_bug_tools.rng.lds.sequences import RandomSequence, SobolSequence
from sim_bug_tools import Point, Domain


# How to have multiple envelopes in the same space
class CompositeScorable(Scorable):
    def __init__(self, scorables: list[Scorable]):
        """
        Note that the resulting score must be identical, and that the scores
        will be summed to get the composite score!
        """
        self.scorables = scorables
        self._input_dims = scorables[0].get_input_dims()
        self._score_dims = scorables[0].get_score_dims()
        # assert all(
        #     [s.get_score_dims() == self._score_dims for s in scorables[1:]] or True
        # ), "Not all of the scores match"
        
    
    def score(self, p: Point) -> ndarray:
        return sum([s.score(p) for s in self.scorables])

    
    def classify_score(self, score: ndarray) -> bool:
        return any([s.classify_score(score) for s in self.scorables])
 
    def get_input_dims(self):
        return self._input_dims 

    
    def get_score_dims(self):
        return self._score_dims

ndim = 3
num_bpoints = 100 #*how many successful envelope search attempts*
domain = Domain.normalized(ndim) # just searching between 0 and 1
seq = SobolSequence(domain, [f'x{i}' for i in range(ndim)])

scorable = CompositeScorable([
    ProbilisticSphere(Point([0.5] * ndim), 0.3, 0.25),
    ProbilisticSphere(Point([0] * ndim), 0.5, 0.25),
    ProbilisticSphere(Point([1] * ndim), 0.2, 0.25),
    ProbilisticSphere(Point([0.2] * (ndim-2) + [0.4, 8]), 0.25, 0.25),
])
    

# Before running gradient descent, we must build an ANN
ann_exp = ANNExperiment()

ann_params = ANNParams("model-1", scorable, seq, 500, 100, 100)

# WARNING: will automatically cache results. If a result cache is found,
# it will load the cache instead of rerunning the experiment. This is 
# true for exp_obj(params) and exp_obj.lazily_run(params). If you want
# to bypass this, run exp_obj.run(params) to run again and cache a new
# result. If you don't want to cache, then use exp_obj.experiment(params)*
ann_results = ann_exp(ann_params) 

# Now we can run gradient descent
gd_exp = GDExplorerExperiment()
gd_params = GDExplorerParams("gd-global_search-1", ann_results, num_bpoints, GDExplorerExperiment.steepest_descent)
gd_results = gd_exp(gd_params)

"""
Gradient descent results contain the paths that were taken to look for
an envelope. There are two types of paths, boundary paths and 
non-boundary paths. Boundary paths are those that have encountered an
envelope, the others failed to do so.

Note that this experiment looks for the boundary, not the maximum
score. The difference is pretty small, so I don't think it matters.
"""

# Let's animate this in step-wise fashion
import matplotlib.pyplot as plt
from sim_bug_tools.graphics import Grapher
g = Grapher(ndim == 3, domain)

get_points = lambda lst_pd: [p_data.point for p_data in lst_pd]

for path in gd_results.boundary_paths:
    #* we can plot out a path all at once
    # path = get_points(path)
    # g.plot_all_points(path)
    # g.draw_path(path) # draw dotted line between points
    
    #* we can plot the final boundary points as well
    g.plot_all_points(get_points(gd_results.boundary))
    
    #* Or we can draw them one-by-one
    tmp = []
    _p_graph = None 
    _path_graph = None
    for pd in path:
        tmp += [pd.point]
        
        if _p_graph:
            # remove previous graphics
            _p_graph.remove()
            for _g in _path_graph:
                _g.remove()
        
        # plot updated graphics
        _p_graph = g.plot_all_points(tmp, color='blue')
        _path_graph = g.draw_path(tmp, color='blue')
        
        plt.pause(0.01) # display / update the window
        input("press enter to continue")
        