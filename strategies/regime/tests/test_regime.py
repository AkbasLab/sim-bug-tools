import unittest
import os, sys

from sklearn.metrics import mean_absolute_error
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(FILE_DIR))

import sim_bug_tools.rng.lds.sequences as sequences
import sim_bug_tools.structs as structs
import sim_bug_tools.utils as utils
import sim_bug_tools.graphics as graphics

import sim_bug_tools.exploration.brrt_v2.adherer as adherer

import regime
import brrt

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns
# import fitter




def target_score_classifier(score) -> bool:
    if isinstance(score, structs.Point):
        score = score.as_series()
    elif not isinstance(score, pd.Series):
        raise ValueError
    return score["e_brake"] > 0 and score["e_brake"] < .5

class TestRegime(unittest.TestCase):

    def test_round_to_limits(self):
        p0 = np.array([-1,2,4])
        _min = np.array([0,0,0])
        _max = np.array([3,3,3])
        x = brrt.round_to_limits(p0, _min, _max)
        self.assertTrue(all(x == np.array([0,2,3])))
        return

    def test_metric_1(self):
        return
        print("\n\n")
        
        # Create the regime
        r = regime.RegimeSUMO(target_score_classifier)

        # Load a dataframe
        df = pd.read_csv("%s/data/metric1.csv" % FILE_DIR)
        # df.describe().to_csv("hhh.csv")
        # return 
        df = df.iloc[:4000]

        mean_dist = r.metric_1(df)

        ax = graphics.new_axes()
        ax.plot(
            [i for i in range(len(mean_dist))],
            mean_dist
        )

        ax.set_xlabel("Iteration")
        ax.set_ylabel("Mean distance between X")
        ax.set_title("Metric 1 (63 normal dimensions)")

        plt.savefig("%s/figures/metric_1.png" % FILE_DIR)

        print("\n\n")
        return

    def test_global_exploration(self):
        return

        # Create the regime
        r = regime.RegimeSUMO(target_score_classifier)
        
        # Use random sequence as a placeholder
        seq = sequences.RandomSequence(
            r.parameter_manager.domain,
            r.parameter_manager.axes_names
        )
        seq.seed = 222
        for i in range(10):
            seq.get_points(1)

        # The exploration should end after 2 steps.
        r.global_exploration(seq)
        return

    def test_boundary_detection(self):
        # return
        print("\n\n")
        # Create the regime
        r = regime.RegimeSUMO(target_score_classifier)

        # Use random sequence as a placeholder
        seq = sequences.RandomSequence(
            r.parameter_manager.domain,
            r.parameter_manager.axes_names
        )
        seq.seed = 222
        for i in range(10):
            seq.get_points(1)

        # The exploration should end after 2 steps.
        r.global_exploration(seq)
        
        # Now do the boundary detection step
        r.boundary_detection()

        print("\n\n")
        return