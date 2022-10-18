import unittest
import os, sys
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(FILE_DIR))

import sim_bug_tools.rng.lds.sequences as sequences
import sim_bug_tools.structs as structs
# import sim_bug_tools.utils as utils

import sim_bug_tools.exploration.brrt_v2.adherer as adherer

import regime
import brrt

import numpy as np
import pandas as pd



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
        return
        print("\n\n")
        # return
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