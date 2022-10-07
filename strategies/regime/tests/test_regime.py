import unittest
import os, sys
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(FILE_DIR))

import sim_bug_tools.rng.lds.sequences as sequences
# import sim_bug_tools.structs as structs
# import sim_bug_tools.utils as utils

from sim_bug_tools.exploration.brrt_v2.adherer \
    import BoundaryAdherer, BoundaryAdherenceFactory

import regime
import numpy as np
import pandas as pd

class TestRegime(unittest.TestCase):

    def test_global_exploration(self):
        # Define the target score classifier
        def target_score_classifier(point : structs.Point) -> bool:
            return score["e_brake"] > 0 and score["e_brake"] < .5

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
        # Define the target score classifier
        def target_score_classifier(score : pd.Series) -> bool:
            return score["e_brake"] > 0 and score["e_brake"] < .5

        # Create the regime
        r = regime.RegimeSUMO(target_score_classifier)


        d = 0.005
        r = 2
        angle = 30 * np.pi / 180
        num = 4

        return