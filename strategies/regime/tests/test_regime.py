import unittest
import os, sys
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(FILE_DIR))

import sim_bug_tools.rng.lds.sequences as sequences
# import sim_bug_tools.structs as structs
# import sim_bug_tools.utils as utils

# import simulator
import regime

class TestRegime(unittest.TestCase):

    def test_global_exploration(self):
        return
        # Create the regime
        r = regime.RegimeSUMO()
        
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

    def test_local_sensitivity_reduction(self):
        # Create the regime
        r = regime.RegimeSUMO()
        return