import unittest
import os, sys
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(FILE_DIR))

# import sim_bug_tools.rng.lds.sequences as sequences
# import sim_bug_tools.structs as structs
# import sim_bug_tools.utils as utils

# import simulator
import regime

class TestRegime(unittest.TestCase):

    def test1(self):
        print("\n\n")
        regime.RegimeSUMO()
        print("\n\n")
        return