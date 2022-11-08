import unittest
import os, sys
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(FILE_DIR))

import simulator

import sim_bug_tools.structs as structs
import sim_bug_tools.rng.lds.sequences as sequences

class TestSimulator(unittest.TestCase):

    def test_simulator(self):
        return
        print("\n\n")

        s = simulator.HighwayPassTest()

        print("\n\n")
        return

    def test_parameter_manager(self):
        print("\n\n")
        # return
        # Initialize the manager
        manager = simulator.HighwayPassTestParameterManager()

        # Sequence Generator
        rng = sequences.RandomSequence(
            domain = structs.Domain.normalized(len(manager.params_df.index)),
            axes_names = manager.params_df["feature"].tolist()    
        )
        rng.seed = 555

        for i in range(81):
            rng.get_points(1)

        # Test with a random point
        for i in range(1):
            print("SIM", i)
            p = rng.get_points(1)[0]
            s = simulator.HighwayPassTest(manager.map_parameters(p))
            print("")

        

        print("\n\n")
        return