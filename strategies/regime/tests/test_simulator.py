import unittest
import os, sys
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(FILE_DIR))

import simulator

import sim_bug_tools.rng.lds.sequences as sequences
import sim_bug_tools.structs as structs

class TestSimulator(unittest.TestCase):

    def testTLRPM(self):
        # Initialize the Parameter Manager
        manager = simulator.TrafficLightRaceParameterManager()

        # Choose a Random Sequence
        seq = sequences.RandomSequence(
            manager.domain,
            manager.axes_names
        )
        seq.seed = 222

        # Get the first point in the sequence
        point = seq.get_points(1)[0]

        # Obtain dataframes of concrete parameters for vehicles and TL
        manager.map_parameters(point)
        return

    def testTLRTest(self):
        # Initialize the Parameter Manager
        manager = simulator.TrafficLightRaceParameterManager()

        # Choose a Random Sequence
        seq = sequences.RandomSequence(
            manager.domain,
            manager.axes_names
        )
        seq.seed = 222

        # Get the first point in the sequence
        point = seq.get_points(1)[0]

        # Obtain dataframes of concrete parameters for vehicles and TL
        manager.map_parameters(point)

        # Simulation Test
        test = simulator.TrafficLightRaceTest()
        return