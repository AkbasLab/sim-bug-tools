import unittest
import os, sys
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(FILE_DIR))

import simulator

import sim_bug_tools.rng.lds.sequences as sequences
import sim_bug_tools.structs as structs
import sim_bug_tools.utils as utils

class TestSimulator(unittest.TestCase):

    

    def testTLRPM(self):
        # return
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
        params = manager.map_parameters(point)

        # Flatten parameters
        flat_params = manager.flatten_params_df(
            params["veh"]["concrete"], params["tl"]["concrete"])
            
        # The flat sequence should be the same length as the number of 
        # dimensions
        self.assertEqual(len(flat_params.index), manager.n_dim)
        return

    def testTLRTest(self):
        return
        # Initialize the Parameter Manager
        manager = simulator.TrafficLightRaceParameterManager()

        # Choose a Random Sequence
        seq = sequences.RandomSequence(
            manager.domain,
            manager.axes_names
        )
        seq.seed = 222

        for i in range(100):
            # Get the first point in the sequence
            point = seq.get_points(1)[0]

            # Obtain dataframes of concrete parameters for vehicles and TL
            params = manager.map_parameters(point)

            # Simulation Test
            test = simulator.TrafficLightRaceTest(
                veh_params = params["veh"]["concrete"],
                tl_params = params["tl"]["concrete"]
            )
            # break

        return

    def test_10k_random(self):
        return
        # Initialize the Parameter Manager
        manager = simulator.TrafficLightRaceParameterManager()

        # Choose a Random Sequence
        seq = sequences.RandomSequence(
            manager.domain,
            manager.axes_names
        )
        seq.seed = 222

        # Output dir
        out_dir = "%s/temp" % FILE_DIR

        # Test structure is:
        #  test_id
        #  | + params
        #  | | + veh
        #  | | | + concrete
        #  | | | + normal
        #  | | + tl
        #  | |   + concrete
        #  | |   + normal
        #  | + scores
        #  |   + veh
        #  |   + scores

        data = dict() 

        n_tests = 10_000

        for i in range(n_tests):
            print("\n TEST %d of %d\n" % (i, n_tests - 1))
            # Get the first point in the sequence
            point = seq.get_points(1)[0]

            # Obtain dataframes of concrete parameters for vehicles and TL
            params = manager.map_parameters(point)

            # Simulation Test
            test = simulator.TrafficLightRaceTest(
                veh_params = params["veh"]["concrete"],
                tl_params = params["tl"]["concrete"]
            )

            data[i] = {
                "params" : params,
                "scores" : {
                    "veh" : test.veh_score_df,
                    "scores" : test.scores
                }
            }
            continue
        
        fn = "%s/random_10k.pkl" % out_dir
        utils.save(data, fn)
        return