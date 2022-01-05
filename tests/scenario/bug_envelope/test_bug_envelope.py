import unittest
from sim_bug_tools.rng.rrt import RapidlyExploringRandomTree
from sim_bug_tools.scenario.controllers import ControllerYieldConstantSpeed
import sim_bug_tools.rng.lds.sequences as sequences
import sim_bug_tools.scenario.units as units
import sim_bug_tools.utils as utils
import numpy as np
import sim_bug_tools.graphics as graphics
import sim_bug_tools.scenario.clients as clients
import os



class TestBugEnvelope(unittest.TestCase):
        
    
    def test_generate_bugs(self):
        
        generate_bugs = False
        if generate_bugs:

            controller = ControllerYieldConstantSpeed(
                tau = np.float64(.1),
                scenarios_to_run = np.int32(1000),
                offset = utils.prime(333),
                speed = [units.Speed(kph=1), units.Speed(kph=100)],
                distance_from_junction = [units.Distance(meter=0), units.Distance(meter=1000)],
                local_searches = 5
            )

            SEQUENCES = {
                "Sobol" : sequences.SobolSequence,
                "Halton" : sequences.HaltonSequence,
                "Faure" : sequences.FaureSequence,
                "Random" : sequences.RandomSequence
            }

            print("\nLegend: LONG-LOCAL\n")

            
            long_seq = SEQUENCES["Halton"]
            local_seq = SEQUENCES["Random"]

            rrt = RapidlyExploringRandomTree(
                local_seq(
                    domain = controller.domain,
                    axes_names = controller.axes_names
                ),
                step_size = 0.01,
                seed = 555,
                exploration_radius = 0.1
            )
            points, statuses = controller.test_sequence(long_seq, rrt)

            fn = os.path.join("sim_bug_tools","tests","scenario","bug_envelope","bugs.pkl")
            utils.save([points,statuses],fn)

            # points, statuses = utils.load(fn)
            
            # def is_bug(status : str):
            #     if status.upper() == clients.Scenario.YIELD:
            #         return True
            #     return False

            # for i, point in enumerate(points):
            #     print(point, statuses[i])

            # bug_envelopes = graphics.Voronoi(
            #     points,
            #     bugs = [is_bug(status) for status in statuses]
            # ).bug_envelopes

            # print(
            #     type(bug_envelopes)
            # )
        print()

        return

def main():
    unittest.main()

if __name__ == "__main__":
    main()