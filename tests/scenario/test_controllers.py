from sim_bug_tools.rng.rrt import RapidlyExploringRandomTree
from sim_bug_tools.scenario.controllers import ControllerYieldConstantSpeed
import sim_bug_tools.scenario.units as units
import unittest
import numpy as np
import sim_bug_tools.utils as utils
import sim_bug_tools.rng.lds.sequences as sequences
import sim_bug_tools.structs as structs

class TestControllers(unittest.TestCase):

    def test_controller_yield_constant_speed(self):
        print()

        # Offset should be a prime number
        self.assertRaises(
            ValueError,
            ControllerYieldConstantSpeed,
            tau = np.float64(1),
            scenarios_to_run = np.int32(10),
            offset = 2048,
            speed = [units.Speed(kph=1), units.Speed(kph=100)],
            distance_from_junction = [units.Distance(meter=0), units.Distance(meter=1000)],
            local_searches = 5
        )

        controller = ControllerYieldConstantSpeed(
            tau = np.float64(1),
            scenarios_to_run = np.int32(10),
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

        for name, seq in SEQUENCES.items():
            for _name, _seq in SEQUENCES.items():
                print("\n [%s-%s]\n" % (name.upper(), _name.upper()))
                rrt = RapidlyExploringRandomTree(
                    _seq(
                        domain = controller.domain,
                        axes_names = controller.axes_names
                    ),
                    step_size = 0.01,
                    seed = 555,
                    exploration_radius = 0.1
                )
                controller.test_sequence(seq, rrt)
        

        print()
        return

def main():
    unittest.main()

if __name__ == "__main__":
    main()
