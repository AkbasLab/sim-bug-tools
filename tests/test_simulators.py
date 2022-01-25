import unittest
import sim_bug_tools.simulators as simulators
import sim_bug_tools.structs as structs

class TestSimulators(unittest.TestCase):

    def test_simulator(self):
        print("\n\n")

        n_dim = 4
        sim = simulators.Simulator(
            structs.Domain([(0,1) for n in range(n_dim)])
        )
        
        for i in range(10):
            sim.update()
            break

        print("\n\n")
        return