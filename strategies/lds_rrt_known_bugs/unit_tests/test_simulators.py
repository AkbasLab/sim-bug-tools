# Add the parent directory to the path
import os, sys
UNITTEST_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(UNITTEST_DIR))
import simulators as simulator


import unittest
import sim_bug_tools.structs as structs
import sim_bug_tools.rng.lds.sequences as sequences
import json
import sim_bug_tools.utils as utils
from sim_bug_tools.rng.rrt import RapidlyExploringRandomTree
import pandas as pd

class TestSimulators(unittest.TestCase):
    def test_simulator_known_bugs(self):
        with open("%s/test_bugs.json" % UNITTEST_DIR, "r") as f:
            for line in f:
                hhh = json.loads(line)
                break
        bug_profile = [structs.Domain.from_json(d) for d in hhh[0]]
        
        n_dim = len(bug_profile[0])
        domain = structs.Domain([(0,1) for n in range(n_dim)])
        seq = sequences.RandomSequence(
            domain, 
            ["dim_%d" % n for n in range(n_dim)],
            seed = 300
        )

        sim = simulator.SimpleSimulatorKnownBugs(
            bug_profile, seq, 
            file_name = "%s/out/sskb.tsv" % UNITTEST_DIR
        )

        if os.path.exists(sim.file_name):
            os.remove(sim.file_name)

        sim.run(10)
        sim.long_walk_on_enter()
        sim.run(10)

        self.assertEqual(utils.rawincount(sim.file_name), 21)

        
        return



    def test_simulator_known_bugs_rrt(self):
        bug_profile = [structs.Domain([(0,1) for n in range(4)])]
        
        n_dim = len(bug_profile[0])
        domain = structs.Domain([(0,1) for n in range(n_dim)])
        axes_names = ["dim_%d" % n for n in range(n_dim)]
        seq = sequences.RandomSequence(
            domain, axes_names, seed = 300
        )
        rrt = RapidlyExploringRandomTree(
            sequences.RandomSequence(
                domain, axes_names, seed = 555
            ),
            step_size = 0.01,
            exploration_radius = 1
        )
        
        sim = simulator.SimpleSimulatorKnownBugsRRT(
            bug_profile = bug_profile,
            sequence = seq,
            rrt = rrt,
            n_branches = 5,
            file_name = "%s/out/sskbrrt.tsv" % UNITTEST_DIR,
            log_to_console = False
        )

        if os.path.exists(sim.file_name):
            os.remove(sim.file_name)

        sim.run(10)
        sim.local_search_on_enter()
        sim.run(14)

        self.assertEqual(utils.rawincount(sim.file_name), 25)

        df = pd.read_csv(sim.file_name, sep="\t")
        self.assertEqual( df[df.state == "LONG_WALK"].state.count(), 4 )
        self.assertEqual( df[df.state == "LOCAL_SEARCH"].state.count(), 20 )
        return


if __name__ == "__main__":
    unittest.main()
