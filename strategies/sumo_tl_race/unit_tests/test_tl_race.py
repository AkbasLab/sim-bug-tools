import os, sys
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(FILE_DIR))
from tl_race import TrafficLightRace

import unittest
import sim_bug_tools.rng.lds.sequences as sequences

class TestTrafficLightRace(unittest.TestCase):
    def test_tl_race(self):
        print("\n\n")

        sim = TrafficLightRace(
            sequences.RandomSequence,
            file_name = "%s/out/tl-test.tsv" % FILE_DIR
        )
        
        sim.run(2)
        
        d = sim.as_dict()
        sim2 = TrafficLightRace.from_dict(d)
        sim2.resume()
        sim2.run(2)

        print("\n\n")
        return

if __name__ == "__main__":
    unittest.main()