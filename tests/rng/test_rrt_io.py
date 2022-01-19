import unittest
from sim_bug_tools.rng.rrt import RapidlyExploringRandomTree
import sim_bug_tools.rng.lds.sequences as sequences
import sim_bug_tools.structs as structs
import json

class TestRRTIO(unittest.TestCase):

    def test_rrt_io(self):
        n_dim = 4
        seed = 555
        seq = sequences.RandomSequence(
            domain = structs.Domain([(0,1) for n in range(n_dim)]),
            axes_names = ["dim%d" % n for n in range(n_dim)],
            seed = seed
        )

        empty_rrt = RapidlyExploringRandomTree(
            seq = seq,
            step_size = 0.01,
            exploration_radius = 1
        )

        empty_rrt_dict = empty_rrt.as_dict()
        empty_rrt_new = RapidlyExploringRandomTree.from_dict(empty_rrt_dict)

        self.assertEqual(
            empty_rrt.as_json(),
            empty_rrt_new.as_json()
        )

        
        rrt = empty_rrt.copy()
        point = structs.Point([1.,2.,3.,4.])
        rrt.reset(point)
        [rrt.step() for n in range(10)]
        rrt_dict = rrt.as_dict()
        rrt_new = RapidlyExploringRandomTree.from_dict(rrt_dict)

        self.assertEqual(rrt.as_json(), rrt_new.as_json())
        return

def main():
    unittest.main()