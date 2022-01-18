import unittest
from sim_bug_tools.rng.rrt import RapidlyExploringRandomTree
import sim_bug_tools.rng.lds.sequences as sequences
import sim_bug_tools.structs as structs
import json

class TestRRTIO(unittest.TestCase):

    def test_rrt_io(self):
        print("\n\n")

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

        # print(json.dumps(empty_rrt_dict))

        print("\n\n")
        return