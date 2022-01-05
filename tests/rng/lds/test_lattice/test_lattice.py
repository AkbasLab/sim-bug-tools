import unittest
import sim_bug_tools.structs as structs
import sim_bug_tools.utils as utils
from sim_bug_tools.rng.lds.sequences import LatticeSequence

class TestLattice(unittest.TestCase):
    def test_lattice(self):
        print("\n\n")


        n_dim = 3
        domain = structs.Domain.normalized(n_dim)
        axes_names = ["x", "y"]
        n_pts = 1000
        skip = utils.prime(40)

        seq = LatticeSequence(domain, axes_names)
        
        points = seq.get_points(101**n_dim + 5)

        print(len(points))

        # [print(p) for p in points[:]]
        fn = "sim_bug_tools/tests/rng/lds/test_lattice/dim3.txt"
        with open(fn,"w") as f:
            for p in points:
                f.write("%s\n" % str(p))

        print("\n\n")
        # quit()
        return