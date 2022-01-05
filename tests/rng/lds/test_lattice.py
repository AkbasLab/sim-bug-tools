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

        [print(p) for p in points[:]]
        

        print("\n\n")
        # quit()
        return