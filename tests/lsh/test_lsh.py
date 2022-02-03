import unittest
import sim_bug_tools.lshash as lshash
import numpy as np
import sim_bug_tools.structs as structs

class TestLSH(unittest.TestCase):

    def test_lsh(self):
        # print("\n\n")

        n_dim = 2
        lsh = lshash.LSHash(6,n_dim,n_hash_generators=10)

        rng = np.random.default_rng(3)
        points = [structs.Point(np.round(p,1)) for p in rng.uniform(size=(1000, n_dim))]
        [lsh.index(p) for p in points]
        

        # print("\n\n")
        return