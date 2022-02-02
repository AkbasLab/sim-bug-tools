import unittest
import sim_bug_tools.lshash as lshash

class TestLSH(unittest.TestCase):

    def test_lsh(self):
        print("\n\n")

        lsh = lshash.LSHash(6,2)

        print("\n\n")
        return