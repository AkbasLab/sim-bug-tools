import os, sys
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(FILE_DIR))
from simple import Simple

import unittest

class TestSimple(unittest.TestCase):
    def test_simple(self):
        print("\n\n")
        sim = Simple(file_name = "%s/out/simple.tsv" % FILE_DIR)

        sim._run_sumo_scenario(1000, 100)
        print("\n\n")
        return

if __name__ == "__main__":
    unittest.main()