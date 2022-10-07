import unittest
import sim_bug_tools.structs as structs
import pandas as pd

class TestPoint(unittest.TestCase):


    def test_point(self):
        s = pd.Series([1,2,3,4,5], index=["a", "b", "c", "d", "e"])
        p = structs.Point(s)
        self.assertEqual(p.as_series()["a"], 1)

        p = structs.Point([1,2,3,4,5])
        self.assertEqual(p.as_series()[0], 1)
        return 
