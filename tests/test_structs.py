import unittest
import sim_bug_tools.structs as structs

class TestDiscreteCollection(unittest.TestCase):

    def test_discrete_range(self):
        # print("\n\n")
        
        dc1 = structs.DiscreteCollection(1.3, 8.1, .5)
        self.assertEqual(len(dc1.keys), 15)

        dc2 = structs.DiscreteCollection(1, 2, .5)
        self.assertEqual(len(dc2.keys), 3)

        # print("\n\n")
        return
