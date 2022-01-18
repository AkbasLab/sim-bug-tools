import unittest

from sim_bug_tools.structs import Point, Domain

class TestStructIO(unittest.TestCase):

    def test_point_io(self):
        arr = [1., 2., 3., 4.]
        p1 = Point(arr)
        self.assertEqual(p1.to_list(), arr)
        p1.as_json()       
        return

    def test_domain_io(self):
        domain = Domain([(0,1) for n in range(4)])
        domain.as_dict()
        domain.as_json()
        return