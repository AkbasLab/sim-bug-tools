import unittest
import sim_bug_tools as sbt


class TestUtils(unittest.TestCase):

    def test_prime(self):
        print("\n\n")
        
        with self.assertRaises(ValueError):
            sbt.prime(1601)
        
        self.assertEqual(13, sbt.prime(5))

        print("\n\n")
        return
    
    def test_1(self):
        return
    
    def test_2(self):
        return
    
