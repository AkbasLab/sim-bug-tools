import sim_bug_tools.utils as utils
import numpy as np
import unittest
from unittest import TestCase

class TestUtils(TestCase):

    def test_as_position(self):
        self.assertTrue(
            isinstance(utils.as_position(np.array([.1,.2,.3,.4])), np.ndarray)
        )

        self.assertTrue(
            isinstance(utils.as_position([.1,.2,.3,.4]), np.ndarray)
        )

        self.assertRaises(TypeError, utils.as_position, None)

        self.assertRaises(TypeError, utils.as_position, "hhh")
        return


    def test_denormalize(self):
        
        a = 10
        b = 110
        
        self.assertEqual(utils.denormalize(a,b,0.1), 20.)

        self.assertEqual(
            list(utils.denormalize(a,b,np.array([.1, .2, .3]))),
            list(np.array([20.,30.,40.]))
        )
        return

    def test_prime(self):

        self.assertEqual(utils.prime(80), 419)
        self.assertRaises(ValueError, utils.prime, -1)
        self.assertRaises(ValueError, utils.prime, 1601)

        self.assertTrue(utils.is_prime(4217))
        self.assertFalse(utils.is_prime(2048))
        return


    def test_project(self):
        print("\n\n")

        utils.project(1.2, 5.1, 0.3, by=0.5)

        print("\n\n")
        return

def main():
    unittest.main()


if __name__ == "__main__":
    main()