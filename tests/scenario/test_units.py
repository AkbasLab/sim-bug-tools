import unittest
from unittest import TestCase
import sim_bug_tools.scenario.units as units

class TestUnits(TestCase):

    def test_distance(self):
        
        self.assertEqual(units.Distance(meter = 1).meter, 1)
        self.assertEqual(units.Distance(meter = 1).nanometer, 1e9)
        self.assertEqual(units.Distance(meter = 1).METER_TO_NANOMETER, 1e9)
        self.assertEqual(units.Distance(meter = 1).NANOMETER_TO_METER, 1e-9)
        self.assertRaises(ValueError,units.Distance)
        return

    def test_speed(self):

        self.assertAlmostEqual(
            units.Speed(mps = 15).mps, 
            15, places = 12
        )

        self.assertAlmostEqual(
            units.Speed(meters_per_second = 15).mps, 
            15, places = 12
        )

        self.assertAlmostEqual(
            units.Speed(kph=30).kph, 
            30, places = 6
        )

        self.assertAlmostEqual(
            units.Speed(kilometers_per_hour = 30).kph, 
            30, places = 6
        )

        self.assertEqual(units.Speed(mps = 1).MPS_TO_NMPS, 1e9)
        self.assertEqual(units.Speed(mps = 1).NMPS_TO_MPS, 1/1e9)
        self.assertEqual(units.Speed(mps = 1).KPH_TO_MPS, 10./36.)
        self.assertEqual(units.Speed(mps = 1).MPS_TO_KPH, 3.6)
        self.assertEqual(units.Speed(mps = 1).NMPS_TO_KPH, 1/1e9 * 3.6)
        self.assertRaises(ValueError,units.Speed)
        return

    def test_time(self):

        self.assertEqual(units.Time(second = 1).second,1)
        self.assertEqual(units.Time(hour = 1).second, 3600)
        self.assertEqual(units.Time(second = 1).nanosecond, 1e9)
        self.assertRaises(ValueError,units.Time)
        return



def main():
    unittest.main()


if __name__ == "__main__":
    main()