import unittest
from unittest import TestCase
import sim_bug_tools.scenario.units as units
from sim_bug_tools.scenario.actors import ActorConstantSpeed

class TestActor(TestCase):
    
    def test_actor_constant_speed(self):
        speed = units.Speed(kph=30)
        distance = units.Distance(meter = 300)

        self.assertRaises(TypeError,ActorConstantSpeed, None , None)
        self.assertRaises(TypeError,ActorConstantSpeed, None , distance)
        self.assertRaises(TypeError,ActorConstantSpeed, speed , None)
        
        actor = ActorConstantSpeed(speed,distance)

        self.assertTrue(  isinstance(actor.speed, units.Speed)  )
        self.assertTrue(  isinstance(actor.distance_from_junction, units.Distance)  )
        self.assertTrue(  isinstance(actor.vehicle_id, str)  )

        return

def main():
    unittest.main()


if __name__ == "__main__":
    main()