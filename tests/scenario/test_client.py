import unittest
from unittest import TestCase
import sim_bug_tools.scenario.config as config
import sim_bug_tools.scenario.clients as clients
import sim_bug_tools.scenario.units as units
from sim_bug_tools.scenario.actors import ActorConstantSpeed

class TestClient(TestCase):

    def test_client(self):
        c1 = clients.Client(config.SUMO)
        self.assertEqual(c1.priority,1)
        c1.test()

        cfg = config.SUMO.copy()
        cfg["gui"] = True
        c2 = clients.Client(cfg)
        c2.test()
        return

    def test_vehicle_population(self):
        vp = clients.VehiclePopulation(config.SUMO,2)
        vp.test()
        return

    def test_scenario(self):
        speed1 = units.Speed(kph = 50)
        speed2 = units.Speed(kph = 10)
        dist = units.Distance(meter=200)
        dut = ActorConstantSpeed(speed1, dist)
        npc1 = ActorConstantSpeed(speed1, dist)
        npc2 = ActorConstantSpeed(speed2, dist)

        scenario_yield = clients.Scenario(dut, npc1, config.SUMO)
        self.assertEqual(scenario_yield.run_until_yield(1),
             scenario_yield.YIELD)

        scenario_no_yield = clients.Scenario(dut, npc2, config.SUMO)
        self.assertEqual(scenario_no_yield.run_until_yield(1), 
            scenario_no_yield.NO_YIELD)
        return


def main():
    unittest.main()

if __name__ == "__main__":
    main()