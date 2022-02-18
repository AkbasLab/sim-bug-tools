import unittest
import sim_bug_tools.simulator as simulator
import sim_bug_tools.structs as structs
import sim_bug_tools.sumo as sumo
import sim_bug_tools.rng.lds.sequences as sequences
import sim_bug_tools.utils as utils
import json
import os
from sim_bug_tools.rng.rrt import RapidlyExploringRandomTree
import pandas as pd

class TestSimulators(unittest.TestCase):

    def test_simulator(self):

        n_dim = 4
        sim = simulator.Simulator(
            domain = structs.Domain([(0,1) for n in range(n_dim)])
        )
        sim.long_walk_to_local_search_trigger = lambda : True
        sim.local_search_to_long_walk_trigger = lambda  : True
        sim.local_search_to_paused_trigger = lambda : True
        sim._n_steps_to_run = 10
        sim._log_to_console = False
        
        


        states = []
        for i in range(10):            
            sim.update()
            states.append(sim.state)

        self.assertEqual(states,[
            simulator.State.LOCAL_SEARCH,
            simulator.State.LONG_WALK,
            simulator.State.LOCAL_SEARCH,
            simulator.State.LONG_WALK,
            simulator.State.LOCAL_SEARCH,
            simulator.State.LONG_WALK,
            simulator.State.LOCAL_SEARCH,
            simulator.State.LONG_WALK,
            simulator.State.LOCAL_SEARCH,
            simulator.State.PAUSED
        ])

        

        sim.local_search_to_paused_trigger = lambda : False
        sim.local_search_to_long_walk_trigger = lambda : False
        sim._n_steps_to_run = 5
        sim.long_walk_on_enter()

        states = []
        for i in range(10):            
            sim.update()
            states.append(sim.state)
        states.append(sim.state)
        
        self.assertEqual(states[-1], simulator.State.INCOMPLETE_LOCAL_SEARCH)

        self.assertEqual(
            sim.as_json(),
            simulator.Simulator.from_dict(sim.as_dict()).as_json()
        )
        
        sim.run(10)


        sim.paused_on_enter()
        sim.resume()
        self.assertEqual(sim.state, simulator.State.LONG_WALK)

        sim.long_walk_on_enter()
        self.assertFalse(sim.resume())

        sim.incomplete_local_search_on_enter()
        sim.resume()
        self.assertEqual(sim.state, simulator.State.LOCAL_SEARCH)
        
        sim.incomplete_local_search_on_enter()
        sim.cancel()
        self.assertEqual(sim.state, simulator.State.PAUSED)

        sim.local_search_on_enter()
        self.assertFalse(sim.cancel())
        return






    def test_traci_client(self):
        # print("\n\n")

        map_dir = "sumo/tl_race"
        config = {
            "gui" : True,

            # Street network
            "--net-file" : "%s/tl-race.net.xml" % map_dir,

            # Logging
            "--error-log" : "%s/error-log.txt" % map_dir,

            # Traci Connection
            "--num-clients" : 1,
            "--remote-port" : 5522,

            # GUI Options
            "--delay" : 100,
            "--start" : "--quit-on-end",

            # RNG
            "--seed" : 333
        }

        # Should run fine.
        sim = sumo.TraCIClient(config)        
        sim.run_to_end()
        sim.close()



        

        # print("\n\n")
        return