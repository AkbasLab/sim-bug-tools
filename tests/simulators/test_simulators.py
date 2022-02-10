import unittest
import sim_bug_tools.simulators.base as sim_base
import sim_bug_tools.simulators.sumo as sim_sumo
import sim_bug_tools.structs as structs
import sim_bug_tools.rng.lds.sequences as sequences
import sim_bug_tools.utils as utils
import json
import os
from sim_bug_tools.rng.rrt import RapidlyExploringRandomTree
import pandas as pd
import traci
import sys

class Testsim_base(unittest.TestCase):

    def test_simulator(self):

        n_dim = 4
        sim = sim_base.Simulator(
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
            sim_base.State.LOCAL_SEARCH,
            sim_base.State.LONG_WALK,
            sim_base.State.LOCAL_SEARCH,
            sim_base.State.LONG_WALK,
            sim_base.State.LOCAL_SEARCH,
            sim_base.State.LONG_WALK,
            sim_base.State.LOCAL_SEARCH,
            sim_base.State.LONG_WALK,
            sim_base.State.LOCAL_SEARCH,
            sim_base.State.PAUSED
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
        
        self.assertEqual(states[-1], sim_base.State.INCOMPLETE_LOCAL_SEARCH)

        self.assertEqual(
            sim.as_json(),
            sim_base.Simulator.from_dict(sim.as_dict()).as_json()
        )
        
        sim.run(10)
        return






    def test_simulator_known_bugs(self):
        with open("tests/simulators/test_bugs.json", "r") as f:
            for line in f:
                hhh = json.loads(line)
                break
        bug_profile = [structs.Domain.from_json(d) for d in hhh[0]]
        
        n_dim = len(bug_profile[0])
        domain = structs.Domain([(0,1) for n in range(n_dim)])
        seq = sequences.RandomSequence(
            domain, 
            ["dim_%d" % n for n in range(n_dim)],
            seed = 300
        )

        sim = sim_base.SimpleSimulatorKnownBugs(
            bug_profile, seq, 
            file_name = "tests/simulators/out/sskb.tsv"
        )

        if os.path.exists(sim.file_name):
            os.remove(sim.file_name)

        sim.run(10)
        sim.long_walk_on_enter()
        sim.run(10)

        self.assertEqual(utils.rawincount(sim.file_name), 21)
        return



    def test_simulator_known_bugs_rrt(self):
        bug_profile = [structs.Domain([(0,1) for n in range(4)])]
        
        n_dim = len(bug_profile[0])
        domain = structs.Domain([(0,1) for n in range(n_dim)])
        axes_names = ["dim_%d" % n for n in range(n_dim)]
        seq = sequences.RandomSequence(
            domain, axes_names, seed = 300
        )
        rrt = RapidlyExploringRandomTree(
            sequences.RandomSequence(
                domain, axes_names, seed = 555
            ),
            step_size = 0.01,
            exploration_radius = 1
        )
        
        sim = sim_base.SimpleSimulatorKnownBugsRRT(
            bug_profile = bug_profile,
            sequence = seq,
            rrt = rrt,
            n_branches = 5,
            file_name = "tests/simulators/out/sskbrrt.tsv",
            log_to_console = False
        )

        if os.path.exists(sim.file_name):
            os.remove(sim.file_name)

        sim.run(10)
        sim.local_search_on_enter()
        sim.run(14)

        self.assertEqual(utils.rawincount(sim.file_name), 25)

        df = pd.read_csv(sim.file_name, sep="\t")
        self.assertEqual( df[df.state == "LONG_WALK"].state.count(), 4 )
        self.assertEqual( df[df.state == "LOCAL_SEARCH"].state.count(), 20 )
        return



    def test_traci_client(self):
        print("\n\n")

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

        # SHould raise error when no config is given.
        self.assertRaises(ValueError, sim_sumo.TraCIClient)

        # Should run fine.
        sim = sim_sumo.TraCIClient(config=config)        
        sim.run_to_end()
        sim.close()



        

        print("\n\n")
        return