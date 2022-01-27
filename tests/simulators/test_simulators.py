import unittest
import sim_bug_tools.simulators as simulators
import sim_bug_tools.structs as structs
import sim_bug_tools.rng.lds.sequences as sequences
import sim_bug_tools.utils as utils
import json

class TestSimulators(unittest.TestCase):

    def test_simulator(self):

        n_dim = 4
        sim = simulators.Simulator(
            structs.Domain([(0,1) for n in range(n_dim)])
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
            simulators.State.LOCAL_SEARCH,
            simulators.State.LONG_WALK,
            simulators.State.LOCAL_SEARCH,
            simulators.State.LONG_WALK,
            simulators.State.LOCAL_SEARCH,
            simulators.State.LONG_WALK,
            simulators.State.LOCAL_SEARCH,
            simulators.State.LONG_WALK,
            simulators.State.LOCAL_SEARCH,
            simulators.State.PAUSED
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
        
        self.assertEqual(states[-1], simulators.State.INCOMPLETE_LOCAL_SEARCH)

        self.assertEqual(
            sim.as_json(),
            simulators.Simulator.from_dict(sim.as_dict()).as_json()
        )
        
        sim.run(10)
        return






    def test_simulator_known_bugs(self):
        print("\n\n")

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

        sim = simulators.SimpleSimulatorKnownBugs(
            bug_profile, seq, 
            file_name = "tests/simulators/out/sskb.sim"
        )
        with open(sim.file_name, "w") as f:
            f.write("")
        sim.run(10)
        sim.run(10)

        # self.assertEqual(utils.rawincount(sim.file_name), 20)

        

        print("\n\n")
        return

