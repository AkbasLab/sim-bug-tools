import unittest
import sim_bug_tools.simulators as simulators
import sim_bug_tools.structs as structs

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