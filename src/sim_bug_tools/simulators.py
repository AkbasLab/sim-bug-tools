from abc import abstractmethod
from abc import ABC
import enum
import sim_bug_tools.structs as structs
import sim_bug_tools.rng.lds.sequences as sequences
from sim_bug_tools.rng.rrt import RapidlyExploringRandomTree
import pandas as pd

class State(enum.Enum):
    PAUSED = "PAUSED"
    LONG_WALK = "LONG_WALK"
    LOCAL_SEARCH = "LOCAL_SEARCH"
    NO_SIMULATOR_LOADED = "NO_SIMULATOR_LOADED"
    INCOMPLETE_LOCAL_SEARCH = "INCOMPLETE_LOCAL_SEARCH"

class Simulator():
    def __init__(self, domain : structs.Domain):
        self._step = 0
        self._domain = domain
        self._n_long_walks = 0
        self._n_temp_long_walks = 0
        self._n_local_searches = 0
        self._n_temp_local_searches = 0
        self._n_bugs = 0
        self._history = pd.DataFrame({
            "step" : [],
            "point" : [],
            "is_bug" : [],
            "state" : []
        })
        self._n_steps_to_run = 0
        self._last_observed_point = None
        self._local_search_enabled = False
        self.paused()
        return

    @property
    def state(self) -> State:
        return self._state

    @property
    def step(self) -> int:
        return self._step

    @property
    def domain(self) -> structs.Domain:
        return self._domain

    @property
    def n_long_walks(self) -> int:
        return self._n_long_walks
        
    @property
    def n_temp_long_walks(self) -> int:
        return self._n_temp_long_walks
    
    @property
    def n_local_searches(self) -> int:
        return self._n_local_searches

    @property
    def n_temp_local_searches(self) -> int:
        return self._n_temp_local_searches

    @property
    def n_bugs(self) -> int:
        return self._n_bugs

    @property
    def history(self) -> pd.DataFrame:
        return self._history

    @property
    def n_steps_to_run(self) -> int:
        return self._n_steps_to_run

    @property
    def last_observed_point(self) -> structs.Point:
        return self._last_observed_point

    @property
    def local_search_enabled(self) -> bool:
        return self._local_search_enabled
    
    def enable_local_search(self):
        self._local_search_enabled = True
        return

    def paused(self):
        self._state = State.PAUSED
        return

    @abstractmethod
    def long_walk(self, point : structs.Point, is_bug : bool):
        self._state = State.LONG_WALK
        self._step += 1
        self._n_long_walks += 1
        self._n_steps_to_run -= 1
        self.add_to_history(point, is_bug)
        print("long walk")

        is_bug = True

        # Change state
        if self.n_steps_to_run <= 0:
            self.local_search_to_paused()
            self.paused()
            return
        elif is_bug:
            self.long_walk_to_local_search()
            self.local_search()
        self.long_walk()
        return

    
    def local_search(self):
        if self.local_search_enabled:
            self._state = State.LOCAL_SEARCH
            self._step += 1
            self._n_local_searches += 1
            self._n_steps_to_run -= 1
            print("local search")
        return

    
    def incomplete_local_search(self):
        self._state = State.INCOMPLETE_LOCAL_SEARCH
        return

    # Transitions
    def long_walk_to_local_search(self):
        return 

    def local_search_to_paused(self):
        return 
    
    def run(self, n : int):
        self._n_steps_to_run = n
        del n

        if self.state == State.PAUSED:
            self.long_walk()
        elif self.stae == State.INCOMPLETE_LOCAL_SEARCH:
            self.local_search()
        return    

    def add_to_history(self, point : structs.Point, is_bug : bool):
        self._history = self._history.append({
            "step" : self.step,
            "point" : point,
            "is_bug" : is_bug,
            "state" : self.state,
        }, ignore_index = True)
        self._last_observed_point = point
        return
    


class SimpleSimulatorKnownBugs(Simulator):
    """
    A simple simulator with known bugs.
    Samples from a normal domain.
    """
    def __init__(self,
        bug_profile : list[structs.Domain],
        sequence : sequences.Sequence
    ):
        n_dim = len(sequence.domain)
        super().__init__(
            domain = structs.Domain([(0,1) for _ in range(n_dim)])
        )

        self._bug_profile = bug_profile
        self._sequence = sequence
        return

    @property
    def bug_profile(self) -> list[structs.Domain]:
        return self._bug_profile
    
    @property
    def sequence(self) -> sequences.Sequence:
        return self._sequence

    def long_walk(self):
        # Sample from the sequence
        point = self.sequence.get_points(1)[0]

        # Check if it's in any bug boundary
        is_bug = any([point in bug_envelope for bug_envelope in self.bug_profile])

        # Call the parent function
        super().long_walk(point, is_bug)         
        return 


class SimpleSimulatorKnownBugsRRT(SimpleSimulatorKnownBugs):
    """
    A simple simulator with known bugs.
    Samples from a normal domain.
    RRT Version
    """
    def __init__(self, 
        bug_profile : list[structs.Domain],
        sequence : sequences.Sequence,
        rrt : RapidlyExploringRandomTree,
        n_branches : int
    ):
        super().__init__(bug_profile, sequence)
        self._rrt = rrt
        self._n_branches = n_branches
        self.enable_local_search()
        return

    @property
    def rrt(self) -> RapidlyExploringRandomTree:
        return self._rrt

    @property
    def n_branches(self) -> int:
        return self._n_branches
    

    def long_walk_to_local_search(self):
        # Reset the RRT
        self.rrt.reset(self.last_observed_point)
        return

    def local_search(self):

        super().local_search()
        return

    
    


    
    

    