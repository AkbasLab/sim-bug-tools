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
        self._id = "0"
        self._log_to_console = True
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

    @property
    def log_to_console(self) -> bool:
        return self._log_to_console

    @property
    def id(self) -> str:
        return self._id
    
    def enable_local_search(self):
        self._local_search_enabled = True
        return
    
    def enable_log_to_console(self):
        self._log_to_console = True
        return

    def set_id(self, id : str):
        self._id = id
        return

    def log(self, msg : str) -> str:
        msg = "%s:%s" % (self.id, msg)
        if self.log_to_console:
            print(msg)
        return msg

    # States
    def paused(self):
        self._state = State.PAUSED
        return


    def long_walk(self, point : structs.Point, is_bug : bool):
        self._state = State.LONG_WALK
        self._step += 1
        self._n_long_walks += 1
        self._n_steps_to_run -= 1
        self.add_to_history(point, is_bug)
        is_bug = True
        self.log("Long Walk")

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

    
    def local_search(self, point : structs.Point = None, is_bug : bool = None):
        if not self.local_search_enabled:
            return
        elif point is None:
            raise ValueError("point is None.")
        elif is_bug is None:
            raise ValueError("is_bug is None.")

        self._state = State.LOCAL_SEARCH
        self._step += 1
        self._n_local_searches += 1
        self._n_steps_to_run -= 1
        self.add_to_history(point, is_bug)
        self.log("Local Search")

        # Change state
        if self.n_steps_to_run <= 0:
            if self.local_search_exit_condition():
                self.local_search_to_paused()
                self.paused()
                return
            else:
                self.local_search_to_incomplete_local_search()
                self.incomplete_local_search()
                return
        elif self.local_search_exit_condition():
            self.local_search_to_long_walk()
            self.long_walk()
        self.local_search()
        return

    
    def incomplete_local_search(self):
        self._state = State.INCOMPLETE_LOCAL_SEARCH
        return

    # Transitions
    def long_walk_to_local_search(self):
        return 

    def local_search_to_paused(self):
        return

    def local_search_to_long_walk(self):
        return
    
    def local_search_to_paused(self):
        return

    def local_search_to_incomplete_local_search(self):
        return


    #  Exit conditions
    def local_search_exit_condition(self) -> bool:
        return True
    



    def run(self, n : int):
        self._n_steps_to_run = n
        del n

        if self.state == State.PAUSED:
            self.long_walk()
        elif self.state == State.INCOMPLETE_LOCAL_SEARCH:
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

        # Check if it's in the bug profile
        is_bug = self.is_point_in_bug_profile(point)

        # Call the parent function
        return super().long_walk(point, is_bug)         
        


    def is_point_in_bug_profile(self, point : structs.Point) -> bool:
        return any([point in bug_envelope for bug_envelope in self.bug_profile])


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
        self._n_branches_remaining = n_branches
        self.enable_local_search()
        return

    @property
    def rrt(self) -> RapidlyExploringRandomTree:
        return self._rrt

    @property
    def n_branches(self) -> int:
        return self._n_branches

    @property
    def n_branches_remaining(self) -> int:
        return self._n_branches_remaining
    

    def long_walk_to_local_search(self):
        # Reset the RRT
        self.rrt.reset(self.last_observed_point)
        self._n_branches_remaining = self.n_branches
        return

    def local_search(self):
        # Generate the next point
        point = self.rrt.step()[2]

        # Check if it's in the bug profile
        is_bug = self.is_point_in_bug_profile(point)

        # Reduce branches remaining.
        self._n_branches_remaining -= 1

        #  Call parent function
        return super().local_search(point, is_bug)
        
    
    def local_search_exit_condition(self) -> bool:
        return self.n_branches_remaining <= 0

    
    


    
    

    