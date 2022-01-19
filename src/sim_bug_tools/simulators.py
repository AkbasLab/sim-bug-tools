from abc import abstractmethod
import enum
import sim_bug_tools.structs as structs
import sim_bug_tools.rng.lds.sequences as sequences
from sim_bug_tools.rng.rrt import RapidlyExploringRandomTree
import sim_bug_tools.utils as utils
import pandas as pd
import json
import os

def from_dict(d : dict):
    return

class State(enum.Enum):
    PAUSED = "PAUSED"
    LONG_WALK = "LONG_WALK"
    LOCAL_SEARCH = "LOCAL_SEARCH"
    NO_SIMULATOR_LOADED = "NO_SIMULATOR_LOADED"
    INCOMPLETE_LOCAL_SEARCH = "INCOMPLETE_LOCAL_SEARCH"
    INITIALIZED = "INITIALIZED"

class Simulator():
    """
    Simulator base class. This base class collects common statistics
    and defines the four states of operation. In addition, this base
    class provides methods of I/O.
    """
    def __init__(self, domain : structs.Domain):
        self._state = State.INITIALIZED
        self._step = 0
        self._domain = domain
        self._n_long_walks = 0
        self._n_local_searches = 0
        self._n_bugs = 0
        self._local_search_enabled = False
        self._id = "0"
        self._log_to_console = True
        self._file_name = "hhh.sim"

        # Temp Data
        self._history = pd.DataFrame({
            "step" : [],
            "point" : [],
            "is_bug" : [],
            "state" : []
        })
        self._n_steps_to_run = 0
        self._last_observed_point = None
        return

    @property
    def state(self) -> State:
        """
        The simulation state.
        """
        return self._state

    @property
    def step(self) -> int:
        """
        The current simulation step.
        """
        return self._step

    @property
    def domain(self) -> structs.Domain:
        """
        The domain which defines the simulation space.
        """
        return self._domain

    @property
    def n_long_walks(self) -> int:
        """
        The number of long walk steps perfomed so far.
        """
        return self._n_long_walks
    
    @property
    def n_local_searches(self) -> int:
        """
        The number of local search steps performed so far.
        """
        return self._n_local_searches

    @property
    def n_bugs(self) -> int:
        """
        The number of bugs observed so far.
        """
        return self._n_bugs

    @property
    def history(self) -> pd.DataFrame:
        """
        A temporary history of simulation results of the current run.
        """
        return self._history

    @property
    def n_steps_to_run(self) -> int:
        """
        Number of steps left to run.
        """
        return self._n_steps_to_run

    @property
    def last_observed_point(self) -> structs.Point:
        """
        The last observed point from the previous step.
        """
        return self._last_observed_point

    @property
    def local_search_enabled(self) -> bool:
        """
        If local search is enabled.
        """
        return self._local_search_enabled

    @property
    def log_to_console(self) -> bool:
        """
        If the self.log() function will print to console.
        """
        return self._log_to_console

    @property
    def id(self) -> str:
        """
        A unique ID.
        """
        return self._id

    @property
    def file_name(self) -> str:
        """
        Filename of the simulator record file.
        """
        return self._file_name
    
    def enable_local_search(self):
        """
        Enables local search.
        """
        self._local_search_enabled = True
        return
    
    def enable_log_to_console(self):
        """
        Enables logging messages to console with self.log()
        """
        self._log_to_console = True
        return

    def set_id(self, id : str):
        """
        Set the id.
        """
        self._id = id
        return


    @abstractmethod
    def as_dict(self) -> dict:
        return {
            "class" : self.__class__.__name__,
            "step": self.step,
            "domain": self.domain.as_dict(),
            "n_long_walks" : self.n_long_walks,
            "n_local_searches" : self.n_local_searches,
            "n_bugs" : self.n_bugs,
            "local_search_enabled" : self.local_search_enabled,
            "id" : self.id,
            "state" : self.state.value
        }

    def as_json(self) -> str:
        return json.dumps(self.as_dict())        

    @staticmethod
    @abstractmethod
    def from_dict(d : dict, sim = None):
        """
        Create a simulator instance from a dictionary

        -- Parameters --
        d : dict
            Instance properties
        sim : Simulator (default = None)
            simulator to inherit properties. Will construct a new instance
            if None.
        """
        if sim is None:
            sim = Simulator(  structs.Domain.from_dict(d["domain"])  )
        sim._step = int(d["step"])
        sim._n_long_walks = int(d["n_long_walks"])
        sim._n_local_searches = int(d["n_local_searches"])
        sim._n_bugs = int(d["n_bugs"])
        sim._local_search_enabled = bool(d["local_search_enabled"])
        sim._id = str(d["id"])
        sim._state = State(str(d["state"]))
        return sim

    # Internal
    def _clear_temp_data(self):
        """
        Clears tempory data.
        """
        self._history = pd.DataFrame({
            "step" : [],
            "point" : [],
            "is_bug" : [],
            "state" : []
        })
        self._n_steps_to_run = 0
        self._last_observed_point = None
        return

    def _write_to_file(self):
        # File does not exist
        if not os.path.exists(self.file_name):
            with open(self.file_name, "r+") as f:
                f.write(self.as_json())
                f.write(self.history.to_csv(index=False))

        # File exists.
        else:
            # The first line is the simulation configuration.
            with open(self.file_name, "r+") as f:
                f.seek(0) # Point to first line
                f.write(self.as_json())
            
            # Now add the temporary records to the end.
            with open(self.file_name, "a") as f:
                f.write(self.history.to_csv(index=False, header=False))
        return


    # States
    def paused(self):
        """
        Paused State
        """
        self._state = State.PAUSED
        self._clear_temp_data()
        self.log("Paused")
        self._write_to_file()
        return

    def incomplete_local_search(self):
        """
        Incomplete Local Search State
        """
        self._state = State.INCOMPLETE_LOCAL_SEARCH
        self._clear_temp_data()
        self.log("Incomplete Local Search")
        return


    def long_walk(self, point : structs.Point, is_bug : bool):
        """
        Long Walk State.

        -- Parameters --
        point : structs.Point
            Point observed.
        is_bug : bool
            If the observed point was a bug.
        """
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
        """
        Local Search State

        -- Parameters --
        point : structs.Point
            Point observed.
        is_bug : bool
            If the observed point was a bug.
        """
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

    
    

    # Transitions
    def long_walk_to_local_search(self):
        """
        Transition function.
        Long Walk State -> Local Search State
        """
        return 

    def local_search_to_paused(self):
        """
        Transition function.
        Local Search State -> Paused State
        """
        return

    def local_search_to_long_walk(self):
        """
        Transition function.
        Local Search State -> Long Walk State
        """
        return
    
    def local_search_to_paused(self):
        """
        Transition function.
        Local Search State -> Paused State
        """
        return

    def local_search_to_incomplete_local_search(self):
        """
        Transition function.
        Local Search State -> Incomplete Local Search State
        """
        return


    #  Exit conditions
    def local_search_exit_condition(self) -> bool:
        """
        Exit Condition function to leave Local Search State.

        -- Return --
        bool
            Will leave Local Search State when True.
        """
        return True
    


    #  Other Functions
    def run(self, n : int):
        """
        Runs the simulation for n_steps

        --- Parameters --
        n : int
            Steps to run.
        """
        self._n_steps_to_run = n
        del n

        if self.state == State.PAUSED:
            self.long_walk()
        elif self.state == State.INCOMPLETE_LOCAL_SEARCH:
            self.local_search()
        return  


    def add_to_history(self, point : structs.Point, is_bug : bool):
        """
        Adds a new row to the history dataframe

        --- Parameters ---
        point : structs.Point
            A point in space.
        is_bug : bool
            If a bug was observed at that point in space.
        """
        self._history = self._history.append({
            "step" : self.step,
            "point" : point,
            "is_bug" : is_bug,
            "state" : self.state.value,
        }, ignore_index = True)
        self._last_observed_point = point
        return

    def log(self, msg : str) -> str:
        """
        A wrapper for print() to log messages to various I/O.
        Appends the ID to the front of the message.

        --- Parameters ---
        msg : str
            Message to be printed.

        --- Return --
        str
            The transformed message.
        """
        msg = "%s:%s" % (self.id, msg)
        if self.log_to_console:
            print(msg)
        return msg


class SimpleSimulatorKnownBugs(Simulator):
    """
    A simple simulator with known bugs.
    Samples from a normal domain.
    """
    def __init__(self,
        bug_profile : list[structs.Domain],
        sequence : sequences.Sequence,
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
        """
        A list of domains which form the bug profile.
        """
        return self._bug_profile
    
    @property
    def sequence(self) -> sequences.Sequence:
        """
        Sequence used to generate points.
        """
        return self._sequence

    @abstractmethod
    def as_dict(self) -> dict:
        d = {
            "bug_profile" : [domain.as_dict() for domain in self.bug_profile],
            "sequence" : self.sequence.as_dict()
        }
        return utils.flatten_dicts([d, super().as_dict()])

    @staticmethod
    @abstractmethod
    def from_dict(d : dict, sim = None):
        """
        Create a simulator instance from a dictionary

        -- Parameters --
        d : dict
            Instance properties
        sim : Simulator (default = None)
            simulator to inherit properties. Will construct a new instance
            if None.
        """
        if sim is None:
            sim = SimpleSimulatorKnownBugs(
                bug_profile = [structs.Domain.from_dict(domain) \
                    for domain in d["bug_profile"]],
                sequence = sequences.from_dict(d["sequence"])
            )
        return Simulator.from_dict(d, sim)


    def long_walk(self):
        """
        The sequence generates a point which is compared to the bug profile.
        """
        # Sample from the sequence
        point = self.sequence.get_points(1)[0]

        # Check if it's in the bug profile
        is_bug = self.is_point_in_bug_profile(point)

        # Call the parent function
        return super().long_walk(point, is_bug)         
        


    def is_point_in_bug_profile(self, point : structs.Point) -> bool:
        """
        Checks if a point is within the bug profile.

        --- Parameters ---
        point : structs.Point
            A point in space

        --- Return --
        bool
            True when the point exists within the bug profile.
        """
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
        """
        A Rapidly Exploring Random Tree
        """
        return self._rrt

    @property
    def n_branches(self) -> int:
        """
        The number of branches the RRT will grow before resetting to 0 branches.
        """
        return self._n_branches

    @property
    def n_branches_remaining(self) -> int:
        """
        The number of branches left for the RRT to grow.
        """
        return self._n_branches_remaining

    def as_dict(self) -> dict:
        d = {
            "rrt" : self.rrt.as_dict(),
            "n_branches" : self.n_branches,
            "n_branches_remaining" : self.n_branches_remaining
        }
        return utils.flatten_dicts([d, super().as_dict()])

    @staticmethod
    @abstractmethod
    def from_dict(d : dict, sim = None):
        """
        Create a simulator instance from a dictionary

        -- Parameters --
        d : dict
            Instance properties
        sim : Simulator (default = None)
            simulator to inherit properties. Will construct a new instance
            if None.
        """
        if sim is None:
            sim = SimpleSimulatorKnownBugsRRT(
                bug_profile = [structs.Domain.from_dict(domain) \
                    for domain in d["bug_profile"]],
                sequence = sequences.from_dict(d["sequence"]),
                rrt = RapidlyExploringRandomTree.from_dict(d["rrt"]),
                n_branches = int(d["n_branches"])
            )
        sim._n_branches_remaining = int(d["n_branches_remaining"])
        return SimpleSimulatorKnownBugs.from_dict(d, sim)
    

    def long_walk_to_local_search(self):
        """
        The RRT is reset and centered on the last observed point.
        """
        # Reset the RRT
        self.rrt.reset(self.last_observed_point)
        self._n_branches_remaining = self.n_branches
        return

    def local_search(self):
        """
        Local Exploration using the RRT for point selection, until the
        specified amount of branches are grown.
        """
        # Generate the next point
        point = self.rrt.step()[2]

        # Check if it's in the bug profile
        is_bug = self.is_point_in_bug_profile(point)

        # Reduce branches remaining.
        self._n_branches_remaining -= 1

        #  Call parent function
        return super().local_search(point, is_bug)
        
    
    def local_search_exit_condition(self) -> bool:
        """
        The local search ends when the RRT does not need to grow anymore
        branches.
        """
        return self.n_branches_remaining <= 0

    
    


    
    

    