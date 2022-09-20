from abc import abstractmethod
import enum
import sim_bug_tools.structs as structs
# import sim_bug_tools.rng.lds.sequences as sequences
# from sim_bug_tools.rng.rrt import RapidlyExploringRandomTree
# import sim_bug_tools.utils as utils
import pandas as pd
# import numpy as np
import json
import os


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
    def __init__(self, **kwargs):
        self._state = State.INITIALIZED
        self._step = 0
        self._n_long_walks = 0
        self._n_local_searches = 0
        self._n_bugs = 0


        try:
            self._domain = kwargs["domain"]
        except KeyError:
            self._domain = None

        
        try:
            self._id = kwargs["id"]
        except KeyError:
            self._id = ""
        assert isinstance(self.id, str)
        
        try:
            self._log_to_console = kwargs["log_to_console"]
        except KeyError:
            self._log_to_console = False
        assert isinstance(self.log_to_console, bool)

        try:
            self._file_name = kwargs["file_name"]
            if not self.file_name[-4:] == ".tsv":
                self._file_name = "%s.tsv" % self.file_name
        except KeyError:
            self._file_name = ""
        assert isinstance(self.file_name, str)

        # Temp Data
        self._history = pd.DataFrame({
            "step" : [],
            "is_bug" : [],
            "state" : [],
            "point_normal" : [],
            "point_concrete" : []
        })
        self._n_steps_to_run = 0
        self._last_observed_point_normal = None
        self._last_observed_point_concrete = None
        self._last_observed_point_is_bug = False
        return

    def ___GETTERS_AND_SETTERS___(self):
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
    def last_observed_point_normal(self) -> structs.Point:
        """
        The last observed normal point from the previous step.
        """
        return self._last_observed_point_normal

    @property
    def last_observed_point_concrete(self) -> structs.Point:
        """
        The last observed concrete point from the previous step.
        """
        return self._last_observed_point_concrete

    @property
    def last_observed_point_is_bug(self) -> bool:
        """
        If the last observed point is a bug.
        """
        return self._last_observed_point_is_bug

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
        Filename of the history file.
        """
        return self._file_name

    @property
    def run_complete(self) -> bool:
        """
        True when there are no steps left to run.
        """
        return self.n_steps_to_run <= 0
    
    
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



    def ___IO___(self):
        return

    
    def as_dict(self) -> dict:
        return {
            "class" : self.__class__.__name__,
            "step": self.step,
            "domain": self.domain.as_dict(),
            "n_long_walks" : self.n_long_walks,
            "n_local_searches" : self.n_local_searches,
            "n_bugs" : self.n_bugs,
            "id" : self.id,
            "state" : self.state.value,
            "file_name" : self.file_name
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
            sim = Simulator(  domain = structs.Domain.from_dict(d["domain"])  )
        sim._step = int(d["step"])
        sim._n_long_walks = int(d["n_long_walks"])
        sim._n_local_searches = int(d["n_local_searches"])
        sim._n_bugs = int(d["n_bugs"])
        sim._id = str(d["id"])
        sim._state = State(str(d["state"]))
        sim._file_name = d["file_name"]
        return sim



    def ___INTERNAL___(self):
        return
    
    def _clear_temp_data(self):
        """
        Clears tempory data.
        """
        self._history = pd.DataFrame({
            "step" : [],
            "is_bug" : [],
            "state" : [],
            "point_normal" : [],
            "point_concrete" : []
        })
        self._n_steps_to_run = 0
        self._last_observed_point_normal = None
        self._last_observed_point_normal = None
        return

    def _write_to_file(self):
        """
        Append the history to the .tsv file and update the .json file with
        the simulator statistics.
        """
        if not self.file_name:
            return

        use_header = not os.path.exists(self.file_name)
        with open(self.file_name, "a") as f:
            f.write(
                self.history.to_csv(header=use_header, index=False, sep = "\t")
            )
        
        settings_fn = "%s.json" % self.file_name[:-4]
        with open(settings_fn, "w") as f:
            f.write(json.dumps(self.as_dict(), indent=4))
        return



    def ____STATES___(self):
        return

    # States
    def paused_on_update(self):
        """
        Paused State. Called on update.
        """
        return

    def paused_on_enter(self):
        """
        Paused State. Called on enter.
        """
        self._state = State.PAUSED
        self.log("Paused")
        self._write_to_file()
        self._clear_temp_data()
        return

    def paused_on_exit(self):
        """
        Paused State. Called on exit.
        """
        return



    def incomplete_local_search_on_update(self):
        """
        Incomplete Local Search State. Called on update.
        """
        return

    def incomplete_local_search_on_enter(self):
        """
        Incomplete Local Search State. Called on enter.
        """
        self._state = State.INCOMPLETE_LOCAL_SEARCH
        self.log("Incomplete Local Search")
        self._write_to_file()
        self._clear_temp_data()
        return

    def incomplete_local_search_on_exit(self):
        """
        Incomplete Local Search State. Called on exit.
        """
        return




    def long_walk_on_update(self, 
            point_normal : structs.Point = None, 
            point_concrete : structs.Point = None,
            is_bug : bool = None):
        """
        Long Walk State. Called on update.

        -- Parameters --
        point : structs.Point
            Point observed.
        is_bug : bool
            If the observed point was a bug.
        """
        self._step += 1
        self._n_long_walks += 1
        self._n_steps_to_run -= 1
        self.add_to_history(point_normal, point_concrete, is_bug)
        self.log("Long Walk")
        return

    def long_walk_on_enter(self):
        """
        Long Walk State. Called on enter.
        """
        self._state = State.LONG_WALK
        return

    def long_walk_on_exit(self):
        """
        Long Walk State. Called on exit.
        """
        return




    def local_search_on_update(self, 
            point_normal : structs.Point = None, 
            point_concrete : structs.Point = None,
            is_bug : bool = None):
        """
        Local Search State. Called on update.

        -- Parameters --
        point : structs.Point
            Point observed.
        is_bug : bool
            If the observed point was a bug.
        """
        self._step += 1
        self._n_local_searches += 1
        self._n_steps_to_run -= 1
        self.add_to_history(point_normal, point_concrete, is_bug)
        self.log("Local Search")
        return

    def local_search_on_enter(self):
        """
        Local Search. Called on enter.
        """
        self._state = State.LOCAL_SEARCH
        return

    def local_search_on_exit(self):
        """
        Local Search. Called on exit.
        """
        return

    
    
    
    def ___TRANSITIONS___(self):
        return

    
    def paused_to_long_walk_on_enter(self):
        """
        Transition function.
        Paused State -> Long Walk State
        Called on enter.
        """
        return

    def paused_to_long_walk_trigger(self) -> bool:
        """
        Transition function.
        Paused State -> Long Walk State.
        Trigger condition.
        """
        return False
    

    def long_walk_to_local_search_on_enter(self):
        """
        Transition function.
        Long Walk State -> Local Search State
        Called on enter.
        """
        return 

    def long_walk_to_local_search_trigger(self) -> bool:
        """
        Transition function.
        Long Walk State -> Local Search State
        Trigger condition.
        """
        return False


    def long_walk_to_paused_on_enter(self):
        """
        Transition function.
        Long Walk State -> Paused State
        Called on enter.
        """
        return

    def long_walk_to_paused_trigger(self) -> bool:
        """
        Transition function.
        Long Walk State -> Paused State
        Trigger condition.
        """
        return False


    def local_search_to_paused_on_enter(self):
        """
        Transition function.
        Local Search State -> Paused State
        Called on enter.
        """
        return

    def local_search_to_paused_trigger(self) -> bool:
        """
        Transition function.
        Local Search State -> Paused State
        Trigger condition.
        """
        return False

    def local_search_to_long_walk_on_enter(self):
        """
        Transition function.
        Local Search State -> Long Walk State
        Called on enter.
        """
        return

    def local_search_to_long_walk_trigger(self) -> bool:
        """
        Transition function.
        Local Search State -> Long Walk State
        Trigger Condition
        """
        return False
    
    def local_search_to_paused_on_enter(self):
        """
        Transition function.
        Local Search State -> Paused State
        Called on enter.
        """
        return

    def local_search_to_paused_trigger(self) -> bool:
        """
        Transition function.
        Local Search State -> Paused State
        Trigger Condition
        """
        return False


    def local_search_to_incomplete_local_search_on_enter(self):
        """
        Transition function.
        Local Search State -> Incomplete Local Search State
        Called on enter.
        """
        return

    def local_search_to_incomplete_local_search_trigger(self) -> bool:
        """
        Transition function.
        Local Search State -> Incomplete Local Search State
        Trigger Condition.
        """
        return False

    def incomplete_local_search_to_local_search_on_enter(self):
        """
        Transition function.
        Incomplete Local Search State --> Local Search State
        Called on enter.
        """
        return

    def incomplete_local_search_to_local_search_trigger(self) -> bool:
        """
        Transition function.
        Incomplete Local Search State --> Local Search State
        Trigger Condition
        """
        return False

    def incomplete_local_search_to_paused_on_enter(self):
        """
        Transition function.
        Incomplete Local Search State --> Paused State
        Called on enter.
        """
        return

    def incomplete_local_search_to_paused_trigger(self) -> bool:
        """
        Transition function.
        Incomplete Local Search State --> Paused State
        Trigger Condition
        """
        return False




    def ___OTHER_FUNCTIONS___(self):
        return

    
    def update(self):
        """
        Update the simulation.
        1 simulation step. 
        """

        if self.state is State.INITIALIZED:
            self.long_walk_on_enter()
        
        if self.state is State.LONG_WALK:
            self.long_walk_on_update()
            if self.run_complete or self.long_walk_to_paused_trigger():
                self.local_search_on_exit()
                self.long_walk_to_paused_on_enter()
                self.paused_on_enter()
            elif self.long_walk_to_local_search_trigger():
                self.local_search_on_exit()
                self.long_walk_to_local_search_on_enter()
                self.local_search_on_enter()

        elif self.state is State.LOCAL_SEARCH:
            self.local_search_on_update()
            if self.run_complete:
                self.local_search_on_exit()
                
                if self.local_search_to_paused_trigger():
                    self.local_search_to_paused_on_enter()
                    self.paused_on_enter()
                else:
                    self.local_search_to_incomplete_local_search_on_enter()
                    self.incomplete_local_search_on_enter()
            elif self.local_search_to_long_walk_trigger():
                self.local_search_on_exit()
                self.local_search_to_long_walk_on_enter()
                self.long_walk_on_enter()

        elif self.state is State.PAUSED:
            self.paused_on_update()
            if self.paused_to_long_walk_trigger():
                self.paused_to_long_walk_on_enter()
                self.long_walk_on_enter()

        elif self.state is State.INCOMPLETE_LOCAL_SEARCH:
            self.incomplete_local_search_on_update()
            if self.incomplete_local_search_to_paused_trigger():
                self.incomplete_local_search_to_paused_on_enter()
                self.paused_on_enter()
        return



    def run(self, n : int):
        """
        Runs the simulation for n_steps

        --- Parameters --
        n : int
            Steps to run.
        """
        self._n_steps_to_run = n

        for i in range(n):
            if self.run_complete:
                break
            self.update()
        return  

    def resume(self) -> bool:
        """
        Transitions the simulation from Pause to Long Walk or from
        Incomplete Local Search to Local Search.
        Ignores triggers.

        --- Return --
        bool
            True if successful. False otherwise.
        """
        if self.state is State.PAUSED:
            self.paused_on_exit()
            self.paused_to_long_walk_on_enter()
            self.long_walk_on_enter()
            return True
        elif self.state is State.INCOMPLETE_LOCAL_SEARCH:
            self.incomplete_local_search_on_exit()
            self.incomplete_local_search_to_local_search_on_enter()
            self.local_search_on_enter()
            return True
        return False

    def cancel(self) -> bool:
        """
        Transitions the simulation from Incomplete Local Search to Paused.
        Ignores Triggers

        -- Return --
        bool
            True if successful. False otherwise.
        """
        if self.state is State.INCOMPLETE_LOCAL_SEARCH:
            self.incomplete_local_search_on_exit()
            self.incomplete_local_search_to_paused_on_enter()
            self.paused_on_enter()
            return True
        return False

    def add_to_history(self, 
            point_normal : structs.Point = None, 
            point_concrete : structs.Point = None,
            is_bug : bool = None):
        """
        Adds a new row to the history dataframe

        --- Parameters ---
        point : structs.Point
            A point in space.
        is_bug : bool
            If a bug was observed at that point in space.
        """
        point_normal_data = None
        if not point_normal is None:
            point_normal_data = point_normal.array.tolist()

        point_concrete_data = None
        if not point_concrete is None:
            point_concrete_data = point_concrete.array.tolist()

        self._history = self._history.append({
            "step" : self.step,
            "is_bug" : is_bug,
            "state" : self.state.value,
            "point_normal" : point_normal_data,
            "point_concrete" : point_concrete_data,
        }, ignore_index = True)
        self._last_observed_point_normal = point_normal
        self._last_observed_point_concrete = point_concrete
        self._last_observed_point_is_bug = is_bug
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


    


