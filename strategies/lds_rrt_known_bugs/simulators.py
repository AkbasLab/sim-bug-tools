from abc import abstractmethod
import sim_bug_tools.simulator as simulator
import sim_bug_tools.rng.lds.sequences as sequences
import sim_bug_tools.structs as structs
import sim_bug_tools.utils as utils
from sim_bug_tools.rng.rrt import RapidlyExploringRandomTree

class SimpleSimulatorKnownBugs(simulator.Simulator):
    """
    A simple simulator with known bugs.
    Samples from a normal domain.
    """
    def __init__(self,
        bug_profile : list[structs.Domain],
        sequence : sequences.Sequence,
        **kwargs
    ):
        n_dim = len(sequence.domain)

        
        kwargs["domain"] = structs.Domain([(0,1) for _ in range(n_dim)])
        super().__init__(
            **kwargs
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
        return simulator.Simulator.from_dict(d, sim)


    def long_walk_on_update(self):
        """
        The sequence generates a point which is compared to the bug profile.
        """
        # Sample from the sequence
        point = self.sequence.get_points(1)[0]

        # Check if it's in the bug profile
        is_bug = self.is_point_in_bug_profile(point)

        # Call the parent function
        return super().long_walk_on_update(point_normal = point, is_bug = is_bug)         
        


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
        n_branches : int,
        **kwargs
    ):
        super().__init__(bug_profile, sequence, **kwargs)

        self._rrt = rrt
        self._n_branches = n_branches
        self._n_branches_remaining = n_branches
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
    

    def long_walk_to_local_search_on_enter(self):
        """
        The RRT is reset and centered on the last observed point.
        """
        # Reset the RRT
        self.rrt.reset(self.last_observed_point_normal)
        self._n_branches_remaining = self.n_branches
        return

    def local_search_on_update(self):
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
        return super().local_search_on_update(point_normal = point, is_bug = is_bug)         
        
    
    def local_search_to_long_walk_trigger(self) -> bool:
        """
        The local search ends when the RRT does not need to grow anymore
        branches.
        """
        return self.n_branches_remaining <= 0

    def local_search_to_paused_trigger(self) -> bool:
        """
        When the sim is complete (implied in base class), and there are no
        branches remaining. Transition to Paused State.
        """
        return self.n_branches_remaining <= 0

    def long_walk_to_local_search_trigger(self) -> bool:
        """
        Move to local search when a bug is observed
        """
        return self.last_observed_point_is_bug
    
