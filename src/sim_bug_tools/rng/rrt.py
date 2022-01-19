from selectors import EpollSelector
import sim_bug_tools.structs as structs
import sim_bug_tools.rng.lds.sequences as sequences
import numpy as np
import json

class RapidlyExploringRandomTree:
    def __init__(self, 
                seq : sequences.Sequence,
                step_size : np.float64,
                exploration_radius : np.float64):
        """
        Rapidly exploring random tree implementation
        
        -- Parameters --
        root : Point
            Starting point.
        seq : Sequence
            Sequence for exploration space sampling.
        step_size : np.float64
            Amount of exploration performed during each step.
        exploration_radius : np.float64
            The size of the RRT is this square radius value measured from the root. 
        """
        self._seq = seq
        self._step_size = np.float64(step_size)
        self._exploration_radius = np.float64(exploration_radius)
        self._root = structs.Point([])
        self._contents = []
        return

    @property
    def root(self) -> structs.Point:
        return self._root
    
    @property
    def seq(self) -> sequences.Sequence:
        return self._seq

    @property
    def step_size(self) -> np.float64:
        return self._step_size

    @property
    def contents(self) -> list[structs.Point]:
        return self._contents

    @property
    def size(self) -> int:
        return len(self.contents)

    @property
    def exploration_radius(self) -> np.float64:
        return self._exploration_radius

    def as_dict(self) -> dict:
        return {
            "seq" : self.seq.as_dict(),
            "step_size" : self.step_size,
            "exploration_radius" : self.exploration_radius,
            "root" : self.root.to_list(),
            "contents" : [point.to_list() for point in self.contents]
        }
    
    def as_json(self) -> str:
        return json.dumps(self.as_dict())

    @staticmethod
    def from_dict(d : dict):
        rrt = RapidlyExploringRandomTree(
            seq = sequences.from_dict(d["seq"]),
            step_size = d["step_size"],
            exploration_radius = d["exploration_radius"] 
        )
        rrt._root = structs.Point(d["root"])
        rrt._contents = [structs.Point(arr) for arr in d["contents"]]
        return rrt

    def copy(self):
        rrt = RapidlyExploringRandomTree(
            seq = self.seq,
            step_size = self.step_size,
            exploration_radius = self.exploration_radius
        )
        rrt._root = self.root
        rrt._contents = self.contents.copy()
        return rrt

        


    def step(self):
        """
        Performs a single exploration step
        """
        # Get a random point in the space
        random_point = self.seq.get_points(1)[0]

        # Find the nearest point to this spot
        nearest_point_in_contents = self.nearest_point(random_point)

        # Get the point one step size away from the nearest point along the line of the random point
        projected_point = nearest_point_in_contents.project_towards_point(
            point=random_point, x=self.step_size
        )

        self._contents.append(projected_point)
        return random_point, nearest_point_in_contents, projected_point

    def nearest_point(self, p):
        """
        Find the nearest point in contents to point p.

        -- Parameters --
        p : Point
            Point being compared

        -- Return --
        Point
            Nearest point to p within contents
        """
        distances = [
            point_in_contents.distance_to(p) for point_in_contents in self.contents
        ]
        return self.contents[distances.index(min(distances))]
    
    def reset(self, root : structs.Point):
        """
        Resets the RRT centered on a new root.
        """
        self._root = root
        self._contents = [self._root]
        self._seq.domain = structs.Domain.from_bounding_points(
            pointA = self._root.array - self.exploration_radius,
            pointB = self._root.array + self.exploration_radius
        )
        return