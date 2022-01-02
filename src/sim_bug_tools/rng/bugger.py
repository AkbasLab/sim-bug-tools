from sim_bug_tools.rng.lds.sequences import Sequence
from sim_bug_tools.structs import Domain, Point
from random import Random
import json
import numpy as np

class BugBuilder:
    def __init__(
        self, 
        location_domain: Domain, 
        size_domain: Domain, 
        sequence: Sequence, 
        random : Random = Random(0), 
    ):
        """
        Generates domains that represent a N-D volume.

        Args:
            location_domain (Domain): The domain in which the bug domain can 
             reside.
            size_domain (Domain): The domain in which the size of the bug can 
             be selected.
            sequence (Sequence): The 1D sequence that will be used to randomly* 
             generate the bugs.
            seed (Random, optional): The Random sequence to vary the sequence's 
             construction. Defaults to Random(0).
        """
        self._location_domain = location_domain
        self._size_domain = size_domain
        self._seq = sequence

        ## Ugly way to set up a seeded sequence :(
        # random.seed(seed)
        self._sequence_params = {
            "skip": int(random.random() * 1000),
            "offset": int(random.random() * 1000),
        }

    def _generate_number(self):
        return self._seq.get_sample(1, **self._sequence_params).points[0][0]

    def build_bug(self) -> Domain:
        # random.seed(self._seed)
        """
        Returns a domain that is within the given size, and at
        with the given location domain.


        """

        size_array = list()
        loc_array = list()

        for i, loc_limits in enumerate(self._location_domain):
            size_limits = self._size_domain[i]

            concrete_coord = (
                loc_limits[1] - loc_limits[0]
            ) * self._generate_number() + loc_limits[0]
            loc_array += [concrete_coord]

            concrete_size = (
                size_limits[1] - size_limits[0]
            ) * self._generate_number() + size_limits[0]
            if concrete_size + concrete_coord > loc_limits[1]:
                concrete_size = loc_limits[1] - concrete_coord

            size_array += [concrete_size]

        return Domain.from_dimensions(size_array, Point(loc_array))

    def build_bugs(self, n):
        for i in range(n):
            yield self.build_bug()

    @property
    def size_domain(self) -> Domain:
        return self._side_domain
    
    @size_domain.setter
    def size_domain(self, value : Domain):
        self._size_domain = value
        return

    def __str__(self):
        return "+BugBuilder" + \
            "\n+--Location Domain: " + str(self._location_domain[0]) + \
            "\n+--Size Domain: " + str(self._size_domain)


def profiles_from_json(_json: str) -> list[list[Domain]]:
    """
    Loads profiles from json
    """
    if _json[-5:] == ".json":
        with open(_json,"r") as f:
            data = json.loads(f.read())
    else:
        data = json.loads(_json)
    return [[Domain.from_json(domain) for domain in profile] \
        for profile in data]


# if __name__ == "__main__":
#     from sim_bug_tools.rng.lds.sequences import SobolSequence

#     loc = Domain([(0, 1), (0, 1), (0, 1)])
#     size = Domain([(0, 0.1), (0, 0.1), (0, 0.1)])
#     sobol = SobolSequence(loc, ["x", "y", "z"])

#     b = BugBuilder(loc, size, sobol, 4)

#     for i in range(10):
#         print(f"----- Bug #{i} -----")
#         bug = b.build_bug()
#         print("Bounds:", bug.bounding_points, "Dimensions:", bug.dimensions)
