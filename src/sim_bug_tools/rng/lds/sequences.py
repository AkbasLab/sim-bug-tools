
import numpy as np
from numpy import int32
from abc import abstractmethod
import logging
import openturns as ot

from sim_bug_tools.rng.lds.coverage import Sample
from sim_bug_tools.structs import Point, Domain


class Sequence:
    """
    Used to provide a uniform interface for providing a sequence of points.
    Classes that implement SequenceAdapter will be compatible with our
    sequence operations and tools.
    """

    @abstractmethod
    def __init__(self, domain: Domain, axes_names: tuple[str], seed : int = -1):
        "Initializes the sequence with the number of dimensions of the points."
        self._domain = domain
        self._axes_names = axes_names
        self._seed = np.int32(seed)
        self._offset = 0
        # self._granularity = np.float63(0.01)
        self.__post_init__()

    @abstractmethod
    def __post_init__(self):
        pass

    @abstractmethod
    def get_points(self, n: int32) -> list[Point]:
        "Return n-number of points from the sequence."
        raise NotImplementedError(
            f"{__class__} Sequence has no implementation of get_points method!"
        )
        return None

    def get_sample(
        self, num_points: int32, skip: int = 0, batch: int = 1, offset: int = 0
    ) -> Sample:
        """
        Generates points using it while following a pattern defined by the parameters.

        Args:
            sequence ():            OT Sample object, or function that
                                    has a functional interface of
                                    point foo(int)
                                    point foo()

            skip (int):             How many points to skip over before yielding a point.
            num_points (int):       The number of points to yield.
            batch (int, optional):  How many points to yield at a time. Defaults to 1.
            offset (int, optional): Which iteration

        Yields:
            list: The list of points according to batch size
        """

        if offset:
            self.get_points(offset)

        points = []
        count = 0
        i = 0

        while count < num_points:
            if i % skip == 0:
                points += self.get_points(batch)
                count += 1

            else:
                self.get_points(1)

            i += 1

        return Sample(points, self._axes_names, self._domain)

    @property
    def domain(self) -> Domain:
        return self._domain
    @domain.setter
    def domain(self, d : Domain):
        self._domain = d
        return

    @property
    def axes_names(self) -> tuple[str]:
        return self._axes_names

    @property
    def seed(self) -> np.int32:
        return self._seed

    @seed.setter
    def seed(self, n: np.int32):
        self._seed = np.int32(n)
        np.random.seed(self._seed)
        return

    @property
    def offset(self) -> int:
        return self._offset

    @abstractmethod
    def as_dict(self) -> dict:
        return {
            "class" : self.__class__.__name__,
            "domain" : self.domain.as_dict(),
            "axes_names" : self.axes_names,
            "seed" : self.seed,
            "offset" : self.offset
        }


## Currently Supported Sequences ##
class HaltonSequence(Sequence):
    def __post_init__(self):
        self._ot_sequence = ot.HaltonSequence(len(self._domain))

    def get_points(self, n: int32) -> list[Point]:
        points = list(
            map(
                lambda point: Point(np.array(point)), self._ot_sequence.generate(int(n))
            )
        )
        self._offset += n
        return points


class SobolSequence(Sequence):
    def __post_init__(self):
        self._ot_sequence = ot.SobolSequence(len(self._domain))

    def get_points(self, n: int32) -> list[Point]:
        points = list(
            map(
                lambda point: Point(np.array(point)), self._ot_sequence.generate(int(n))
            )
        )
        self._offset += n
        return points


class FaureSequence(Sequence):
    def __post_init__(self):
        self._ot_sequence = ot.FaureSequence(len(self._domain))

    def get_points(self, n: int32) -> list[Point]:
        points = list(
            map(
                lambda point: Point(np.array(point)), self._ot_sequence.generate(int(n))
            )
        )
        self._offset += n
        return points


class RandomSequence(Sequence):
    def __post_init__(self):
        return

    def get_points(self, n: np.int32) -> list[Point]:
        points = [Point(point) for point in np.random.rand(n, len(self._domain))]
        self._offset += n
        return points


class LatticeSequence(Sequence):
    def __post_init__(self):
        self._reset()
        return
    
    def _reset(self):
        x = self._domain.granularity

        self._lower_point = np.array([arr[0] for arr in self.domain.array])
        self._upper_point = np.array([arr[1] for arr in self.domain.array])
        self._next_point = np.array([np.nan for i in range(len(self._domain.array[0]))])

        n_buckets = np.ceil((self._upper_point - self._lower_point)/self._domain.granularity)
        n_dim = len(n_buckets)
        self._n_max_points = (int(n_buckets[0]) + 1)**n_dim 

        self._current_point_index = 0
        return

    def _get_next_point(self) -> Point:
        self._current_point_index += 1
        # if self._current_point_index > self._n_max_points:
        #     self._reset()

        if np.all(np.isnan(self._next_point)):
            self._next_point = self._lower_point
            return Point(self._next_point)

        
        increment = np.array([0. for i in range(len(self._domain.array))])
        increment[0] = self._domain.granularity


        # print(self._next_point, increment, self.domain.array)

        self._next_point = self._next_point + increment
        

        size = len(self._next_point)
        for i in range(size):
            if self._next_point[i] > self._upper_point[i] + self._domain.granularity:
                self._next_point[i] = self._lower_point[i]
                next_index = ((i+1) % size)
                self._next_point[next_index] += self._domain.granularity

        # print(self._next_point)
        # print(x)
        return Point(self._next_point)

    def get_points(self, n: np.int32) -> list[Point]:
        points = [self._get_next_point() for i in range(n)]
        self._offset += n
        return points




def from_dict(d : dict) -> Sequence:
    """
    Reconstructs a sequence from a dictionary

    --- Parameters ---
    d : dict
        Dictionary which defines a sequence

    --- Return ---
    Sequence
        A sequence generated from the given dictionary.
    """
    sequence_constructors = {
        "HaltonSequence" : HaltonSequence,
        "SobolSequence" : SobolSequence,
        "FaureSequence" : FaureSequence,
        "RandomSequence" : RandomSequence,
        "LatticeSequence" : LatticeSequence
    }
    seq : Sequence = sequence_constructors[d["class"]](
        domain = Domain.from_dict(d["domain"]),
        axes_names = d["axes_names"],
        seed = d["seed"]
    )
    seq.get_points(d["offset"])
    return seq


def main():
    import matplotlib.pyplot as plt

    domain = Domain.normalized(3)
    names = "x y z".split()
    seq = HaltonSequence(domain, names)
    points = seq.get_points(10)

    sample = Sample(points, names, domain)
    axes = sample.transpose

    plt.scatter(axes[0], axes[1])
    plt.show()


if __name__ == "__main__":
    main()
