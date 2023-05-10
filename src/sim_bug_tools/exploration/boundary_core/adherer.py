from abc import ABC
from abc import abstractmethod as abstract
from typing import Callable, Generic, TypeVar, Type

import numpy as np
from numpy import ndarray

from sim_bug_tools.structs import Domain, Point

T_BNODE = tuple[Point, ndarray]


class SampleOutOfBoundsException(Exception):
    "When a boundary Adherer samples out of bounds, this exception may be thrown"

    def __init__(self, msg="Sample was out of bounds!"):
        self.msg = msg
        super().__init__(msg)

    def __str__(self):
        return f"<BoundaryLostException: {self.msg}>"


class BoundaryLostException(Exception):
    "When a boundary Adherer fails to find the boundary, this exception is thrown"

    def __init__(self, msg="Failed to locate boundary!"):
        self.msg = msg
        super().__init__(msg)

    def __str__(self):
        return f"<BoundaryLostException: {self.msg}>"


class _AdhererIterator:
    def __init__(self, ba: "Adherer"):
        # global _v, _g, _p, _s, _exp # DEBUG
        self.ba = ba

    def __next__(self):
        # global _s, _g, _exp
        if self.ba.has_next():
            return self.ba.sample_next()
        else:
            raise StopIteration


class Adherer(ABC):
    """
    An Adherer provides the ability to identify a point that lies on the
    boundary of an N-D volume (i.e. target envelope). Furthermore, it allows
    for the incremental stepping through the process and the collection
    of intermediate samples. A "classifier" function describes whether
    or not a sampled Point is within or outside of the target envelope.

    Object Variables:
        self._classifier (Callable[[Point], bool]): Classifying function that
            determines if a set of parameters results in target performance.
        self._pivot (Point): The boundary point to pivot from to find a
            neighboring boundary point.
        self._n (ndarray): The OSV of the boundary point self.pivot

    """

    def __init__(
        self,
        classifier: Callable[[Point], bool],
        bnode: T_BNODE,
        direction: ndarray,
        domain: Domain = None,
        fail_out_of_bounds=True,
    ):
        """
        Adherence Strategy interface. Describes the methods necessary
        to implement a surface adherer, sampling in some direction
        from a known boundary node to find a neighboring boundary node.

        WARNING: __init__ is for setup only, do NOT sample yet. All sampling
        should be done by "sample_next" exclusively to maintain interface
        contracts.

        Args:
            classifier (Callable[[Point], bool]): Classifying function that
                determines if a set of parameters results in target performance.
            bnode (T_BNODE): The boundary point and osv to pivot from for finding
                a neighboring boundary node.
            domain (Domain): The domain to constrain sampling to (Defaults to None)
        """
        assert (
            len(bnode[0]) == len(bnode[1])
        ), "Node's location and OSV have mismatching number of dimensions?"
        assert len(bnode[0]) == len(domain)

        self._classifier = classifier
        self._domain = domain

        self._pivot, self._n = bnode
        self._direction = direction

        self._fail_out_of_bounds = fail_out_of_bounds

        self._new_b = None
        self._new_n = None

    @property
    def classifier(self):
        return self._classifier

    @property
    def domain(self):
        return self._domain

    @property
    def direction(self):
        return self._direction

    @property
    def parent_bnode(self):
        return self._pivot, self._n

    @abstract
    def sample_next(self) -> tuple[Point, bool]:
        """
        Takes the next sample to find the boundary. When the boundary is found,
        (property) b will be set to that point and sample_next will no longer
        return anything.

        Raises:
            BoundaryLostException: This exception is raised if the adherer
                fails to acquire the boundary.
            SampleOutOfBoundsException: Iff fail_out_of_bounds is true, this
                exception is raised if the adherer attempts to sample outside

        Returns:
            Point, bool: The next sample and target class
            None: If the boundary was acquired or lost
        """
        pass

    def has_next(self) -> bool:
        """
        Returns:
            bool: True if the boundary has not been found and has remaining
                samples.
        """
        return self.bnode is None

    @property
    def bnode(self) -> T_BNODE:
        """
        Boundary point and its orthonormal surface vector, returns None
        if the boundary hasn't been found yet.
        """
        return (self._new_b, self._new_n) if self._new_b is not None else None

    def _classify(self, p: Point):
        in_domain = self._domain is None or p in self._domain
        if self._fail_out_of_bounds and not in_domain:
            raise SampleOutOfBoundsException()
        return in_domain and self._classifier(p)

    def __iter__(self):
        return _AdhererIterator(self)


A = TypeVar("A", bound=Adherer)


class AdherenceFactory(Generic[A]):
    """
    Different adherence strategies can require different initial parameters.
    Since the Explorer does not know what these parameters are, we must decouple
    the construction of the Adherer from the explorer, allowing for initial
    parameters to be defined prior to the execution of the exploration alg.
    """

    def __init__(
        self,
        classifier: Callable[[Point], bool],
        domain: Domain = None,
        fail_out_of_bounds: bool = True,
    ):
        """
        An abstract base clase for Adherer factories. This factory
        constructs an Adherer of type A from class using a relative boundary
        point and OSV to find a neighboring boundary node.

        Args:
            classifier (Callable[[Point], bool]): The target performance
                classifying function.
            domain (Domain): The domain to constrain the adherence to
            fail_out_of_bounds (bool, optional): If set to True, the resulting
                Adherer will fail if it samples out of bounds. Defaults to True.
        """
        self._classifier = classifier
        self._domain = domain
        self._fail_out_of_bounds = fail_out_of_bounds

    @property
    def classifier(self):
        return self._classifier

    @property
    def domain(self):
        return self._domain

    @abstract
    def adhere_from(self, b: Point, n: ndarray, direction: ndarray) -> A:
        """
        Find a boundary point that neighbors the given point, p, in the
        provided direction, given the provided surface vector, n.

        Args:
            p (Point): A boundary point to use as a pivot point
            n (ndarray): The orthonormal surface vector for boundary point p
            direction (ndarray): The direction to sample towards

        Returns:
            Adherer: The adherer object that will find the boundary given the
                above parameters.
        """
        pass
