from copy import copy
from typing import Callable

import numpy as np
from numpy import ndarray
from sim_bug_tools.structs import Point

DATA_LOCATION = "location"
DATA_NORMAL = "normal"

ANGLE_90 = np.pi / 2

class BoundaryLostException(Exception):
    def __init__(self, msg="Failed to locate boundary!"):
        self.msg = msg
        super().__init__(msg)

    def __str__(self):
        return (
            f"<BoundaryLostException: Angle: {self.theta}, Jump Distance: {self.d}>"
        )


class _AdhererIterator:
    def __init__(self, ba: "BoundaryAdherer"):
        self.ba = ba
        
    def __next__(self):
        if self.ba.has_next:
            return self.ba.sample_next()        
        else:
            raise StopIteration

class BoundaryAdherer:
    def __init__(self, classifier: Callable[[Point], bool], p: Point, n: ndarray, direction: ndarray, d: float, theta: float):
        self._classifier = classifier
        self._p = p
        
        n = BoundaryAdherer.normalize(n)
            
        self._rotater_function = self.generateRotationMatrix(n, direction)
        
        self._s: ndarray = copy(n) * d
        self._s = np.dot(self._rotater_function(-ANGLE_90), self._s)
                
        self._prev: Point = None
        self._prev_class = None
        
        self._cur: Point = p + Point(self._s)
        self._cur_class = classifier(self._cur)
        
        if self._cur_class:
            self._rotate = self._rotater_function(theta)
        else:
            self._rotate = self._rotater_function(-theta)
            
        self._b = None
        
        self._iteration = 0
        self._max_iteration = (np.pi / 2) // theta
    
    @property
    def b(self) -> Point | None:
        return self._b
    
    def has_next(self) -> bool:
        """
        Returns:
            bool: True if the boundary has not been found and has remaining
                samples.
        """
        self.sample_next = lambda: None
        return self._b is None
    
    def sample_next(self) -> Point | None:
        """
        Takes the next sample to find the boundary. When the boundary is found,
        (property) b will be set to that point and sample_next will no longer
        return anything.
        
        Raises:
            BoundaryLostException: This exception is raised if the adherer 
                fails to acquire the boundary.

        Returns:
            Point: The next sample 
            None: If the boundary was acquired or lost
        """
        self._prev = self._cur
        self._s = np.dot(self._rotate, self._s)
        self._cur = self._p + Point(self._s)
        
        self._prev_class = self._cur_class
        self._cur_class = self._classifier(self._cur)
        
        if self._cur_class != self._prev_class:
            self._b = self._cur if self._cur_class else self._prev
            self.sample_next = lambda: None
            
        elif self._iteration > self._max_iteration:
            raise BoundaryLostException()
            
        return self._cur
    
    def find_boundary(self) -> Point:
        """
        Samples until the boundary point is found.
        
        Returns:
            Point: The estimated boundary point
        """
        while self.has_next():
            self.sample_next()
        
        return self._b
    
    def __iter__(self):
        return _AdhererIterator(self)
    
    @staticmethod
    def normalize(u: ndarray):
        return u / np.linalg.norm(u)
    
    @staticmethod
    def orthonormalize(u: ndarray, v: ndarray) -> tuple[ndarray, ndarray]:
        """
        Generates orthonormal vectors given two vectors @u, @v which form a span.

        -- Parameters --
        u, v : np.ndarray
            Two n-d vectors of the same length
        -- Return --
        (un, vn)
            Orthonormal vectors for the span defined by @u, @v
        """
        assert len(u) == len(v)
        
        un = BoundaryAdherer.normalize(u)        
        vn = v - np.dot(un, v.T) * un
        vn = BoundaryAdherer.normalize(vn)
        
        if not (np.dot(un, vn.T) < 1e-4):
            raise Exception("Vectors %s and %s are already orthogonal." % (un, vn))
        
        return un, vn
    
    @staticmethod
    def generateRotationMatrix(u: ndarray, v: ndarray) -> Callable[[float], ndarray]:
        """
        Creates a function that can construct a matrix that rotates by a given angle.

        Args:
            u, v : ndarray
                The two vectors that represent the span to rotate across.

        Raises:
            Exception: fails if u and v aren't vectors or if they have differing
                number of dimensions.

        Returns:
            Callable[[float], ndarray]: A function that returns a rotation matrix
                that rotates that number of degrees using the provided span.
        """
        u = u.squeeze()
        v = v.squeeze()
        
        if u.shape != v.shape:
            raise Exception("Dimension mismatch...")
        elif len(u.shape) != 1:
            raise Exception("Arguments u and v must be vectors...")
            
        u = u[np.newaxis]
        v = v[np.newaxis]
        
        u, v = BoundaryAdherer.orthonormalize(u, v)
        
        I = np.identity(len(u))
        
        coef_a = v * u.T - u * v.T 
        coef_b = u * u.T + v * v.T
        
        return lambda theta: I * np.sin(theta) * coef_a + (np.cos(theta) - 1) * coef_b
        