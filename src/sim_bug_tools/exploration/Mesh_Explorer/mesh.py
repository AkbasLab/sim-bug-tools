import numpy as np 

from numpy import ndarray
from itertools import permutations
from rtree.index import Index, Property
from typing import Callable, NewType

from sim_bug_tools.exploration.boundary_core.explorer import Explorer
from sim_bug_tools.exploration.boundary_core.adherer import AdherenceFactory
from sim_bug_tools.structs import Domain, Point, Scaler

# A path is a boundary point and a direction to travel along.
# It indicates where to go next.
ENUM_CARDINAL = NewType('cardinal', int)
PATH = tuple[Point, ndarray, ENUM_CARDINAL] 

class MeshExplorer(Explorer):
    
    def __init__(self, b0: Point, n0: ndarray, adhererF: AdherenceFactory, scaler: Scaler, margin: float = -0.01):
        super().__init__(b0, n0, adhererF)
        self._scaler = scaler 
        self._margin = margin
        
        p = Property()
        p.set_dimension(self.ndims)
        self._index = Index(properties=p)
        
        # ENUM_CARDINAL = index of basis vector
        self._BASIS_VECTORS = tuple(np.identity(self.ndims))
        
        self._next_id = 0
        
        self._next_paths: list[PATH] = []
        self._add_child(b0, n0)
        
    def _add_child(self, bk: Point, nk: ndarray):
        self._next_paths.extend(self._get_next_paths_from(bk, nk))
        self._index.insert(self._next_id, bk)
    
    def _select_parent(self) -> tuple[Point, ndarray]:
        finding_gap = True
        while finding_gap:   
            # Find a point that doesn't overlap with others
            bk, self._v = self._next_paths.pop()
            b_new = bk + self._v 
            finding_gap = not self._check_overlap(b_new)
    
    def _pick_direction(self) -> ndarray:
        return self._v
    
    def _get_next_paths_from(self, bk: Point, nk: ndarray) -> list[PATH]:
        return [(bk, cardinal) for cardinal in self._find_cardinals(nk)]
    
    
    def _find_cardinals(self, n: ndarray) -> list[ndarray]:
        """
        Gets the cardinal vectors and their enum type from a given OSV. 

        Args:
            n (ndarray): _description_
            exclude (ENUM_CARDINAL, optional): _description_. Defaults to None.

        Returns:
            list[ndarray]: _description_
        """
        unit_vectors = list(self._BASIS_VECTORS)
        
        align_vector = unit_vectors[0]
        unit_vectors = unit_vectors[1:]
        
        angle = self.angle_between(align_vector, n)
        
        A = self.generateRotationMatrix(align_vector, n)(angle)
        
        cardinals = []
        
        for uv in unit_vectors:
            c = np.dot(A, uv)
            cardinals.extend((c, -c))
            
        return cardinals

    def _check_overlap(self, bk: Point) -> bool:
        "Returns True if there is an overlap"
        bnear_id = next(self._index.nearest(bk, 1))
        bnear = self._boundary[bnear_id][0]
        
        displacement: ndarray = (bk - bnear).array
        
        min_distance = np.linalg.norm(self._scaler.scale(displacement) * (1 + self._margin))
        distance = np.linalg.norm(displacement)       
        
        return distance < min_distance
        
    @classmethod
    def angle_between(cls, v1, v2) -> np.float64:
        """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
        """
        v1_u = cls.normalize(v1)
        v2_u = cls.normalize(v2)
        return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))    
        
    @staticmethod
    def normalize(u: ndarray):
        return u / np.linalg.norm(u)

    @classmethod
    def orthonormalize(cls, u: ndarray, v: ndarray) -> tuple[ndarray, ndarray]:
        """
        Generates orthonormal vectors given two vectors @u, @v which form a span.

        -- Parameters --
        u, v : np.ndarray
            Two n-d vectors of the same length
        -- Return --
        (un, vn)
            Orthonormal vectors for the span defined by @u, @v
        """
        u = u.squeeze()
        v = v.squeeze()

        assert len(u) == len(v)

        u = u[np.newaxis]
        v = v[np.newaxis]

        un = cls.normalize(u)
        vn = v - np.dot(un, v.T) * un
        vn = cls.normalize(vn)

        if not (np.dot(un, vn.T) < 1e-4):
            raise Exception("Vectors %s and %s are already orthogonal." % (un, vn))

        return un, vn

    @classmethod
    def generateRotationMatrix(cls, u: ndarray, v: ndarray) -> Callable[[float], ndarray]:
        """
        Creates a function that can construct a matrix that rotates by a given angle.

        Args:
            u, v : ndarray
                The two vectors that represent the span to rotate across.

        Raises:
            Exception: fails if @u and @v aren't vectors or if they have differing
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

        u, v = cls.orthonormalize(u, v)

        I = np.identity(len(u.T))

        coef_a = v * u.T - u * v.T
        coef_b = u * u.T + v * v.T

        return lambda theta: I + np.sin(theta) * coef_a + (np.cos(theta) - 1) * coef_b

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from sim_bug_tools.graphics import Grapher
    from sim_bug_tools.structs import Spheroid
    from sim_bug_tools.exploration.brrt_std.adherer import ConstantAdherenceFactory
    ndims = 3
    domain = Domain.normalized(ndims)
    
    g = Grapher(ndims == 3, domain)
    
    d = 0.05
    scaler = Spheroid(d)
    delta_theta = np.pi * 5 / 180
    
    radius = 0.4
    loc = Point([0.5 for i in range(ndims)])
    classifier = lambda p: loc.distance_to(p) <= radius 
    
    b0 = [0.5 for i in range(ndims - 1)] + [radius]
    n0 = np.array([0 for i in range(ndims - 1)] + [-1])
    
    adhf = ConstantAdherenceFactory(classifier, scaler, delta_theta, domain, True)
    
    exp = MeshExplorer(b0, n0, adhf, scaler)
    
    i = 0
    points = []
    while True:
        while len(exp.boundary) == i:
            points.append(exp.step())
        
        g.plot_point(exp.boundary[-1])
        
        plt.pause(0.01)
        input("enter")
        