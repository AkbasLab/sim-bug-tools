# Just playing around with polar exploration strategy concept
import numpy as np

from numpy import ndarray
from rtree.index import Index, Property

from sim_bug_tools.structs import Point, Domain



class TopDownPolarExplorer:
    
    
    def __init__(self, t0: Point, d: float, min_dist: float):
        self._t0 = t0 
        self._d = d
        self._min_dist = min_dist
        
        self._points: list[Point] = []
        
        p = Property()
        p.set_dimension(len(t0))
        self._index = Index(properties=p)
        
        self._BASIS_VECTORS = list(np.identity(len(t0)))
    
        self._next_paths = self._initial_rose(t0)
        
    def _initial_rose(self, t0: Point):
        paths = []
        for bv in self._BASIS_VECTORS:
            paths.extend([(t0, bv), (t0, -bv)])
        return paths
        
    def step(self):
        finding_gap = True
        
        while finding_gap:
            p, v = self._next_paths.pop()
            
            cur = p + Point(v)
            
            if not self.has_overlap(cur):
                cards = self.find_cardinals(cur)
                # enqueue new paths
                paths = [(cur, card) for card in cards]
                self._next_paths = paths + self._next_paths
                finding_gap = False
        
        return cur 
    
    def find_cardinals(self, p: Point):
        # unit_vectors = list(self._BASIS_VECTORS)
        # alignment_vector = unit_vectors[0]
        # unit_vectors = unit_vectors[1:]
        
        s = self.normalize((p - self._t0).array)
        v = s.copy()
        v[0] *= 0.25
        v = self.normalize(v)
        if np.dot(v, s.T) > 1e-03:
            s, v = self.orthonormalize(s, v)
        
        return list(map(lambda x: x * self._d, [s, -s, v, -v]))
    
    @staticmethod
    def normalize(v: ndarray) -> ndarray:
        return v / np.linalg.norm(v)
        
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
            
        
    def has_overlap(self, p: Point) -> Point:
        p_near_id = next(self._index.nearest(p, 1), None)
        p_near = self._points[p_near_id] if p_near_id is not None else None
        
        return p_near is not None and p.distance_to(p_near) < self._min_dist
        
    
    
    



def test_orthonormalize():
    u = np.array([0, 1])
    v = np.array([1, 1])
    print(u, v)
    
    u, v = TopDownPolarExplorer.orthonormalize(u, v)
    
    print(u)
    print(v)
    
def test_polar_exp():
    import matplotlib.pyplot as plt 
    from sim_bug_tools.graphics import Grapher
    ndims = 2
    domain = Domain.normalized(ndims)
    g = Grapher(ndims == 3, domain)
    
    d = 0.05
    t0 = Point(0.5, 0.5)
    exp = TopDownPolarExplorer(t0, d, d * 0.9)
    
    chunk = 1
    
    g.plot_point(t0, color="red")
    
    while True:
        points = []
        for i in range(chunk):
            points.append(exp.step())
        
        g.plot_all_points(points, color='blue')
        
        plt.pause(0.01)
        continue
    
    
    
    
if __name__ == '__main__':
    test_polar_exp()