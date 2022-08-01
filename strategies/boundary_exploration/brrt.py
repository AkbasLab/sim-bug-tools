from copy import copy
from typing import Callable

import numpy as np
from numpy import ndarray
from sim_bug_tools.rng.lds.sequences import RandomSequence, Sequence
from sim_bug_tools.structs import Domain, Point
from treelib import Node, Tree

from adherer import BoundaryAdherer

DATA_LOCATION = "location"
DATA_NORMAL = "normal"

class BoundaryRRT:
    def __init__(self, classifier: Callable[[Point], bool], t0: Point, d: float, theta: float):
        self._classifier = classifier
        self._d = d
        self._theta = theta
        self._ndims = len(t0)
        
        self._tree = Tree()
        
        p0, n0 = self._surface(t0)
        self._root = Node(identifier=0, data=self._create_data(p0, n0))
        self._next_id = 1
        
        self._tree.add_node(self._root)
        
        
    def grow(self, num: int = 1):
        
        for i in range(num):
            r = self._random_point()
            p, n = self._find_nearest(r)
            ba = BoundaryAdherer(self._classifier, p, n, r, self._d, self._theta)
            ba.find_boundary()
            
            pk, nk = ba.bn 
            
            self._add_node(pk, nk)
            
            
    def _add_node(self, p: Point, n: ndarray):
        node = Node(identifier=self._next_id, data=self._create_data(p, n))
        self._next_id += 1
    
    def _random_point(self) -> Point:
        return Point(np.random.rand(self._ndims))
    
    def _find_nearest(self, p: Point) -> tuple[Point, ndarray]:
        closest = {
            "id": 0,
            "distance": self._tree.nodes[0].data[DATA_LOCATION].distance_to(p),
        }
        
        for id, node in self._tree.nodes.items():
            dist = node.data[DATA_LOCATION].distance_to(p)
            if dist < closest["distance"]:
                closest["id"] = id
                closest["distance"] = dist
        
        node = self._tree.get_node(closest["id"])
        pk = node.data[DATA_LOCATION] 
        nk = node.data[DATA_NORMAL] 
        
        return pk, nk
        
    
    @staticmethod
    def _create_data(location, normal) -> dict:
        return {DATA_LOCATION: location, DATA_NORMAL: normal}
        
    
    @staticmethod
    def _surface(t0):
        v = np.random.rand(len(t0))
        
        s = v * self._d
        
        prev = None 
        cur = t0
        
        while self._classifier(cur):
            prev = cur 
            cur = prev + Point(s)
            
        return prev, v
        
