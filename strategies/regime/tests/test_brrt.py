import unittest
import os, sys
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(FILE_DIR))

import sim_bug_tools.utils as utils
import sim_bug_tools.structs as structs
import brrt

import rtree

class TestRegime(unittest.TestCase):
    def test_metrics(self):
        print("\n\n")

        rrt : brrt.BoundaryRRT = utils.load("%s/data/brrt.pkl" % FILE_DIR)

        # Initialize an index
        # root = rrt.tree.get_node(0).data["location"]
        # p = rtree.index.Property()
        # p.set_dimension(len(root))
        # index = rtree.index.Index(properties=p)
        
        [rrt.index.insert(node.identifier, node.data["location"]) \
            for node in rrt.tree.all_nodes()]
        print(len(rrt.index))
        

        # for node in rrt.tree.all_nodes():
        #     p : structs.Point = node.data["location"]
        #     x = next(rrt.index.nearest(p.array, 2))
            
        #     break
        

        print("\n\n")
        return