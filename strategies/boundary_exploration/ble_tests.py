import os
import sys
from itertools import product

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from copy import copy
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray
from sim_bug_tools.structs import Point
from tools.grapher import Grapher
from treelib import Node, Tree

from adherer import DATA_LOCATION, DATA_NORMAL, BoundaryAdherer
from brrt import BoundaryRRT
from brrt_2 import BoundaryRRT as BoundaryRRT2

ndims = 3

g = Grapher(True)

num_iterations = 100

theta_min = 1 * np.pi / 180 
theta_max = 10 * np.pi / 180 
theta_count = 10
thetas = [(theta_max - theta_min)/theta_count * i + theta_min for i in range(theta_count)]

delta_theta_min = 45 * np.pi / 180 
delta_theta_max = 90 * np.pi / 180 
delta_theta_count = 5
delta_thetas = [(delta_theta_max - delta_theta_min)/delta_theta_count * i + delta_theta_min for i in range(delta_theta_count)]

rate_min = 1.5 
rate_max = 5
rate_count = 7
rates = [(rate_max - rate_min)/rate_count * i + rate_min for i in range(rate_count)]

num_min = 1
num_max = 6
num_count = 5
nums = [(num_max - num_min)//num_count * i + num_min for i in range(num_count)]

d_min = 0.01
d_max = 0.05
d_count = 5
ds = [(d_max - d_min)/d_count * i + d_min for i in range(d_count)]

pairs = product(ds, thetas)
pairs2 = product(ds, delta_thetas, rates, nums)

sphere_loc = Point([0.5 for x in range(ndims)])
sphere_rad = 0.25
g.create_sphere(sphere_loc, sphere_rad)

classifier = lambda p: sphere_loc.distance_to(p) <= sphere_rad

pairs = list(pairs)
pairs2 = list(pairs2)
print(pairs2[658])
input()

def test_error_rate():
    results = []
    for i, pair in enumerate(pairs[:70]):
        d, theta = pair
        print("i, d, theta =", i, d, theta)
        brrt = BoundaryRRT(classifier, sphere_loc, d, theta)
        points = []
        count = 0
        info = {"errs": 0, "avg": 0}
        
        while count < num_iterations:
            try:
                data = brrt.grow()
                points += [data[0]]
                count += 1
                
            except:
                info["errs"] += 1
                
        info["avg"] = info["errs"] / (info["errs"] + count)
        results.append(info)
                
    # print("results...", list(pairs), results)
    # print("zip:", list(zip(list(pairs), results)))
    for pair, r in zip(list(pairs), results):
        print (pair, "\t", r)

def test_error_rate_2():
    results = []
    for i, pair in enumerate(pairs2):
        d, theta, rate, num = pair
        print("i, d, theta, rate, num =", i, d, theta, rate, num)
        brrt = BoundaryRRT2(classifier, sphere_loc, d, theta, rate, num)
        points = []
        count = 0
        info = {"errs": 0, "avg": 0}
        
        while count < num_iterations:
            try:
                data = brrt.grow()
                points += [data[0]]
                count += 1
                
            except:
                info["errs"] += 1
                
        info["avg"] = info["errs"] / (info["errs"] + count)
        results.append(info)
                
    # print("results...", list(pairs), results)
    # print("zip:", list(zip(list(pairs), results)))
    for pair, r in zip(list(pairs2), results):
        print (pair, "\t", r)
        
        
        

def test_visual(d, theta):

    print(f"d, theta = {d}, {theta * 180 / np.pi}")
    
    brrt = BoundaryRRT(classifier, sphere_loc, d, theta)
    count = 0
    err_count = 0
    points = []
    
    while count < num_iterations:
        try: 
            data = brrt.grow()
            points += [data[0]]
            count += 1
        except:
            if len(points) > 0:
                g.plot_all_points(points)
                points = []
            print("Boundary lost, showing how the vector moves...")
            node: Node = brrt.previous_node
            direction = brrt.previous_direction
            pk, nk = node.data[DATA_LOCATION], node.data[DATA_NORMAL]
            p_graphic = g.plot_point(pk)
            n_graphic = g.add_arrow(pk, nk, color="green")
            direction_graphic = g.add_arrow(pk, direction, color="yellow")
            print("n =", nk)
            print("direction =", direction)
            
            print(f"Distance from surface = {sphere_rad - pk.distance_to(sphere_loc)}")
                        
            adh = BoundaryAdherer(classifier, pk, nk, direction, d, theta)
            print(f"Previous was in envelope: {classifier(pk + Point(adh._s))}")
            s = None
            done = False
            while not done and adh.has_next():
                if s is not None:
                    s.remove()

                try:
                    print(adh.sample_next())    
                except:
                    print("Done")
                    done = True
                    
                s = g.add_arrow(pk, adh._s*100, color="b")
                plt.pause(0.001)
                print("s =", adh._s)
                print("A =", adh._rotate)
                code = input("press enter...")
                
                if code == "t":
                    _s = adh._s 
                    _A = adh._rotate 
                    for x in range(10):
                        print("s:", _s := np.dot(_A, _s))
                    print("s:", _s.astype(str))
                    print("A:", _A.astype(str))
            
            err_count += 1
            p_graphic.remove()
            n_graphic.remove()
            s.remove()

if __name__ == "__main__":
    test_error_rate_2()
    
    # test_visual(0.05, 1.4311699866353502)
    
