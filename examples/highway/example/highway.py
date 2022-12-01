"""
This python script serves as an example for using SUMO simulator class for the
"highway" scenario. The following files are required:
- highway.net.xml
    The SUMO network.
- params.csv
    A csv file with the parameter ranges for the highway scenario.
- simulator.py
    Contains the parameter manager and SUMO simulator classes
"""
# Adds the parent directory "highway" to the system path.
import os
import sys
from datetime import datetime

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(FILE_DIR))

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import simulator

import sim_bug_tools.structs as structs
from sim_bug_tools.exploration.boundary_core.surfacer import find_surface
# from sim_bug_tools.exploration.brrt_std.adherer import (
#     BoundaryAdherenceFactory, BoundaryLostException)
# import sim_bug_tools.rng.lds.sequences as sequences
from sim_bug_tools.exploration.brrt_std.brrt import BoundaryRRT
from sim_bug_tools.exploration.brrt_v2.adherer import (
    BoundaryAdherenceFactory, BoundaryLostException)
from sim_bug_tools.graphics import Grapher


def to_json(name: str, num_errs: int, num_results: int, results: list[tuple[structs.Point, bool]], meta: dict = {}) -> str:
    json_dict = {'name': name, 'num_results': num_results, 'num_errs': num_errs, 'results': [], "meta-data": meta}
    
    for p, cls in results:
        json_dict['results'].append(
            (
                [x for x in p],
                cls
            )
        )
    
    return json.dumps(json_dict, indent=4)

def inject_points(path: str, grapher: Grapher):
    json_dict = None
    with open(path, "r") as f:
        json_dict = json.loads(f.read())
    target = []
    nontarget = []
    for array, cls in json_dict["results"]:
        if cls:
            target.append(structs.Point(array))
        else:
            nontarget.append(structs.Point(array))
    
    if len(target) > 0:
        grapher.plot_all_points(target, color="cyan")
    
    if len(nontarget) > 0:
        grapher.plot_all_points(nontarget, color="orange")
    
    
def get_inital_node(path: str, classifier, t0, d, domain, v):
    "Get initial node from cache or sim"
    from os.path import exists 
    node = None
    if exists(path):
        print("Cache found, using cached initial boundary point and OSV...")
        with open(path, "r") as f:
            json_dict = json.loads(f.read())
            node = structs.Point(json_dict["b0"]), np.array(json_dict["n0"])
            
    else:    
        print("No cached initial point, finding surface...")
        node, interm_points = find_surface(classifier, t0, d, domain, v)
        b0, n0 = node
        json_dict = {"b0": tuple(b0.array), "n0": tuple(n0), "interm": [tuple(p.array) for p in interm_points]}
        with open(path, "w") as f:
            f.write(json.dumps(json_dict, indent=4))
    
    return node
        
    

def main():
    # Create a parameter manager object 
    manager = simulator.HighwayPassTestParameterManager()
    
    # Classifies a point using a test
    def classifier(p: structs.Point) -> bool:
        # Use the manager to generate discrete concrete values 
        concrete_params = manager.map_parameters(p)
        # An entire SUMO simulation will run from start-to-end
        test = simulator.HighwayPassTest(concrete_params)
        return test.scores["e_brake"] > 0 or test.scores["collision"] > 0

    grapher = Grapher(True, structs.Domain.normalized(3))    
    axes_names = ["displacement", "rel. velocity", "max brake"]
    grapher.ax.set_xlabel(axes_names[0])
    grapher.ax.set_ylabel(axes_names[1])
    grapher.ax.set_zlabel(axes_names[2])
    
    inject_points("14-25-54-results.txt", grapher)
    
    d = 0.05
    theta = 5 * np.pi / 180 # 5 degrees
    delta_theta = 90 * np.pi / 180 # 5 degrees
    domain = structs.Domain.normalized(3)
    
    # Initial point:
    # Min distance, max rel_vel, min braking force.
    t0 = structs.Point(0, 1, 0) 
    
    # Go in the direction of least risk to find surface
    v = np.array([1, 0, 1])
    v = v / np.linalg.norm(v) # normalize
    
    # Find the surface of the envelope
    
    b0, n0 = get_inital_node(".3d-highway_cache-node.json", classifier, t0, d, domain, v)
    grapher.plot_point(b0, color="orange")
    
    
    # Create the adherence strategy
    adherer_f = BoundaryAdherenceFactory(classifier, domain, d, delta_theta, 2, 4)
    # adherer_f = BoundaryAdherenceFactory(classifier, structs.Domain.normalized(3), d, theta)
    
    # Create the BRRT
    brrt = BoundaryRRT(b0, n0, adherer_f)    
    done = False
    i = 0
    all_points = []
    while len(input("Press enter...")) == 0:
        points = []
        errs = 0
        while i < 135:
            try:
                p, n = brrt.expand()
                i += 1
                points.append(p)
            except BoundaryLostException as e:
                print(f"Lost boundary: {e.msg}")
                errs += 1
                
                
        grapher.plot_all_points(points, color="orange")  
        all_points.extend(points)
        plt.pause(0.01) 
        
    print("Saving results to json file")
    meta = {
        "total_samples": len(brrt.sub_samples),
        "ratio": len(points) / len(brrt.sub_samples)
    }
    
    with open(f"{datetime.now().strftime('%H-%M-%S')}-results.txt", "w") as f:
        f.write(to_json("BRRT Sim Test - Boundary points only", errs, i, [(p, True) for p in all_points], meta))
    input("Done. Press enter to exit...")
        

    # # The parameters and scores of many tests can be grouped together into a 
    # # dataframe
    # print("\nRunning many tests...")
    # points = []
    # params = []
    # scores = []
    # for i in range(10):
    #     p = rng.get_points(1)[0]
    #     points.append(p)

    #     concrete_params = manager.map_parameters(p)
    #     params.append(concrete_params)

    #     test = simulator.HighwayPassTest(concrete_params)
    #     scores.append(test.scores)
    #     continue

    # params_df = pd.DataFrame(params)
    # print("\n: Parameters :")
    # print(params_df)

    # scores_df = pd.DataFrame(scores)
    # print("\n: Scores :")
    # print(scores_df)



    # # If needed, a pandas Series can be cast into a point
    # print()
    # p = structs.Point(params_df.iloc[0])
    # print(p)

    # # And visa versa
    # s = p.as_series()
    # print()
    # print(s)

if __name__ == "__main__":
    
    main()