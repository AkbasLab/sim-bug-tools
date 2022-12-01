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

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(FILE_DIR))

import json
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import simulator

import sim_bug_tools.structs as structs
from sim_bug_tools.graphics import Grapher
# import sim_bug_tools.rng.lds.sequences as sequences
from sim_bug_tools.rng.lds.sequences import SobolSequence


def to_json(name: str, num_errs: int, num_results: int, results: list[tuple[structs.Point, bool]]) -> str:
    json_dict = {'name': name, 'num_results': num_results, 'num_errs': num_errs, 'results': []}
    for p, cls in results:
        json_dict['results'].append(
            (
                [x for x in p],
                cls
            )
        )
    
    return json.dumps(json_dict, indent=4)




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
    
    d = 0.1
    theta = 5 * np.pi / 180 # 5 degrees
    domain = structs.Domain.normalized(3)
    
    seq = SobolSequence(domain, axes_names)
    
    done = False
    
    while len(input("Press enter...")) == 0:
        points = []
        target = []
        nontarget = []
        for i in range(500):
            p = seq.get_sample(1).points[0]
            cls = classifier(p)
            
            if cls:
                target.append(p)
            else:
                nontarget.append(p)
            
        grapher.plot_all_points(target, color="red")          
        grapher.plot_all_points(nontarget, color="blue")          
                
                
        plt.pause(0.01) 
    
    all_points = [(p, True) for p in target]
    all_points.extend([(p, False) for p in nontarget])
    print("Saving results to json file")
    with open(f"{datetime.now().strftime('%H-%M-%S')}-random_results.txt", "w") as f:
        f.write(to_json("Sobol Sampling of 3D test", 0, len(all_points), all_points))
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