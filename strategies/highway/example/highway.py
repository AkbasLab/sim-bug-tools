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
import os, sys
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(FILE_DIR))


import sim_bug_tools.rng.lds.sequences as sequences
import sim_bug_tools.structs as structs

import simulator

import pandas as pd


def main():
    # Create a parameter manager object 
    manager = simulator.HighwayPassTestParameterManager()


    # Any normalized Point is acceptable.
    # For this example, we use a randomly generated point.
    rng = sequences.RandomSequence(
            domain = structs.Domain.normalized(len(manager.params_df.index)),
            axes_names = manager.params_df["feature"].tolist()    
    )
    rng.seed = 555

    for i in range(80):
        rng.get_points(1)

    p = rng.get_points(1)[0]
    print(p)


    # Use the manager to generate discrete concrete values 
    concrete_params = manager.map_parameters(p)
    print(": Concrete Parameters :")
    print(concrete_params)

    # Run a test
    # An entire SUMO simulation will run from start-to-end
    test = simulator.HighwayPassTest(concrete_params)

    # The scores are automatically generated.
    print(test.scores)


    # The parameters and scores of many tests can be grouped together into a 
    # dataframe
    print("\nRunning many tests...")
    points = []
    params = []
    scores = []
    for i in range(10):
        p = rng.get_points(1)[0]
        points.append(p)

        concrete_params = manager.map_parameters(p)
        params.append(concrete_params)

        test = simulator.HighwayPassTest(concrete_params)
        scores.append(test.scores)
        continue

    params_df = pd.DataFrame(params)
    print("\n: Parameters :")
    print(params_df)


    scores_df = pd.DataFrame(scores)
    print("\n: Scores :")
    print(scores_df)



    # If needed, a pandas Series can be cast into a point
    print()
    p = structs.Point(params_df.iloc[0])
    print(p)

    # And visa versa
    s = p.as_series()
    print()
    print(s)
    return

if __name__ == "__main__":
    main()