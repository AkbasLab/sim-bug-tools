from sim_bug_tools.sumo import TraCIClient
import sim_bug_tools.utils as utils
import sim_bug_tools.rng.lds.sequences as sequences

import errors
import simulator

import traci
import os
import pandas as pd
import numpy as np
import time

FILE_DIR = os.path.dirname(os.path.abspath(__file__))

class RegimeSUMO:
    def __init__(self):
        # This class is the parameter manager tool for the 
        #  Tl-Race scenario
        tlrpm = simulator.TrafficLightRaceParameterManager()


        # Test Sequence
        sequences.RandomSequence()
        return

    
    def _test_timing(self):

        run_ids = [i for i in range(1)]

        timing = []
        for i in run_ids:
            before = time.time()


            # test = simulator.TrafficLightRace()



            elapsed = time.time() - before
            timing.append(elapsed)

        utils.save(timing, "stats/timing.pkl")
        return


if __name__ == "__main__":
    RegimeSUMO()