import os, sys
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(FILE_DIR))
from simple import Simple

import pandas as pd
import unittest

class TestSimple(unittest.TestCase):
    def test_simple(self):
        print("\n\n")
        sim = Simple(file_name = "%s/out/simple.csv" % FILE_DIR)
        
        dist_from_stop = []
        speed_start = []
        score = []

        
        i = 0; total = 100*51
        for dfs in range(1,100+1):
            for ss in range(0,50+1):
                i += 1
                print("RUN %d of %d" % (i,total))
                dist_from_stop.append(dfs)
                speed_start.append(ss)
                score.append(
                    sim._run_sumo_scenario(dfs, ss)
                )
                print("\nScore: %.2f\n" % score[-1])
                

                

        df = pd.DataFrame({
            "dist_from_stop" : dist_from_stop,
            "speed_start" : speed_start,
            "score" : score
        })
        df.to_csv("%s/out/simple.csv" % FILE_DIR, index=False)
        print("\n\n")
        return

if __name__ == "__main__":
    unittest.main()