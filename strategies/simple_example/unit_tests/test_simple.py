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
        is_comfortable = []

        
        i = 0; total = 100*51
        for dfs in range(1,100+1):
            for ss in range(0,50+1):
                i += 1
                print("RUN %d of %d" % (i,total))
                dist_from_stop.append(dfs)
                speed_start.append(ss)
                is_comfortable.append(
                    sim._run_sumo_scenario(dfs, ss)
                )
                break
            break

        df = pd.DataFrame({
            "dist_from_stop" : dist_from_stop,
            "speed_start" : speed_start,
            "is_comfortable" : is_comfortable
        })
        df.to_csv("%s/out/simple.csv" % FILE_DIR, index=False)
        print("\n\n")
        return

if __name__ == "__main__":
    unittest.main()