import unittest
import pandas as pd
import sim_bug_tools.graphics as graphics
import matplotlib.pyplot as plt

class TestSignals(unittest.TestCase):
    def test_signal(self):
        print()

        fn = "tests/graphics/data/random.tsv"
        df = pd.read_csv(fn, sep="\t")
        s1 = graphics.Signal(
            time = df["step"].tolist(), 
            amplitude = df["is_bug"].tolist()
        )
        

        print()
        return