import unittest
import pandas as pd
import sim_bug_tools.stats as stats

class TestStats(unittest.TestCase):
    def test_rectangle_wave(self):
        print()

        fn = "tests/stats/data/random.tsv"
        df = pd.read_csv(fn, sep="\t")
        stats.rectangle_wave(df["step"].tolist(), df["is_bug"].tolist())

        print()
        return