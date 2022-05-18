import os
FILE_DIR = os.path.dirname(os.path.abspath(__file__))

import unittest
import json
import pandas as pd
import re


class TestReadTSV(unittest.TestCase):

    def test_read_tsv(self):
        # Load Metadata
        fn = "%s/out/tl-test.json" % FILE_DIR
        with open(fn) as f:
            metadata = json.load(f)
        
        # Load the Dataframe
        fn = "%s/out/tl-test.tsv" % FILE_DIR
        df = pd.read_csv(fn, sep="\t")
        
        # Point concrete should be the size of the parameter names
        # Currently point_normal is invalid.
        # Parse point as a list of floats
        point = [float(n) for n in \
            re.findall(r'[0-9]*\.[0-9]*', df["point_concrete"].iloc[0])]
        print(
            len(metadata["parameter_names"]),
            len(point)
        )

        # TODO: Use this in catboost and then try SHAP

        return