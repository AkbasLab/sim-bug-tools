import unittest
import os, sys
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(FILE_DIR))

import shap_tools as st
import sim_bug_tools.graphics as graphics

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

class TestShapTools(unittest.TestCase):

    def test_one(self):
        print("\n\n")
        # Read Data
        params_df = pd.read_csv("%s/data/b_params.csv" % FILE_DIR)
        scores_df = pd.read_csv("%s/data/b_scores.csv" % FILE_DIR)

        # Select Scores
        scores = scores_df["e_brake"].apply(lambda x: x > 0 and x < .5)

        # Get Feature Impact
        fih = st.FeatureImpactHelper(
            data_df=params_df, 
            scores=scores,
            test_size=0.7
        )


        # Plot
        plt.figure(figsize = (5,10))
        ax = plt.axes()

        y = [i for i in range(len(fih.feature_impact["impact"]))]
        height = fih.feature_impact["impact"]
        features = fih.feature_impact["feature"]

        bars = ax.barh(y, height, label=features, color="black")
        ax.set_yticks(y)
        ax.set_yticklabels(features)

        labels = ["%.3f" % np.round(x, decimals=3) \
            for x in fih.feature_impact["impact"]]
        ax.bar_label(bars, labels = labels)

        ax.invert_yaxis()
        
        ax.set_xlabel("Feature Impact (out of 1.0)")
        ax.set_ylabel("Feature")
        ax.set_title("Feature Impact for Envelope $E$")

        plt.tight_layout()
        plt.savefig("%s/figures/impact.png" % FILE_DIR)

        print("\n\n")
        return