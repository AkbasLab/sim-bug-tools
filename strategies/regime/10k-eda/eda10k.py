import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import sim_bug_tools.utils as utils
import sim_bug_tools.graphics as graphics

import pandas as pd
import matplotlib.pyplot as plt

class EDA10K:
    def __init__(self):
        # self.generate_tables()
        self.params_df = pd.read_csv("data/param.csv")
        self.scores_df = pd.read_csv("data/score.csv")

        for feat in self.scores_df.columns:
            vals = self.scores_df[self.scores_df[feat] > 0][feat]
            ax = graphics.new_axes()
            ax.hist(vals, bins=30)
            plt.savefig("figs/%s.png" % feat)
        return



    def generate_tables(self):
        # Load the data
        raw = utils.load("data/random_10k.pkl")

        # Test structure is:
        #  test_id
        #  | + params
        #  | | + veh
        #  | | | + concrete
        #  | | | + normal
        #  | | + tl
        #  | |   + concrete
        #  | |   + normal
        #  | + scores
        #  |   + veh
        #  |   + scores

        # Collect data for each test.
        all_params = []
        all_scores = []
        for test_id, test in raw.items():
            # Parameter data
            flat_veh = self.flatten_veh_params_df( test["params"]["veh"]["normal"] )
            flat_tl = self.flatten_tl_params_df( test["params"]["tl"]["normal"] )
            params = flat_veh.append(flat_tl)

            # scores
            scores = test["scores"]["scores"]
            
            # Add test id to each series.
            params["test_id"] = test_id
            scores["test_id"] = test_id

            # Add to the list
            all_params.append(params)
            all_scores.append(scores)
            continue

        # Make the tables
        param_df = pd.DataFrame(all_params)
        score_df = pd.DataFrame(all_scores)

        # Normalize the collision field
        # score_df["collision"] = score_df["collision"].apply(
        #     lambda x: x / score_df["collision"].max()
        # )

        param_df.to_csv("data/param.csv", index=False)
        score_df.to_csv("data/score.csv", index=False)
        return

    def flatten_tl_params_df(self, df : pd.DataFrame) -> pd.Series:
        data = {}
        for i in range(len(df.index)):
            data["TL_%s" % df["state"].iloc[i]] = df["dur"].iloc[i]
        return pd.Series(data)

    def flatten_veh_params_df(self, df : pd.DataFrame) -> pd.Series:
        features = df.columns.to_list()[:-1]
        data = {}
        for i in range(len(df.index)):
            vid = df["veh_id"].iloc[i]
            for feat in features:
                data["AV%s_%s" % (vid, feat)] = df[feat].iloc[i]
            continue
        return pd.Series(data)




if __name__ == "__main__":
    EDA10K()