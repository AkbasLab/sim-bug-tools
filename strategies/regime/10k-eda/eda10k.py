import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import sim_bug_tools.utils as utils
import pandas as pd

class EDA10K:
    def __init__(self):
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
        for test_id, test in raw.items():
            # Parameter data
            flat_veh = self.flatten_veh_params_df( test["params"]["veh"]["normal"] )
            flat_tl = self.flatten_tl_params_df( test["params"]["tl"]["normal"] )
            params = flat_veh.append(flat_tl)

            # scores
            scores = test["scores"]["scores"]
            print(scores)
            break
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