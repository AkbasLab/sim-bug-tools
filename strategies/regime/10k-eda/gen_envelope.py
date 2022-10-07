import pandas as pd
import sim_bug_tools.utils as utils
import sim_bug_tools.structs as structs

def main():
    params_df = pd.read_csv("data/param.csv")
    score_df = pd.read_csv("data/score.csv")

    # Grab the first test which an e_brake score < .5
    for i in range(len(score_df.index)):
        if score_df["e_brake"].iloc[i] > 0 and score_df["e_brake"].iloc[i] < .5:
            root = params_df.iloc[i]
            break

    # Choose a root
    root = structs.Point(root.tolist())
    
    # Compare each record
    envelope_indices = []
    for i in range(len(params_df.index)):
        p = structs.Point(params_df.iloc[i].tolist())
        x = root.distance_to(p)
        if x < 5000.:
            envelope_indices.append(i)
        continue

    print(len(envelope_indices))
    return



if __name__ == "__main__":
    main()