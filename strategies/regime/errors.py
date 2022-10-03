import re
import pandas as pd



def categorize_errors(errors : str) -> pd.DataFrame:
    """
    Categorize a string of errors.
    """
    scores = []
    for err in errors.split("\n"):
        if "because of a red traffic light" in err:
            scores.append({
                "veh_id" : veh_id(err),
                "type" : "e_stop",
                "score" : 1
            })

        if "performs emergency braking on lane" in err:
            scores.append({
                "veh_id" : veh_id(err),
                "type" : "e_stop",
                "score" : e_brake(err)
            })

        if "(jam)" in err:
            scores.append({
                "veh_id" : veh_id(err),
                "type" : "jam",
                "score" : 1
            })
    if scores:
        return pd.DataFrame(scores)
    return pd.DataFrame(columns=["veh_id", "type", "score"])

def e_brake(err : str, emergency_decel : float) -> float:
    """
    Parse E_brake score from an e_brake warning.

    0 = ideal, no ebrake used.
    1 = worst case

    Score is calculated by 1 - (wished/decel)
    """
    decel = float(re.findall(r"decel=\d*\.\d*",err)[0].split("=")[-1])
    wished = float(re.findall(r"wished=\d*\.\d*", err)[0].split("=")[-1])
    return 1 - (wished/decel)




