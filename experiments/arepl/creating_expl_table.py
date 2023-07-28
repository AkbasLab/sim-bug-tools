# from arepl_dump import dump

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from sim_bug_tools import Point, Domain
from sim_bug_tools.graphics import Grapher

pairs = (
    # ("sphere-cluster", "const"),
    # ("sphere-cluster", "exp"),
    ("sphere", "const"),
    ("sphere", "exp"),
    ("cube", "const"),
    ("cube", "exp"),
)


def avg(lst: list):
    return sum(lst) / len(lst)


for e_type, adh_type in pairs:
    with open(f".tmp/results/expl_results-{e_type}-rrt-{adh_type}.json", "r") as f:
        dct = json.loads(f.read())

    result = {
        "ndim": [],
        "eff": [],
        "err": [],
        "BLEs": [],
        "OOBs": [],
    }

    for key, value in dct.items():
        eff, err, bles, oobs = np.array(
            [
                (test["eff"], test["avg-err"], test["BLEs"], test["OOBs"])
                for test in value
            ]
        ).T

        result["ndim"].append(key)
        result["eff"].append(avg(eff))
        result["err"].append(avg(err))
        result["BLEs"].append(avg(bles))
        result["OOBs"].append(avg(oobs))

    pd.DataFrame(result).to_csv(f"table-{e_type}-{adh_type}.csv")
