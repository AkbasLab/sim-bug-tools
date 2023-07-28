# from arepl_dump import dump

import json
import pandas as pd
import matplotlib.pyplot as plt


from sim_bug_tools import Point, Domain
from sim_bug_tools.graphics import Grapher

with open("../../expl_results-sphere-cluster-const.json", "r") as f:
    df = pd.DataFrame(json.loads(f.read()))

ndims = [3, 5, 10, 15, 20, 25, 30, 50, 75, 100]

convert = lambda lst: [Point(x, y) for x, y in zip(ndims, lst)]


def create_subdf(name: str) -> pd.DataFrame:
    data = {}
    for row in df[name]:
        for key, value in row.items():
            if key in data:
                data[key].append(value)
            else:
                data[key] = [value]

    print(data)
    return pd.DataFrame(data)


# print(create_subdf('sphere-const-3d'))

# print(df2)


def plot_eff():
    domain = Domain.from_bounding_points(Point(0, 0), Point(101, 1))

    effs = [abs(create_subdf(sub_result)["eff"].mean()) for sub_result in df]
    data = convert(effs)
    print(data)

    g = Grapher(False, domain, ["dimensions", "efficiency"])
    g.set_yformat("{:,.0%}")
    g.plot_all_points(data)
    g.draw_path(data)


def plot_err():
    errs = [abs(create_subdf(sub_result)["avg-err"].mean()) for sub_result in df]
    domain = Domain.from_bounding_points(Point(0, 0), Point(101, max(errs) * 1.1))
    data = convert(errs)
    print(data)

    g = Grapher(False, domain, ["dimensions", "error"])
    # g.set_yformat("{:,.0%}")
    g.plot_all_points(data)
    g.draw_path(data)


plot_eff()
plot_err()

plt.show()
