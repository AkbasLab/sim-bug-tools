import json
import pandas as pd
import matplotlib.pyplot as plt


from sim_bug_tools import Point, Domain
from sim_bug_tools.graphics import Grapher


def avg(lst):
    return sum(lst) / len(lst)


# size_tests = [100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600]
dim_tests = [3, 5, 10, 15, 20, 25, 30, 50, 75, 100]


result = {
    "err": [],
    "eff": [],
    "failures": [],
    "dimensions": [],
}

for dimensions in dim_tests:
    with open(f".gd_exp/gd_exp-{dimensions}.json", "r") as f:
        data = json.loads(f.read())

    err, eff, failures = zip(
        *[
            (test["avg-err"], test["post-train-eff"], test["failure-count"])
            for test in data
        ]
    )

    result["err"].append(avg(err))
    result["eff"].append(avg(eff))
    result["failures"].append(avg(failures))
    result["dimensions"].append(dimensions)

df = pd.DataFrame(result)

print(df)

convert = lambda lst: [Point(x, y) for x, y in zip(dim_tests, lst)]


def plot_eff():
    domain = Domain.from_bounding_points(Point(0, 0), Point(101, 0.05))
    g = Grapher(False, domain, ["dimensions", "efficiency"])
    # g.ax.set_xscale("log", base=2)
    g.set_yformat("{:,.1%}")
    g.plot_all_points(convert(df["eff"].values))
    g.draw_path(convert(df["eff"].values))


def plot_err():
    domain = Domain.from_bounding_points(
        Point(0, min(df["err"] - 0.01)), Point(101, max(df["err"] + 0.01))
    )
    g = Grapher(False, domain, ["dimensions", "error from boundary"])
    g.plot_all_points(convert(df["err"].values), color="red")
    g.draw_path(convert(df["err"].values), color="red")
    # g.ax.set_xscale("log", base=2)


def plot_fc():
    domain = Domain.from_bounding_points(
        Point(0, 0), Point(101, max(df["failures"]) + 10)
    )
    g = Grapher(False, domain, ["dimensions", "failure count"])
    # g.ax.set_xscale("log", base=2)
    g.plot_all_points(convert(df["failures"]), color="red")
    g.draw_path(convert(df["failures"]), color="red")


plot_eff()
plot_fc()
plot_err()
plt.show()
