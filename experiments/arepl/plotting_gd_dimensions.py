import json
import pandas as pd
import matplotlib.pyplot as plt


from sim_bug_tools import Point, Domain
from sim_bug_tools.graphics import Grapher

with open("../../gd-dimension-test.json", "r") as f:
    df = pd.DataFrame(json.loads(f.read()))

ndims = [3, 5, 10, 15, 20, 25, 30, 50, 75, 100]


convert = lambda lst: [Point(x, y) for x, y in zip(ndims, lst)]


def plot_eff():
    domain = Domain.from_bounding_points(Point(0, 0), Point(101, 0.03))
    g = Grapher(False, domain, ["dimensions", "efficiency"])
    g.set_yformat("{:,.0%}")
    g.plot_all_points(convert(map(lambda val: val, df["eff"])))
    g.draw_path(convert(map(lambda val: val, df["eff"])))


def plot_err():
    domain = Domain.from_bounding_points(Point(0, 0), Point(101, max(df["avg-err"])))
    g = Grapher(False, domain, ["dimensions", "error"])
    g.plot_all_points(convert(df["avg-err"]), color="red")
    g.draw_path(convert(df["avg-err"]), color="red")


def plot_fc():
    domain = Domain.from_bounding_points(
        Point(0, 0), Point(101, max(df["n-failures"]) + 100)
    )
    g = Grapher(False, domain, ["dimensions", "failure count"])
    g.plot_all_points(convert(df["n-failures"]), color="red")
    g.draw_path(convert(df["n-failures"]), color="red")


def plot_nonb_count():
    domain = Domain.from_bounding_points(
        Point(0, 0), Point(max(n_samples) + 512, 14000)
    )
    g = Grapher(False, domain, ["train size", "non-boundary sample count"])
    g.ax.set_xscale("log", base=2)
    g.plot_all_points(convert(df["nonb-cound"]), color="red")
    g.draw_path(convert(df["nonb-cound"]), color="red")


plot_eff()
plot_fc()
plot_err()
plt.show()
