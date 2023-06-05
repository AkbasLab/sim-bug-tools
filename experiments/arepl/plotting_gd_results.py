from arepl_dump import dump

import json
import pandas as pd
import matplotlib.pyplot as plt


from sim_bug_tools import Point, Domain
from sim_bug_tools.graphics import Grapher

with open("tmp-sd.json", "r") as f:
    df = pd.DataFrame(json.loads(f.read()))

n_samples = [100, 200, 400, 800, 1600, 3200, 6400]


convert = lambda lst: [Point(x, y) for x, y in zip(n_samples, lst)]


def plot_eff():
    domain = Domain.from_bounding_points(
        Point(96, 0), Point(max(n_samples) + 512, 0.05)
    )
    g = Grapher(False, domain, ["train size", "efficiency"])
    g.ax.set_xscale("log", base=2)
    g.set_yformat("{:,.0%}")
    g.plot_all_points(convert(map(lambda val: val / 100, df["post-train-eff%"])))
    g.draw_path(convert(map(lambda val: val / 100, df["post-train-eff%"])))


def plot_err():
    domain = Domain.from_bounding_points(
        Point(96, -0.2), Point(max(n_samples) + 512, 0.4)
    )
    g = Grapher(False, domain, ["train size", "error"])
    g.ax.set_xscale("log", base=2)
    g.plot_all_points(convert(df["avg-err"]), color="red")
    g.draw_path(convert(df["avg-err"]), color="red")


def plot_fc():
    domain = Domain.from_bounding_points(Point(96, 0), Point(max(n_samples) + 512, 20))
    g = Grapher(False, domain, ["train size", "failure count"])
    g.ax.set_xscale("log", base=2)
    g.plot_all_points(convert(df["failure-count"]), color="red")
    g.draw_path(convert(df["failure-count"]), color="red")


def plot_nonb_count():
    domain = Domain.from_bounding_points(
        Point(96, 1000), Point(max(n_samples) + 512, 14000)
    )
    g = Grapher(False, domain, ["train size", "non-boundary sample count"])
    g.ax.set_xscale("log", base=2)
    g.plot_all_points(convert(df["nonb-cound"]), color="red")
    g.draw_path(convert(df["nonb-cound"]), color="red")


plot_eff()
plot_err()
plot_fc()
plot_nonb_count()


plt.show()
