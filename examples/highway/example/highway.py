"""
This python script serves as an example for using SUMO simulator class for the
"highway" scenario. The following files are required:
- highway.net.xml
    The SUMO network.
- params.csv
    A csv file with the parameter ranges for the highway scenario.
- simulator.py
    Contains the parameter manager and SUMO simulator classes
"""
# Adds the parent directory "highway" to the system path.
import os
import sys
from datetime import datetime
from rtree.index import Property, Index
from numpy import ndarray

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(FILE_DIR))

OUTPUT_FOLDER = ".results/.saved/3d"

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import simulator

import sim_bug_tools.structs as structs
from sim_bug_tools.structs import Domain, Point
from sim_bug_tools.exploration.boundary_core.surfacer import find_surface


# import sim_bug_tools.rng.lds.sequences as sequences
from sim_bug_tools.exploration.brrt_std.brrt import BoundaryRRT

from sim_bug_tools.exploration.brrt_std.adherer import (
    ConstantAdherenceFactory as ConstAdherer,
)
from sim_bug_tools.exploration.brrt_v2.adherer import (
    ExponentialAdherenceFactory as ExpAdherer,
)

from sim_bug_tools.exploration.boundary_core.adherer import (
    BoundaryLostException,
    SampleOutOfBoundsException,
)
from sim_bug_tools.graphics import Grapher

ADHERER_VERSION = "v2"


def to_json(
    name: str,
    num_errs: int,
    num_results: int,
    results: list[tuple[structs.Point, bool]],
    meta: dict = {},
) -> str:
    json_dict = {
        "name": name,
        "num_results": num_results,
        "num_errs": num_errs,
        "results": [],
        "meta-data": meta,
    }

    for p, cls in results:
        json_dict["results"].append(([x for x in p], cls))

    return json.dumps(json_dict, indent=4)


def jsonable_points(points: list[structs.Point]):
    return [[x for x in p] for p in points]


def inject_points(path: str, grapher: Grapher):
    json_dict = None
    with open(path, "r") as f:
        json_dict = json.loads(f.read())
    target = []
    nontarget = []
    for array, cls in json_dict["results"]:
        if cls:
            target.append(structs.Point(array))
        else:
            nontarget.append(structs.Point(array))

    if len(target) > 0:
        grapher.plot_all_points(target, color="cyan")

    if len(nontarget) > 0:
        grapher.plot_all_points(nontarget, color="orange")


def get_inital_node(path: str, classifier, t0, d, domain, v):
    "Get initial node from cache or sim"
    from os.path import exists

    node = None
    if exists(path):
        print("Cache found, using cached initial boundary point and OSV...")
        with open(path, "r") as f:
            json_dict = json.loads(f.read())
            node = structs.Point(json_dict["b0"]), np.array(json_dict["n0"])

    else:
        print("No cached initial point, finding surface...")
        node, interm_points = find_surface(classifier, t0, d, domain, v)
        b0, n0 = node
        json_dict = {
            "b0": tuple(b0.array),
            "n0": tuple(n0),
            "interm": [tuple(p.array) for p in interm_points],
        }
        with open(path, "w") as f:
            f.write(json.dumps(json_dict, indent=4))

    return node


def main():
    # Create a parameter manager object
    manager = simulator.HighwayPassTestParameterManager()

    # Classifies a point using a test
    def classifier(p: structs.Point) -> bool:
        # Use the manager to generate discrete concrete values
        concrete_params = manager.map_parameters(p)
        # An entire SUMO simulation will run from start-to-end
        test = simulator.HighwayPassTest(concrete_params)
        return test.scores["e_brake"] > 0 or test.scores["collision"] > 0

    axes_names = ["displacement", "rel. velocity", "max brake"]
    grapher = Grapher(True, structs.Domain.normalized(3), axes_titles=axes_names)

    d = 0.05
    theta = 15 * np.pi / 180  # 5 degrees
    delta_theta = 90 * np.pi / 180  # 90 degrees
    r = 2
    N = 4
    domain = structs.Domain.normalized(3)

    # Initial point:
    # Min distance, max rel_vel, min braking force.
    t0 = structs.Point(0, 1, 0)

    # Go in the direction of least risk to find surface
    v = np.array([1, 0, 1])
    v = v / np.linalg.norm(v)  # normalize

    # Find the surface of the envelope

    b0, n0 = get_inital_node(
        ".3d-highway_cache-node.json", classifier, t0, d, domain, v
    )
    grapher.plot_point(b0, color="orange")

    # Create the adherence strategy
    if ADHERER_VERSION == "v1":
        adherer_f = ConstAdherer(
            classifier, Domain.normalized(3), d, theta, fail_out_of_bounds=True
        )
    elif ADHERER_VERSION == "v2":
        adherer_f = ExpAdherer(
            classifier,
            Domain.normalized(3),
            d,
            delta_theta,
            r,
            N,
            fail_out_of_bounds=True,
        )

    b_i = 0
    nsamples = 200

    out_of_bounds_count = 0
    errs = 0
    bp_k = 1
    backprop_enabled = True

    # Create the BRRT
    brrt = BoundaryRRT(b0, n0, adherer_f)
    all_points: list[tuple[Point, bool]] = [(b0, True)]
    b_i = 0
    while brrt.boundary_ < nsamples:
        try:
            all_points.append(brrt.step())

        except BoundaryLostException as e:
            errs += 1

        except SampleOutOfBoundsException as e:
            out_of_bounds_count += 1

        if b_i != brrt.boundary_:
            b_i = brrt.boundary_
            if backprop_enabled:
                brrt.back_propegate_prev(bp_k)

    # grapher.plot_all_points(list(map(lambda node: node[0], points)),
    # color="black")

    # grapher.draw_tree(brrt._tree, color="blue")
    # grapher.save(f"{OUTPUT_FOLDER}/tree.png")
    # plt.show()

    print("Setting up output data...")
    non_boundary_points = [
        (p, cls)
        for p, cls in all_points
        if p not in map(lambda node: node[0], brrt.boundary)
    ]

    data = {
        "name": f"BRRT{ADHERER_VERSION} 3D Scenario - {nsamples}",
        "brrt-type": ADHERER_VERSION,
        "brrt-params": {"d": d, "theta": theta}
        if ADHERER_VERSION == "v1"
        else {"d": d, "delta_theta": theta, "r": r, "N": N},
        "backprop-enabled": backprop_enabled,
        "backprop-params": {"k": bp_k} if backprop_enabled else None,
        "meta-data": {
            "errs": errs,
            "out-of-bounds": out_of_bounds_count,
            # Excludes samples that weren't classified / used by SUMO
            "ratio": brrt.boundary_ / len(non_boundary_points),
            "b-count": brrt.boundary_,
            "nonb-count": len(non_boundary_points),
            "total-samples": len(all_points),
        },
        "dimensions": axes_names,
        "b-nodes": [(tuple(b), tuple(n)) for b, n in brrt.boundary],
        "nonb-points": [(tuple(p), bool(cls)) for p, cls in non_boundary_points],
    }

    print("Saving results to json file...")
    with open(
        f"{OUTPUT_FOLDER}/{datetime.now().strftime(f'%y%m%d-%H%M%S')}-{ADHERER_VERSION}-3d-{'' if backprop_enabled else 'no_'}bp.json",
        "w",
    ) as f:
        f.write(json.dumps(data, indent=4))

    input("Done. Press enter to exit...")


if __name__ == "__main__":
    main()
