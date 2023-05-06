import highway2
import numpy as np
import sim_bug_tools.structs as structs
import sim_bug_tools.rng.lds.sequences as sequences
import json

from os.path import exists
from typing import Callable, Any
from sim_bug_tools.exploration.boundary_core.surfacer import find_surface

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

from sim_bug_tools.exploration.brrt_std.brrt import BoundaryRRT

from datetime import datetime
import pickle

from sim_bug_tools.structs import Point, Domain
from numpy import ndarray
from rtree.index import Index, Property

# Constants
OUTPUT_FOLDER = ".results/.saved/pred"
#


def provide_data(path: str, foo: Callable[[Any], dict], *args, **kwargs):
    result = None
    if exists(path):
        print("Cache found! Providing data...")
        with open(path, "r") as f:
            result = json.loads(f.read())
    else:
        print("No cache found. Processing request and caching result...")
        result = foo(*args, **kwargs)
        with open(path, "w") as f:
            f.write(json.dumps(result, indent=4))

    return result


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


def test1():
    manager = highway2.HighwayTrafficParameterManager()

    domain = structs.Domain.normalized(len(manager.params_df.index))
    print(len(manager.params_df.index))

    def classifier(p: structs.Point) -> bool:
        # Use the manager to generate discrete concrete values
        concrete_params = manager.map_parameters(p)
        # An entire SUMO simulation will run from start-to-end
        test = highway2.HighwayTrafficTest(concrete_params)
        return test.scores["e_brake"] > 0 or test.scores["collision"] > 0

    axes_names = [
        "speed_limit",
        "lc_dur",
        "dut_e_decel",
        "dut_sosl",
        "npcl1_init_disp",
        "npcl1_sosl",
        "npcl1_vtype",
        "npcl2_init_disp",
        "npcl2_sosl",
        "npcl2_vtype",
        "npcl3_init_disp",
        "npcl3_sosl",
        "npcl3_vtype",
        "npcl4_init_disp",
        "npcl4_sosl",
        "npcl4_vtype",
        "npcl5_init_disp",
        "npcl5_sosl",
        "npcl5_vtype",
        "npcr1_init_disp",
        "npcr1_sosl",
        "npcr1_vtype",
        "npcr2_init_disp",
        "npcr2_sosl",
        "npcr2_vtype",
        "npcr3_init_disp",
        "npcr3_sosl",
        "npcr3_vtype",
        "npcr4_init_disp",
        "npcr4_sosl",
        "npcr4_vtype",
        "npcr5_init_disp",
        "npcr5_sosl",
        "npcr5_vtype",
    ]

    rng = sequences.SobolSequence(domain, axes_names)
    rng.seed = 555

    # Adherer params
    d = 0.05
    theta = 15 * np.pi / 180  # 5 degrees
    delta_theta = 90 * np.pi / 180  # 90 degrees
    r = 2
    N = 4

    ndims = len(axes_names)
    print(ndims)
    domain = structs.Domain.normalized(ndims)

    # Find initial target value using sobol
    def find_t0_nt0(cut_off=50) -> dict:
        t0 = None
        nt0 = None
        p: structs.Point = None

        i = 0
        while (t0 is None or nt0 is None) and (cut_off is None or i < cut_off):
            p = rng.get_points(1)[0]
            cls = classifier(p)
            if t0 is None and cls:
                t0 = p
                print("Found a target point...")
            elif nt0 is None and not cls:
                nt0 = p
                print("Found a non-target point...")

        print("Found a target and non-target point!")

        return {"t0": tuple(t0), "nt0": tuple(nt0)}

    # Only executes if no t0 was cached
    tmp = provide_data(f"{OUTPUT_FOLDER}/nd-highway_cache-t0_n0-new.json", find_t0_nt0)
    t0 = structs.Point(tmp["t0"])
    nt0 = structs.Point(tmp["nt0"])

    v = (nt0 - t0).array
    v /= np.linalg.norm(v)

    def get_surface_node(cut_off=10) -> dict:
        result = None
        while result is None:
            # try:
            #     result = find_surface(classifier, t0, d, domain, cut_off=10)
            # except Exception as e:
            #     print("Failed to find boundary...", e)

            result = find_surface(classifier, t0, d, domain, v, cut_off=None)
            node, interm, outter_sample_is_in_domain = result
            print(result)
            # if not outter_sample_is_in_domain:
            #     print("Direction not pointing to edge of envelope")
            #     result = None

        return {
            "b0": tuple(node[0]),
            "n0": tuple(node[1]),
            "interm": [tuple(p) for p in interm],
        }

    node_cache = provide_data(
        f"{OUTPUT_FOLDER}/nd-highway_cache-node2-new.json", get_surface_node
    )
    b0 = structs.Point(node_cache["b0"])
    n0 = np.array(node_cache["n0"])

    ADHERER_VERSION = "v2"
    if ADHERER_VERSION == "v1":
        adherer_f = ConstAdherer(classifier, domain, structs.Spheroid(d), theta, True)
    else:
        adherer_f = ExpAdherer(classifier, domain, d, delta_theta, r, N, True)

    nsamples = 1000

    bp_k = 1
    backprop_enabled = bp_k > 0
    brrt = BoundaryRRT(b0, n0, adherer_f)

    all_points: tuple[structs.Point, bool] = []
    i = 0
    out_of_bounds_count = 0
    errs = 0

    # while len(input("Press enter...")) == 0:
    #    errs = 0
    while brrt.boundary_ < nsamples:
        try:
            all_points.append(brrt.step())
        except BoundaryLostException as e:
            errs += 1
        except SampleOutOfBoundsException as e:
            out_of_bounds_count += 1

        if backprop_enabled and i != brrt.boundary_:
            i = brrt.boundary_
            brrt.back_propegate_prev(bp_k)

    print("Setting up data...")

    non_boundary_points = [
        (tuple(p), bool(cls))
        for p, cls in all_points
        if p not in map(lambda node: node[0], brrt.boundary)
    ]

    data = {
        "name": f"BRRT{ADHERER_VERSION} ND Scenario - {nsamples}",
        "backprop-enabled": backprop_enabled,
        "backprop-params": {"k": bp_k} if backprop_enabled else None,
        "brrt-type": ADHERER_VERSION,
        "brrt-params": {"d": d, "theta": theta}
        if ADHERER_VERSION == "v1"
        else {"d": d, "delta-theta": delta_theta, "r": r, "N": N},
        "meta-data": {
            "err-count": errs,
            "out-of-bounds-count": out_of_bounds_count,
            "ratio": brrt.boundary_ / len(non_boundary_points),
            "b-count": brrt.boundary_,
            "nonb-count": len(non_boundary_points),
        },
        "dimensions": axes_names,
        "b-nodes": [(tuple(b), tuple(n)) for b, n in brrt.boundary],
        "nonb-points": [(tuple(p), bool(cls)) for p, cls in non_boundary_points],
    }

    print(data)
    print("Saving results to json file...")
    with open(
        f"{OUTPUT_FOLDER}/{datetime.now().strftime('%y%m%d-%H%M%S')}-nd-{ADHERER_VERSION}-{'' if backprop_enabled else 'no_'}bp.json",
        "w",
    ) as f:
        f.write(json.dumps(data, indent=4))


test1()
