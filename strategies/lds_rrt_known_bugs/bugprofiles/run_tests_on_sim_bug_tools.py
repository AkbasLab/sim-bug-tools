# Add the parent directory to the path
import os, sys
UNITTEST_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(UNITTEST_DIR))
import sim_bug_tools.structs as structs
import pandas as pd
import sim_bug_tools.rng.bugger as bugger
import sim_bug_tools.structs as structs
import sim_bug_tools.rng.lds.sequences as sequences
import sim_bug_tools.utils as utils
import simulators as simulator
from random import Random
from sim_bug_tools.rng.rrt import RapidlyExploringRandomTree

import numpy as np
import json


def test_simulator_known_bugs(i : int):
    with open(UNITTEST_DIR + "/test_bug_profiles/test_bugs" + str(i) + ".json", "r") as f:
        for line in f:
            hhh = json.loads(line)
            break
    bug_profile = [structs.Domain.from_json(d) for d in hhh[0]]
    
    n_dim = len(bug_profile[0])
    domain = structs.Domain([(0,1) for n in range(n_dim)])
    seq = sequences.RandomSequence(
        domain, 
        ["dim_%d" % n for n in range(n_dim)],
        seed = 300
    )

    sim = simulator.SimpleSimulatorKnownBugs(
        bug_profile, seq, 
        file_name = UNITTEST_DIR + "/out/sskbRandomSequence" + str(i) + ".tsv"
    )

    if os.path.exists(sim.file_name):
        os.remove(sim.file_name)

    sim.run(10)
    sim.long_walk_on_enter()
    sim.run(10)

    
    return

def test_simulator_known_bugs_faure_sequence(i : int):
    with open(UNITTEST_DIR + "/test_bug_profiles/test_bugs" + str(i) + ".json", "r") as f:
        for line in f:
            hhh = json.loads(line)
            break
    bug_profile = [structs.Domain.from_json(d) for d in hhh[0]]
    
    n_dim = len(bug_profile[0])
    domain = structs.Domain([(0,1) for n in range(n_dim)])
    seq = sequences.FaureSequence(
        domain, 
        ["dim_%d" % n for n in range(n_dim)],
        seed = 300
    )

    sim = simulator.SimpleSimulatorKnownBugs(
        bug_profile, seq, 
        file_name = UNITTEST_DIR + "/out/sskbFaureSequence" + str(i) + ".tsv"
    )

    if os.path.exists(sim.file_name):
        os.remove(sim.file_name)

    sim.run(10)
    sim.long_walk_on_enter()
    sim.run(10)

        
    return

def test_simulator_known_bugs_halton_sequence(i : int):
    with open(UNITTEST_DIR +"/test_bug_profiles/test_bugs" + str(i) + ".json", "r") as f:
        for line in f:
            hhh = json.loads(line)
            break
    bug_profile = [structs.Domain.from_json(d) for d in hhh[0]]
    
    n_dim = len(bug_profile[0])
    domain = structs.Domain([(0,1) for n in range(n_dim)])
    seq = sequences.HaltonSequence(
        domain, 
        ["dim_%d" % n for n in range(n_dim)],
        seed = 300
    )

    sim = simulator.SimpleSimulatorKnownBugs(
        bug_profile, seq, 
        file_name = UNITTEST_DIR + "/out/sskbHaltonSequence" + str(i) + ".tsv"
    )

    if os.path.exists(sim.file_name):
        os.remove(sim.file_name)

    sim.run(10)
    sim.long_walk_on_enter()
    sim.run(10)

    
    
    return

def test_simulator_known_bugs_sobol_sequence(i : int):
    with open(UNITTEST_DIR + "/test_bug_profiles/test_bugs" + str(i) + ".json", "r") as f:
        for line in f:
            hhh = json.loads(line)
            break
    bug_profile = [structs.Domain.from_json(d) for d in hhh[0]]
    
    n_dim = len(bug_profile[0])
    domain = structs.Domain([(0,1) for n in range(n_dim)])
    seq = sequences.SobolSequence(
        domain, 
        ["dim_%d" % n for n in range(n_dim)],
        seed = 300
    )

    sim = simulator.SimpleSimulatorKnownBugs(
        bug_profile, seq, 
        file_name = UNITTEST_DIR + "/out/sskbSobolSequence" + str(i) + ".tsv"
    )

    if os.path.exists(sim.file_name):
        os.remove(sim.file_name)

    sim.run(10)
    sim.long_walk_on_enter()
    sim.run(10)

        
    return


def test_simulator_known_bugs_lattice_sequence(i : int):
    with open(UNITTEST_DIR + "/test_bug_profiles/test_bugs" + str(i) + ".json", "r") as f:
        for line in f:
            hhh = json.loads(line)
            break
    bug_profile = [structs.Domain.from_json(d) for d in hhh[0]]
    
    n_dim = len(bug_profile[0])
    domain = structs.Domain([(0,1) for n in range(n_dim)])
    seq = sequences.LatticeSequence(
        domain, 
        ["dim_%d" % n for n in range(n_dim)],
        seed = 300
    )

    sim = simulator.SimpleSimulatorKnownBugs(
        bug_profile, seq, 
        file_name = UNITTEST_DIR + "/out/sskbLatticeSequence" + str(i) + ".tsv"
    )

    if os.path.exists(sim.file_name):
        os.remove(sim.file_name)

    sim.run(10)
    sim.long_walk_on_enter()
    sim.run(10)

       
    return


def test_simulator_known_bugs_rrt(i : int):
    bug_profile = [structs.Domain([(0,1) for n in range(4)])]
    
    n_dim = len(bug_profile[0])
    domain = structs.Domain([(0,1) for n in range(n_dim)])
    axes_names = ["dim_%d" % n for n in range(n_dim)]
    seq = sequences.RandomSequence(
        domain, axes_names, seed = 300
    )
    rrt = RapidlyExploringRandomTree(
        sequences.RandomSequence(
            domain, axes_names, seed = 555
        ),
        step_size = 0.01,
        exploration_radius = 1
    )
    
    sim = simulator.SimpleSimulatorKnownBugsRRT(
        bug_profile = bug_profile,
        sequence = seq,
        rrt = rrt,
        n_branches = 5,
        file_name = "%s/out/sskbrrt.tsv" % UNITTEST_DIR,
        log_to_console = False
    )

    if os.path.exists(sim.file_name):
        os.remove(sim.file_name)

    sim.run(10)
    sim.local_search_on_enter()
    sim.run(14)

    

    df = pd.read_csv(sim.file_name, sep="\t")
    
    return


def random_size_domain(n_dim : int, a : float, b : float, random : Random):
    return structs.Domain( 
        np.round( [(0, random.uniform(a, b,)) for _ in range(n_dim)], decimals=2)  
    ) 



def generate_bug_profile(n_dim : int, n_clusters : int, i : int):
    
    file_name = UNITTEST_DIR + "/test_bug_profiles/test_bugs" + str(i) + ".json"

    if os.path.exists(file_name):
                os.remove(file_name)


    cluster_size_min = 0.1
    cluster_size_max = 0.2
    n_clusters = 1
    seed = 0
    n_profiles = 30
    random = Random(seed)

    normal_domain = structs.Domain([(0,1) for _ in range(n_dim)])
    random = Random(seed) 

    # We need to use the random sequence or it will conflict with our results.
    seq = sequences.RandomSequence(
                normal_domain, 
                ["dim_%d" % n for n in range(n_dim)],
                seed = 300
            )

    # Construct the bug builder object
    bug_builder = bugger.BugBuilder(
            location_domain = normal_domain,
            size_domain = random_size_domain(
                n_dim, cluster_size_min, cluster_size_max, random),
            sequence = seq,
            random = random
        )

    # Generate the profiles
    profiles = []
    for i_profile in range(n_profiles):
        clusters : list[structs.Domain] = []
        for i_bug in range(n_clusters):
            # Generate a cluster
            clusters.append( bug_builder.build_bug().as_json() )

            # Scramble the size of the next cluster.
            bug_builder.size_domain = random_size_domain(
                n_dim, cluster_size_min, cluster_size_max, random)
            continue
        profiles.append(clusters)
        
        continue

    #  Dump to file
    with open(file_name, "w") as f:
        f.write(json.dumps(profiles)) 

    print("Profiles saved to \"%s\"." % file_name)
        
        
    
    return

#Test number
i = 0
dimension_range = [2,5,10,50]
cluster_range = range(10)

for dimension in dimension_range:
    for cluster in cluster_range:
        generate_bug_profile(dimension,cluster,i)
        test_simulator_known_bugs(i)
        test_simulator_known_bugs_faure_sequence(i)
        test_simulator_known_bugs_lattice_sequence(i)
        test_simulator_known_bugs_halton_sequence(i)
        test_simulator_known_bugs_rrt(i)
        test_simulator_known_bugs_sobol_sequence(i)
        i = i + 1


