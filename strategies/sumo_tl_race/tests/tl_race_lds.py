import os, sys
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(FILE_DIR))
from tl_race import TrafficLightRace

import sim_bug_tools.rng.lds.sequences as sequences

def main():
    SEQUENCES_GENERATORS = [
        sequences.RandomSequence,
        sequences.FaureSequence,
        sequences.HaltonSequence,
        sequences.SobolSequence
    ]
    SEQUENCE_NAMES = ["random", "faure", "halton", "sobol"]
    seed = 500
    checkpoint = 100
    tests_total = 5000
    

    for i, seq_gen in enumerate(SEQUENCES_GENERATORS):
        print("\n\n### %s ###\n\n" % SEQUENCE_NAMES[i])
        sim = TrafficLightRace(
            sequence_generator = seq_gen,
            file_name = "%s/out/%s.tsv" % (FILE_DIR, SEQUENCE_NAMES[i])
        )
        for n in range(checkpoint, tests_total, checkpoint):
            sim.resume()
            sim.run(n)
            continue
        continue

    return

if __name__ == "__main__":
    main()