import sim_bug_tools.simulators.base as sim_base
import sim_bug_tools.simulators.sumo as sim_sumo
import sim_bug_tools.rng.lds.sequences as sequences

class Tiger:
    def __init__(self):
        seq_gen = sequences.HaltonSequence
        sim = sim_sumo.TrafficLightRace(seq_gen)
        return



if __name__ == "__main__":
    Tiger()