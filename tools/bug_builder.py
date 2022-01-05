import sim_bug_tools.rng.bugger as bugger
import sim_bug_tools.structs as structs
import sim_bug_tools.rng.lds.sequences as sequences

import PySimpleGUI as sg
from random import Random
import numpy as np
import json

class BugGeneratorWindow:

    NUM_CHARS = "0123456789."

    def __init__(self):
        sg.theme("Default1")

        layout = [
            [sg.T("Seed", s=20), sg.InputText(default_text="0",s=10, key="seed")],
            [sg.T("Number of dimensions.", s=20), 
                sg.InputText(s=10, key="n_dim")],
            [sg.T("Number of bug clusters.", s=20), 
                sg.InputText(s=10, key="n_clusters")],
            [sg.T("Cluster size.", s=20), 
                sg.T("Min."), sg.InputText(s=5,key="cluster_size_min"), 
                sg.T("Max."), sg.InputText(s=5, key="cluster_size_max")],
            [sg.T("Number of profiles.",s=20), sg.InputText(s=10, key="n_profiles")],
            [sg.T("Output filename.", s=20), sg.Input(key="output_fn"), 
                sg.FileSaveAs(default_extension=".json", 
                    file_types=[("JSON files", "*.json")])],
            [sg.Button("Generate")],
            [sg.ProgressBar(1, orientation="h", s=(50,20), key="prog")],
            [sg.Output(s=(75,5), key="output", )]
        ]

        self.window = sg.Window("Bug Builder", layout)

        while True:
            event, values = self.window.read()
            if event == "Generate":
                # Check input
                # TODO
                try:
                    self.check_input(values)
                except ValueError:
                    continue

                # Finally generate the bugs
                self.generate_bugs(values)
            if event == sg.WIN_CLOSED or event == 'Cancel': # if user closes window or clicks cancel
                break

        self.window.close()
        return

    def check_input(self, values : dict) -> bool:
        flag = True
        if not isinstance(self.get_int(values["seed"]), int):
            print("ERR: Seed must be of type int.")
            flag = False
        
        n_dim = self.get_int(values["n_dim"])
        if not all([n_dim >= 1, isinstance(n_dim, int)]):
            print("ERR: Dimensions should be >= 1 and by of type int.")
            flag = False

        n_clusters = self.get_int(values["n_clusters"])
        if not all([n_clusters >= 1, isinstance(n_clusters, int)]):
            print("ERR: # Clusters should be >= 1 and by of type int.")
            flag = False

        cluster_size_min = self.get_float(values["cluster_size_min"])
        cluster_size_max = self.get_float(values["cluster_size_max"])
        if not all([
            cluster_size_min > 0., 
            cluster_size_min <= 1,
            cluster_size_min <= cluster_size_max,
            isinstance(cluster_size_min, float), 
        ]):
            print("ERR: Min cluster size should be 1 >= x > 0 \
                and by of type float and <= max cluster size.")
            flag = False
        if not all([
            cluster_size_max > 0., 
            cluster_size_max <= 1,
            cluster_size_max >= cluster_size_min,
            isinstance(cluster_size_max, float), 
        ]):
            print("ERR: Max cluster size should be 1 >= x > 0 \
                and by of type float and >= min cluster size.")
            flag = False

        n_profiles = self.get_int(values["n_profiles"])
        if not all( [n_profiles > 0, isinstance(n_profiles, int)] ):
            print("ERR: n_profiles must be > 0 and be of type int.")
            flag = False

        if not values["output_fn"].strip()[-5:] == ".json":
            print("ERR: Output file must end in .json")
            flag = False

        return flag

    def get_float(self, s : str) -> float:
        return float("".join([x for x in s if (x in self.NUM_CHARS)]))
    
    def get_int(self, s : str) -> int:
        return int(self.get_float(s))



    def random_size_domain(self, n_dim : int, a : float, b : float, random : Random):
        return structs.Domain( 
            np.round( [(0, random.uniform(a, b,)) for _ in range(n_dim)], decimals=2)  
        ) 

    def generate_bugs(self, values : dict):
        print("Generating profiles.")

        n_dim = int(values["n_dim"].strip())
        cluster_size_min = float(values["cluster_size_min"].strip())
        cluster_size_max = float(values["cluster_size_max"].strip())
        seed = int(values["seed"].strip())
        n_profiles = int(values["n_profiles"].strip())

        # The domain is a normal dommain
        normal_domain = structs.Domain([(0,1) for _ in range(n_dim)])
        random = Random(seed) 

        # We need to use the random sequence or it will conflict with our results.
        seq = sequences.RandomSequence(normal_domain, 
            ["dim_%d" % (i) for i in range(n_dim)])    
        seq.seed = seed

        # Construct the bug builder object
        bug_builder = bugger.BugBuilder(
                location_domain = normal_domain,
                size_domain = self.random_size_domain(
                    n_dim, cluster_size_min, cluster_size_max, random),
                sequence = seq,
                random = random
            )

        # Generate the profiles
        profiles = []
        for i_profile in range(n_profiles):
            clusters : list[structs.Domain] = []
            for i_bug in range(int(values["n_clusters"].strip())):
                # Generate a cluster
                clusters.append( bug_builder.build_bug().as_json() )

                # Scramble the size of the next cluster.
                bug_builder.size_domain = self.random_size_domain(
                    n_dim, cluster_size_min, cluster_size_max, random)
                continue
            profiles.append(clusters)
            self.window["prog"].update((i_profile)/float(n_profiles))
            continue

        #  Dump to file
        with open(values["output_fn"].strip(), "w") as f:
            f.write(json.dumps(profiles)) 
        
        print("Profiles saved to \"%s\"." % values["output_fn"])

        self.window["prog"].update(1)
        return


def main():
    BugGeneratorWindow()    
    return


if __name__ == "__main__":
    main()
