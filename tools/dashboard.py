"""
For development
Add sim_bug_tools to the python path.
"""
import os, sys
sys.path.insert(0, "D:\\git-projects\\sim-bug-tools\\src\\sim_bug_tools")


import PySimpleGUI as sg
import pathlib
import sim_bug_tools.structs as structs
import sim_bug_tools.rng.bugger as bugger
import sim_bug_tools.rng.lds.sequences as sequences
import sim_bug_tools.utils as utils
import re

print()

class DashboardWindow:

    def __init__(self):
        # self._domain = structs.Domain([])

        self._init_window()
        self._window_loop()
        return

    @property
    def window(self) -> sg.Window:
        return self._window

    @property
    def values(self) -> dict:
        return self._values

    @property
    def bug_profiles(self) -> list[list[structs.Domain]]:
        return self._bug_profiles

    @property
    def n_dim(self) -> int:
        return self._n_dim

    @property
    def domain(self) -> structs.Domain:
        return self._domain

    @property
    def seq(self) -> sequences.Sequence:
        return self._seq

    def _init_window(self):
        """
        Intitialize the GUI window
        """
        sg.theme("Default1")

        tab_long_walk = [
            [sg.T("Sequence:"), 
            sg.Combo(["sobol", "faure", "halton", "random"],
                default_value="random", key="LW:sequence", readonly=True)],
            [sg.T("Seed:"), sg.InputText(s=10, default_text="0", key="LW:seed")],
            [sg.T("# Skip at start."), 
                sg.InputText(s=10, default_text="0", key="LW:n_skip_at_start")]
        ]

        tab_local_search = [
            [sg.T("Strategy"),
                sg.Combo(["No local search.", "RRT"], readonly=True,
                    default_value="No local search.", key="LS:strategy")],
            [sg.T("Branch size."), sg.InputText(s=10, key="RRT:branch_size")],
            [sg.T("# Branches."), sg.InputText(s=10, key="RRT:n_branches")]
        ]

        layout = [
            [ 
                sg.TabGroup([[
                    sg.Tab('Long Walk', tab_long_walk),
                    sg.Tab('Local Search', tab_local_search)
                ]],
                expand_x=True)
            ],
            [sg.T("Bug profiles:"), sg.In(key="bug_profiles_fn"),
                sg.FileBrowse(target="bug_profiles_fn", 
                file_types=([("JSON files", "*.json")]) )],
            [sg.Button("Tada")]
        ] 

        self._window = sg.Window("Dashboard", layout)
        return



    def _window_loop(self):
        """
        This is the window loop with end-user IO
        """
        while True:
            event, self._values = self.window.read()

            if event == sg.WIN_CLOSED:
                break

            self._values["bug_profiles_fn"] = "D:/git-projects/sim-bug-tools/tools/test_bugs.json"
            self._values["RRT:n_branches"] = "5"
            self._values["RRT:branch_size"] = "0.01"
            print(self.values)

            if self._check_input(self.values):
                self._initialize_simulation()
        self.window.close()
        return


    def _check_input(self, values : dict) -> bool:
        """
        Check the values/validates forms.
        """
        flag = True

        bug_profiles_fn = values["bug_profiles_fn"].strip()
        if bug_profiles_fn in ["",None]:
            print("ERR: Bug profile field is blank.")
            flag = False
        elif bug_profiles_fn[-5:] != ".json":
            print("ERR: Bug profile file must end in .json")
            flag = False
        elif not pathlib.Path(bug_profiles_fn).is_file():
            print("ERR: Bug profile file does not exist.")
            flag = False

        seed = values["LW:seed"].strip()
        if seed in ["",None]:
            print("ERR: Long Walk Seed field is blank.")
            flag = False
        else:
            try: 
                utils.parse_int(seed)
            except IndexError:
                print("ERR: Long Walk Seed must be an int.")
                flag = False

        n_skip_at_start = values["LW:n_skip_at_start"].strip()
        if n_skip_at_start in ["",None]:
            print("ERR: n_skip_at_start field is blank.")
            flag = False
        else:
            try: 
                n_skip_at_start = utils.parse_int(n_skip_at_start)
                if n_skip_at_start < 0:
                    print("ERR: n_skip_at_start must be >= 0.")
                    flag = False    
            except IndexError:
                print("ERR: n_skip_at_start must be an int.")
                flag = False

        branch_size = values["RRT:branch_size"].strip()
        if branch_size in ["",None]:
            print("ERR: branch_size field is blank.")
            flag = False
        else:
            try: 
                branch_size = utils.parse_float(branch_size)
                if any([branch_size < 0, branch_size > 1]):
                    print("ERR: branch_size must be  0 >= x >= 1.")
                    flag = False    
            except IndexError:
                print("ERR: branch_size must be a float.")
                flag = False

        n_branches = values["RRT:n_branches"].strip()
        if n_branches in ["",None]:
            print("ERR: n_branches field is blank.")
            flag = False
        else:
            try: 
                n_branches = utils.parse_int(n_branches)
                if n_branches < 0:
                    print("ERR: n_branches must be >= 0.")
                    flag = False    
            except IndexError:
                print("ERR: n_branches must be an int.")
                flag = False
        
        return flag


    def _initialize_simulation(self):
        # Load bugs
        print("Loading bug profiles from %s" % self.values["bug_profiles_fn"].strip())
        bug_profile_fn = self.values["bug_profiles_fn"].strip()
        self._bug_profiles = bugger.profiles_from_json(bug_profile_fn)

        # The number of dimensions is selected automatically from the bug profiles.
        self._n_dim = len(self.bug_profiles[0][0])
        print("# Dimensions:", self.n_dim)
        print("   # Cluster:", len(self.bug_profiles[0]))
        print("  # Profiles:", len(self.bug_profiles))
        print()

        # Create a normalized domain 
        self._domain = structs.Domain([(0,1) for _ in range(self.n_dim)])
        assert len(self.domain) == self.n_dim

        # Seed
        seed = utils.parse_int(self.values["LW:seed"])

        # Sequences
        choice = self.values["LW:sequence"]        
        if choice == "random":
            seq = sequences.RandomSequence
        elif choice == "halton":
            seq = sequences.HaltonSequence
        elif choice == "sobol":
            seq = sequences.SobolSequence
        elif choice == "faure":
            seq = sequences.FaureSequence
        else:
            raise Exception("Unsupported sequence > %s", choice)
        self._seq = seq(self.domain, ["dim_%d" % (n+1) for n in range(self.n_dim)])
        self._seq.seed = seed # For random sequence

        # Skip ahead
        n_skip_at_start = utils.parse_int( self.values["LW:n_skip_at_start"] )
        self.seq.get_points(n_skip_at_start)

        strategy = self.values["LS:strategy"].lower()
        if strategy == "no local search.":
            print("NO LOCAL SEARCH")
        elif strategy == "rrt":
            print("RRT")
        return

    

def main():
    DashboardWindow()
    return

if __name__ == "__main__":
    main()

