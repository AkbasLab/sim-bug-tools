"""
For development
Add sim_bug_tools to the python path.
"""
# import os, sys
# sys.path.insert(0, "D:\\git-projects\\sim-bug-tools\\src\\sim_bug_tools")


import PySimpleGUI as sg
import pathlib
import sim_bug_tools.structs as structs
import sim_bug_tools.rng.bugger as bugger
import sim_bug_tools.rng.lds.sequences as sequences
import sim_bug_tools.utils as utils
import sim_bug_tools.simulators as simulators

from sim_bug_tools.rng.rrt import RapidlyExploringRandomTree


print()

class DashboardWindow:

    _window = None
    _values = None
    _bug_profiles = None
    _n_dim = None
    _domain = None
    _seq = None
    _axis_name = None
    _rrt = None
    _simulators = None

    def __init__(self):
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

    @property
    def axis_names(self) -> list[str]:
        return self._axis_names

    @property
    def rrt(self) -> RapidlyExploringRandomTree:
        return self._rrt

    @property
    def simulators(self) -> list[simulators.Simulator]:
        return self._simulators

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
            [sg.T("# Branches."), sg.InputText(s=10, key="RRT:n_branches")],
            [sg.T("Seed."), sg.InputText(s=10, key="RRT:seed", default_text="0")]
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
            [
                sg.Button("New", k="new"), sg.Button("Run", k="run", disabled=True),
                sg.InputText("5", k="n_runs", s=5)
            ],
            [sg.Column([[]], k="col_simulation_status")]
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

            self._values["bug_profiles_fn"] = "tools/test_bugs.json"
            self._values["RRT:n_branches"] = "5"
            self._values["RRT:branch_size"] = "0.01"
            self._values["LS:strategy"] = "RRT"
            # print(event)
            print(self.values)

            if self._check_input(self.values, event):
                if event == "new":
                    self._new_simulation()
                if event == "run":
                    self._run_simulation()
        self.window.close()
        return


    def _check_input(self, values : dict, event : str = "") -> bool:
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
        
        seed_rrt = values["RRT:seed"].strip()
        if seed_rrt in ["",None]:
            print("ERR: seed_rrt field is blank.")
            flag = False
        else:
            try: 
                seed_rrt = utils.parse_int(seed_rrt)  
            except IndexError:
                print("ERR: seed_rrt must be an int.")
                flag = False

        if event == "run":
            n_runs = values["n_runs"].strip()
            if n_runs in ["", None]:
                print("ERR:n_runs field is blank.")
                flag = False
            else:
                try: 
                    n_runs = utils.parse_int(n_runs)
                    if n_runs < 0:
                        print("ERR:n_runs must be >= 0.")
                        flag = False    
                except IndexError:
                    print("ERR:n_runs must be an int.")
                    flag = False

        return flag


    def _new_simulation(self):
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
        self._axis_names = ["dim_%d" % (n+1) for n in range(self.n_dim)]
        self._seq = seq(self.domain, self.axis_names)
        self._seq.seed = seed # For random sequence

        # Skip ahead
        n_skip_at_start = utils.parse_int( self.values["LW:n_skip_at_start"] )
        self.seq.get_points(n_skip_at_start)

        # Select Strategy
        strategy = self.values["LS:strategy"].lower()
        if strategy == "no local search.":
            # print("NO LOCAL SEARCH")
            self._simulators = [simulators.SimpleSimulatorKnownBugs(
                bug_profile = bug_profile,
                sequence = self.seq,
            ) for bug_profile in self.bug_profiles]
        elif strategy == "rrt":
            # print("RRT")
            seq_rrt = sequences.RandomSequence(self.domain, self.axis_names)
            seq_rrt.seed = utils.parse_int( self.values["RRT:seed"] )
            self._rrt = RapidlyExploringRandomTree(
                seq = seq_rrt,
                step_size = utils.parse_int( self.values["RRT:branch_size"] ),
                exploration_radius = 1
            )
            self._simulators = [simulators.SimpleSimulatorKnownBugsRRT(
                bug_profile = bug_profile,
                sequence = self.seq,
                rrt = self.rrt,
                n_branches = utils.parse_int( self.values["RRT:n_branches"] )
            ) for bug_profile in self.bug_profiles]

        # Set IDs for all simulators and transition into paused state.
        for i, sim in enumerate(self.simulators):
            sim.set_id("%d" % i)
            sim.paused()
            continue

        # Update simulator state
        msg = "".join(["%d: %s\n" % (i_sim, sim.state.value) for \
                i_sim, sim in enumerate(self.simulators)])[:-1]


        # # Add state and bars for all simulations
        # [self.window.extend_layout(
        #     self.window["col_simulation_status"],
        #     self._new_simulation_status_row_layout(i)
        # ) for i in range(len(self.bug_profiles))]


        # Disable new button
        self.window["new"].update(disabled=True)
        self.window["run"].update(disabled=False)
        return

    
    def _new_simulation_status_row_layout(self, i):
        return [[
            sg.T("%d: %s" % (i, simulators.State.NO_SIMULATOR_LOADED.value), 
                s=len(simulators.State.NO_SIMULATOR_LOADED.value), 
                k="SS:state_%d" % i),
            sg.ProgressBar(1, expand_x = True, orientation = "h",
                key="SS:progress_%d" % i)
        ]]


    def _run_simulation(self):
        n_runs = utils.parse_int(self.values["n_runs"])
        for sim in self.simulators:
            sim.run(n_runs)
            break
        return
    

def main():
    DashboardWindow()
    return

if __name__ == "__main__":
    main()

