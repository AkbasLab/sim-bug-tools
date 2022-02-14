import shutil
import warnings
if shutil.which("sumo") is None:
    warnings.warn("Cannot find sumo/tools in the system path. Please verify that the lastest SUMO is installed from https://www.eclipse.org/sumo/")

import sim_bug_tools.simulators.base as sim_base
import sim_bug_tools.rng.lds.sequences as sequences
import sim_bug_tools.utils as utils
import sim_bug_tools.constants as constants
import sim_bug_tools.structs as structs
import traci
import pandas as pd
import numpy as np

class TraCIClient:
    def __init__(self, config : dict, priority : int = 1):
        """
        Barebones TraCI client.

        --- Parameters ---
        priority : int
            Priority of clients. MUST BE UNIQUE
        config : dict
            SUMO arguments stored as a python dictionary.
        """
        
        self._config = config
        self._priority = priority
        

        self.connect()
        return

    @property
    def priority(self) -> int:
        """
        Priority of TraCI client.
        """
        return self._priority

    @property
    def config(self) -> dict:
        """
        SUMO arguments stored as a python dictionary.
        """
        return self._config

    def run_to_end(self):
        """
        Runs the client until the end.
        """
        while traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()
            # more traci commands
        return

    def close(self):
        """
        Closes the client.
        """
        traci.close()
        return


    def connect(self):
        """
        Start or initialize the TraCI connection.
        """
        warnings.simplefilter("ignore", ResourceWarning)
        # Start the traci server with the first client
        if self.priority == 1:
            cmd = []

            for key, val in self.config.items():
                if key == "gui":
                    sumo = "sumo"
                    if val: sumo +="-gui"
                    cmd.append(sumo)
                    continue
                
                if key == "--remote-port":
                    continue

                cmd.append(key)
                cmd.append(str(val))
                continue

            traci.start(cmd,port=self.config["--remote-port"])
            traci.setOrder(self.priority)
            return
        
        # Initialize every client after the first.
        traci.init(port=self.config["--remote-port"])
        traci.setOrder(self.priority)
        return    

    


class TrafficLightRace(sim_base.Simulator):

    def __init__(self, sequence_generator : sequences.Sequence, **kwargs):
        super().__init__(**kwargs)

        map_dir = "sumo/tl_race"
        config = {
            "gui" : False,
            # "gui" : True,

            # Street network
            "--net-file" : "%s/tl-race.net.xml" % map_dir,

            # Route files
            # "--route-files" : "%s/default.rou.xml" % map_dir,

            # Logging
            "--error-log" : "%s/error-log.txt" % map_dir,

            # Traci Connection
            "--num-clients" : 1,
            "--remote-port" : 5522,

            # GUI Options
            "--delay" : 100,
            "--start" : "--quit-on-end",

            # RNG
            "--seed" : 333
        }
        self._client = TraCIClient(config)
        self._tlid = traci.trafficlight.getIDList()[0]
        

        # Setup sequence
        self._n_parameters = self.n_vehicles * 7
        self._seq = sequence_generator(
            domain = [(0,1) for n in range(self.n_parameters)],
            axes_names = ["dim%d" % n for n in range(self.n_parameters)]
        )

        # Add the route
        traci.route.add(self.route_id, ["before_tl","after_tl"])

        # Add vehicles
        self._vehicles = []
        self._n_vehicles_added = 0

        # Add Vehicles
        point = self.seq.get_points(1)[0]
        self.add_vehicles(point)
        
        # Traffic Light
        print(
            traci.trafficlight.getCompleteRedYellowGreenDefinition(self.tlid)
        )


        # TODO: Remove later
        self.client.run_to_end()
        self.client.close()
        return

    @property
    def client(self) -> TraCIClient:
        """
        TraCI CLient
        """
        return self._client

    @property
    def n_vehicles(self) -> int:
        """
        Number of vehicles in the race.
        """
        return 10

    @property
    def n_parameters(self) -> int:
        """
        Number of input parameters
        """
        return self._n_parameters

    @property
    def n_dim(self) -> int:
        """
        Alias for n_parameters.
        """
        return self._n_parameters

    @property
    def vehicle_id_prefix(self) -> str:
        """Vehicle prefix ID"""
        return "veh_"

    @property
    def route_id(self) -> str:
        """
        Route ID
        """
        return "r0"

    @property
    def seq(self) -> sequences.Sequence:
        """
        Sequence for sampling points
        """
        return self._seq

    @property
    def n_vehicles_added(self) -> int:
        """
        Number of vehicles added to simulation.
        """
        return self._n_vehicles_added

    @property
    def vehicles(self) -> list[str]:
        """
        List of vehicle ids in the sim.
        """
        return self._vehicles

    @property
    def tlid(self) -> str:
        """
        TraCI Traffic Light ID
        """
        return self._tlid


    def add_vehicles(self, point : structs.Point):
        n_veh_params = 7
        [ self.add_veh(point[i*n_veh_params : i*n_veh_params+n_veh_params]) \
            for i in range(self.n_vehicles) ]
        return
    
    def add_veh(self, point : np.ndarray ):
        """
        Add a vehicle

        --- Parameters --
        point : np.ndarray
            normal array of size (1,7)
        """
        vid = "%s_%d" % (self.vehicle_id_prefix, self.n_vehicles_added)
        self.vehicles.append(vid)
        self._n_vehicles_added += 1

        traci.vehicle.add(vid, self.route_id)

        setters = [
            traci.vehicle.setLength,
            traci.vehicle.setWidth,
            traci.vehicle.setMinGap,
            traci.vehicle.setAccel,
            traci.vehicle.setDecel,
            traci.vehicle.setEmergencyDecel,
            traci.vehicle.setMaxSpeed
        ]

        attributes = [
            [.215, 16.5], [.478, 2.55], [.25, 2.5],
            [1.1, 6.], [2., 10.], [5., 10.], [5.4, 200]
        ]

        concrete_values = [utils.project(*attrib, point[i]) \
            for i, attrib in enumerate(attributes)]

        [setters[i](vid, val) for i, val in enumerate(concrete_values)]
        return




    