from sim_bug_tools.sumo import TraCIClient
import sim_bug_tools.utils as utils

import traci
import os
import pandas as pd
import numpy as np

FILE_DIR = os.path.dirname(os.path.abspath(__file__))

class RegimeSUMO:
    def __init__(self):
        errors = []
        for i in range(1000):
            test = TrafficLightRace()
            errors.append(":::TEST_%d:::\n" % i)
            [errors.append(msg)  for msg in test.errors]
        with open("%s/hhh.txt" % FILE_DIR, "w") as f:
            [f.write(err) for err in errors]
        return
    

class TrafficLightRace:
    def __init__(self):
        # SUMO configuration
        map_dir = "%s" % FILE_DIR
        self._config = {
            "gui" : False,
            # "gui" : True,

            # Street network
            "--net-file" : "%s/tl-race.net.xml" % map_dir,

            # Logging
            "--error-log" : "%s/error-log.txt" % map_dir,
            "--log" : "%s/log.txt" % map_dir,

            # Traci Connection
            "--num-clients" : 1,
            "--remote-port" : 5522,

            # GUI Options
            "--delay" : 100,
            "--start" : "--quit-on-end",

            # RNG
            "--seed" : 333
        }

        # Start Client
        self._client = TraCIClient(self.config)

        # Add the route
        traci.route.add(self.route_id, ["before_tl","after_tl"])


        # Add vehicles
        self._veh_param_df = pd.read_excel(
            "%s/parameters.ods" % FILE_DIR, 
            engine="odf", 
            sheet_name="vehicle"
        )
        self._n_vehicles_added = 0

        for i in range(10):
            normal_values = dict()
            for feat in self.veh_param_df["feature"]:
                normal_values[feat] = np.random.rand()
            self._add_veh(normal_values)


        # Initialize the Traffic Signals
        self._tl_param_df = pd.read_excel(
            "%s/parameters.ods" % FILE_DIR,
            engine = "odf",
            sheet_name = "traffic signal"
        )

        normal_values = dict()
        for feat in self.tl_param_df["feature"]:
            normal_values[feat] = np.random.rand()
        self._init_traffic_signal(normal_values)
        
        
        # Run Simuations
        while traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()
            continue

        # Close client
        self.client.close()

        # Read the error log
        with open("%s/error-log.txt" % map_dir) as f:
            self._errors = f.readlines()
        return

    @property
    def config(self) -> dict:
        """
        SUMO Configuration setting.
        """
        return self._config

    @property
    def client(self) -> TraCIClient:
        """
        TraCI CLient
        """
        return self._client

    @property
    def errors(self) -> list[str]:
        return self._errors

    @property
    def n_vehicles_added(self) -> int:
        return self._n_vehicles_added

    @property
    def route_id(self) -> str:
        """
        Route ID
        """
        return "r0"

    @property
    def tl_param_df(self) -> pd.DataFrame:
        return self._tl_param_df

    @property
    def vehicle_id_prefix(self) -> str:
        """Vehicle prefix ID"""
        return "veh_"

    @property
    def veh_param_df(self) -> pd.DataFrame:
        return self._veh_param_df
    

    def _add_veh(self, normal_values : dict) -> dict:
        """
        Adds a vehicle to the client. 
        @normal value is a hash-map of vehicle parameter-->normal value
        """
        # There should be the same keys and values in norm_val as the vehicle
        # feature configuration file
        assert(len(normal_values) == len(self.veh_param_df.index))

        # The keys should match the values in the dataframe
        for key in normal_values.keys():
             assert any(self.veh_param_df["feature"].str.contains(key))

        # The values should be between 0 and 1.
        for val in normal_values.values():
            assert val >= 0 and val <= 1

        # SUMO Vehicle ID
        vid = "%s%d" % (self.vehicle_id_prefix, self.n_vehicles_added)
        self._n_vehicles_added += 1

        # Add the vehicle to the sumo sim
        traci.vehicle.add(vid, self.route_id)

        # Select concrete values.
        concrete_values = dict()
        for i in range(len(self.veh_param_df.index)):
            row = self.veh_param_df.iloc[i]    
            val = utils.project(
                a = row["min"],
                b = row["max"],
                n = normal_values[row["feature"]],
                by = row["inc"]
            )
            # print("%8s: %6.2f %s" %  (row["feature"], val, row["uom"]))
            concrete_values[row["feature"]] = val
            continue
    
        # Set the values for the vehicle
        setters_map = {
            "length" : traci.vehicle.setLength,
            "width" : traci.vehicle.setWidth,
            "min_gap" : traci.vehicle.setMinGap,
            "accel" : traci.vehicle.setAccel,
            "decel" : traci.vehicle.setDecel,
            "max_speed" : traci.vehicle.setMaxSpeed
        }
        [setters_map[key](vid, val) for key, val in concrete_values.items()]

        return concrete_values


    def _init_traffic_signal(self, normal_values : dict) -> dict:
        """
        Initializes the phase durations of the traffic signal. 
        @normal value is a hash-map of traffic_signal parameter-->normal value
        """
        # There should be the same keys and values in norm_val as the vehicle
        # feature configuration file
        assert(len(normal_values) == len(self.tl_param_df.index))

        # The keys should match the values in the dataframe
        for key in normal_values.keys():
             assert any(self.tl_param_df["feature"].str.contains(key))

        # The values should be between 0 and 1.
        for val in normal_values.values():
            assert val >= 0 and val <= 1

        # Get concrete values
        concrete_values = dict()
        for i in range(len(self.tl_param_df.index)):
            row = self.tl_param_df.iloc[i]
            val = utils.project(
                a = row["min"],
                b = row["max"],
                n = normal_values[row["feature"]],
                by = row["inc"]
            )
            concrete_values[row["feature"]] = val
            continue

        # Configure phases
        states = self.tl_param_df["state"].tolist()
        phases = [ traci.trafficlight.Phase(
            duration = dur,
            state = states[i],
            minDur = dur,
            maxDur = dur 
        ) for i, dur in enumerate(concrete_values.values())]

        # Set the program logic.
        tlsid = traci.trafficlight.getIDList()[0]
        traci.trafficlight.setProgramLogic(
            tlsid,
            traci.trafficlight.Logic(
                "prg0",
                type = 0,
                currentPhaseIndex = 0,
                phases = phases
            )
        )

        return concrete_values
    

if __name__ == "__main__":
    RegimeSUMO()