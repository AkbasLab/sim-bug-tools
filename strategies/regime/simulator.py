from sim_bug_tools.sumo import TraCIClient
import sim_bug_tools.utils as utils
import sim_bug_tools.structs as structs

import errors
import simulator

import traci
import os
import pandas as pd
import numpy as np
import time

FILE_DIR = os.path.dirname(os.path.abspath(__file__))


class TrafficLightRaceParameterManager:
    def __init__(self):
        """
        This class has helper functions to manage the logical parameters for
        the TL-Race logical scenario.
        """
        # Vehicle Parameter Ranges
        self._veh_param_df = pd.read_excel(
            "%s/parameters.ods" % FILE_DIR, 
            engine="odf", 
            sheet_name="vehicle"
        )

        # Traffic Light Parameter Ranges
        self._tl_param_df = pd.read_excel(
            "%s/parameters.ods" % FILE_DIR,
            engine = "odf",
            sheet_name = "traffic signal"
        )

        # Create a normal Domain
        self._n_av = 10
        self._n_dim = self.n_av * len(self.veh_param_df.index) \
            + len(self.tl_param_df.index)
        self._domain = structs.Domain.normalized(self.n_dim)

        # Axis Names
        self._axes_names = []
        for n in range(self.n_av):
            for feat in self.veh_param_df["feature"]:
                self.axes_names.append("AV%d %s" % (n, feat))
        for feat in self.tl_param_df["feature"]:
            self.axes_names.append("TL %s" % feat)
        return



    def ___PARAMETERS___(self):
        return

    @property
    def axes_names(self) -> list[str]:
        return self._axes_names

    @property
    def domain(self) -> structs.Domain:
        return self._domain

    @property
    def n_av(self) -> int:
        return self._n_av

    @property
    def n_dim(self) -> int:
        return self._n_dim

    @property
    def tl_param_df(self) -> pd.DataFrame:
        return self._tl_param_df

    @property
    def veh_param_df(self) -> pd.DataFrame:
        return self._veh_param_df

    


    def ___PUBLIC_METHODS___(self):
        return

    def map_parameters(self, point : structs.Point) -> dict:
        """
        Maps a normal @point to concrete parameters.
        Produces 4 dataframes for vehicle and TL parameters in the following
         datastructure
            + veh
            | + normal
            | + concrete
            + tl
              + normal
              + concrete
        """
        assert len(point) == self.n_dim

        # Generate Concrete Vehicle Values
        all_concrete_values = []
        all_normal_values = []
        n_features = len(self.veh_param_df.index)
        for n in range(self.n_av):
            # Construct the hash-map data structure
            normal_values = dict()
            for i, feat in enumerate(self.veh_param_df["feature"]):
                val = point[n*n_features + i]
                normal_values[feat] = val

            # Get Concrete values for each AV
            concrete_values = self._map_vehicle_parameters(normal_values)
            
            # Add the AV id. 
            concrete_values["veh_id"] = n
            normal_values["veh_id"] = n

            # Add to the list
            all_concrete_values.append(concrete_values)
            all_normal_values.append(normal_values)
            continue

        # Combine into a dataframe
        veh_concrete_df = pd.DataFrame(all_concrete_values)
        veh_normal_df = pd.DataFrame(all_normal_values)


        # Generate Concrete TL Values
        # The 3 phases of the TL are determined by the last 3 values
        p = point[-3:]
        normal_values = dict()
        for i,feat in enumerate(self.tl_param_df["feature"]):
            normal_values[feat] = p[i]
        concrete_values = self._map_tl_parameters(normal_values)
        
        # Cobine into a dataframe
        tl_concrete_df = pd.DataFrame(concrete_values)
        tl_normal_df = pd.DataFrame({
            "state" : tl_concrete_df["state"],
            "dur" : p
        })

        return {
            "veh" : {
                "normal" : veh_normal_df,
                "concrete" : veh_concrete_df
            },
            "tl" : {
                "normal" : tl_normal_df,
                "concrete" : tl_concrete_df
            }
        }




    def ___PRIVATE_METHODS___(self):
        return

    
    def _map_vehicle_parameters(self, normal_values : dict) -> dict:
        # There should be the same keys and values in norm_val as the vehicle
        # feature configuration file
        assert(len(normal_values) == len(self.veh_param_df.index))

        # The keys should match the values in the dataframe
        for key in normal_values.keys():
             assert any(self.veh_param_df["feature"].str.contains(key))

        # The values should be between 0 and 1.
        for val in normal_values.values():
            assert val >= 0 and val <= 1

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
            concrete_values[row["feature"]] = val

        return concrete_values

    def _map_tl_parameters(self, normal_values : dict) -> dict:
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
        concrete_values = []
        for i in range(len(self.tl_param_df.index)):
            row = self.tl_param_df.iloc[i]
            val = utils.project(
                a = row["min"],
                b = row["max"],
                n = normal_values[row["feature"]],
                by = row["inc"]
            )
            concrete_values.append({
                "state" : row["state"],
                "dur" : val
            })
            continue

        return concrete_values






    
class TrafficLightRaceTest:
    def __init__(self):
        # SUMO configuration
        self._map_dir = "%s" % FILE_DIR
        self._config = {
            "gui" : False,
            # "gui" : True,

            # Street network
            "--net-file" : "%s/tl-race.net.xml" % self.map_dir,

            # Logging
            "--error-log" : "%s/error-log.txt" % self.map_dir,
            # "--log" : "%s/log.txt" % map_dir,

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

        self.client.close()
        # self._add_all_veh()
        return

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

        # Construct the error dataframe
        self._categorize_errors()
        return

    def ___FEATURES___(self):
        return


    @property
    def concrete_values_df(self) -> pd.DataFrame:
        return self._concrete_values_df


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
    def errors(self) -> pd.DataFrame:
        return self._errors

    @property
    def map_dir(self) -> str:
        return self._map_dir

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

    



    def ___PRIVATE_METHODS____(self):
        return
    
    def _add_all_veh(self) :
        # Lists to hold values
        all_normal_values = []
        all_concrete_values = []

        # Add 10 AVs
        for i in range(10):

            # Select normal values
            normal_values = dict()
            for feat in self.veh_param_df["feature"]:
                normal_values[feat] = np.random.rand()

            # Select Concrete values and add veh to sim.
            concrete_values = self._add_veh(normal_values)

            # Add the unique VID to the dictionary
            vid = "%s%d" % (self.vehicle_id_prefix, i)
            normal_values["veh_id"] = vid
            concrete_values["veh_id"] = vid

            # Add the value dictionaries to their list
            all_normal_values.append(normal_values)
            all_concrete_values.append(concrete_values)
            continue

        # Generate normal + concrete dataframes


        return


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
    

    def _categorize_errors(self):
        # Read from file
        with open("%s/error-log.txt" % self.map_dir) as f:
            errors = f.readlines() 
        
        for err in errors:
            return
        return


    def _verify_concrete_values(self, concrete_values : pd.DataFrame):
        veh_ids = ["%s%d" % (self.vehicle_id_prefix,i) for i in range(10)]

        return