from sim_bug_tools.sumo import TraCIClient
import sim_bug_tools.utils as utils
import sim_bug_tools.structs as structs


import traci
import os
import re
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None

import warnings
import abc



FILE_DIR = os.path.dirname(os.path.abspath(__file__))


class ParameterManager(abc.ABC):
    @abc.abstractmethod
    def map_parameters(self, point : structs.Point):
        pass

    @abc.abstractproperty
    def parameter_summary(self) -> pd.DataFrame:
        pass




class TrafficLightRaceParameterManager(ParameterManager):
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
        

        # Parameter Summary
        self._param_summary = self._init_metadata()
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
    def param_summary(self) -> pd.DataFrame:
        return self._param_summary

    @property
    def tl_param_df(self) -> pd.DataFrame:
        return self._tl_param_df

    @property
    def veh_param_df(self) -> pd.DataFrame:
        return self._veh_param_df

    


    def ___PUBLIC_METHODS___(self):
        return

    def flatten(self, ls : list[list[any]]) -> list[any]:
        return [item for sublist in ls for item in sublist]

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

    def flatten_params_df(
            self, 
            veh_params_df : pd.DataFrame, 
            tl_params_df : pd.DataFrame
        ) -> pd.Series:
        """
        Flattens a @veh_params_df and @tl_params_df into a Series so that
        they may be concatenated easily.
        """
        warnings.simplefilter(action='ignore', category=FutureWarning)
        flat_veh_s = self._flatten_veh_params_df(veh_params_df)
        flat_tl_s = self._flatten_tl_params_df(tl_params_df)
        return flat_veh_s.append(flat_tl_s)



    def ___PRIVATE_METHODS___(self):
        return

    def _flatten_tl_params_df(self, df : pd.DataFrame) -> pd.Series:
        """
        Flattens a TL parameters @df into a series.
        """
        data = {}
        for i in range(len(df.index)):
            data["TL_%s" % df["state"].iloc[i]] = df["dur"].iloc[i]
        return pd.Series(data)

    def _flatten_veh_params_df(self, df : pd.DataFrame) -> pd.Series:
        """
        Flattens a vehicle parameters @df into a series.
        """
        features = df.columns.to_list()[:-1]
        data = {}
        for i in range(len(df.index)):
            vid = df["veh_id"].iloc[i]
            for feat in features:
                data["AV%s_%s" % (vid, feat)] = df[feat].iloc[i]
            continue
        return pd.Series(data)

    def _init_metadata(self):
        point = structs.Point(np.zeros(self.n_dim))
        params = self.map_parameters(point)
        flat_params = self.flatten_params_df(
            params["veh"]["concrete"], params["tl"]["concrete"])
        index = flat_params.index

        # Granularity
        inc = self.flatten([self.veh_param_df["inc"] \
            for n in range(self.n_av)] + [self.tl_param_df["inc"]])

        normalize = lambda df : list(df["inc"] / (df["max"] - df["min"]))
        inc_norm = self.flatten([normalize(self.veh_param_df) \
            for n in range(self.n_av)] + [normalize(self.tl_param_df)])

        feat_min = self.flatten([self.veh_param_df["min"] \
            for n in range(self.n_av)] + [self.tl_param_df["min"]])

        feat_max = self.flatten([self.veh_param_df["max"] \
            for n in range(self.n_av)] + [self.tl_param_df["max"]])

        df = pd.DataFrame({
            "feat" : index,
            "min" : feat_min,
            "max" : feat_max,
            "inc" : inc,
            "inc_norm" : inc_norm
        })

        return df

    def _map_vehicle_parameters(self, normal_values : dict) -> dict:
        """
        Maps @normal values to concrete values, as defined in the veh_params_df
            length
            width
            min_gap
            accel
            decel
            max_speed
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
        """
        Maps @normal values to concrete values as deined in the tl_params_df
            G
            Y
            R
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
    def __init__(self, veh_params : pd.DataFrame, tl_params : pd.DataFrame):
        # SUMO configuration
        self._map_dir = "%s" % FILE_DIR
        self._error_log_fn = "%s/error-log.txt" % self.map_dir
        self._config = {
            "gui" : False,
            # "gui" : True,

            # Street network
            "--net-file" : "%s/tl-race.net.xml" % self.map_dir,

            # Logging
            "--error-log" : self.error_log_fn,
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

        # Configure Traffic Light
        self._tl_param_df = tl_params
        self._init_traffic_light()

        # Add the route
        traci.route.add(self.route_id, ["before_tl","after_tl"])

        # Add vehicles to the route
        self._veh_param_df = veh_params
        self._add_all_veh()

        # Run Simuations
        while traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()
            continue

        # Close the Client
        self.client.close()

        # Score the performance of the test.
        self._veh_score_df = self._calc_veh_scores()
        self._scores = self.veh_score_df.apply(max) \
            .drop("veh_id")
        return

    def ___FEATURES___(self):
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
    def error_log_fn(self) -> str:
        return self._error_log_fn

    @property
    def map_dir(self) -> str:
        return self._map_dir

    @property
    def route_id(self) -> str:
        """
        Route ID
        """
        return "r0"

    @property
    def scores(self) -> pd.Series:
        return self._scores

    @property
    def tl_param_df(self) -> pd.DataFrame:
        return self._tl_param_df

    @property
    def veh_param_df(self) -> pd.DataFrame:
        return self._veh_param_df

    @property
    def veh_score_df(self) -> pd.DataFrame:
        return self._veh_score_df

    



    def ___PRIVATE_METHODS____(self):
        return
    
    def _add_all_veh(self) :
        """
        Adds all vehicles to the simulation.
        """
        for i in range(len(self.veh_param_df.index)):
            self._add_veh(self.veh_param_df.iloc[i])
        return


    def _add_veh(self, veh_params : pd.Series):
        """
        Adds a vehicle to the client, then adjust the vehicle to specs in
        @veh_params 
        """
        # Add a vehicle to the sim
        vid = str(int(veh_params["veh_id"]))
        traci.vehicle.add(vid, self.route_id)
    
        # Set the values for the vehicle
        setters_map = {
            "length" : traci.vehicle.setLength,
            "width" : traci.vehicle.setWidth,
            "min_gap" : traci.vehicle.setMinGap,
            "accel" : traci.vehicle.setAccel,
            "decel" : traci.vehicle.setDecel,
            "max_speed" : traci.vehicle.setMaxSpeed
        }

        # Get feature name, minus veh_id
        features = self.veh_param_df.columns.tolist()[:-1]

        # Set vehicle attributes
        [setters_map[feat](vid, veh_params[feat]) for feat in features]

        # Set emergency decel to 2x decel
        traci.vehicle.setEmergencyDecel(vid, 2 * veh_params["decel"])
        return


    def _init_traffic_light(self):
        """
        Initializes the phase durations of the traffic light. 
        """
        # Configure phases
        phases = [ traci.trafficlight.Phase(
            duration = self.tl_param_df["dur"].iloc[i],
            state = self.tl_param_df["state"].iloc[i],
            minDur = self.tl_param_df["dur"].iloc[i],
            maxDur = self.tl_param_df["dur"].iloc[i] 
        ) for i in range(len(self.tl_param_df.index))]

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
        return

    def _verify_veh_params(self, vid : str):
        """
        A helper method used to verify that the veh params are set correctly.
        This method is not called during a simulation test
        """
        # Traci Getters
        getters = {
            "length" : traci.vehicle.getLength,
            "width" : traci.vehicle.getWidth,
            "min_gap" : traci.vehicle.getMinGap,
            "accel" : traci.vehicle.getAccel,
            "decel" : traci.vehicle.getDecel,
            "max_speed" : traci.vehicle.getMaxSpeed
        }

        # Get the values from the concrete value dataframe
        series = self.veh_param_df[
            self.veh_param_df["veh_id"] == float(vid)
        ].iloc[0]
        
        # Compare each value
        print()
        print("VEH", vid)
        for key, val in getters.items():
            print( "%8s %5.3f %5.3f" % (key, val(vid), series[key]))
        print( "%8s %5.3f %5.3f" % ("E Decel", 
            traci.vehicle.getEmergencyDecel(vid), 
            series["decel"] * 2))
        print()
        return


    def __parse_veh_id(self, err : str) -> str:
        return re.findall(r"vehicle '\d'", err.lower())[0].strip("'").split("'")[-1]


    def _calc_veh_scores(self) -> pd.DataFrame:
        """
        Calculate the test score of this scenario for each vehicle by parsing
        the error log.

        Returns a pandas Dataframe of vehicle scores.
        """
        # There are 3 performance metrics. 
        # First, the default score is assigned to each vehicle.
        df = pd.DataFrame({
            "veh_id" : self.veh_param_df["veh_id"].astype(str),
            "jam" : 0.,
            "e_brake" : 0.,
            "e_stop" : 0.,
            "collision" : 0.
        })

        # Parse the error log to fill in the vehicle scores
        errors = open(self.error_log_fn).readlines()
        for err in errors:
            # Parse the vehicle ID
            vid = self.__parse_veh_id(err)
            
            # Get the index of the veh id in the score dataframe
            i = df[df["veh_id"] == vid].index[0]

            # Determine the error
            if "(jam)" in err:
                df["jam"].iloc[i] = 1.

            elif "because of a red traffic light" in err:
                df["e_stop"].iloc[i] = 1.

            elif "performs emergency braking" in err:
                wished = float(re.findall(r"wished=-*\d*\.*\d*", err)[0]\
                    .split("=")[-1])
                observed = float(re.findall(r"decel=-*\d*\.*\d*", err)[0]\
                    .split("=")[-1])
                df["e_brake"].iloc[i] = abs(wished/observed)

            elif "collision with vehicle" in err:
                observed = float(re.findall(r"gap=-*\d*\.*\d*", err)[0]\
                    .split("=")[-1])
                min_gap = self.veh_param_df["min_gap"].iloc[i]
                df["collision"].iloc[i] = abs(observed/min_gap)
            continue

        return df

    