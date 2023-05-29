import pandas as pd

pd.options.mode.chained_assignment = None

import os

FILE_DIR = os.path.dirname(os.path.abspath(__file__))

import sim_bug_tools.structs as structs
import sim_bug_tools.utils as utils
from sim_bug_tools.sumo import TraCIClient
import traci
import re
import traceback
import numpy as np


def project(a: float, b: float, n: float, inc: float = None) -> float:
    """
    Project a normal val @n between @a and @b with an discretization
    increment @inc.
    """
    assert (
        n >= 0 and n <= 1
    ), "projecting normalized value n between a and b must be between 0 and 1"
    assert b >= a, "projecting between two values, a and b, a must be <= b"

    # If no increment is provided, return the projection
    if inc is None:
        return n * (b - a) + a

    # Otherwise, round to the nearest increment
    n_inc = (b - a) / inc

    x = np.round(n_inc * n)
    return min(a + x * inc, b)


class HighwayTrafficParameterManager:
    def __init__(self):
        """
        The parameter manager class is helper class which projects a normal
        Point to a usable input for the Highway Pass Test class.
        """
        self._params_df = pd.read_csv("%s\\highway-traffic\\params-ht.csv" % FILE_DIR)
        return

    @property
    def params_df(self) -> pd.DataFrame:
        return self._params_df

    def map_parameters(self, point: structs.Point) -> pd.Series:
        """
        Maps a normal @point to concrete parameters.
        Returns a pandas series
        """
        s = pd.Series(
            [
                project(
                    a=self.params_df.iloc[i]["min"],
                    b=self.params_df.iloc[i]["max"],
                    n=x,
                    inc=self.params_df.iloc[i]["inc"],
                )
                for i, x in enumerate(point)
            ],
            index=self.params_df["feature"],
        )
        return s


class HighwayTrafficTest:
    def __init__(self):
        """
        This class is a fully encapsulated black box scenario which performs a
        simple AV scenario in SUMO traffic simulator.

        concrete parameters --> run test --> score test

        -- Parameters --
        @param_s : pd.Series
            Concrete parameters for the black box scenario.
        """
        # SUMO configuration
        self._map_dir = "%s\\highway-traffic" % FILE_DIR
        self._error_log_fn = "%s\\error-log.txt" % self.map_dir
        self._config = {
            # "--no-warnings": "",
            # "--no-step-log": "",
            "gui": True,
            # "gui": True,
            # Street network
            "--net-file": "%s\\highway-traffic.net.xml" % self.map_dir,
            "-r": "%s\\highway-traffic.rou.xml" % self.map_dir,
            # Logging
            "--error-log": self.error_log_fn,
            # "--log" : "%s\\log.txt" % map_dir,
            # Traci Connection
            "--num-clients": 1,
            "--remote-port": 5522,
            # GUI Options
            "--delay": 100,
            # "--start" : "--quit-on-end",
            # RNG
            "--seed": 333,
            ## Lane Change Duration
            "--lanechange.duration": 1.1,
        }

        # Start Client
        self._client = TraCIClient(self.config)
        with open(self.error_log_fn, "r") as f:
            self._error_log_cache = f.readline()
        # Set speed limit
        # self._set_speed_limit()

    def run(self, params_s: pd.Series) -> pd.Series:
        self._params_s = params_s.copy()
        if self.client.is_cached:
            self._refresh_error_log()
            self.client.load_cached_state()
        # Add Vehicles
        try:
            self._add_all_vehicles()
        except Exception:
            traceback.print_exc()
            self.client.close()
            raise Exception("Error adding vehicles. Closed client.")

        # Run Simuations
        try:
            self.client.run_to_end()
        except Exception:
            traceback.print_exc()
            self.client.close()
            raise Exception("Error encountered when runnng simulation. Closed client.")

        # Scores
        self._scores_s = self._calc_scores()
        return self._scores_s

    def __enter__(self):
        self.client.cache_state()
        return self

    def __exit__(self):
        self.client.clear_cache()
        self.client.close()

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
    def params(self) -> pd.Series:
        return self._params_s

    @property
    def params_s(self) -> pd.Series:
        return self._params_s

    @property
    def route_id(self) -> str:
        """
        Route ID
        """
        return "r0"

    @property
    def scores(self) -> pd.Series:
        return self._scores_s

    @property
    def scores_s(self) -> pd.Series:
        return self._scores_s

    # @property
    # def speed_limit_kph(self) -> float:
    #     return self._speed_limit_kph

    def ___PUBLIC_METHODS___(self):
        return

    def kph2mps(self, kph: float):
        return kph / 3.6

    def ___PRIVATE_METHODS___(self):
        return

    def _set_speed_limit(self):
        traci.edge.setMaxSpeed("highway", self.kph2mps(self._params_s["speed_limit"]))
        return

    def _add_all_vehicles(self):
        # Prepare Route
        self._warmup_rid = "r:warmup"
        traci.route.add(self._warmup_rid, ["warmup", "highway"])

        # Add the DUT
        self._add_dut()

        self._add_npcs()
        return

    def _add_npcs(self):
        #  Prepare
        v_types = ["passenger", "motorcycle", "van", "truck", "semitrailer"]
        npc_data = []
        for lr in "lr":
            for i in range(1, 5 + 1):
                vid = "npc%s%d" % (lr, i)

                i_vtype = int(self.params_s["%s_vtype" % vid])
                if i_vtype > 4:
                    raise ValueError("Invalid vtype '%d'. Must be in [0,4]" % i_vtype)
                vtype = v_types[i_vtype]

                sosl = self.params_s["%s_sosl" % vid]
                speed = self.params_s["speed_limit"] + sosl

                try:
                    prev_init_disp = self.params_s["npc%s%d_init_disp" % (lr, i - 1)]
                except KeyError:
                    prev_init_disp = 0
                init_disp = self.params_s["%s_init_disp" % vid] + prev_init_disp

                lane = int(lr == "l")
                npc_data.append(
                    pd.Series(
                        {
                            "vid": vid,
                            "vtype": vtype,
                            "speed": speed,
                            "init_pos": init_disp,
                            "lane": lane,
                        }
                    )
                )
                continue
            continue
        df = pd.DataFrame(npc_data)
        # print(df)

        # Add NPCS
        for npc in npc_data:
            traci.vehicle.add(
                npc["vid"],
                self._warmup_rid,
                typeID=npc["vtype"],
                departSpeed=self.kph2mps(npc["speed"]),
            )
            traci.vehicle.setLaneChangeMode(npc["vid"], 512)
            traci.vehicle.setMaxSpeed(npc["vid"], npc["speed"])
            traci.vehicle.moveTo(
                npc["vid"], "highway_%d" % npc["lane"], npc["init_pos"]
            )
            continue

        return

    def _add_dut(self):
        dut_init_speed = self.kph2mps(
            self.params_s["dut_sosl"] + self.params_s["speed_limit"]
        )
        dut_init_pos = 0
        dut_e_decel = self.kph2mps(self.params_s["dut_e_decel"])
        traci.vehicle.add("dut", self._warmup_rid, departSpeed=dut_init_speed)
        traci.vehicle.setMaxSpeed("dut", dut_init_speed)
        traci.vehicle.setDecel("dut", dut_e_decel / 2)
        traci.vehicle.setEmergencyDecel("dut", dut_e_decel)
        traci.vehicle.moveTo("dut", "highway_0", dut_init_pos)
        traci.vehicle.setColor("dut", (0, 255, 255, 255))
        return

    def _refresh_error_log(self):
        with open(self.error_log_fn, "w") as f:
            f.writelines(self._error_log_cache)

    def _calc_scores(self):
        scores = pd.Series({"e_brake": 0, "collision": 0})

        # Parse log for scores
        errors = open(self.error_log_fn).readlines()

        # Filter out errors with DUT in them
        errors = [err for err in errors if "dut" in err]

        # Then parse
        for err in errors:
            if "collision" in err:
                scores["collision"] = 1
            elif ("emergency braking" in err) and (not "time=0.0" in err):
                if len(wished := re.findall(r"wished=-*\d*\.*\d*", err)) == 0:
                    continue
                elif len(observed := re.findall(r"decel=-*\d*\.*\d*", err)) == 0:
                    continue
                wished = float(wished[0].split("=")[-1])
                observed = float(observed[0].split("=")[-1])
                scores["e_brake"] = min(abs(wished / observed) * 2, 1)
            continue

        return scores

    def _verify(self):
        vids = traci.vehicle.getIDList()

        df = pd.DataFrame(
            {
                "vid": vids,
                "vtype": [traci.vehicle.getTypeID(vid) for vid in vids],
                "speed": [traci.vehicle.getSpeed(vid) * 3.6 for vid in vids],
                "pos": [traci.vehicle.getLanePosition(vid) for vid in vids],
                "lane": [traci.vehicle.getLaneID(vid) for vid in vids],
                "e_decel": [traci.vehicle.getEmergencyDecel(vid) * 3.6 for vid in vids],
            }
        )

        print(df)
        self.client.close()
        return
