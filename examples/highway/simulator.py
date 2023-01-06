"""
Simplistic, 3D collision logical scenario:
The intent here is to provide a scenario that can be easily visualized in 
3-dimensions. The intent here is to provide a intuitive example before getting 
too abstract. Since it involves collisions, people already have an understanding
of how each parameter might effect the 3D graph, thus allowing them to visually 
validate our results without much effort.

Scenario:
- There is a car (NPC) in front of the DUT on a single-lane highway (same lane)
- The parameters to the scenario are
    - Initial displacement between the cars
    - Relative velocity of the DUT to the NPC
    - Maximum breaking force of the DUT
- Some constants
    - The speed of the NPC
    - The highway track
    - The weather
    - etc.

Testing:
The target behavior is a near-collision or collision. Distance < 6"
"""
import re

import pandas as pd
import traci

import sim_bug_tools.structs as structs
import sim_bug_tools.utils as utils
from sim_bug_tools.sumo import TraCIClient

pd.options.mode.chained_assignment = None

import os

FILE_DIR = os.path.dirname(os.path.abspath(__file__))


class HighwayPassTestParameterManager:
    def __init__(self):
        """
        The parameter manager class is helper class which projects a normal
        Point to a usable input for the Highway Pass Test class.
        """
        self._params_df = pd.read_csv("%s/params.csv" % FILE_DIR)
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
                utils.project(
                    a=self.params_df.iloc[i]["min"],
                    b=self.params_df.iloc[i]["max"],
                    n=x,
                    by=self.params_df.iloc[i]["inc"],
                )
                for i, x in enumerate(point)
            ],
            index=self.params_df["feature"],
        )
        return s


class HighwayPassTest:
    def __init__(self, params: structs.Point):
        """
        This class is a fully encapsulated black box scenario which performs a
        simple AV scenario in SUMO traffic simulator.

        concrete parameters --> run test --> score test

        -- Parameters --
        @param_s : pd.Series
            Concrete parameters for the black box scenario.
        """

        # Check input
        params_s = pd.Series(params)
        assert all(
            [
                any(params_s.index.str.contains(feat))
                for feat in ["init_disp", "rel_vel", "e_decel"]
            ]
        )
        self._params_s = params

        # SUMO configuration
        self._map_dir = "%s" % FILE_DIR
        self._error_log_fn = "%s/error-log.txt" % self.map_dir
        self._config = {
            "gui": False,
            # "gui": True,
            # Street network
            "--net-file": "%s/highway.net.xml" % self.map_dir,
            # Logging
            "--error-log": self.error_log_fn,
            # "--log" : "%s/log.txt" % map_dir,
            # Traci Connection
            "--num-clients": 1,
            "--remote-port": 5522,
            # GUI Options
            "--delay": 100,
            # "--start" : "--quit-on-end",
            # RNG
            "--seed": 333,
        }

        # Start Client
        self._client = TraCIClient(self.config)

        # Add actors
        self._add_actors()

        # Run Simuations
        while traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()

        # Close the Client
        self.client.close()

        # Determine scores
        self._scores_s = self._calc_scores()

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
    def params(self) -> structs.Point:
        return self._params_s

    @property
    def params_s(self) -> structs.Point:
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

    def kph2mps(self, kph: float):
        return kph / 3.6

    def _add_actors(self):
        """
        Adds the actors to the simulation
        """

        # Prepare Route
        warmup_rid = "r:warmup"
        traci.route.add(warmup_rid, ["warmup", "highway"])

        # Configure
        base_speed = 10

        npc_init_speed = self.kph2mps(base_speed)
        npc_init_pos = self.params_s["init_disp"]

        dut_init_speed = self.kph2mps(base_speed + self.params_s["rel_vel"])
        dut_init_pos = 0
        dut_e_decel = self.kph2mps(self.params_s["e_decel"])

        # Add the npc
        traci.vehicle.add(
            "npc",
            warmup_rid,
            departSpeed=npc_init_speed,
            departPos=500,
        )
        traci.vehicle.setMaxSpeed("npc", npc_init_speed)
        traci.vehicle.moveTo("npc", "highway_0", npc_init_pos)

        # Add the DUT
        traci.vehicle.add("dut", warmup_rid, departSpeed=dut_init_speed)
        traci.vehicle.setMaxSpeed("dut", dut_init_speed)
        traci.vehicle.setDecel("dut", dut_e_decel / 2)
        traci.vehicle.setEmergencyDecel("dut", dut_e_decel)
        traci.vehicle.moveTo("dut", "highway_0", dut_init_pos)

    def _calc_scores(self):
        scores = pd.Series({"e_brake": 0, "collision": 0})

        # Parse log for scores
        errors = open(self.error_log_fn).readlines()
        for err in errors:
            if "collision" in err:
                scores["collision"] = 1
            elif "emergency braking" in err:
                wished = float(re.findall(r"wished=-*\d*\.*\d*", err)[0].split("=")[-1])
                observed = float(
                    re.findall(r"decel=-*\d*\.*\d*", err)[0].split("=")[-1]
                )
                scores["e_brake"] = min(abs(wished / observed) * 2, 1)

        return scores
