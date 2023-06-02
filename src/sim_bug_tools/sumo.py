import os
import shutil
import warnings
import traci
import pandas as pd
import json
from abc import ABC, abstractmethod as abstract

from sim_bug_tools.structs import Point, Domain, Grid
import sim_bug_tools.utils as utils
from pandas import Series, DataFrame


if shutil.which("sumo") is None:
    warnings.warn(
        "Cannot find sumo\\tools in the system path. Please verify that the lastest SUMO is installed from https://www.eclipse.org/sumo/"
    )


class TraCIClient:
    CACHE_DIR = "/tmp"
    CACHE_NAME = "cached_state.xml"
    CACHE_PATH = f"{CACHE_NAME}"

    def __init__(self, config: dict, priority: int = 1):
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
        self._cached = False

        self.connect()

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

    @property
    def is_cached(self) -> bool:
        return self._cached

    def run_to_end(self):
        """
        Runs the client until the end.
        """

        while traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()
            # more traci commands

    def close(self):
        """
        Closes the client.
        """
        traci.close()

    # def connect(self):
    #     """
    #     Start or initialize the TraCI connection.
    #     """
    #     warnings.simplefilter("ignore", ResourceWarning)
    #     # Start the traci server with the first client
    #     if self.priority == 1:
    #         cmd = []

    #         for key, val in self.config.items():
    #             if key == "gui":
    #                 sumo = "sumo"
    #                 if val:
    #                     sumo += "-gui"
    #                 cmd.append(sumo)
    #                 continue

    #             if key == "--remote-port":
    #                 continue

    #             cmd.append(key)
    #             cmd.append(str(val))

    #         traci.start(cmd, port=self.config["--remote-port"])
    #         traci.setOrder(self.priority)

    #     # Initialize every client after the first.
    #     traci.init(port=self.config["--remote-port"])
    #     traci.setOrder(self.priority)
    def connect(self):
        """
        Start or initialize the TraCI connection.
        """
        warnings.simplefilter("ignore", ResourceWarning)
        # Start the traci server with the first client
        if self.priority == 1:
            cmd = []  # ["--no-step-log", "--no-warnings"]

            for key, val in self.config.items():
                if key == "gui":
                    sumo = "sumo"
                    if val:
                        sumo += "-gui"
                    cmd.append(sumo)
                    continue

                if key == "--remote-port":
                    continue

                cmd.append(key)
                cmd.append(str(val))
                continue

            traci.start(cmd, port=self.config["--remote-port"])
            traci.setOrder(self.priority)
            return

        # Initialize every client after the first.
        traci.init(port=self.config["--remote-port"])
        traci.setOrder(self.priority)
        return

    def cache_state(self):
        traci.simulation.saveState(self.CACHE_PATH)
        # traci.simulation_saveState(f"{self.CACHE_DIR}/{self.CACHE_NAME}")
        self._cached = True

    def load_cached_state(self):
        print("before:", traci.simulation.getLoadedNumber())
        traci.simulation.loadState(self.CACHE_PATH)
        print("after:", traci.simulation.getLoadedNumber())
        # traci.simulation_loadState(f"{self.CACHE_DIR}/{self.CACHE_NAME}")

    def clear_cache(self):
        self._cached = False
        os.remove(f"{self.CACHE_DIR}\\{self.CACHE_NAME}")
        os.removedirs(f"{self.CACHE_DIR}")


class ScenarioReport:
    pass


class ConcreteScenario(ABC):
    def __init__(self, traci: TraCIClient):
        self._traci = traci

    @abstract
    def setup(self):
        pass

    @abstract
    def teardown(self):
        pass

    @abstract
    def run(self, traci: TraCIClient) -> ScenarioReport:
        pass

    def __enter__(self):
        self.setup()

    def __exit__(self):
        self.teardown()


class LogicalScenario(ABC):
    @abstract
    def __call__(self, params) -> ConcreteScenario:
        pass


class Simulation:
    PARAM_DEFAULT = "params.csv"
    CONFIG_DEFAULT = "config.json"
    MAP_DIR_DEFAULT = ""

    def __init__(self, dir: str, config: dict = None, **kwargs):
        """
        Args:
            dir (str): _description_
            config (dict, optional): _description_. Defaults to None.
            params_name (_type_, optional): _description_. Defaults to PARAM_DEFAULT.
            config_name (_type_, optional): _description_. Defaults to CONFIG_DEFAULT.
        """
        params_name = kwargs.get("params_name", self.PARAM_DEFAULT)
        config_name = kwargs.get("config_name", self.CONFIG_DEFAULT)
        map_dir = kwargs.get("map_dir", self.MAP_DIR_DEFAULT)
        map_name = kwargs.get("map_name", None)

        self._dir = dir
        self._params_df = pd.read_csv(f"{dir}\\{params_name}")
        self._config = json.loads(f"{dir}\\{config_name}") if config is None else config
        self._map_dir = f"{dir}\\{map_dir}"

        self._gui_enabled = (
            kwargs["gui"]
            if "gui" in kwargs
            else self._config["gui"]
            if "gui" in self._config
            else False
        )

        self._validate_setup(map_name)

        self._error_log_fn = f"{map_dir}\\error-log.txt"

        bounds, res = zip(
            *[((row["min"], row["max"]), row["inc"]) for row in self._params_df.iloc]
        )
        self._domain = Domain(bounds)
        self._grid = Grid(res)
        self._ndims = len(bounds)

        self._client = TraCIClient(self._config)

    def run_scenario(self, scenario: ConcreteScenario):
        with scenario as s:
            results = s.run(self.traci)

        return results

    def _validate_setup(self, map_name: str):
        assert (
            map_name is not None or "--net-file" in self._config
        ), "Map must be in config or provided by parameter!"

        if "--net-file" not in self._config:
            self._config["--net-file"] = f"{self._map_dir}\\{map_name}"

    @property
    def traci(self) -> TraCIClient:
        return self._client

    @property
    def ndims(self) -> float:
        return self._ndims

    @property
    def domain(self) -> Domain:
        return self._domain

    @property
    def grid(self) -> Grid:
        return self._grid

    def map_parameters(self, p: Point, src_domain: Domain = None) -> Series:
        """
        Maps a normal @p to concrete parameters. Returns a pandas series.
        @src_domain defaults to a normalized domain of the dimensionality as self.
        """
        src_domain = Domain.normalized(self.ndims) if src_domain is None else src_domain
        mapped_p = Domain.translate_point_domains(p, src_domain, self._domain)
        disc_p = self._grid.descretize_point(mapped_p)

        return Series(disc_p, index=self.params_df["feature"])

    def unmap_parameters(self, s: Series, dst_domain: Domain = None) -> Point:
        """
        Maps a normal @point to concrete parameters.
        Returns a pandas series
        """
        dst_domain = Domain.normalized(self.ndims) if dst_domain is None else dst_domain
        return Domain.translate_point_domains(Point(s), self._domain, dst_domain)
