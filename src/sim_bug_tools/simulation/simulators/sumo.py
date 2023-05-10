import shutil
import warnings
import traci
import pandas as pd
import json
from abc import ABC, abstractmethod as abstract
from typing import Generic, TypeVar

from sim_bug_tools.structs import Point, Domain, Grid
import sim_bug_tools.utils as utils
from pandas import Series, DataFrame

from ..simulation_core import (
    Simulator,
    ScenarioBuilder,
    TargetSDL,
    ConcreteScenario,
    LogicalScenario,
    ScenarioReport,
)


if shutil.which("sumo") is None:
    warnings.warn(
        "Cannot find sumo/tools in the system path. Please verify that the lastest SUMO is installed from https://www.eclipse.org/sumo/"
    )

PY_SDL = "python"


# Sumo-specific scenario format
class SumoConcreteScenario(ConcreteScenario):
    DEFAULT_CONFIG = {
        "gui": False,
        # Street network
        "--net-file": "map/network.net.xml",
        # Logging
        "--error-log": "map/error-log.txt",
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

    def __init__(self, name: str, desc: str, params: dict, config: dict = None):
        super().__init__(name, desc, PY_SDL, params)

    def validate_configuration(self, config: dict) -> bool:
        return all(map(lambda x: x in self.DEFAULT_CONFIG, config))


## Supported SDLs
### Raw Python Scenarios
class PySumoConcreteScenario(ConcreteScenario):
    def __init__(self, name, desc, params):
        super().__init__(name, desc, PY_SDL, params)

    @abstract
    def execute(self, sim: "Sumo") -> ScenarioReport:
        raise NotImplementedError()


class PyLogicalScenario(LogicalScenario):
    @abstract
    def run(self, *params, **kwargs) -> PySumoConcreteScenario:
        raise NotImplementedError()


class PyBuilder(ScenarioBuilder["Sumo"]):
    def build(self, scenario: PySumoConcreteScenario, sim: "Sumo"):
        return scenario.execute(sim)

    @property
    def target_sdl(self) -> TargetSDL:
        return PY_SDL


### e.g. Scenic


# Simulator
class Sumo(Simulator):
    def __init__(self, config: dict, priority: int = 1):
        """
        Barebones TraCI client.

        --- Parameters ---
        priority : int
            Priority of clients. MUST BE UNIQUE
        config : dict
            SUMO arguments stored as a python dictionary.
        """
        super().__init__({PY_SDL: PyBuilder()})

        self._config = config
        self._priority = priority

        self.connect()

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

    def _start_first_client(self):
        cmd = []

        for key, val in self.config.items():
            if key == "gui":
                sumo = "sumo"
                if val:
                    sumo += "-gui"
                cmd.append(sumo)

            elif key != "--remote-port":
                cmd.append(key)
                cmd.append(str(val))

        traci.start(cmd, port=self.config["--remote-port"])
        traci.setOrder(self.priority)

    def connect(self):
        """
        Start or initialize the TraCI connection.
        """
        warnings.simplefilter("ignore", ResourceWarning)
        # Start the traci server with the first client
        if self.priority == 1:
            self._start_first_client()
        else:
            # Initialize every client after the first.
            traci.init(port=self.config["--remote-port"])
            traci.setOrder(self.priority)

    def config(
        self,
        scenario: SumoLogicalScenario,
        gui: bool = False,
        num_clients: int = 1,
        delay: int = 100,
        seed: int = 333,
    ):
        """
        Configuration for a given logical scenario is assumed
        to have the following structure:

        root_sim_folder/
            <map-folder-name>/
                <network-file>
                <error-log>
        """
        net_rel_path: str = "highway.net.xml"
        error_log_path: str = "map/error-log.txt"

        return {
            "gui": gui,
            # Street network
            "--net-file": net_rel_path,
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

    @property
    def builders(self) -> dict[TargetSDL, "ScenarioBuilder"]:
        return self._builders

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
