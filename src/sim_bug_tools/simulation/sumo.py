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


if shutil.which("sumo") is None:
    warnings.warn(
        "Cannot find sumo/tools in the system path. Please verify that the lastest SUMO is installed from https://www.eclipse.org/sumo/"
    )


class TraCIClient:
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
