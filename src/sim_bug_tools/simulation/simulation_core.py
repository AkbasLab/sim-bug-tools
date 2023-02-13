import pandas as pd
import json

from abc import ABC, abstractmethod as abstract
from typing import TypeVar, Generic, NewType
from dataclasses import dataclass

from sim_bug_tools.structs import Point, Domain, Grid
from sim_bug_tools.decorators import copy_signature


class ScenarioReport:
    pass


@abstract
@dataclass
class ConcreteScenario(ABC):
    name: str
    desc: str
    sdl: str
    params: tuple


class LogicalScenario(ABC):
    @abstract
    def actualizeScenario(self, *params, **kwargs) -> ConcreteScenario:
        pass

    __call__ = actualizeScenario


TargetSDL = NewType("TargetSDL", str)


class Simulator(ABC):
    @abstract
    @property
    def builders(self) -> dict[TargetSDL, "ScenarioBuilder"]:
        raise NotImplementedError()

    def run(self, scenario: ConcreteScenario):
        with self.builders[scenario.sdl].build(scenario, self) as executable:
            executable()  # just thoughts


S = TypeVar("S", bound=Simulator)


class ScenarioBuilder(ABC, Generic[S]):
    def __init__(self, target_sdl: str):
        self._target_sdl = target_sdl

    @abstract
    def build(self, scenario: ConcreteScenario, sim: S):
        pass

    @property
    def target_sdl(self):
        return self._target_sdl

    __call__ = build


# class Simulation:
# PARAM_DEFAULT = "params.csv"
# CONFIG_DEFAULT = "config.json"
# MAP_DIR_DEFAULT = ""

# def __init__(self, dir: str, config: dict = None, **kwargs):
#     """
#     Args:
#         dir (str): The
#         config (dict, optional): _description_. Defaults to None.
#     """
#     params_name = kwargs.get("params_name", self.PARAM_DEFAULT)
#     config_name = kwargs.get("config_name", self.CONFIG_DEFAULT)
#     map_dir = kwargs.get("map_dir", self.MAP_DIR_DEFAULT)
#     map_name = kwargs.get("map_name", None)

#     self._dir = dir
#     self._params_df = pd.read_csv(f"{dir}/{params_name}")
#     self._config = json.loads(f"{dir}/{config_name}") if config is None else config
#     self._map_dir = f"{dir}/{map_dir}"

#     self._gui_enabled = (
#         kwargs["gui"]
#         if "gui" in kwargs
#         else self._config["gui"]
#         if "gui" in self._config
#         else False
#     )

#     self._validate_setup(map_name)

#     self._error_log_fn = f"{map_dir}/error-log.txt"

#     bounds, res = zip(
#         *[((row["min"], row["max"]), row["inc"]) for row in self._params_df.iloc]
#     )
#     self._domain = Domain(bounds)
#     self._grid = Grid(res)
#     self._ndims = len(bounds)

#     self._client = TraCIClient(self._config)

# def run_scenario(self, scenario: ConcreteScenario):
#     with scenario as s:
#         results = s.run(self.traci)

#     return results

# def _validate_setup(self, map_name: str):
#     assert (
#         map_name is not None or "--net-file" in self._config
#     ), "Map must be in config or provided by parameter!"

#     if "--net-file" not in self._config:
#         self._config["--net-file"] = f"{self._map_dir}/{map_name}"

# @property
# def traci(self) -> TraCIClient:
#     return self._client

# @property
# def ndims(self) -> float:
#     return self._ndims

# @property
# def domain(self) -> Domain:
#     return self._domain

# @property
# def grid(self) -> Grid:
#     return self._grid

# def map_parameters(self, p: Point, src_domain: Domain = None) -> Series:
#     """
#     Maps a normal @p to concrete parameters. Returns a pandas series.
#     @src_domain defaults to a normalized domain of the dimensionality as self.
#     """
#     src_domain = Domain.normalized(self.ndims) if src_domain is None else src_domain
#     mapped_p = Domain.translate_point_domains(p, src_domain, self._domain)
#     disc_p = self._grid.discretize_point(mapped_p)

#     return Series(disc_p, index=self.params_df["feature"])

# def unmap_parameters(self, s: Series, dst_domain: Domain = None) -> Point:
#     """
#     Maps a normal @point to concrete parameters.
#     Returns a pandas series
#     """
#     dst_domain = Domain.normalized(self.ndims) if dst_domain is None else dst_domain
#     return Domain.translate_point_domains(Point(s), self._domain, dst_domain)
