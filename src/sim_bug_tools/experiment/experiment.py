from typing import Generic, TypeVar
from datetime import datetime
from abc import ABC, abstractmethod as abstract
from sim_bug_tools.structs import Point, Domain
import json

T = TypeVar("T")


class ParameterSpace(ABC, Generic[T]):
    def __init__(self, domain: Domain, axes_names: list[str] = None) -> None:
        self._domain = domain
        self._axes_names = (
            axes_names
            if axes_names is not None
            else [chr(ord("x") + i) for i in range(len(domain))]
        )

    @abstract
    def evaluate(p: Point) -> T:
        raise NotImplementedError()


class ExperimentParams(ABC):
    """
    A class that represents input parameters into an experiment. The purpose of
    this class is to enable type-hinting for interdependencies between
    Experiments, allowing for a dependent Experiment to know what another
    Experiment's parameters are.`
    """

    def __init__(self, dimensions: list[str], name: str = None):
        """
        Args:
            dimensions (list[str]): A list of names for each parameter.
            name (str, optional): A name to give the resulting concrete
                experiment. Defaults to None.
        """
        self.dimensions = dimensions
        self.name = "None" if name is None else name

    def to_dict(self):
        return self.__dict__


P = TypeVar("P", bound=ExperimentParams)


class Experiment(ABC, Generic[P]):
    """
    A class that provides tools for creating, documenting, and varying
    experiments.
    """

    KW_EXP_NAME = "name"

    def __init__(self, name: str, abbrevation: str, output_path: str):
        self._name = name
        self._abbreviation = abbrevation
        self._output_path = output_path

        self.setup()

    @abstract
    def experiment(self, params: P) -> dict:
        raise NotImplementedError()

    def setup(self):
        """
        Input data,
        """
        pass

    def execute(self, variations: list[P], save_separately=False):
        """
        Executes the experiment with the provided variations. If a "name" key is
        found within a variation (i.e. experiment params) then it will be used
        to label that experiment

        Args:
            variations (list[ExperimentParams]): The list of parameters to
                execute the experiment with.
            save_separately (bool, optional): Save each variation's results in a
                separate file. Files will be labeled with their variation's
                name. Defaults to False.
        """
        _name_counts = {}

        results: dict = {}

        for i, params in enumerate(variations):
            completion = "OK"
            try:
                result = self.experiment(params)

            except Exception as e:
                completion = f"Exception: {e}"

            result = {} if result is None else result

            result["exit"] = completion
            result["params"] = params.to_dict()

            # Handle name duplicates...
            if params.name in _name_counts:
                results[f"{params.name}-{_name_counts[params.name]}"] = result
                _name_counts[params.name] += 1
            else:
                results[params.name] = result
                _name_counts[params.name] = 1

        self.teardown(completion)

        return result

    def teardown(self, completion: str):
        """
        Will automatically output the data
        """
        with open(
            f"{self._output_path}/{datetime.strftime('%y%m%d-%H%M%S')}-{self._abbreviation}",
            "w",
        ) as f:
            data = self.data
            data["exit"] = completion
            f.write(json.dumps(data))

    @property
    def reserved_names(self) -> set[str]:
        """
        The set of reserved names that cannot be used in the result of an
        experiments.
        """
        return set(["exit", self.KW_EXP_NAME])

    @property
    def data(self) -> dict:
        """
        The json-able data dictionary
        """
        return {}
