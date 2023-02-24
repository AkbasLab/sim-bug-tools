from typing import Generic, TypeVar
from datetime import datetime
from abc import ABC, abstractmethod as abstract
from sim_bug_tools.structs import Point, Domain
import json

T = TypeVar('T')
class ParameterSpace(ABC, Generic[T]):
    def __init__(self, domain: Domain, axes_names: list[str] = None) -> None:
        self._domain = domain
        self._axes_names = (
            axes_names 
            if axes_names is not None
            else [chr(ord('x') + i) for i in range(len(domain))]
        )
    
    @abstract
    def evaluate(p : Point) -> T:
        raise NotImplementedError()
    
    
class Experiment(ABC):
    """
    A class that provides tools for creating, documenting, and varying
    experiments.
    """
    KW_EXP_NAME = 'name'

    def __init__(self, name: str, abbrevation: str, output_path: str):
        self._name = name
        self._abbreviation = abbrevation
        self._output_path = output_path

        self.setup()

    @abstract
    def experiment(self, params: dict) -> dict:
        raise NotImplementedError()

    def setup(self):
        """
        Input data,
        """
        pass

    def execute(self, variations: list[dict], save_separately=False):
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
        variations = [variations] if type(variations) is dict else variations
        if type(variations) is not list:
            raise TypeError("Variations must be an experiment params (dict) or a list of params (list[dict])")
            
        _name_counts = {}

        results: list[dict] = []

        for i, params in enumerate(variations):
            completion = "OK"
            try:
                result = self.experiment(params)

            except Exception as e:
                completion = f"Exception: {e}"

            # Ensure dict
            result = result if result is not None else {}

            name = params[self.KW_EXP_NAME] if self.KW_EXP_NAME in params else str(i)
            result["exit"] = completion

            # Handle name duplicates...
            if name in _name_counts:
                results[f"{name}-{_name_counts[name]}"] = result
                _name_counts[name] += 1
            else:
                results[name] = result
                _name_counts[name] = 1

        self.teardown(completion)
    
    


    def teardown(self, completion: str):
        """
        Will automatically output the data
        """
        with open(
            f"{datetime.strftime('%y%m%d-%H%M%S')}-{self._abbreviation}", "w"
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
