from typing import Callable
from datetime import datetime
import json


class ExperimentParams:
    """
    The params into an experiment. These are also referred to as
    "variations," since they vary the experiment.
    """

    def __init__(self, variation_name: str, data: dict):
        self._name = variation_name
        self._data = data

    @property
    def data(self):
        return self._data

    @property
    def name(self):
        return self._name

    def __repr__(self):
        return f'<ExperimentParams: "{self.name}">'


class Experiment:
    """
    A class that provides tools for creating, documenting, and varying
    experiments.
    """

    def __init__(self, name: str, abbrevation: str, output_path: str):
        self._name = name
        self._abbreviation = abbrevation
        self._output_path = output_path

        self.setup()

    @property
    def reserved_names(self) -> set[str]:
        """
        The set of reserved names that cannot be used in the result of an
        experiments.
        """
        return set(["exit"])

    @property
    def data(self) -> dict:
        """
        The json-able data dictionary
        """
        return {}

    def setup(self):
        """
        Input data,
        """
        pass

    def execute(self, variations: list[ExperimentParams], save_separately=False):
        """
        Executes the experiment with the provided variations

        Args:
            variations (list[ExperimentParams]): The list of parameters to
                execute the experiment with.
            save_separately (bool, optional): Save each variation's results in a
                separate file. Files will be labeled with their variation's name.
                Defaults to False.
        """

        _name_counts = {}

        results: list[dict] = []

        for variation in variations:
            completion = "OK"
            try:
                result = self.experiment()

            except Exception as e:
                completion = f"Exception: {e}"

            name = variation.name

            # Ensure dict
            result = result if result is not None else {}
            result["exit"] = completion

            # Handle name duplicates...
            if name in _name_counts:
                results[f"{name}-{_name_counts[name]}"] = result
                _name_counts[name] += 1
            else:
                results[name] = result
                _name_counts[name] = 1

        self.teardown(completion)

    def experiment(self, params: ExperimentParams):
        pass

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
