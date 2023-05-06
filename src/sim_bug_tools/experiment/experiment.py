import json
import os
import sys
import re
import pickle

from typing import Generic, TypeVar, NewType
from datetime import datetime
from abc import ABC, abstractmethod as abstract

from sim_bug_tools.structs import Point, Domain


class ExperimentParams(ABC):
    def __init__(self, name: str, desc: str):
        """
        Args:
            name (str): A UNIQUE name for this set of parameters. Used for
                fetching previous results derived from these parameters.
            desc (str): A simple description for what the experiment is aiming
                for.
        """
        self.name = self._format_name(name)
        self.desc = desc

    # def copy_with(self, **kwargs):

    def _format_name(self, name: str):
        return name.strip().replace(" ", "_")

    def __repr__(self) -> str:
        return (
            f"<{self.__class__.__name__} | {self.name}>\n{self.desc}\n"
            + "{\n"
            + str(self.__dict__)
            + "\n}"
        )

    __str__ = __repr__


P = TypeVar("P", bound=ExperimentParams)


class ExperimentResults(ABC, Generic[P]):
    """
    Implement this class for defining dependencies for sequencing
    experiments. This provides type hints for when you depend on the result
    of another experiment, also enables caching of results. Make this a
    dataclass for easy boiler plate.
    """

    EXP_NAME = "experiment-name"
    SPEC_NAME = "specific-name"
    DATE = "date"
    PARAMS = "experiment-parameters"

    def __init__(self, params: P):
        self.date = datetime.now()
        self.params = params

    def _misc_json_parser(self, o):
        "Add additional cases here for handling sim-bug-tools objects"
        if type(o) is Point:
            return tuple(o)
        elif type(o) is Domain:
            return tuple(map(lambda x: tuple(x), tuple(o)))
        else:
            return o.__dict__

    def to_json(self):
        "Use this method to save in a readable format."
        d = {}

        # Format keys to use - instead of _
        for key, value in self.__dict__.items():
            key = key.replace("_", "-").strip("-", " ")
            d[key] = value

        return json.dumps(d, default=self._misc_json_parser, sort_keys=True, indent=4)

    def __repr__(self):
        return f"{self.params.name}-{self.date.strftime('%Y%m%d_%H%M%S')}"

    __str__ = __repr__


ResultName = NewType("ResultName", str)
R = TypeVar("R", bound=ExperimentResults)
E = TypeVar("E", bound="Experiment")


class ExperimentIndex(Generic[P, R]):
    EXPERIMENT_NAME = "name"
    EXPERIMENT_DESCRIPTION = "description"
    RESULT_IDS = "results"

    RE_INDEX = re.compile("_*-index.json")

    def __init__(self, parent_path: str, name: str, desc: str = None, force_init=False):
        """
        Constructs a new index within the parent_path. The index folder will
        have a name of @name, and will construct an index json file
        _@name-index.json. This json will manage the different experiment caches
        for future use. Keep in mind, although the index itself is .json, the
        caches are pkl'd. This is to improve type-hinting.

        Note: if load=False and the index already exists, the index will be
        wiped. However, the caches (.pkl) will remain, despite being cleared
        from the index. This is to prevent accidental deletion of experiment
        results, but also to simplify the interface.

        Args:
            parent_path (str): Path to a folder the store the new index
            name (str): The name of the experiment that the index is for
            desc (str, optional): The description of the experiment. Needed only
                if the index is being created. Defaults to None.
            load (bool, optional): If True, it will attempt to load the index
                instead of creating it. If False, it will create (and potentially
                overwrite) the index. Defaults to False.
        """
        self._root = os.path.join(parent_path, name)
        os.makedirs(self._root, exist_ok=True)

        self._path = os.path.join(self._root, self.get_index_name(name))

        if not force_init and os.path.exists(self._path):
            self._json = self._load_index()
        else:
            self._json = self._init_index(name, desc)

    def _init_index(self, name: str, desc: str):
        _json = {
            self.EXPERIMENT_NAME: name,
            self.EXPERIMENT_DESCRIPTION: desc,
            self.RESULT_IDS: [],
        }

        with open(self._path, "w") as f:
            f.write(json.dumps(_json, indent=4))

        return _json

    def _load_index(self):
        with open(self._path, "r") as f:
            _json = json.loads(f.read())

        return _json

    def update(self, name: str = None, desc: str = None):
        if name is not None:
            self._json[self.EXPERIMENT_NAME] = name
        if desc is not None:
            self._json[self.EXPERIMENT_DESCRIPTION] = desc

        self._write_json()

    def add_result(self, result: R):
        "Add results to index and pickle result for future retrieval"
        with open(os.path.join(f"{self._root}", f"{str(result)}.pkl"), "wb") as f:
            pickle.dump(result, f)

        self._json[self.RESULT_IDS].append(str(result))

        self._write_json()

    def get_result(self, filename: str) -> R:
        with open(os.path.join(self._root, f"{filename}.pkl"), "rb") as f:
            result = pickle.load(f)

        return result

    def get_previous_result_by_name(self, param_name: str) -> R:
        most_recent = sorted(
            [name for name in self.result_names if name.startswith(param_name)]
        )

        if len(most_recent) > 0:
            with open(os.path.join(self._root, f"{most_recent[-1]}.pkl"), "rb") as f:
                result = pickle.load(f)
        else:
            result = None

        return result

    def _write_json(self):
        with open(self._path, "w") as f:
            f.write(json.dumps(self._json, indent=4))

    @property
    def result_names(self) -> list[ResultName]:
        return self._json[self.RESULT_IDS]

    @property
    def previous_result_name(self) -> ResultName:
        return self.result_names[-1]

    @property
    def previous_result(self) -> R:
        return self.get_result(self.previous_result_name)

    @property
    def first_result_name(self) -> ResultName:
        return self.result_names[0]

    @property
    def first_result(self) -> R:
        return self.get_result(self.first_result_name)

    @property
    def experiment_name(self) -> str:
        return self._json[self.EXPERIMENT_NAME]

    @property
    def experiment_description(self) -> str:
        return self._json[self.EXPERIMENT_DESCRIPTION]

    @property
    def json(self):
        return self._json

    @staticmethod
    def from_experiment_name(
        parent_path: str, name: str, desc: str = None
    ) -> "ExperimentIndex":
        return ExperimentIndex(parent_path, name, desc)

    # @staticmethod
    # def from_experiment(parent_path: str, experiment: E) -> "ExperimentIndex":
    #     return ExperimentIndex(parent_path, experiment.name, experiment.description)

    @staticmethod
    def load_from_index_path(index_path: str) -> "ExperimentIndex":
        parent_path, name = os.path.split(index_path)
        return ExperimentIndex(parent_path, name)

    @staticmethod
    def get_index_name(exp_name: str) -> str:
        return f"_{exp_name}-index.json"

    @classmethod
    def load_from_root_folder(cls, root_path: str) -> "ExperimentIndex":
        file_names = os.listdir(root_path)
        index_paths = list(filter(cls.RE_INDEX, file_names))
        if len(index_paths) > 1:
            raise FileNotFoundError(
                "There were too many index files within the folder! Please remove one and try again..."
            )
        elif len(index_paths) == 0:
            raise FileNotFoundError("No index file found within the specified folder!")

        return cls.load_from_index_path(index_paths[0])


class Experiment(ABC, Generic[P, R]):
    CACHE_FOLDER = ".experiment_caches"

    def __init__(self):
        self._index: ExperimentIndex[P, R] = ExperimentIndex(
            self.CACHE_FOLDER, self.get_name()
        )  # .from_experiment(self.CACHE_FOLDER, self)

    @abstract
    def experiment(self, params: P) -> R:
        "Implement this with your experiment's code"
        raise NotImplementedError()

    def setup(self):
        "Implement this to support with statement"
        raise NotImplementedError("Setup method was not defined for this experiment!")

    def teardown(self):
        "Implement this to support with statement"
        raise NotImplementedError(
            "Teardown method was not defined for this experiment!"
        )

    def run(self, params: P) -> R:
        "Execute this to run your experiment"
        result = self.experiment(params)
        self._index.add_result(result)
        return result

    def lazily_run(self, params: P) -> R:
        "Execute if no params with the same name have already been executed"
        result = self.index.get_previous_result_by_name(params.name)
        if result is None:
            result = self.run(params)

        return result

    __call__ = lazily_run

    __enter__ = setup
    __exit__ = teardown

    @property
    def index(self):
        "The experiment index object for managing the cached results."
        return self._index

    @property
    def state(self):
        return {key: value for key, value in self.__dict__.items() if key[0] != "_"}

    @property
    def has_cache(self) -> bool:
        return os.path.exists(self.index_path)

    @property
    def index_root_path(self):
        "The root folder of the cache index."
        return os.path.join(f"{self.CACHE_FOLDER}", self.get_name())

    @property
    def index_path(self):
        "The path to the index file for the cache."
        return os.path.join(
            f"{self.CACHE_FOLDER}",
            self.get_name(),
            ExperimentIndex.get_index_name(self.__name__),
        )

    @classmethod
    def get_name(cls):
        return cls.__name__

    @classmethod
    def get_index(cls) -> ExperimentIndex[P, R]:
        return ExperimentIndex(cls.CACHE_FOLDER, cls.__name__)
