from typing import Callable, Generic, TypeVar, NewType
from datetime import datetime
import json
from abc import ABC, abstractmethod as abstract
import os
import re
import pickle
from sim_bug_tools.structs import Point, Domain

ResultName = NewType("ResultName", str)
R = TypeVar("R", bound="Experiment.ExperimentResults")
E = TypeVar("E", bound="Experiment")


class ExperimentIndex(Generic[R]):
    EXPERIMENT_NAME = "name"
    EXPERIMENT_DESCRIPTION = "description"
    RESULT_NAMES = "results"

    RE_INDEX = re.compile("_*-index.json")

    def __init__(self, parent_path: str, name: str, desc: str = None, load=False):
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

        if load:
            self._json = self._load_index()
        else:
            self._json = self._init_index(name, desc)

    def _init_index(self, name: str, desc: str):
        _json = {
            self.EXPERIMENT_NAME: name,
            self.EXPERIMENT_DESCRIPTION: desc,
            self.RESULT_NAMES: [],
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
        with open(os.path.join(f"{self._root}", f"{str(result)}.pkl"), "wb") as f:
            pickle.dump(result, f)

        self._json[self.RESULT_NAMES].append(str(result))

        self._write_json()

    def get_result(self, name: ResultName) -> R:
        result: R
        with open(os.path.join(self._root, f"{name}.pkl"), "r") as f:
            result = pickle.load(f)

        return result

    def _write_json(self):
        with open(self._path, "w") as f:
            f.write(json.dumps(self._json, indent=4))

    @property
    def result_names(self) -> list[ResultName]:
        return self._json[self.RESULT_NAMES]

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
    def from_experiment(parent_path: str, experiment: E) -> "ExperimentIndex"[E]:
        return ExperimentIndex(parent_path, experiment.name, experiment.desc)

    @staticmethod
    def load_from_index_path(index_path: str) -> "ExperimentIndex":
        parent_path, name = os.path.split(index_path)
        return ExperimentIndex(parent_path, name, load=True)

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


class Experiment(ABC):
    CACHE_FOLDER = "experiment_caches"

    def __init__(self, name: str, desc: str):
        self._name = name
        self._desc = desc

        self._setup()
        self._lazily_init_index()

    @property
    def desc(self):
        return self._desc

    @property
    def name(self):
        return self._name

    @property
    def state(self):
        return self.__dict__

    @property
    def has_cache(self) -> bool:
        return os.path.exists(self.index_path)

    def _setup(self):
        """
        Wraps the run method with setup and teardown operations at runtime.
        """
        run = self.run

        def new_run(*args, **kwargs):
            # setup
            self.fetch_dependencies()

            # run
            run(*args, **kwargs)

            # teardown
            self._cache()

        return new_run

    # Caching
    class ExperimentResults(ABC):
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

        def __init__(self, exp_name: str, params: dict, spec_name: str = None):
            self.date = datetime.now()
            self.exp_name = exp_name
            self.spec_name = spec_name
            self.params = params

        def _misc_json_parser(self, o):
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

            return json.dumps(
                d, default=self._misc_json_parser, sort_keys=True, indent=4
            )

        def __repr__(self):
            if self.spec_name is not None:
                s = f"{self.date.strftime('%Y%m%d_%H%M%S')}-{self.exp_name}-{self.spec_name}"
            else:
                s = f"{self.date.strftime('%Y%m%d_%H%M%S')}-{self.exp_name}"

            return s

        __str__ = __repr__

    def _lazily_init_index(self):
        "Inits the cache index for managing cache files if it's not already init'd."
        if os.path.exists(self.index_root_path):
            self._index = ExperimentIndex.load_from_index_path(self.index_path)
        else:
            self._index = ExperimentIndex.from_experiment(self.CACHE_FOLDER, self)

    # Running
    @abstract
    def run(self) -> ExperimentResults:
        """
        Implement this method to execute your experiment.
        Notes:
        - all object variables will be part of the "state" of the experiment, if
        you want to save that information as a result, store it in self.

        Returns:
            ExperimentResults: The results of the experiment.
        """
        pass

    def lazily_run(self) -> ExperimentResults:
        """
        Runs only if there is no cached result, otherwise returns the cached
        result.

        Returns:
            ExperimentResults: The results of the experiment.
        """
        if self.has_cache:
            return self._cache()
        else:
            return self.run()

    def run_all(self, sequence: tuple) -> list[ExperimentResults]:
        """
        Runs a sequence of parameters. Each parameter is ran by the experiment,
        the results from which are returned as a list in the order of their execution.

        Args:
            sequence (tuple): The sequence of parameters. Parameters must match
                the signature of the self.run command.

        Returns:
            list[ExperimentResults]: The list of results, where each result corresponds to a
                parameter within @sequence.
        """
        return [self.run(params) for params in sequence]

    @property
    def index_root_path(self):
        "The root folder of the cache index."
        return os.path.join(f"{self.CACHE_FOLDER}", self._name)

    @property
    def index_path(self):
        "The path to the index file for the cache."
        return os.path.join(f"{self.CACHE_FOLDER}", self._name)

    @property
    def index(self):
        "The experiment index object for managing the cached results."
        return self._index
