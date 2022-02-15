"""
Misc. functions 

Author: Quentin Goss
"""
import numpy as np
import pickle
from sim_bug_tools.constants import *
import re
from itertools import (takewhile,repeat)

def as_position(position: np.ndarray) -> np.ndarray:
    """
    Checks if the given position is valid. If so, returns a np.ndarray.

    -- Parameter --
    position : list or np.ndarray
        Position

    -- Return --
    position as an np.ndarray if valid. Otherwise throws a type error
    """
    if isinstance(position, list):
        return np.array(position)
    elif isinstance(position, np.ndarray):
        return position
    else:
        raise TypeError(
            "Position is %s instead of %s" % (type(position), type(np.ndarray))
        )


def denormalize(a: np.float64, b: np.float64, x: np.float64) -> np.float64:
    """
    Maps a normal value x between values a and b

    -- Parameters --
    a : float or np.ndarray
        Lower bound
    b : float or np.ndarray
        Upper bound
    x : float or np.ndarray
        Normal value between 0 and 1

    -- Return --
    float or np.ndarray
        x applied within the range of a and b
    """
    return x * (b - a) + a


def project(
        a: float, 
        b: float, 
        x: float, 
        by: float = None
    ) -> float:
    """
    Project a normal value @x between @a and @b.

     -- Parameters --
    a : float or np.ndarray
        Lower bound
    b : float or np.ndarray
        Upper bound
    x : float or np.ndarray
        Normal value between 0 and 1
    by : float
        Granularity of range 


    -- Return --
    float or np.ndarray
        x applied within the range of a and b
    """
    if by is None:
        return x * (b - a) + a

    
    return


def pretty_dict(d: dict, indent: np.int32 = 0):
    """
    Pretty print python dictionary

    -- Parameters --
    d : dictionary
        Dictionary to print
    indent : int
        Number of spaces in indent
    """
    for key, value in d.items():
        print(" " * indent + str(key))
        if isinstance(value, dict):
            pretty_dict(value, indent + 1)
        else:
            print(" " * (indent + 1) + str(value))


def save(obj, fn: str):
    """
    Save an object to file.

    -- Parameters --
    obj : python object
        Object to save
    fn : str
        Filename
    """
    with open(fn, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    return


def load(fn: str):
    """
    Load a python object from a file

    --- Parameters --
    fn : str
        Filename

    --- Return ---
    unpickled python object.
    """
    with open(fn, "rb") as f:
        return pickle.load(f)


def transposeList(lst: list) -> list:
    return np.asarray(lst).T.tolist()


def convert_to_tuples(array):
    return tuple(map(lambda x: tuple(x), array))


def get_column(array, index):
    return np.array(map(lambda ele: ele[index], array))


## Dictionary Tools ##


def dictSubtract(a: dict, b: dict) -> dict:
    "Set subtraction between dictionary objects."
    return {key: value for key, value in a.items() if key not in b}


def dictIntersect(a: dict, b: dict):
    "Set intersection between dictionary objects."
    return {key: value for key, value in a.items() if key in b}


def sortByDict(a: dict, b: dict) -> dict:
    """
    Sorts A entries by the values in B, where A's keys are a subset of B's.
    Example:
        a = {'a': 5, 'b': 23, 'c': 2}
        b = {'c': 0, 'a': 4, 'b': 2}

        c = sortByDict(a, b).items() = [('c', 2), ('b', 23), ('a', 5)]

    Args:
        a (dict)
        b (dict)

    Returns:
        dict: a (sorted)
    """
    result = {}

    # Sort the dectionary by value
    keys = list(dict(sorted(b.items(), key=lambda x: x[1])).keys())
    for i in range(len(a)):
        key = keys[i]
        result[key] = a[key]

    return result


def prime(n: int) -> np.int32:
    """
    Returns the n-th position prime number.

    -- Parameter --
    n : int
        n-th position. Where 1- < n < 1601.

    -- Return --
    n-th position prime number
    """
    prime_max = 1600
    if n < 0 or n > 1600:
        raise ValueError("%d-th value is not within 0 < n < 1601")
    return np.int32(PRIME_VECTOR[n])


def is_prime(x: np.int32) -> bool:
    """
    Checks if x is in the first 1600 prime numbers.

    -- Parameter --
    x : np.int32
        Any number

    -- Return --
    Whether x is in the first 1600 prime numbers.
    """
    return x in PRIME_VECTOR_SET





def filter_unique(array : list[np.ndarray]) -> list[np.ndarray]:
        """
        Filters unique values from a numpy array

        -- Parameter --
        array : list[np.ndarray]
            List of numpy arrays.

        -- Return --
        list[np.ndarray]
            The unique arrays within array.
        """
        unique, counts = np.unique(np.sort(np.array(array)), axis=0, return_counts=True)
        return unique[counts == 1]


def parse_float(s : str) -> float:
    return float(re.findall(r'-?\d+\.?\d*', s)[0])

def parse_int(s : str) -> int:
    return int(parse_float(s))

def flatten_dicts(dicts : list[dict]) -> dict:
    """
    Flatten multiple dicts by keys
    """
    for i in range(1,len(dicts)):
        for key, value in dicts[i].items():
            dicts[0][key] = value
    return dicts[0]

def rawincount(filename : str) -> int:
    """
    Returns the number of \n in a file.

    -- Parameters --
    filename : str
        Filename to count newline characters
    
    -- Return --
    int
        Number of newline charachters in file.
    """
    with open(filename, 'rb') as f:
        bufgen = takewhile(lambda x: x, (f.raw.read(1024*1024) for _ in repeat(None)))
        return sum( buf.count(b'\n') for buf in bufgen )