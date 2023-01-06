"""
Misc. functions 

Author: Quentin Goss
"""
import numpy as np
import pickle
from sim_bug_tools.constants import *
from sim_bug_tools.structs import Point
import re
from itertools import takewhile, repeat
from numpy import ndarray, float64
from rtree.index import Index


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


def project(a: float, b: float, n: float, by: float = None) -> float:
    """
    Project a normal value @x between @a and @b.

     -- Parameters --
    a : float or np.ndarray
        Lower bound
    b : float or np.ndarray
        Upper bound
    n : float or np.ndarray
        Normal value between 0 and 1
    by : float
        Granularity of range

    -- Return --
    float or np.ndarray
        x applied within the range of a and b
    """
    if not (b > a):
        raise ValueError

    # Continous
    if by is None:
        return n * (b - a) + a

    #  Discrete
    norm_interval = by / (b - a)
    interval = np.round(n / norm_interval, decimals=0)
    return a + interval * by


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


def filter_unique(array: list[np.ndarray]) -> list[np.ndarray]:
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


def parse_float(s: str) -> float:
    return float(re.findall(r"-?\d+\.?\d*", s)[0])


def parse_int(s: str) -> int:
    return int(parse_float(s))


def flatten_dicts(dicts: list[dict]) -> dict:
    """
    Flatten multiple dicts by keys
    """
    for i in range(1, len(dicts)):
        for key, value in dicts[i].items():
            dicts[0][key] = value
    return dicts[0]


def rawincount(filename: str) -> int:
    """
    Returns the number of \n in a file.

    -- Parameters --
    filename : str
        Filename to count newline characters

    -- Return --
    int
        Number of newline charachters in file.
    """
    with open(filename, "rb") as f:
        bufgen = takewhile(lambda x: x, (f.raw.read(1024 * 1024) for _ in repeat(None)))
        return sum(buf.count(b"\n") for buf in bufgen)


def find_sample_by_depth(
    boundary: list[tuple[Point, ndarray]],
    start_i: int,
    index: Index,
    d: float64,
    is_target: bool = True,
    threshold: float64 = 0.05,
):
    """
    Find an sample a certain distance from or into an envelope.

    Attempts to ensure that the sample is

    Args:
        boundary (list[tuple[Point, ndarray]]): The list of bounday nodes.
        i_start (int): The index of the node to start from.
        d (float64): The jump size of each step.
        is_target (bool): If True, it will sample into the envelope, otherwise.
            it will sample away from the envelope.
        threshold (float64): Percent error allowed from d [defaults to 5%]

    Returns None if the sample was not far enough away from other boundary
    points; otherwise, returns the desired point.
    """
    b, n = boundary[start_i]
    p = b - Point(n * d) if is_target else b + Point(n * d)

    b_near, _ = boundary[index.nearest(p, 1).__next__()]

    if p.distance_to(b_near) < d * (1 - threshold):
        return None

    return p


def k_nearest_density(
    boundary: list[tuple[Point, ndarray]], index: Index, k: int, max_distance: float64
) -> float64:
    """
    Finds the average distance per point. This is an analogue to density,
    although not the same.

    Args:
        boundary (list[tuple[Point, ndarray]]): The list of boundary nodes.
        index (Index): The boundary R-Tree
        k (int): The (max) number of neighbors to consider
        max_distance (float64): The maximum distance that points can be considered.

    Returns:
        float64: The average distance per point
    """
    if len(boundary) == 0:
        return None

    density = 0

    for b, _ in boundary:
        ids = index.nearest(b, k + 1)  # +1 to ignore b's self

        tmp = 0
        for id in ids:
            tmp += b.distance_to(boundary[id])

        density += tmp / len(ids)

    return density / len(boundary)


def find_intermediate(p: Point, nodes: tuple[Point, ndarray]):
    """
    Using a collection of neighboring nodes, an average of those nodes is
    determined via a weighted sum. They are weighted based on their distance,
    where the closest node will be given the greatest preference.

    Args:
        p (Point): The point which is determining the weights
        nodes (tuple[Point, ndarray]): The boundary nodes that neighbor one another

    Returns:
        tuple[Point, ndarray]: The virtual boundary node that is the average of
            the neighbors, weighted according to their distance from p.
    """
    dists: float64 = [p.distance_to(b) for b, n in nodes]
    total = sum(dists)
    weights: float64 = [p.distance_to(b) / total for b, n in nodes]

    new_b = Point.zeros(len(p))
    new_n = np.zeros(new_b.array.shape)
    for weight, node in zip(weights, nodes):
        b, n = node
        new_b += Point(b * weight)
        new_n += n * weight

    return new_b, new_n


def predict_class(
    p: Point,
    boundary: list[tuple[Point, ndarray]],
    err: float64,
    threshold: float64,
    k: int = 3,
    rt_index: Index = None,
):
    """
    Using a list of boundary nodes, this algorithm predicts whether or not a
    given sample lies within or without the envelope and if that point lies on
    the boundary. The k-value represents k-nearest neighboring boundary nodes,
    which are used to improve the estimation.

    Args:
        p (Point): The point to be (predictively) classified boundary
        (list[tuple[Point, ndarray]]): The list of boundary nodes err (float64):
        How much error to expect from the boundary threshold (float64): The max
        distance k-neighbors can be from initial
            boundary point.
        k (int, optional): The number of neighboring boundary points to improve
            estimations with. Defaults to 3.
        rt_index (Index, Optional): The pre-loaded R-Tree index that stores the
            positional information of the boundary nodes (faster than rebuilding
            it each time, if already created.)

    Returns:
        tuple[bool, meta-data]: Returns the result of being a target-value or
        not followed by meta-data. The meta-data includes [is_on_boundary,
        virtual_boundary_node, neighbors]. The virtual_boundary_node is
        constructed by a weighted sum of the neighbors. Read find_intermediate
        for more info.
    """
    # find nearest boundary point node.
    node_ids = tuple(rt_index.nearest(p, k))
    nearest = boundary[node_ids[0]]

    # Eliminate distant boundary points
    neighbors = [
        boundary[id]
        for id in node_ids
        if nearest[0].distance_to(boundary[id][0]) <= threshold
    ]
    b, n = find_intermediate(p, neighbors)

    rel_pos_vector = (p - b).array

    # Dot product between n and displacement from b reveals which side of the
    # boundary we are on. Positive values are outside, negative are inside.
    alignment = np.dot(rel_pos_vector, n)

    is_in_envelope = alignment <= 0

    # If we are inside the envelope AND within error margin of the boundary...
    is_boundary_point = is_in_envelope and np.linalg.norm(rel_pos_vector) <= err

    return is_in_envelope, (is_boundary_point, (b, n), neighbors)


def random_point_in_sphere(min_r: float64, max_r: float64, loc: Point):
    "Another way to generate a sample. Helps with testing high-dimension hyperspheres"
    ndims = len(loc)
    v = np.random.rand(ndims)
    v /= np.linalg.norm(v)
    rand_r = np.random.rand(1)[0] * (max_r - min_r) + min_r
    return Point(v * rand_r) + loc
