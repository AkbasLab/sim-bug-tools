"""
Contains a general collection of classes to provide necessary data structures.
"""
import numpy
from sim_bug_tools.decorators import typecheck
from functools import reduce
import logging
import numpy as np
from numpy import int32, float64, ndarray
import matplotlib.pyplot as plt
import matplotlib.axes
import json



class Point:
    """
    An N-dimensional point in space.
    """

    def __init__(self, *args):
        """
        Arguments:
            An iterable,

            A series of numbers
        """
        ITERABLE = (list, tuple, map, ndarray)

        self._vector: ndarray

        if len(args) == 1 and isinstance(args[0], ITERABLE):
            self._vector = self._format_array(args[0])
        elif Point.is_point(args):
            self._vector = self._format_array(args)
        else:
            raise ValueError(
                f"{__class__.__name__}.__init__: Invalid arguments (args = {args})."
            )

    @property
    def array(self) -> ndarray:
        return self._vector

    @property
    def floored_int_array(self) -> ndarray:
        return ndarray(map(lambda x: int32(x), self._vector))

    @property
    def size(self) -> np.int32:
        return np.int32(len(self._vector))

    def __iter__(self):
        return self._vector.__iter__()

    def __len__(self):
        return len(self._vector)

    def __floor__(self):
        new_vector = [round(axis) for axis in self]
        return Point(new_vector)

    def __round__(self, ndigits=None):
        new_vector = [round(axis, ndigits) for axis in self]
        return Point(new_vector)

    def __sub__(self, other):
        if not isinstance(other, Point):
            raise ValueError("Can only subtract a point from another point!")

        return Point(
            list(map(lambda axis_self, axis_other: axis_self - axis_other, self, other))
        )

    def __add__(self, other):
        if not isinstance(other, Point):
            raise ValueError("Can only add a point to another point!")

        return Point(
            list(map(lambda axis_self, axis_other: axis_self + axis_other, self, other))
        )

    def __getitem__(self, key: int) -> float64:
        return self._vector[key]

    def __str__(self):
        return f"{__class__.__name__}: {self._vector}"

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        return np.array_equal(self.array, other.array)

    def _format_array(self, array):
        """
        Convert an arbitrary iterable into a valid ndarray.
        Will throw
        """
        return np.array([float64(x) for x in array])

    def to_list(self):
        return self.array.tolist()

    def as_json(self):
        return json.dumps(self.to_list())

    @staticmethod
    def is_point(array: tuple) -> bool:
        """
        Returns True if the array is a valid point.
        """
        NUMERIC = (int, float, float64)

        return all(map(lambda x: isinstance(x, NUMERIC), array))

    @classmethod
    def zeros(cls, num_dimensions: int):
        return Point([0 for x in range(num_dimensions)])

    def distance_to(self, point) -> np.float64:
        """
        Distance of this point to another point.

        -- Parameters --
        point : Point
            Another point in space

        -- Return --
        np.float64
            Distance from this point to another point
        """
        self.enforce_same_dimension(point)
        dim_dist = np.array(
            [(self.array[i] - point.array[i]) ** 2 for i in range(self.array.size)]
        ).sum()
        return np.float64(np.sqrt(dim_dist))

    def enforce_same_dimension(self, point):
        """
        Enforces that this point and another are the same dimension. Throws an error if not the same

        -- Parameters --
        point : Point
            Another point
        """
        if self.size != point.size:
            raise Exception("Points are not the same size.")
        return

    def project_towards_point(self, point, x: np.float64):
        """
        Projects this point towards another point by x amount of cartesian distance.

        -- Parameters --
        point : Point
            The target Point where this point will be projected towards.
        x : float
            Amount to be projected

        -- Return --
        Point
            A point on the straight line between this point and the target point
            that is x amount towards the target point. If x amount is a point on
            the line that passes the target point, then the target point is returned.
        """
        self.enforce_same_dimension(point)
        x = np.float64(x)

        d = self.distance_to(point)
        p0 = self.array
        p1 = point.array
        p = x / d

        return Point(p0 + p * (p1 - p0))

class Domain:
    """
    A domain defines an n-dimensional volume. It is an immutable
    iterable containing a series of tuples (length of 2) that
    represent upper and lower limits, one for each axis. A domain
    in array form is defined as follows:
        domain = [(lower_i, upper_i) for each dimension]
    """

    def __init__(self, arr: tuple, granularity: np.float64 = 0.01):
        """
        Arguments:
            arr             (iterable):
                Defines a domain from an array in the form
                [(low_i, high_i) for i in range(N)]
                If defined, all other arguments are ignored.
            granularity  np.int64:
                Used for projecting points into a discrete domain.
        """

        self._inclusion_bounds: tuple[bool] = tuple(
            [(True, False) for x in range(len(arr))]
        )

        if Domain.is_domain(arr):
            self._arr: ndarray = np.array(arr)

        else:
            raise ValueError(
                f"Invalid array format for Domain!\nForm: [(low, high) for i in range(num_dimensions)]\nGot: {arr}"
            )

        self._granularity = granularity

        a = np.array([arr[0] for arr in self.array])
        b = np.array([arr[1] for arr in self.array])
        self._n_buckets = np.ceil((b - a) / self.granularity).astype(np.int32) + 1

    def as_dict(self) -> dict[str, Point]:
        arr = self.bounding_points
        return {
            "lower_bounds": arr[0].array.tolist(), 
            "upper_bounds": arr[1].array.tolist(),
            "granularity": self.granularity
        }

    def as_json(self) -> str:
        arr = self.bounding_points
        return json.dumps({
            "lower_bounds": arr[0].array.tolist(), 
            "upper_bounds": arr[1].array.tolist(),
            "granularity": self.granularity
        })

    @property
    def bounding_points(self) -> tuple[Point]:
        pA: Point
        pB: Point

        pA = Point([bounds[0] for bounds in self])
        pB = Point([bounds[1] for bounds in self])

        return (pA, pB)

    @property
    def dimensions(self) -> ndarray:
        "The dimensions of the array"
        f = lambda limits: limits[1] - limits[0]
        return np.array(tuple(map(f, self._arr)))

    @property
    def origin(self) -> Point:
        return Point([low for low, high in self])

    @property
    def array(self) -> ndarray:
        "The domain in array form."
        return self._arr

    @property
    def volume(self) -> float64:
        "The volume enclosed by the domain."
        return float64(reduce(lambda vol, axis: vol * axis, self.dimensions))

    @property
    def inclusion_bounds(self) -> tuple[bool]:
        return self._inclusion_bounds

    @inclusion_bounds.setter
    def inclusion_bounds(self, bounds: tuple[bool]):
        self._inclusion_bounds = bounds

    def get_inclusion_upper_bounds(self):
        return np.array(tuple(map(lambda bound: bound[1], self._inclusion_bounds)))

    def get_inclusion_lower_bounds(self):
        return np.array(map(lambda bound: bound[0], self._inclusion_bounds))

    @property
    def include_lower_bounds(self):
        return all(map(lambda bound: bound[0], self._inclusion_bounds))

    @include_lower_bounds.setter
    def include_lower_bounds(self, is_included: bool):
        self._inclusion_bounds = tuple(
            [(is_included, bound[1]) for bound in self._inclusion_bounds]
        )

    @property
    def include_upper_bounds(self):
        return all(map(lambda bound: bound[1], self._inclusion_bounds))

    @include_upper_bounds.setter
    def include_upper_bounds(self, is_included: bool):
        self._inclusion_bounds = tuple(
            [(bound[0], is_included) for bound in self._inclusion_bounds]
        )

    @property
    def granularity(self) -> np.float64:
        return self._granularity

    @property
    def n_buckets(self) -> np.ndarray:
        return self._n_buckets

    def __len__(self):
        # How to get len() of Domain
        return len(self._arr)

    def __iter__(self):
        # Define how to iterate through a Domain
        return self._arr.__iter__()

    def __mul__(self, scalar):
        # Defines how to multiply a Domain by a scalar
        if type(scalar) is int or type(scalar) is float:
            arr = []
            for limits in self:
                new_limits = (limits[0] * scalar, limits[1] * scalar)
                arr += [new_limits]

            return Domain(arr)

    def __getitem__(self, key: int32):
        # Define how to index a domain
        return self._arr[key]

    def __contains__(self, point: Point):

        if not isinstance(point, Point):
            print("Error happened due to point not being Point!")
            print(point)

        contains = []
        for i, x in enumerate(point.array):
            low = self.array[i].min()
            high = self.array[i].max()
            contains.append((x >= low) and (x <= high))

        return all(contains)

    def __str__(self):
        dims = [high - low for low, high in self]
        dims_str = str(dims[0])

        for i in range(len(dims) - 1):
            dims_str += f"x{dims[i]}"

        return f"{__class__.__name__}: {dims_str} at {self.origin}"

    @classmethod
    def from_dimensions(cls, dimensions: tuple[float64], origin: Point = None):
        """
        Create a domain from its domensions, and optionally tranlated according
        to its origin (i.e. bottom most corner.) Note: dimensions in this
        context simply means the lengths of its axes.
            # Dimensions (axes) = len(dimensions)

        Args:
            dimensions (iterable): The length of each axis.

            origin (Point, optional): The bottom most corner of the domain.
                Defaults to 0 for each axis.
        """
        dims = len(dimensions)

        if origin is None:
            origin = Point([0 for i in range(dims)])

        arr = list(map((lambda length, o: (o, length + o)), dimensions, origin))

        return Domain(arr)

    @classmethod
    def from_point_cloud(cls, points: tuple[Point]):
        """
        Returns the smallest domain that encloses the provided set of points.

        Args:
            points (list/tuple)
        """

        arr = [[x, x] for x in points[0]]
        for point in points:
            for d, axis in enumerate(point):
                if axis < arr[d][0]:
                    arr[d][0] = axis
                elif axis > arr[d][1]:
                    arr[d][1] = axis

        return Domain(arr)

    @classmethod
    def from_bounding_points(cls, pointA: Point, pointB: Point):
        """
        Returns a Domain that lies between point A and B. A and B must
        have the same number of dimensions.

        Args:
            pointA (tuple[float, float] | Point)
            pointB (tuple[float, float] | Point)
        """

        result = None

        if len(pointA) == len(pointB):
            result = Domain(
                [
                    (min(pointA[n], pointB[n]), max(pointA[n], pointB[n]))
                    for n in range(len(pointA))
                ]
            )
        else:
            raise ValueError(
                f"""
                    {__class__.__name__} Dimension mismatch between the two points!\n
                    Got: len(A) = {len(pointA)}, len(B) = {len(pointB)}
                """
            )

        return result

    @classmethod
    def normalized(cls, num_dimensions):
        "Returns a normalized domain with the given number of dimensions."
        return Domain([(0, 1) for x in range(num_dimensions)])

    @staticmethod
    def from_json(string: str):
        d = json.loads(string)
        return Domain.from_dict(d)

    @staticmethod
    def from_dict(d : dict):
        a = d["lower_bounds"]
        b = d["upper_bounds"]
        return Domain.from_bounding_points(a, b)
    

    @staticmethod
    def is_domain(array):
        """
        Checks formatting of the array to see if it is compatible with
        the Domain type.
        """

        NUMERIC = (int, float, int32, float64)

        isValid = True
        i = 0
        while i < len(array) and isValid:
            limits = array[i]

            if len(limits) != 2:
                isValid = False
                logging.info(
                    "Invalid limit, must have two elements (one lower limit and one upper limit.)"
                )

            # Are not both lower/upper limits numbers?
            elif not all(map(lambda l: isinstance(l, NUMERIC), limits)):
                isValid = False
                logging.info("Not all limits are of numeric type.")

            elif limits[0] > limits[1]:
                isValid = False
                logging.info("Lower limit is not lower than upper limit.")

            i += 1

        return isValid

    def project(self, point: Point) -> Point:
        """
        Project a point in the domain's space,
        to optimize a selection in the discretized space.
        """
        a = np.array([arr[0] for arr in self.array])
        b = np.array([arr[1] for arr in self.array])
        x = point.array

        # Amount of buckets
        n_buckets = np.ceil((b - a) / self.granularity)
        # n_buckets = self.n_buckets

        # Bucket index
        i = (x * n_buckets).astype(int)

        return Point(a + i * self.granularity)


class Grid:
    """
    An n-dimensional grid, defined by a resolution and origin.
    """

    def __init__(self, resolution: ndarray, origin: Point = None):
        """
        A "resolution" is an n-dimensional vector that describes the dimensions of a single voxel
        within an n-dimensional grid.

        The "origin" enables the translation of a grid.
        """

        self._origin = Point.zeros(len(resolution)) if origin is None else origin
        self._res = np.array(resolution)

    @property
    def resolution(self) -> ndarray:
        return self._res

    @property
    def origin(self) -> Point:
        return self._origin

    def __len__(self):
        return len(self._res)

    def calculate_index_domain(self, continuous_domain: Domain):
        pointA, pointB = continuous_domain.bounding_points

        pointA_index = self.calculate_point_index(pointA)
        pointB_index = self.calculate_point_index(pointB)

        axis_falls_on_grid = lambda axis, dim_index: numpy.isclose(
            (axis - self._origin[dim_index]) % self._res[dim_index], 0
        )

        inc_upper_bounds = continuous_domain.get_inclusion_upper_bounds()

        lst = []
        for i in range(len(pointB_index)):
            axis = pointB_index[i]
            is_included = inc_upper_bounds[i]

            on_grid = axis_falls_on_grid(axis, i)

            if not on_grid or is_included:
                lst += [axis]
            else:
                lst += [axis - 1]

        pointB_index_corrected = Point(lst)

        index_domain = Domain.from_bounding_points(pointA_index, pointB_index_corrected)
        index_domain.include_upper_bounds = True

        return index_domain

    def discretize_point(self, point: Point) -> Point:
        """
        Gets a point afixed to the grid. Rounds down to match grid coordinates.

        Returns:
            Point
        """
        return Point(
            [
                (axis - (axis % self._res[i]) - self._origin[i])
                for i, axis in enumerate(point)
            ]
        )

    def calculate_point_index(self, point: Point) -> ndarray:
        return np.array(
            tuple(
                map(
                    lambda axis, step, o: int32(
                        (axis - ((axis - o) % step) - o) / step
                    ),
                    point,
                    self._res,
                    self._origin,
                ),
            )
        )

    def convert_index_to_point(self, index: list[int32]):
        return Point(
            map(lambda i, step, o: i * step + o, index, self._res, self._origin)
        )


# class Edge:

#     def __init__(self, a : Point, b : Point):
#         """
#         An bidirectional edge between two points.

#         -- parameters --
#         a : Point
#             First point
#         b : Point
#             Second point.
#         """
#         return


class PolyLine:
    def __init__(self, points: list[Point]):
        self._points = points
        self._shape = np.array([p.array for p in points]).shape
        return

    @property
    def points(self) -> list[Point]:
        return self._points

    @property
    def shape(self):
        return self._shape

    def __str__(self) -> str:
        return str(np.array([p.array for p in self.points]))

    def copy(self):
        """
        Shallow copy of this Polyline.
        """
        return PolyLine(self.points)

    def plot(self) -> matplotlib.axes.Axes:
        """
        Plots the polyline.
        Requires plt.show() afterwords.
        Returns the axes.
        """

        # For now, only plot in 2 dimensions
        if not self.shape[1] == 2:
            raise NotImplementedError(
                "%d-D plotting is not yet implemented." % self.shape[1]
            )

        plt.figure(figsize=(5, 5))
        ax = plt.axes()

        x = [p.array[0] for p in self.points]
        y = [p.array[1] for p in self.points]
        ax.plot(x, y)
        return ax

    @staticmethod
    def plot_many(polylines):

        # For now, only plot in 2 dimensions
        if not polylines[0].shape[1] == 2:
            raise NotImplementedError(
                "%d-D plotting is not yet implemented." % polylines[0].shape[1]
            )

        plt.figure(figsize=(5, 5))
        ax = plt.axes()

        for polyline in polylines:
            x = [p.array[0] for p in polyline.points]
            y = [p.array[1] for p in polyline.points]
            ax.plot(x, y)
        return


def main():
    point = Point([0, 0.5, 1])
    domain = Domain.normalized(3)

    print(point in domain)


if __name__ == "__main__":
    main()
