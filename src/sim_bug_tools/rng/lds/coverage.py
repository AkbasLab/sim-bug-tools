import numpy as np
from numpy import ndarray, int32, float64
from dataclasses import dataclass
from sim_bug_tools.structs import Domain, Point, Grid
import sim_bug_tools.utils as utils
import logging


@dataclass
class CoverageData:
    "Represents all of the information related to coverage."

    coverage_map: dict
    coverage_hits: int32
    num_points: int32
    bucket_domain: Domain


@dataclass
class Sample:
    """
    Represents a set of points with a given number of named dimensions and a domain.
    """

    _points: tuple[Point]
    _axes_names: tuple[str]
    _domain: Domain

    def __post_init__(self):
        # Large number of points may not scale well

        if not all([p in self._domain for p in self._points]):
            raise ValueError("Invalid domain! Not all points reside within domain.")

        elif len(self.domain) != len(self.axes_names):
            raise ValueError(
                "Dimension mismatch! Resolution, domain, and number of names do not match."
            )

    @property
    def points(self) -> tuple[Point]:
        return self._points

    @property
    def axes_names(self) -> tuple[str]:
        return self._axes_names

    @property
    def domain(self) -> Domain:
        return self._domain

    @property
    def num_dimensions(self) -> int32:
        return int32(len(self._domain))

    @property
    def transpose(self) -> list[float64]:
        """
        Gets each axis of the sample as a list of floats.
        """
        result = [[] for x in range(len(self._points[0]))]
        for point in self._points:
            for i, axis in enumerate(point):
                result[i] += [axis]

        return result

    def get_coverage_data(self, grid: Grid) -> CoverageData:
        coverage_map: dict[tuple, int32] = {}
        coverage_hits: int = 0
        bucket_domain: Domain = grid.calculate_index_domain(self._domain)

        _ignored_points = 0

        for point in self._points:
            coord = tuple(grid.calculate_point_index(point))

            if coord in coverage_map:
                coverage_map[coord] += 1
            elif point in self._domain:
                coverage_map[coord] = 1
            else:
                _ignored_points += 1

        logging.info(
            f"Ignored coverage for {_ignored_points} points that were outside of domain."
        )

        return CoverageData(
            coverage_map, coverage_hits, len(self._points), bucket_domain
        )


class Heatmap:
    def __init__(self, sample: Sample, grid: Grid):

        num_dimensions = len(sample.axes_names)

        if num_dimensions < 2:
            raise ValueError(
                "Cannot create a heatmap from a sample with fewer than 2 dimensions."
            )

        self._sample = sample
        self._grid = grid

        self._axes_coords: dict[str, int32] = {name: 0 for name in sample.axes_names}
        self._axes_index: dict[str, int32] = {
            axis: num_dimensions - i - 1 for i, axis in enumerate(sample.axes_names)
        }

        self._active_axes: list[str] = [sample.axes_names[0], sample.axes_names[1]]

        self._setup_matrix()

    def _set_coord_in_matrix_by_vector(self, vect, value, matrix=None):
        if matrix is None:
            return self._set_coord_in_matrix_by_vector(vect, value, self._matrix)

        elif len(vect) > 1:
            return self._set_coord_in_matrix_by_vector(
                vect[:-1], value, matrix[vect[-1]]
            )

        else:
            matrix[vect[0]] = value

    def _setup_matrix(self):
        "Builds the n-dimensional matrix from the sample's coverage."

        self._matrix: ndarray
        coverage_data = self._sample.get_coverage_data(self._grid)

        dimensions = coverage_data.bucket_domain.dimensions
        self._matrix = np.zeros(dimensions)

        for coord, num_hits in coverage_data.coverage_map.items():
            self._set_coord_in_matrix_by_vector(coord, num_hits)


class Heatmap_old:
    """
    Provides the ability to take a 2-dimensional slice of an n-dimensional coverage map. Requires
    a sample of points to measure the coverage of said sample.
    """

    def __init__(self, sample: Sample, grid: Grid):
        num_dimensions = len(sample.axes_names)
        self._sample: Sample = sample
        self._grid: Grid = grid
        self._axes_coords: dict[str, int32] = {name: 0 for name in sample.axes_names}
        self._axes_index: dict[str, int32] = {
            axis: num_dimensions - i - 1 for i, axis in enumerate(sample.axes_names)
        }

        self._active_axes: list[str] = [sample.axes_names[0], sample.axes_names[1]]

        self._y: int32 = num_dimensions - 2
        self._x: int32 = num_dimensions - 1

        self._setup_matrix()

    @property
    def active_axes(self) -> list[str]:
        return self._active_axes

    @property
    def axes_index(self) -> dict[str, int32]:
        return self._axes_index

    @property
    def axes_names(self) -> tuple[str]:
        return self._axes_names

    @property
    def axes(self) -> dict[str, int32]:
        return self._axes_coords

    @property
    def constant_axes(self) -> dict[str, int32]:
        return {
            name: value
            for name, value in self._axes_coords.items()
            if name not in self._active_axes
        }

    @property
    def frame(self) -> ndarray:
        coords = list(utils.sortByDict(self.constant_axes, self._axes_index).values())
        return self._get_matrix_slice(self._matrix, coords)

    @property
    def matrix(self) -> ndarray:
        return self._matrix

    def swap_axes(self, axis1: str = None, axis2: str = None):
        if axis1 in self._axes_index and axis2 in self._axes_index:
            axis1_oldIndex = self._axes_index[axis1]
            axis2_oldIndex = self._axes_index[axis2]

            self._matrix = self._matrix.swapaxes(axis1_oldIndex, axis2_oldIndex)

            # If one of the axes is active, swap the other with the one (self.active_axes)
            for i, active_axis in enumerate(self._active_axes):
                if axis1 == active_axis:
                    self._active_axes[i] = axis2
                elif axis2 == active_axis:
                    self._active_axes[i] = axis1

            self._axes_index[axis1] = axis2_oldIndex
            self._axes_index[axis2] = axis1_oldIndex

        else:
            logging.warning(f"[{__class__.__name__}.swapAxes] Invalid axis name.")

    def set_frame_axes(self, new_x_axis: str = None, new_y_axis: str = None):
        """
        Changes the selected axes to display on the x and y axis. The provided axis/axes
        must be a member of self.axes

        Args:
            new_x_axis [str]: Name of the axis to switch to.
            new_y_axis [str]: Name of the axis to switch to.
        """

        if new_x_axis is not None:
            self.swap_axes(self._active_axes[0], new_x_axis)

        # The second condition ensures that a double switch does not occur.
        if new_y_axis is not None and self._active_axes[1] != new_y_axis:
            self.swap_axes(self._active_axes[1], new_y_axis)

    def translate_frame(self, coords: dict[str, int32]):
        for key, val in coords.items():
            if key in self.constant_axes:
                self._axes_coords[key] = val
            else:
                logging.warning(
                    f"[{__class__.__name__}.translateFrame] An axis name for a coordinate was invalid!\n\
                    Ignored: {key}:{val}"
                )

    def _setup_matrix(self):
        "Converts the dict coverage object into a numpy array heatmap."

        coverage_data = self._sample.get_coverage_data(self._grid)
        coverage_map = coverage_data.coverage_map
        self._matrix_domain = coverage_data.bucket_domain

        print(self._matrix_domain)

        # Adding 1 to each axis to include zero as a bucket
        dimensions = tuple(map(lambda axis: axis + 1, self._matrix_domain.dimensions))

        self._matrix = np.zeros(dimensions)
        for coord, hits in coverage_map.items():
            self._set_by_vector(coord, hits)

    def _set_by_vector(
        self, vector: tuple[int32], value: int32, matrix: ndarray = None
    ):
        """
        Traverses the matrix with each proceeding element within the vector as an index, setting the
        final indexed element within the matrix to the provided value.
        """
        if matrix is None:
            self._set_by_vector(vector, value, self._matrix)
        elif len(vector) > 1:
            self._set_by_vector(vector[1:], value, matrix[vector[0]])
        else:
            matrix[vector[0]] = value

    def _get_matrix_slice(self, matrix: ndarray, coord: tuple[int32]) -> ndarray:
        if coord:
            return self._get_matrix_slice(matrix[coord[0]], coord[1:])
        else:
            return matrix
