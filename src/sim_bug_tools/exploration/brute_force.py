# Summer, put your stuff here

import numpy as np
from copy import copy

from numpy import ndarray
import random
from typing import Callable
from scipy.ndimage import label, generate_binary_structure, binary_erosion, iterate_structure

from sim_bug_tools.structs import Point, Domain, Grid
from sim_bug_tools.simulation.simulation_core import Scorable, Graded


# Brute Force Grid Search
def brute_force_grid_search(scorable: Scorable, domain: Domain, grid: Grid) -> ndarray:
    """
    - Brute Force Grid Search
        - Samples all cells of an N-D grid, scoring each.
        - Inputs:
            - `Scoreable` scoreable : the scoreable object that is being explored
            - `Domain` domain : the domain to search
            - `Grid` grid : the grid pattern that discretizes the domain
        - Outputs:
            - `list(ndarray)` : The score matrix for each grid cell
                - Shape is determined by the dimensions of the domain when sliced up by the grid. E.g.
    """
    # bucket matrix contains domain/grid res
    scored_matrix = grid.construct_bucket_matrix(domain)

    # Iterating through the n-dimensional array and getting the score and classification
    for index, item in np.ndenumerate(scored_matrix):
        new_point = grid.convert_index_to_point(index)
        scored_matrix[index] = scorable.score(new_point)

    return scored_matrix


# True-Envelope finding algorithm
def true_envelope_finding_alg(
        scorable: Scorable, score_matrix: ndarray, connectivity=2
) -> list:
    """
    - True-Envelope finding algorithm
        - Finds the set of points that fall within a contiguous envelope.
        - Inputs:
            - `ndarray` classification_matrix : The classification matrix for each grid cell
            - `connectivity` 1,2,rank : Default is 2. 1 excludes diagonals. 2 includes diagonals. 'rank' includes all neighbors.
            - `ndarray | tuple` start_index : The starting index to the score matrix, start_index: ndarray'''
        - Outputs: `list[list[ndarray]]` : A list of groups of the indices of
            the envelopes. There will be one group for each envelope.

    In older versions of SciPy (before version 1.6.0), the generate_binary_structure and iterate_structure functions have a maximum dimension limit of 31. Attempting to generate structures with dimensions higher than this limit may result in an error.
    However, starting from SciPy version 1.6.0, these functions have been updated to support higher-dimensional structures.
    """
    class_matrix = copy(score_matrix)
    for index, item in np.ndenumerate(score_matrix):
        class_matrix[index] = scorable.classify_score(item)
    # Generates a binary matrix to serve as the connectivity stucture for the label function.
    connectivity_matrix = generate_binary_structure(
        rank=class_matrix.ndim, connectivity=connectivity
    )

    # num_clusters is the number of groups found
    # labeled_groups is an ndarray where the clusters of true values are replaced with 1, 2, 3,... depending on what cluster its in
    labeled_groups, num_clusters = label(
        class_matrix, structure=connectivity_matrix
    )
    print("******** LABELED ARRAY ********")
    print(labeled_groups)

    # Grouping all the indices of the matching clusters and putting them all in an array
    unique_labels = np.unique(labeled_groups)
    grouped_indices = []
    print("\nUnique labels (aka groups) : ", unique_labels, "\n")
    for ulabel in range(1, num_clusters + 1):
        # Grouping all the index of the current label into a list
        current_group = []
        for index, item in np.ndenumerate(labeled_groups):
            if ulabel == item:
                current_group.append(index)
        print(" ****** CURRENT GROUP", ulabel, "*******\n", current_group)
        # Appending the current group of indices to the grouped indices array
        grouped_indices.append(current_group)

    discretized_envelopes = grouped_indices

    return discretized_envelopes


# True-Boundary finding algorithm
def true_boundary_algorithm(
    scorable: Scorable, score_matrix: ndarray, envelope_indices: ndarray
) -> ndarray:
    """
    - True-Boundary finding algorithm
        - We have some N-D volume classified into two bodies: Target and Non-Target, this method identifies the cells that lie on the boundary.
        - Inputs:
            - `ndarray` classification_matrix : The classification matrix for each grid cell
            - `ndarray` envelope_indices : The list of indices of cells within a single contiguous envelope
        - Outputs:
            - `ndarray` : The list of indices that fall on the boundary of the N-D envelope's surface.
    """
    # print("Classification matrix:\n",classification_matrix)
    classification_matrix = copy(score_matrix)
    for index, item in np.ndenumerate(score_matrix):
        classification_matrix[index] = scorable.classify_score(item)
    # Changing classification matrix from 0's and 1's to True/False
    class_as_bool = classification_matrix.astype(bool)

    # Apply binary erosion to identify the true values touching false values
    eroded_array = binary_erosion(class_as_bool)
    # print("Erroded array:\n",eroded_array)

    # Find the indices where the classification_matrix is True and eroded_array is False
    all_bound_indices = np.argwhere(class_as_bool & ~eroded_array)
    # print("All bound indices:\n",all_bound_indices)

    # Making array list into ndarray
    envelope_indices = np.stack(envelope_indices)
    # print("Envelope indices:\n", envelope_indices)

    # Reshaping to 2D arrays to compare for the matching indices
    array1_2d = all_bound_indices.reshape(-1, all_bound_indices.shape[-1])
    array2_2d = envelope_indices.reshape(-1, envelope_indices.shape[-1])

    # Getting the indices that are in the evelope and the boundaries arrays
    matching_rows = np.where(np.all(array1_2d[:, None] == array2_2d, axis=-1))[0]

    # Reshape back to all_bound_indices shape
    true_bound_indices = all_bound_indices.reshape(-1, *all_bound_indices.shape[1:])[
        matching_rows
    ]
    # print("True boundary indices:\n",true_bound_indices)

    return true_bound_indices


class ProbilisticSphere(Graded):
    def __init__(self, loc: Point, radius: float, lmbda: float):
        """
        Probability density is formed from the base function f(x) = e^-(x^2),
        such that f(radius) = lmbda and is centered around the origin with a max
        of 1.

        Args:
            loc (Point): Where the sphere is located
            radius (float): The radius of the sphere
            lmbda (float): The density of the sphere at its radius
        """
        self.loc = loc
        self.radius = radius
        self.lmda = lmbda
        self.ndims = len(loc)

        self._c = 1 / radius**2 * np.log(1 / lmbda)

    def score(self, p: Point) -> ndarray:
        "Returns between 0 (far away) and 1 (center of) envelope"
        dist = self.loc.distance_to(p)

        return np.array(1 / np.e ** (self._c * dist**2))

    def classify_score(self, score: ndarray) -> bool:
        return np.linalg.norm(score) > self.lmda

    def gradient(self, p: Point) -> np.ndarray:
        s = p - self.loc
        s /= np.linalg.norm(s)

        return s * self._dscore(p)

    def get_input_dims(self):
        return len(self.loc)

    def get_score_dims(self):
        return 1

    def generate_random_target(self):
        v = np.random.rand(self.get_input_dims())
        v = self.loc + Point(self.radius * v / np.linalg.norm(v) * np.random.rand(1))
        return v

    def generate_random_nontarget(self):
        v = np.random.rand(self.get_input_dims())
        v = self.loc + Point(
            self.radius * v / np.linalg.norm(v) * (1 + np.random.rand(1))
        )
        return v

    def boundary_err(self, b: Point) -> float:
        "Negative error is inside the boundary, positive is outside"
        return self.loc.distance_to(b) - self.radius

    def _dscore(self, p: Point) -> float:
        return -self._c * self.score(p) * self.loc.distance_to(p)


class ProbilisticSphereCluster(Graded):
    def __init__(self, spheres: list[ProbilisticSphere]):
        """
        Probability density is formed from the base function f(x) = e^-(x^2),
        such that f(radius) = lmbda and is centered around the origin with a max
        of 1.

        Args:
            loc (Point): Where the sphere is located
            radius (float): The radius of the sphere
            lmbda (float): The density of the sphere at its radius
        """
        self.spheres = spheres

    def score(self, p: Point) -> ndarray:
        "Returns between 0 (far away) and 1 (center of) envelope"
        return sum(map(lambda s: s.score(p), self.spheres))

    def classify_score(self, score: ndarray) -> bool:
        return any(map(lambda s: s.classify_score(score), self.spheres))

    def gradient(self, p: Point) -> np.ndarray:
        raise NotImplementedError()

    def get_input_dims(self):
        return len(self.spheres[0].loc)

    def get_score_dims(self):
        return 1

    def generate_random_target(self):
        raise NotImplementedError()

    def generate_random_nontarget(self):
        raise NotImplementedError()

    def boundary_err(self, b: Point) -> float:
        raise NotImplementedError()


def test_cluster():
    from sim_bug_tools.graphics import Grapher
    import matplotlib.pyplot as plt

    ndims = 3
    domain = Domain.normalized(ndims)
    grid = Grid([0.1] * ndims)

    sphere1 = ProbilisticSphere(Point(0, 0, 0), 0.2, 0.25)
    sphere2 = ProbilisticSphere(Point([0.5] * ndims), 0.3, 0.25)
    sphere3 = ProbilisticSphere(Point(0, 0, 0.8), 0.2, 0.25)
    scoreable = ProbilisticSphereCluster([sphere1, sphere2, sphere3])
    score_matrix = brute_force_grid_search(scoreable, domain, grid)

    envelopes_list = true_envelope_finding_alg(scoreable, score_matrix, True)
    envelopes = map(
        lambda env: list(map(grid.convert_index_to_point, env)),
        envelopes_list,
    )

    bound = []
    bound1 = true_boundary_algorithm(scoreable, score_matrix, envelopes_list[0])
    bound.append(np.split(bound1, bound1.shape[0]))
    bound2 = true_boundary_algorithm(scoreable, score_matrix, envelopes_list[1])
    bound.append(np.split(bound2, bound2.shape[0]))
    bound3 = true_boundary_algorithm(scoreable, score_matrix, envelopes_list[2])
    bound.append(np.split(bound3, bound3.shape[0]))
    print(bound)
    boundaries = map(
        lambda env2: list(map(grid.convert_index_to_point, env2)),
        bound,
    )

    # print(f"There were {len(envelopes)} envelopes in the space.")

    colors = ["red", "green", "blue", "yellow", "cyan"]

    g = Grapher(ndims == 3, domain)

    # # Plot for envelopes
    for env, color in zip(envelopes, colors):
        g.plot_all_points(env, color=color)
    # plt.show()
    
    # plot for boundary points
    for env2, color in zip(boundaries, colors):
        g.plot_all_points(env2, color="gray")
    plt.show()

if __name__ == "__main__":
    test_cluster()

    
