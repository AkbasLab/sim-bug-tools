# Summer, put your stuff here

import numpy as np
from copy import copy

from numpy import ndarray
import random
from typing import Callable
from scipy.ndimage import label, generate_binary_structure, binary_erosion

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
    scored_matrix: ndarray
    # bucket matrix contains domain/grid res
    scored_matrix = grid.construct_bucket_matrix(domain)

    # Iterating through the n-dimensional array and getting the score and classification
    for index, item in np.ndenumerate(scored_matrix):
        new_point = grid.convert_index_to_point(index)
        scored_matrix[index] = scorable.score(new_point)
        
    return scored_matrix

# True-Envelope finding algorithm
def true_envelope_finding_alg(classification_matrix: ndarray, scoreable: Scorable, include_diagonals: bool = True) -> ndarray:
    """
    - True-Envelope finding algorithm
        - Finds the set of points that fall within a contiguous envelope. 
        - Inputs: 
            - `ndarray` classification_matrix : The classification matrix for each grid cell
            - `scoreable` Scorable : The score classifying function
            - `ndarray | tuple` start_index : The starting index to the score matrix, start_index: ndarray'''
        - Outputs: `ndarray` : array of indices of cells within the contiguous, discretized envelope
    """
    discretized_envelopes: ndarray

    """
    In older versions of SciPy (before version 1.6.0), the generate_binary_structure and iterate_structure functions have a maximum dimension limit of 31. Attempting to generate structures with dimensions higher than this limit may result in an error.
    However, starting from SciPy version 1.6.0, these functions have been updated to support higher-dimensional structures.
    """
    # Generates a binary matrix to serve as the connectivity stucture for the label function. 
    # connectivity=1 is a connectivity without the diagonals
    # connectivity=2 is a connectivity including the diagonals
    if (include_diagonals):
        connectivity_matrix = generate_binary_structure(rank=classification_matrix.ndim, connectivity=2)
    else:
        connectivity_matrix = generate_binary_structure(rank=classification_matrix.ndim, connectivity=1)
    
    # num_clusters is the number of groups found
    # labeled_groups is an ndarray where the clusters of true values are replaced with 1, 2, 3,... depending on what cluster its in
    labeled_groups, num_clusters = label(classification_matrix, structure=connectivity_matrix)
    print("******** LABELED ARRAY ********")
    print(labeled_groups)

    # Grouping all the indices of the matching clusters and putting them all in an array
    unique_labels = np.unique(labeled_groups)
    grouped_indices = []
    print("\nUnique labels (aka groups) : ",unique_labels,"\n")
    for ulabel in range(1, num_clusters+1):
        # Grouping all the index of the current label into a list
        current_group = []
        for index, item in np.ndenumerate(labeled_groups):
            if ulabel == item:
                current_group.append(index)
        print(" ****** CURRENT GROUP", ulabel, "*******\n",current_group)
        # Appending the current group of indices to the grouped indices array
        grouped_indices.append(current_group)

    discretized_envelopes = grouped_indices

    return discretized_envelopes

# True-Boundary finding algorithm
def true_boundary_algorithm(score_matrix: ndarray, envelope_indices: ndarray) -> ndarray:
    """
    - True-Boundary finding algorithm
        - We have some N-D volume classified into two bodies: Target and Non-Target, this method identifies the cells that lie on the boundary.
        - Inputs: 
            - `ndarray` scoreMatrix : The score matrix for each grid cell
            - `list<ndarray>` envelopeIndices : The list of indices of cells within the contiguous envelope
        - Outputs:
            - `list<ndarray>` : The list of indices that fall on the boundary of the N-D envelope's surface.
    """
    classification_matrix: ndarray
    classification_matrix = score_matrix
    # Getting classification matrix from the score matrix:
    # for index, item in np.ndenumerate(score_matrix):
    #     classification_matrix[index] = scorable.classify_score(item)

    # Apply binary erosion to identify the true values touching false values
    eroded_array = binary_erosion(classification_matrix)
    print("Erroded array:\n",eroded_array)

    # Find the indices where the classification_matrix is True and eroded_array is False
    all_bound_indices = np.argwhere(classification_matrix & ~eroded_array)
    print("All bound indices:\n",all_bound_indices)

    # Get the indices that are 
    true_bound_indices = np.intersect1d(all_bound_indices, envelope_indices)
    print("Envelope indices:\n",true_bound_indices)

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


def test_brute_force_grid_search():
    ######### Test values #########
    ndims = 3
    scorable = ProbilisticSphere(Point([0.5]*ndims), 0.5, 0.25)
    domain = Domain.normalized(3)
    grid = Grid(resolution=[0.1]*ndims)
    ###### Testing functions ######
    score_class = brute_force_grid_search(scorable, domain, grid)
    #print("\n**********************\nscore matrix:\n",score_class)

def test_true_envelope_finding_alg(boolean_array: ndarray, scoreable: Scorable):
    # With Diagonals:
    print("\n************************* With Diagonals *****************************\n")
    test_w_diagonal = true_envelope_finding_alg(boolean_array, scorable, True)
    print("**** ENVELOPES ****\n",test_w_diagonal)
    print("\n**********************************************************************\n")
    # Without Diagonals:
    print("\n************************* Without Diagonals *****************************\n")
    test_wo_diagonal = true_envelope_finding_alg(boolean_array, scorable, False)
    print("**** ENVELOPES ****\n",test_wo_diagonal)
    print("\n*************************************************************************\n")
    return test_w_diagonal, test_wo_diagonal

def test_true_boundary_algorithm(score_matrix: ndarray, envelope: ndarray):
    print("Classification matrix:\n",score_matrix)
    print("Envelope indices\n", envelope)
    indices = true_boundary_algorithm(score_matrix, envelope)
    print("Boundary indices:\n", indices)

if __name__ == "__main__":
    ######### Test values #########
    ndims = 3
    scorable = ProbilisticSphere(Point([0.5]*ndims), 0.5, 0.25)
    domain = Domain.normalized(3)
    grid = Grid(resolution=[0.1]*ndims)
    
    boolean_array = np.zeros(shape=(5,5))
    for i, v, in np.ndenumerate(boolean_array):
        boolean_array[i] = bool(random.getrandbits(1))
    boolean_array = np.array(boolean_array)

    # calling test functions:
    #test_brute_force_grid_search()
    print("True envelope finding:\n")
    indices_w_diagonal, indices_w_diagonal = test_true_envelope_finding_alg(boolean_array, scorable)
    print("True boundary with diagonals:\n")
    test_true_boundary_algorithm(boolean_array, indices_w_diagonal)
    

