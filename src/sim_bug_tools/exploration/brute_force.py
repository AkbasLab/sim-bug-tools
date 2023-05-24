# Summer, put your stuff here

import numpy as np

from numpy import ndarray
from typing import Callable
from scipy.ndimage import label, generate_binary_structure

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
            - `list(ndarray)` : The score matrix for each grid cell and the classification matrix for each grid cell
                - Shape is determined by the dimensions of the domain when sliced up by the grid. E.g.
    """
    scored_matrix: ndarray
    classification_matrix: ndarray
    # bucket matrix contains domain/grid res
    scored_matrix = grid.construct_bucket_matrix(domain)
    classification_matrix = scored_matrix

    # Iterating through the n-dimensional array and getting the score and classification
    for index, item in np.ndenumerate(scored_matrix):
        new_point = grid.convert_index_to_point(index)
        scored_matrix[index] = scorable.score(new_point)
        classification_matrix[index] = scorable.classify(new_point)
        
    return [scored_matrix, classification_matrix]

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
    discretized_envelopes = [0]
    # Getting an ndarray of the index in classification_matrix where the classification is true
    true_index = np.argwhere(classification_matrix)

    """
    In older versions of SciPy (before version 1.6.0), the generate_binary_structure and iterate_structure functions have a maximum dimension limit of 31. Attempting to generate structures with dimensions higher than this limit may result in an error.
    However, starting from SciPy version 1.6.0, these functions have been updated to support higher-dimensional structures.
    """
    # Generates a binary matrix to serve as the connectivity stucture for the label function. 
    # structure=1 is a connectivity without the diagonals
    # structure=2 is a connectivity including the diagonals
    if (include_diagonals):
        connectivity_matrix = generate_binary_structure(classification_matrix.ndim, 2)
    else:
        connectivity_matrix = generate_binary_structure(classification_matrix.ndim, 1)
    
    # num_clusters is the number of groups found
    # labels is an ndarray where the clusters of true values are replaced with 1, 2, 3,... depending on what cluster its in
    labels, num_clusters = label(classification_matrix, structure=connectivity_matrix)
    print("************************* LABELS ************************")
    print(labels)

    # Grouping all the indices of the matching clusters and putting them all in an array
    unique_values = np.unique(labels)
    print("\n\nVALUESSSSSSSS : ",unique_values,"\n\n")
    #grouped_indices = [np.where(labels == value)[0] for value in unique_values]
    #discretized_envelopes = np.array(grouped_indices, dtype=ndarray)
    
    return discretized_envelopes

# True-Boundary finding algorithm
def true_boundary_algorithm(score_matrix: ndarray, envelopes: ndarray) -> ndarray:
    """
    - True-Boundary finding algorithm
        - We have some N-D volume classified into two bodies: Target and Non-Target, this method identifies the cells that lie on the boundary.
        - Inputs: 
            - `ndarray` scoreMatrix : The score matrix for each grid cell
            - `list<ndarray>` envelopeIndices : The list of indices of cells within the contiguous envelope
        - Outputs:
            - `list<ndarray>` : The list of indices that fall on the boundary of the N-D envelope's surface.
    """
    pass

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

if __name__ == "__main__":
    ######### Test values #########
    ndims = 3
    scorable = ProbilisticSphere(Point([0.5]*ndims), 0.5, 0.25)
    domain = Domain.normalized(3)
    grid = Grid(resolution=[0.1]*ndims)

    ###### Testing functions ######
    # Score and classification ndarray's where score = [score_matrix, classification_matrix]
    score_class = brute_force_grid_search(scorable, domain, grid)
    print("\n**********************\nscore matrix:\n",score_class[0])
    print("\n**********************\nclassificatio matrix:\n",score_class[1])
    
    e = true_envelope_finding_alg(score_class[1], scorable, include_diagonals=False) 
    print(e)       
