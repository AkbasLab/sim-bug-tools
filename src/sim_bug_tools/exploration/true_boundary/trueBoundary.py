
"""
- Brute Force Grid Search
    - Samples all cells of an N-D grid, scoring each.
    - Inputs: 
        - `Scoreable` scoreable : the scoreable object that is being explored
        - `Domain` domain : the domain to search
        - `Grid` grid : the grid pattern that discretizes the domain
    - Outputs: 
        - `ndarray` : The score matrix for each grid cell
            - Shape is determined by the dimensions of the domain when sliced up by the grid. E.g.
"""


"""
- True-Envelope finding algorithm
    - Finds the set of points that fall within a contiguous envelope. 
    - Inputs: 
        - `ndarray` scoreMatrix : The score matrix for each grid cell
        - `Callable<<float>, bool>` scoreClassifier : The score classifying function
        - `ndarray | tuple` i0 : The starting index to the score matrix
    - Outputs: `list<ndarray>` : list of indices of cells within the contiguous, discretized envelope
"""


"""
- True-Boundary finding algorithm
    - We have some N-D volume classified into two bodies: Target and Non-Target, this method identifies the cells that lie on the boundary.
    - Inputs: 
        - `ndarray` scoreMatrix : The score matrix for each grid cell
        - `list<ndarray>` envelopeIndices : The list of indices of cells within the contiguous envelope
    - Outputs:
        - `list<ndarray>` : The list of indices that fall on the boundary of the N-D envelope's surface.
"""

from abc import ABC
class Scorable(ABC):
    @abstract
    def score(self, p: Point) -> ndarray:
        raise NotImplementedError()

    @abstract
    def classify_score(self, score: ndarray) -> bool:
        raise NotImplementedError()

    def classify(self, p: Point) -> bool:
        return self.classify_score(self.score(p))

    @abstract
    def get_input_dims(self):
        raise NotImplementedError()

    @abstract
    def get_score_dims(self):
        raise NotImplementedError()