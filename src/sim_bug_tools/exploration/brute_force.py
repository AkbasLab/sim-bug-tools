# Summer, put your stuff here

import numpy as np

from numpy import ndarray

from sim_bug_tools.structs import Point, Domain, Grid
from sim_bug_tools.simulation.simulation_core import Scorable

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

if __name__ == "__main__":
    # runs only if this file is the file being executed.
    # Example use of domain with grid
    domain = Domain.normalized(3)
    grid = Grid(resolution=(0.1, 0.2, 0.5))
    
    matrix = grid.construct_bucket_matrix(domain)
    
    print(matrix.shape)
    
    # Another way is to create a grid based on the dimensions
    # of the matrix you want. This simply divides the domain
    # into the segments acording to the shape.
    shape = (4, 4)
    grid = Grid.from_matrix_dimensions(domain, shape)
    
    matrix = grid.construct_bucket_matrix(domain)
    print(matrix.shape)
    exit() 
    
    # You are going to need to sample the Scoreable function
    # for each cell in the matrix.
    scoreable: Scorable
    some_point: Point
    
    some_score = scoreable.score(some_point)
    
    # How do you find a point within the scoreable that lies
    # on the grid? Simple, you can create an index into the
    # matrix and use that to generate a point from the grid.
    # If the matrix is 4x4, then it's indices go from (0,0)
    # to (3,3)
    for x in range(4):
        for y in range(4):
            your_point = grid.convert_index_to_point((x, y))
            

            
