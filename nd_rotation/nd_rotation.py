import sim_bug_tools.structs as structs
import numpy as np

def generate_rotation_matric_function(
    point: structs.Point,
    normal : np.ndarray, 
    other : structs.Point
) -> np.ndarray:
    
    return

def rotate_stackoverflow():
    """
    Source:
    https://math.stackexchange.com/questions/2144153/n-dimensional-rotation-matrix

    Similar to
    https://github.com/davidblitz/torch-rot/blob/main/torch_rot/rotations.py
    """
    # input vectors
    v1 = np.array( [1,1,1,1,1,1] )
    v2 = np.array( [2,3,4,5,6,7] )

    # Gram-Schmidt orthogonalization
    n1 = v1 / np.linalg.norm(v1)
    v2 = v2 - np.dot(n1,v2) * n1
    n2 = v2 / np.linalg.norm(v2)

    

    # rotation by pi/2
    a = np.pi/2

    I = np.identity(6)

    R = I + ( np.outer(n2,n1) - np.outer(n1,n2) ) * np.sin(a) \
        + ( np.outer(n1,n1) + np.outer(n2,n2) ) * (np.cos(a)-1)
    return



def rotation(n, dims, angle):
    """
    https://github.com/scipy/scipy/issues/12693#issuecomment-674419426

    Parameters
    ------------
    n : int
        dimension of the space
    dims : 2-tuple of ints
        the vector indices which form the plane to perform the rotation in
    angle : array_like of shape (M...,)
        broadcasting angle to rotate by

    Returns
    --------
    m : array_like of shape (M..., n, n)
        (stack of) rotation matrix
    """
    i, j = dims
    assert i != j
    c = np.cos(angle)
    s = np.sin(angle)
    arr = np.eye(c.shape + (n, n), dtype=c.dtype)
    arr[..., i, i] = c
    arr[..., i, j] = -s
    arr[..., j, i] = s
    arr[..., j, j] = c
    return arr

def main():
    
    return

if __name__ == "__main__":
    main()