import sim_bug_tools.structs as structs
import numpy as np

def generate_rotation_matric_function(
    point: structs.Point,
    normal : np.ndarray, 
    other : structs.Point
) -> np.ndarray:
    
    return






def rotation_matrix(theta: float, n_1 : np.ndarray, n_2 : np.ndarray) -> np.ndarray:
    """
    This method returns a rotation matrix which rotates any vector 
    in the 2 dimensional plane spanned by 
    @n1 and @n2 an angle @theta. The vectors @n1 and @n2 have to be orthogonal.
    Inspired by 
    https://analyticphysics.com/Higher%20Dimensions/Rotations%20in%20Higher%20Dimensions.htm
    https://github.com/davidblitz/torch-rot/blob/main/torch_rot/rotations.py
    :param @n1: first vector spanning 2-d rotation plane, needs to be orthogonal to @n2
    :param @n2: second vector spanning 2-d rotation plane, needs to be orthogonal to @n1
    :param @theta: rotation angle
    :returns : rotation matrix
    """
    dim = len(n_1)
    assert len(n_1) == len(n_2)
    assert (np.abs(np.dot(n_1, n_2)) < 1e-4)
    return (np.eye(dim) +
        (np.outer(n_2, n_1) - np.outer(n_1, n_2)) * np.sin(theta) +
        (np.outer(n_1, n_1) + np.outer(n_2, n_2)) * (np.cos(theta) - 1)
    )

def rotate_matrix(theta : float, v1 : np.array, v2 : np.array, enforce_orthogonal : bool = True) -> np.array:
    """
    This method returns a rotation matrix which rotates any vector 
    in the 2 dimensional plane spanned by @v1 and @v2 an angle @theta. 
    The vectors @v1 and @v2 should be orthogonal when @enforce_orthogonal is 
    True (default.)

    Inspired by:
    https://math.stackexchange.com/questions/2144153/n-dimensional-rotation-matrix
    https://github.com/davidblitz/torch-rot/blob/main/torch_rot/rotations.py
    https://analyticphysics.com/Higher%20Dimensions/Rotations%20in%20Higher%20Dimensions.htm



    """

    # input vectors
    if len(v1) != len(v2):
        raise ValueError("Input vectors are not the same length.")
    n_dim = len(v1)


    if not enforce_orthogonal:
        # Gram-Schmidt orthogonalization
        n1 = v1 / np.linalg.norm(v1)
        v2 = v2 - np.dot(n1,v2) * n1
        n2 = v2 / np.linalg.norm(v2)
    else:
        n1 = v1
        n2 = v2

    
    if not (np.dot(n1,n2) < 1e-4):
        raise Exception("Vectors %s and %s not orthogonal." % (n1, n2))

    I = np.identity(n_dim)

    R = I + ( np.outer(n2,n1) - np.outer(n1,n2) ) * np.sin(theta) \
        + ( np.outer(n1,n1) + np.outer(n2,n2) ) * (np.cos(theta)-1)
    return R

def main():
    r1 = rotation_matrix(np.pi/2, [0,0,1], [0,1,0])
    print(r1)
    print()

    r2 = rotate_matrix(np.pi/2, [0,0,1], [0,1,0])
    print(r2)
    return

if __name__ == "__main__":
    main()