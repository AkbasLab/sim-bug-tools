import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes

def rotate_matrix(
        theta : float, 
        v1 : np.ndarray, 
        v2 : np.ndarray, 
        enforce_orthogonal : bool = True
    ) -> np.ndarray:
    """
    This method returns a rotation matrix which rotates any vector 
    in the 2 dimensional plane spanned by @v1 and @v2 an angle @theta. 
    The vectors @v1 and @v2 should be orthogonal when @enforce_orthogonal is 
    True (default.)

    Inspired by:
    https://math.stackexchange.com/questions/2144153/n-dimensional-rotation-matrix
    https://github.com/davidblitz/torch-rot/blob/main/torch_rot/rotations.py
    https://analyticphysics.com/Higher%20Dimensions/Rotations%20in%20Higher%20Dimensions.htm

    -- Parameters --
    theta : float
        The angle of rotation in radians.
    v1, v2 : np.array
        Vectors that describe the 2D plane.
    enforce_orthogonal : bool (default=True)
        When False, used Gram-Schmidt orthogonalization to generate
        orthogonal vectors for v1 and v2.
        When True, the function will throw an error if v1 and v2 are not
        orthogonal.
    
    -- Return --
    R : np.ndarray
        Rotation Matrix
        Apply the rotation using np.dot(R, vector)
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

def new_axes() -> matplotlib.axes.Axes:
    """
    Creates a new plot and axes

    -- Return --
    matplotlib.axes
    """
    plt.figure(figsize = (5,5))
    return plt.axes()

def rotate_2d_test():
    # Setup the plot
    ax = new_axes()

    ax.hlines(0, -.5, .5, linestyles="dashed", color="black")
    ax.vlines(0, -.5, .5, linestyles="dashed", color="black")

    # First square.
    square = np.array([
        [.0,.0,.1,.1,.0],
        [.0,.1,.1,.0,.0]
    ])
    ax.plot(square[0], square[1], color = "blue")


    # Rotate 45 degrees clockwise
    r45cw_2d = rotate_matrix(np.pi/4, [0,1], [1,0])
    square2 = np.dot(r45cw_2d, square)
    ax.plot(square2[0], square2[1], color="red")

    ax.set_ylim([-.5,.5])
    ax.set_xlim([-.5,.5])

    plt.savefig("rotate_2d.png")
    return

def rotate_3d_test():
    # Setup figure
    plt.figure(figsize=(5,5))
    ax = plt.axes(projection="3d")

    bounds = [-.2,.2]
    # Center lines
    ax.plot(bounds,[0,0],[0,0],color="grey",linestyle="dashed")
    ax.plot([0,0],bounds,[0,0],color="grey",linestyle="dashed")
    ax.plot([0,0],[0,0],bounds,color="grey",linestyle="dashed")

    # Cube - defined by segments
    segment_templates = "000010 010110 000100 100110 010011 " \
        + "011111 111110 101111 101100 001101 001000 001011"
    cube = []
    for template in segment_templates.split(" "):
        vals = [float(".%s" % c) for c in template]
        seg = np.array([
            [vals[0],vals[3]],
            [vals[1],vals[4]],
            [vals[2], vals[5]]
        ])
        cube.append(seg)
        continue

    [ax.plot(seg[0], seg[1], seg[2], color="blue") for seg in cube]


    # Rotation 45 degrees clockwise around the y axis
    r45cw_3d = rotate_matrix(np.pi/4, [0,1,0], [1,0,0])
    cube2 = [np.dot(r45cw_3d, seg) for seg in cube]
    [ax.plot(seg[0], seg[1], seg[2], color="red") for seg in cube2]


    # Rotation 90 degrees clockwise around the z axis
    r90cw_3d = rotate_matrix(np.pi/2, [0,1,0], [0,0,1])
    cube3 = [np.dot(r90cw_3d, seg) for seg in cube2]
    [ax.plot(seg[0], seg[1], seg[2], color="limegreen") for seg in cube3]


    # Other
    ax.set_ylim(bounds)
    ax.set_xlim(bounds)
    ax.set_zlim(bounds)

    plt.savefig("rotate_3d.png")
    return


def main():
    rotate_2d_test()
    rotate_3d_test()
    return

if __name__ == "__main__":
    main()