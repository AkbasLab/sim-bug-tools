import numpy as np
import sim_bug_tools.graphics as graphics
import matplotlib.pyplot as plt
import matplotlib.axes
from nd_rotation import gram_schmidt_orthogonalization


def plot_point_with_text(
    ax : matplotlib.axes.Axes, 
    p : np.ndarray, 
    text : str,
    color : str = "black"
): 
    ax.plot(p[0], p[1], marker="o", color=color)
    ax.text(p[0] - 0.01, p[1] + 0.015, text, color=color)
    return

def plot_arrow(
    ax : matplotlib.axes.Axes, 
    v0 : np.ndarray, 
    v1 : np.ndarray, 
    ls : str = "-"
): 
    p = v1-v0
    ax.arrow(v0[0], v0[1], p[0], p[1], length_includes_head=True, 
        head_width=0.02, color="black", ls = ls)
    return

def axis_intercept(beta : np.ndarray, beta_0 : float) -> np.ndarray:
    return -beta_0 / beta

def linePoints(a=0,b=0,c=0,ref = [-1.,1.]):
    """given a,b,c for straight line as ax+by+c=0, 
    return a pair of points based on ref values
    e.g linePoints(-1,1,2) == [(-1.0, -3.0), (1.0, -1.0)]
    """
    if (a==0 and b==0):
        raise Exception("linePoints: a and b cannot both be zero")
    return [(-c/a,p) if b==0 else (p,(-c-a*p)/b) for p in ref]


def main():
    ax = graphics.new_axes()

    zero = np.array([0,0])

    s = np.array([0.3, 0.6])
    plot_point_with_text(ax, s, "s")
    plot_arrow(ax, zero, s)

    v = np.array([0.8, 0.5])
    plot_point_with_text(ax, v, "v")
    plot_arrow(ax, zero, v)

    plot_arrow(ax, s, v, ls=":")


    m = np.array([0.9,0.9])
    plot_point_with_text(ax, m, "m")
    plot_arrow(ax, s, m)

    


    n0, n1 = gram_schmidt_orthogonalization(m-s, [0,1])
    plot_point_with_text(ax, n0, "n0")
    plot_point_with_text(ax, n1, "n1")


    

    # Save figure
    ax.set_xlim((-1,1))
    ax.set_ylim((-1,1))
    plt.savefig("plot2d.png")
    return



def main2():
    ax = graphics.new_axes()
    
    ax.hlines(0, -1, 1, linewidth=0.4, color="grey")
    ax.vlines(0, -1, 1, linewidth=0.4, color="grey")


    s = np.array([0.3, 0.6])
    plot_point_with_text(ax, s, "s")

    v = np.array([0.8, 0.5])
    plot_point_with_text(ax, v, "v")

    plot_point_with_text(ax, v-s, "v-s")

    

    # Save figure
    ax.set_xlim((-1,1))
    ax.set_ylim((-1,1))
    plt.savefig("plot2d.png")
    return



def main3():
    ax = graphics.new_axes()

    rng = np.random.default_rng(0)

    xy = rng.uniform(size=(1,2))[0]
    plot_point_with_text(ax, xy, "")
    
    a, b, c = [2,-1,0.5]
    ax.axline(*linePoints(a,b,c, [0,1]), color="black")

    # Save figure
    ax.set_xlim((0,1))
    ax.set_ylim((0,1))
    plt.savefig("plot2d.png")
    return



def main4():
    p0 = np.array([1, 2, 3, 4]) #any numbers, be it 4, 5 or n dimensions
    p1 = np.array([3, 2, 4, 5])

    # note that "p" can be any number from -inf to inf
    straight_line_function = lambda p: p0 + p * (p1 - p0)

    p3 = np.array([0,2,0,1])

    print(straight_line_function(p3))
    return



def main5():
    def hyper4(p1,p2,p3,p4):
        X = np.matrix([p1,p2,p3,p4])
        k = np.ones((4,1))
        a = np.matrix.dot(np.linalg.inv(X), k)
        print ("equation is x *\n%s = 1" % a)
        return a

    a = hyper4([0,0,1,1],[0,3,3,0],[0,5,2,0],[1,0,0,7])

    p = np.array([0, 0, 1, 0])
    dot = np.dot(a.T, p)
    print(dot)
    return


def hyper(points : np.ndarray):
    X = np.matrix(points)
    k = np.ones((X.shape[0],1))
    a = np.matrix.dot(np.linalg.inv(X), k)
    print("equation is x *\n%s = 1" % a)
    return a.T





def main6():
    ax = graphics.new_axes()

    rng = np.random.default_rng(0)

    n_dim = 2
    n_planes = 4
    colors = ["black", "red", "blue", "green"]

    #  Hyperplanes
    planes_points = [points for points in \
        rng.uniform(size=(n_planes, n_dim, n_dim))]
    planes_hyper = [hyper(plane) for plane in planes_points]
    [ax.axline(*points, color = colors[i]) \
        for i, points in enumerate(planes_points)]

    print(planes_hyper)

    # Random points
    n_points = 10
    points = [(0.2, 0.15), (0.5, 0.1), (0.9,0.2)]
    for p in points:

        ax.plot(*p, marker="o", color="black")

        bucket = np.dot(planes_hyper, p) > 0
        hash = "".join([str(int(x)) for x in (bucket)])
        print(hash)

        continue
        # print(bucket)
        hash = "".join([str(int(x)) for x in (bucket)])
        ax.plot(*p, color="black", marker="o")
        for i, char in enumerate(hash):
            ax.text(p[0] + 0.02*i, p[1] + 0.015, char, color = colors[i])
        
        continue

        
        

    


    # Save figure
    ax.set_xlim((0,1))
    ax.set_ylim((0,1))
    plt.savefig("plot2d.png")
    return



def hyper2(points : np.ndarray):
    X = np.array(points)
    k = np.ones((X.shape[0],1))
    a = np.dot(np.linalg.inv(X), k)
    # print("equation is x *\n%s = 1" % a)
    return np.append(a.T, -1)


def main7():
    ax = graphics.new_axes()
    rng = np.random.default_rng(0)

    n_dim = 2
    n_planes = 4
    colors = ["black", "red", "blue", "green"]

    #  Hyperplanes
    planes_points = [points for points in \
        rng.uniform(size=(n_planes, n_dim, n_dim))]
    planes_hyper = [hyper2(plane) for plane in planes_points]
    [ax.axline(*points, color = colors[i]) \
        for i, points in enumerate(planes_points)]


    
        


    #  Points
    n_points = 50
    points = rng.uniform(size=(n_points, n_dim))
    points = [np.append(p, 1) for p in points]
    points.append(np.array([0.69, 0.25, 1]))
    points.append(np.array([0.85, 0.01, 1]))
    for p in points:
        ax.plot(*p, marker="o", color="black")
        bucket = np.dot(planes_hyper, p) > 0
        hash = "".join([str(int(x)) for x in (bucket)])
        ax.plot(*p, color="black", marker="o")
        for i, char in enumerate(hash):
            ax.text(p[0] + 0.02*i, p[1] + 0.015, char, color = colors[i])
        continue
    

    # Save figure
    ax.set_xlim((0,1))
    ax.set_ylim((0,1))
    plt.savefig("plot2d.png")
    return



def hyper3(points : np.ndarray):
    X = np.array(points)
    k = np.ones((X.shape[0],1))
    a = np.dot(np.linalg.inv(X), k)
    return a.T

def main8():
    ax = graphics.new_axes()
    rng = np.random.default_rng(0)

    n_dim = 2
    n_planes = 4
    colors = ["black", "red", "blue", "green"]

    #  Hyperplanes
    planes_points = [points for points in \
        rng.uniform(size=(n_planes, n_dim, n_dim))]
    planes_hyper = [hyper3(plane) for plane in planes_points]
    [ax.axline(*points, color = colors[i]) \
        for i, points in enumerate(planes_points)]


    
        


    #  Points
    n_points = 50
    points = rng.uniform(size=(n_points, n_dim))
    # points = [np.append(p, 1) for p in points]
    # points.append(np.array([0.69, 0.25]))
    # points.append(np.array([0.85, 0.01]))
    for p in points:
        ax.plot(*p, marker="o", color="black")
        bucket = np.dot(planes_hyper, p) > 1
        hash = "".join([str(int(x)) for x in (bucket)])
        ax.plot(*p, color="black", marker="o")
        for i, char in enumerate(hash):
            ax.text(p[0] + 0.02*i, p[1] + 0.015, char, color = colors[i])
        continue
    

    # Save figure
    ax.set_xlim((0,1))
    ax.set_ylim((0,1))
    plt.savefig("plot2d.png")
    return

if __name__ == "__main__":
    main8()