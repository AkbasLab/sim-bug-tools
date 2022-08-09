import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
from numpy import ndarray
from sim_bug_tools.structs import Domain, Point


class Grapher:
    def __init__(self, is3d=False, domain: Domain = None):
        self._fig = plt.figure()
        self._ax : Axes = self._fig.add_subplot(111, projection="3d") if is3d else self._fig.add_subplot()
        if domain is not None:
            print(domain[0])
            self._ax.set_xlim(domain[0])
            self._ax.set_ylim(domain[1])
            if is3d:
                self._ax.set_zlim(domain[2])
            
            
        
        print(self._ax)
        self._is3d = is3d
        
    @property
    def ax(self):
        return self._ax 
    
    def fig(self):
        return self._fig 
    
    def create_sphere(self, loc: Point, radius: float):
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        x = loc[0] + radius * np.cos(u)*np.sin(v)
        y = loc[1] + radius * np.sin(u)*np.sin(v)
        z = loc[2] + radius * np.cos(v)
        return self._ax.plot_wireframe(x, y, z, color="r")
    
    def add_all_arrows(self, locs: list[Point], directions: list[ndarray], **kwargs):
        "Uses matplotlib.pyplot.quiver. Kwargs will be passed to ax.quiver"
        arrows = zip(*map(lambda l, d: np.append(l.array, d), locs, directions))
        return self._ax.quiver(*arrows, **kwargs)
    
    def add_arrow(self, loc: Point, direction: ndarray, **kwargs):
        return self._ax.quiver(*loc, *direction, **kwargs)
    
    def plot_point(self, loc: Point, **kwargs):
        return self._ax.scatter(*loc, **kwargs)
    
    def plot_all_points(self, locs: list[Point], **kwargs):
        return self._ax.scatter(*np.array(locs).T)
    
if __name__ == "__main__":
    print("running")
    g = Grapher(True) 
    
    points = [
        Point(0, 0, 1),
        Point(1, 1, 2),
        Point(1, 2, 0),
        Point(3, 5, 2)
    ]
    
    dirs = [
        np.array([1, 0, 1]),
        np.array([-1, -1, 3]),
        np.array([1, 2, 0]),
        np.array([-3, -5, 2]),
    ]
    
    # g.add_arrow(Point(1, 1), np.array([-0.5, -0.5]), color="red")
    # g.plot_point(Point(0, 0))
    # g.add_all_arrows(points, dirs)
    # g.plot_all_points(points)
    
    a = g.plot_point(points[0])
    b = g.add_arrow(points[0], dirs[0])
    plt.pause(0.01)
    input()
    a.remove()
    b.remove()
    plt.show()
    

        

