"""
Visualization tools
"""
import scipy.spatial
import sim_bug_tools.structs as structs
import sim_bug_tools.utils as utils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import itertools
import networkx as nx
import warnings
import sklearn.decomposition
import matplotlib.axes
import pandas as pd

class Voronoi(scipy.spatial.Voronoi):
    def __init__(self, points : list[structs.Point], bugs : list[bool]):
        """
        A version of the scipy voronoi plot with bug clusters.
        Parent class documentation: 
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.Voronoi.html

        -- Parameters --
        points : list[Point]
            Iterable of points
        bugs : list[bool]
            Iterable of bugs that correspect to the points.
        """
        assert len(points) == len(bugs)
        super().__init__(
            np.array([p.array for p in points])
        )
        
        self._dimension = np.int32(len(points[0]))
        self._bugs = bugs
        self._bug_indices = list(itertools.compress(range(len(bugs)), bugs))
        self._bug_graph = self._init_bug_graph()

        # self.test()

    @property
    def bugs(self) -> list[bool]:
        return self._bugs

    @property
    def bug_indices(self) -> list[int]:
        return self._bug_indices

    @property
    def bug_graph(self) -> nx.Graph:
        return self._bug_graph

    @property
    def dimension(self) -> np.int32:
        return self._dimension


    def _filter_bug_edges(self) -> list[int]:
        """
        Selects and returns the edges around bug regions.

        -- Return --
        list[int]
            Edge indices which border bug regions
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # Get the vertices of the bug regions
            bug_regions = [self.regions[bpr] for bpr in \
                [self.point_region[bri] for bri in self.bug_indices]]


            # Remove bug clusters that are on the outside.
            temp = []
            for br in bug_regions:
                if not np.any(np.isin(-1,br)):
                    temp.append(np.array(br))
            bug_regions = np.array(temp)
                

            # Get edges of the bug regions
            bug_edges = np.concatenate(np.array(list(map(
                lambda br: [  (br[i], br[(i+1) % len(br)]) for i in range(len(br))  ],
                bug_regions
            ))))
        

            # Filter uniqe edges
            bug_edges = utils.filter_unique(bug_edges)

        return bug_edges


    def _init_bug_graph(self) -> nx.Graph:
        """
        Intializes the bug graph.

        -- Return --
        nx.Graph
            Bug envelope in graph representation.
        """
        bug_edges = self._filter_bug_edges()
        
        # Create a graph
        graph = nx.Graph()

        # Add nodes
        graph.add_nodes_from( 
            [(node, {"point": self.vertices[node]}) for node in np.unique(bug_edges)]
         )

        # Add edges
        graph.add_edges_from( bug_edges )

        return graph



def top2pca(points : list[structs.Point]) -> list[structs.Point]:
    """
    Reduces the dimension of a list of point by representing the data 
    with the first two principle components.

    -- Parameters --
    points : list[structs.Point]
        Data Points of n-dimensions.

    -- Return --
    list[structs.Point]
        Dimensionally reduced versions of input points.
    """
    pca = sklearn.decomposition.PCA()
    X = pca.fit_transform(np.array([p.array for p in points]))
    score = X[:, 0:2]
    return [structs.Point(s) for s in score]



def new_axes() -> matplotlib.axes.Axes:
    """
    Creates a new plot and axes

    -- Return --
    matplotlib.axes
    """
    plt.figure(figsize = (5,5))
    return plt.axes()

def apply_pc2_labels_and_limits(ax : matplotlib.axes.Axes):
    """
    Sets the limits of the axes to [-1,1] and gives the lables of PC1 and PC2.

    -- Arguments --
    ax : matplotlib.axes.Axes
        Axes to apply on
    """
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2", labelpad=-3)
    ax.set_xlim([-1,1])
    ax.set_ylim([-1,1])
    return

def plot_bugs(points : list[structs.Point], bugs : list[bool]) -> matplotlib.axes:
    """
    Plot bugs using the first two principle components.

    -- Arguments --
    points : list[structs.Point]
        List of points of 2+ dimensions.
    bugs : list[bool]
        List of bug classification

    -- Return --
    Axis of plot
    """
    points = top2pca(points) # Reduce to 2 dimensions.

    ax = new_axes()

    colors = ["black","red"]
    for i, is_bug in enumerate(bugs):
        ax.plot( 
            points[i][0], points[i][1], 
            color=colors[int(is_bug)],
            marker="."
        )
    
    black_patch = mpatches.Patch(color=colors[0], label='Not a bug.')
    red_patch = mpatches.Patch(color=colors[1], label='Bug')
    ax.legend(handles=[red_patch,black_patch])

    

    apply_pc2_labels_and_limits(ax)
    return ax


def plot_voronoi_only(voronoi : Voronoi) -> matplotlib.axes:
    """
    Plot a voronoi diagram with the lines ones to an axis.

    -- Parameters --
    voronoi : Voronoi
        Voronoi object of 2 dimensions.

    -- Return --
    Plot axis.
    """
    assert voronoi.dimension == 2
    ax = new_axes()

    scipy.spatial.voronoi_plot_2d(
        voronoi,
        ax,
        show_vertices = False,
        show_points = False,
        line_alpha = 1
    )
    
    apply_pc2_labels_and_limits(ax)
    return ax


def plot_voronoi_bug_envelope(voronoi : Voronoi) -> matplotlib.axes:
    """
    Plots the voronoi diagram.
    """
    assert voronoi.dimension == 2
    ax = new_axes()

    scipy.spatial.voronoi_plot_2d(
        voronoi,
        ax,
        show_vertices = False,
        show_points = False,
        line_alpha = 0.7
    )


    # print( len(voronoi.points) )

    edges = voronoi.bug_graph.edges()
    for edge in edges:
        line = [voronoi.vertices[node] for node in edge]
        x = [xy[0] for xy in line]
        y = [xy[1] for xy in line]
        ax.plot(x,y,color="red")
        continue
    
    red_patch = mpatches.Patch(color="red", label='Bug Envelope')
    ax.legend(handles=[red_patch])

    apply_pc2_labels_and_limits(ax)
    return ax


def redudancy_table(
    points : list[structs.Point], 
    bugs : list[bool]) -> pd.DataFrame:
    """
    Generates a bug redundancy table to count unique points

    -- Parameters --
    points : list[structs.Point]
        List of points
    bugs : list[bool]
        List of bugs. Must be the same length as points

    -- Return --
    pf.DataFrame
        Redudancy table
    """
    assert len(points) == len(bugs)


    bugs_amt = 0
    n_bugs = []
    repeated_hits_tracker = dict()
    repeated_hits = []
    for i in range(len(points)):
        key = str(points[i])
        try:
            repeated_hits_tracker[key] += 1
        except KeyError:
            repeated_hits_tracker[key] = 1
        rh = repeated_hits_tracker[key]
        repeated_hits.append(rh)
        if bugs[i] and rh == 1:
            bugs_amt += 1
        n_bugs.append(bugs_amt)


    data = {
        "point" : points,
        "is_bug" : bugs,
        "n_repeated_hits" : repeated_hits,
        "n_bugs" : n_bugs
    }

    return pd.DataFrame(data) 



class Signal:
    def __init__(self, 
        time : list[float], 
        amplitude : list[float],
        on_after : float = 0.5):
        assert len(time) == len(amplitude)
        self._time = time
        self._amplitude = amplitude
        self._on_after = self._on_after
        self._is_on = [amp >= on_after for amp in amplitude]
        return

    @property
    def time(self) -> list[float]:
        return self._time

    @property
    def amplitude(self) -> list[float]:
        return self._amplitude

    @property
    def on_after(self) -> float:
        return self._on_after

    @property
    def is_on(self) -> list[bool]:
        return self._is_on
 
    

# def plot 
    
    




def new_signal_axes() -> matplotlib.axes.Axes:
    """
    Creates a new plot and axes

    -- Return --
    matplotlib.axes
    """
    plt.figure(figsize = (5,1))
    return plt.axes()

def plot_signal(time : list[float], is_bug : list[bool]) -> matplotlib.axes.Axes:
    assert len(time) == len(is_bug)
    ax = new_signal_axes()
    time = np.array(time, dtype=int)
    is_bug = np.array(is_bug, dtype=int)
    ax.plot(time, is_bug, color="black")
    return ax
