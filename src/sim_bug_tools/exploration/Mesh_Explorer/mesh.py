import numpy as np

from numpy import ndarray
from treelib import Tree, Node
from rtree.index import Index, Property
from typing import Callable, NewType

from sim_bug_tools.exploration.boundary_core.explorer import (
    Explorer,
    ExplorationCompletedException,
)
from sim_bug_tools.exploration.boundary_core.adherer import AdherenceFactory
from sim_bug_tools.structs import Domain, Point, Scaler

# A path is a boundary point and a direction to travel along.
# It indicates where to go next.
ENUM_CARDINAL = NewType("cardinal", int)
BID = NewType("id", int)
PATH = NewType("path", tuple[BID, ndarray])

DATA_LOCATION = "location"
DATA_NORMAL = "normal"
ROOT_ID = 0


class MeshExplorer(Explorer):
    def __init__(
        self,
        b0: Point,
        n0: ndarray,
        adhererF: AdherenceFactory,
        scaler: Scaler,
        margin: float = -0.01,
    ):
        super().__init__(b0, n0, adhererF)
        self._scaler = scaler
        self._margin = margin

        # k-nearest using R-tree
        p = Property()
        p.set_dimension(self.ndims)
        self._index = Index(properties=p)

        # a way to represent boundary points
        self._tree = Tree()
        self._prev_id = None

        # ENUM_CARDINAL = index of basis vector
        self._BASIS_VECTORS = tuple(np.identity(self.ndims))

        self._next_paths: list[PATH] = []
        self._add_child(b0, n0)

    @property
    def tree(self):
        return self._tree

    def back_propegate_prev(self, k: int):
        """
        Propegates new information from leaf nodes to k parents.

        Args:
            k (int): How many nodes up the tree to propegate to
        """
        nodes = self._tree.leaves()

        for i in range(k):
            parents = [
                self._tree.parent(node.identifier)
                for node in nodes
                if node.identifier != ROOT_ID
            ]
            for parent in parents:
                osv = self._average_node_osv(parent, 1)
                if osv is not None:
                    parent.data[DATA_NORMAL] = osv
                    self._boundary[parent.identifier] = (
                        self._boundary[parent.identifier][0],
                        osv,
                    )

            nodes = parents

    def _average_node_osv(
        self, node: Node, minimum_children: int = 2, node_weight: np.float64 = 0
    ):
        """
        Averages a node's OSV with it's children and parent OSVs.

        Returns None if min children not met

        Args:
            node_id (int): The id of the node to average
            minimum_children (int, optional): Minimum children necessary to
            average. Defaults to 2.
            node_weight (float64, optional): Will account for the target node's OSV . Defaults to 0.

        Returns:
            ndarray: The new OSV or,
            None: if one could not be created

        Will fail if
        """
        neighbors = self._tree.children(node.identifier)
        if len(neighbors) < minimum_children:
            return None

        new_osv = (
            Point.zeros(self._ndims).array
            if node_weight == 0
            else node.data[DATA_NORMAL]
        )

        if node.identifier != ROOT_ID:
            neighbors.append(self._tree.parent(node.identifier))

        for osv in map(
            lambda neighbor: self._tree.get_node(neighbor.identifier).data[DATA_NORMAL],
            neighbors,
        ):
            new_osv += osv  # self._tree.get_node(node)[DATA_NORMAL]

        if node_weight == 0:
            new_osv /= len(neighbors)
        else:
            new_osv /= len(neighbors) + 1

        return new_osv / np.linalg.norm(new_osv)

    def _add_child(self, bk: Point, nk: ndarray):
        nk = nk / np.linalg.norm(nk)
        next_id = len(self.boundary) - 1
        self._next_paths = self._get_next_paths_from(next_id) + self._next_paths
        self._index.insert(next_id, bk)

        self._tree.add_node(
            Node(identifier=next_id, data=self._create_data(bk, nk)),
            parent=self._prev_id,
        )

    def _select_parent(self) -> tuple[Point, ndarray]:
        finding_gap = True
        while finding_gap and len(self._next_paths) > 0:
            # Find a point that doesn't overlap with others
            bid, self._v = self._next_paths.pop()
            bk, nk = self.boundary[bid]
            b_new = bk + self._scaler.scale(self._v)
            finding_gap = self._check_overlap(b_new)
            self._prev_id = bid

        if finding_gap:
            raise ExplorationCompletedException()

        return bk, nk

    def _pick_direction(self) -> ndarray:
        return self._v

    def _get_next_paths_from(self, bid: BID) -> list[PATH]:
        _, nk = self.boundary[bid]
        return [
            (bid, cardinal)
            for cardinal in self._find_cardinals(nk, self._BASIS_VECTORS)
        ]

    @classmethod
    def _find_cardinals(cls, n: ndarray, basis_vectors: list[ndarray]) -> list[ndarray]:
        """
        Gets the cardinal vectors and their enum type from a given OSV.

        Args:
            n (ndarray): _description_
            exclude (ENUM_CARDINAL, optional): _description_. Defaults to None.

        Returns:
            list[ndarray]: _description_
        """
        unit_vectors = list(basis_vectors)

        align_vector = unit_vectors[0]
        unit_vectors = unit_vectors[1:]

        angle = cls.angle_between(align_vector, n)

        A = cls.generateRotationMatrix(align_vector, n)(-angle)

        cardinals = []

        for uv in unit_vectors:
            c = np.dot(A, uv)
            cardinals.extend((c, -c))

        return cardinals

    def _check_overlap(self, p: Point) -> bool:
        "Returns True if there is an overlap"
        bnear_id = next(self._index.nearest(p, 1))
        bnear = self._boundary[bnear_id][0]

        displacement: ndarray = (p - bnear).array

        min_distance = np.linalg.norm(self._scaler.scale(self._v) * (1 + self._margin))
        distance = np.linalg.norm(displacement)

        return distance < min_distance

    @classmethod
    def angle_between(cls, v1, v2) -> np.float64:
        """Returns the angle in radians between vectors 'v1' and 'v2'::

        >>> angle_between((1, 0, 0), (0, 1, 0))
        1.5707963267948966
        >>> angle_between((1, 0, 0), (1, 0, 0))
        0.0
        >>> angle_between((1, 0, 0), (-1, 0, 0))
        3.141592653589793
        """
        v1_u = cls.normalize(v1)
        v2_u = cls.normalize(v2)
        return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

    @staticmethod
    def normalize(u: ndarray):
        return u / np.linalg.norm(u)

    @classmethod
    def orthonormalize(cls, u: ndarray, v: ndarray) -> tuple[ndarray, ndarray]:
        """
        Generates orthonormal vectors given two vectors @u, @v which form a span.

        -- Parameters --
        u, v : np.ndarray
            Two n-d vectors of the same length
        -- Return --
        (un, vn)
            Orthonormal vectors for the span defined by @u, @v
        """
        u = u.squeeze()
        v = v.squeeze()

        assert len(u) == len(v)

        u = u[np.newaxis]
        v = v[np.newaxis]

        un = cls.normalize(u)
        vn = v - np.dot(un, v.T) * un
        vn = cls.normalize(vn)

        if not (np.dot(un, vn.T) < 1e-4):
            raise Exception("Vectors %s and %s are already orthogonal." % (un, vn))

        return un, vn

    @classmethod
    def generateRotationMatrix(
        cls, u: ndarray, v: ndarray
    ) -> Callable[[float], ndarray]:
        """
        Creates a function that can construct a matrix that rotates by a given angle.

        Args:
            u, v : ndarray
                The two vectors that represent the span to rotate across.

        Raises:
            Exception: fails if @u and @v aren't vectors or if they have differing
                number of dimensions.

        Returns:
            Callable[[float], ndarray]: A function that returns a rotation matrix
                that rotates that number of degrees using the provided span.
        """
        u = u.squeeze()
        v = v.squeeze()

        if u.shape != v.shape:
            raise Exception("Dimension mismatch...")
        elif len(u.shape) != 1:
            raise Exception("Arguments u and v must be vectors...")

        u, v = cls.orthonormalize(u, v)

        I = np.identity(len(u.T))

        coef_a = v * u.T - u * v.T
        coef_b = u * u.T + v * v.T

        return lambda theta: I + np.sin(theta) * coef_a + (np.cos(theta) - 1) * coef_b

    @staticmethod
    def _create_data(location, normal) -> dict:
        return {DATA_LOCATION: location, DATA_NORMAL: normal}


def test_mesh():
    import matplotlib.pyplot as plt
    from sim_bug_tools.graphics import Grapher
    from sim_bug_tools.structs import Spheroid
    from sim_bug_tools.exploration.brrt_std.adherer import ConstantAdherenceFactory
    from sim_bug_tools.exploration.boundary_core.adherer import (
        BoundaryLostException,
        SampleOutOfBoundsException,
    )

    ndims = 3
    domain = Domain.normalized(ndims)

    g = Grapher(ndims == 3, domain)

    d = 0.05
    scaler = Spheroid(d)
    delta_theta = np.pi * 5 / 180

    radius = 0.4
    loc = Point([0.5 for i in range(ndims)])
    g.draw_sphere(loc, radius, color="grey")

    classifier = lambda p: loc.distance_to(p) <= radius

    b0 = Point([0.5 for i in range(ndims - 1)] + [0.5 - radius])
    n0 = np.array([0 for i in range(ndims - 1)] + [-1])
    g.plot_point(b0, color="green")

    adhf = ConstantAdherenceFactory(classifier, scaler, delta_theta, domain, True)

    exp = MeshExplorer(b0, n0, adhf, scaler)

    g_card = None
    g_osv = None

    count = 50
    i = 1
    points = []
    while True:
        while len(exp.boundary) % (count * i):
            try:
                points.append(exp.step())
            except BoundaryLostException as e:
                print("BLE")
            except SampleOutOfBoundsException as e:
                print("SooB")

        g.plot_all_points(list(zip(*exp.boundary[-count:]))[0], color="red")

        if g_card is not None:
            g_card.remove()
            g_osv.remove()

        # path_data = [
        #     (exp.boundary[bid][0], cardinal * 0.1) for bid, cardinal in exp._next_paths
        # ]

        # g_card = g.add_all_arrows(*zip(*path_data))
        # g_osv = g.add_all_arrows(
        #     *zip(*[(b, n * 0.1) for b, n in exp.boundary]), color="yellow"
        # )

        plt.pause(0.01)
        i += 1


g_vec = None
g_uv = None


def test_rotation():
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider
    from sim_bug_tools.graphics import Grapher
    from sim_bug_tools.structs import Spheroid
    from sim_bug_tools.exploration.brrt_std.adherer import ConstantAdherenceFactory
    from sim_bug_tools.exploration.boundary_core.adherer import (
        BoundaryLostException,
        SampleOutOfBoundsException,
    )

    plt.ion()

    fig, ui_ax = plt.subplots()

    ndims = 3
    domain = Domain.normalized(ndims)

    g = Grapher(ndims == 3, domain)

    radius = 0.4
    loc = Point([0.5 for i in range(ndims)])
    basis = tuple(np.identity(ndims))

    phi0 = 0
    theta0 = 0

    phi_slider = Slider(
        ax=ui_ax,
        label="phi",
        valmin=0,
        valmax=np.pi,
        valinit=phi0,
        orientation="vertical",
    )

    theta_slider = Slider(
        ax=ui_ax,
        label="theta",
        valmin=0,
        valmax=np.pi,
        valinit=theta0,
        orientation="horizontal",
    )

    def update(val):
        global g_vec, g_uv
        y = radius * np.cos(phi_slider.val)
        ab = radius * np.sin(phi_slider.val)
        x = ab * np.cos(theta_slider.val)
        z = ab * np.sin(theta_slider.val)

        vec = np.array([x, y, z])

        uvs = MeshExplorer._find_cardinals(vec, basis)

        if g_vec is not None:
            g_vec.remove()

        if g_uv is not None:
            g_uv.remove()

        g_vec = g.add_arrow(loc, vec, color="green")
        g_uv = g.add_all_arrows([loc for i in range(ndims)], uvs, color="blue")

    phi_slider.on_changed(update)
    theta_slider.on_changed(update)

    plt.show(block=True)


if __name__ == "__main__":
    test_mesh()
