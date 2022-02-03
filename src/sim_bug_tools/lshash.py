import numpy as np
import sim_bug_tools.structs as structs

class HashGenerator:
    def __init__(self, 
        hyperplane_points : np.ndarray,
        hyperplane_equations : np.ndarray):
        """
        Hash Generator

        -- Parameter --
        hyperplane_points : np.ndarray
            Points that define the hyperplanes of this hash table
        hyperplane_equations : np.ndarray
            The a values which define the hyperplane equations
        """
        assert hyperplane_points.shape[1] is hyperplane_equations.shape[1]
        self._hyperplane_points = hyperplane_points
        self._hyperplane_equations = hyperplane_equations
        self._table = {}
        return

    @property
    def hyperplane_points(self) -> np.ndarray:
        return self._hyperplane_points

    @property
    def hyperplane_equations(self) -> np.ndarray:
        return self._hyperplane_equations

    @property
    def table(self) -> dict:
        return self._table

    def format_equations(self, decimals : int = 3) -> list[str]:
        """
        Format the hyperplane equations

        -- Return --
            Hyperplane equations as a list
        """
        equations = []
        for hyper_eq in self.hyperplane_equations:
            eq = "".join(["%sx_{%d}+" % (np.round(x, decimals=decimals), i) \
                for i, x in enumerate(hyper_eq)])
            eq = eq[:-1] + "=1"
            equations.append(eq)
        return equations

    def hash(self, p : np.ndarray) -> int:
        """
        Hashes a point @p

        -- Parameter --
        p : np.ndarry
            Some point @p
        
        -- Return --
        int
            Hash as an int
        """
        assert p.shape[0] == self.hyperplane_equations.shape[1]
        return int("".join([str(int(x)) for \
            x in (np.dot(self.hyperplane_equations, p) > 1)]), 2)

    def index(self, p : structs.Point):
        """
        Index a point @p into the hash table.
        """
        key = self.hash(p.array)
        try:
            if not p in self.table[key]:
                self.table[key].append(p)
        except KeyError:
            self._table[key] = [p]
        return


class LSHash:
    def __init__(self, 
        hash_size : int, 
        n_dim : int, 
        n_hash_generators : int = 1,
        seed : int = 0):
        """ LSHash implments locality sensitive hashing using random projection for
        input vectors of dimension `input_dim`.
    
        -- Parameters --
        hash_size : int
            Number of bits in the hash
        n_dim : int
            Number of dimensions of the input.
        (optional) n_hash_generators : int
            Number of hash generators >= 1
        (optional) seed : int
            Seed for random number generator
        """
        self._hash_size = hash_size
        self._n_dim = n_dim
        self._seed = int(seed)
        self._rng = np.random.default_rng(self.seed)
        self._n_hash_generators = n_hash_generators
        assert self._n_hash_generators >= 1
        self._hash_generators = [HashGenerator(*self._generate_hyperplanes()) \
            for _ in range(self.n_hash_generators)]
        

        return

    @property
    def hash_size(self) -> int:
        """
        Number of bits in the hash.
        """
        return self._hash_size

    @property
    def n_dim(self) -> int:
        """
        Number of dimensions of the input
        """
        return self._n_dim

    @property
    def seed(self) -> int:
        """
        Seed
        """
        return self._seed

    @property
    def rng(self) -> np.random.Generator:
        """
        Random number generator
        """
        return self._rng

    @property
    def n_hash_generators(self) -> int:
        """
        Number of hash_generator >= 1
        """
        return self._n_hash_generators

    @property
    def hash_generators(self) -> list[HashGenerator]:
        """
        Hash generator
        """
        return self._hash_generators


    def _generate_hyperplane_equation(self, points : np.ndarray) -> np.ndarray:
        """
        Generate the a values an affine hyperplanes from N-d points where:
            a_0x_0 + a_1x_1 + ... + a_nx_n = 1

        -- Return --
        np.array of shape (1, n_dim)
            a values of the hyperplane equation    
        """
        X = np.array(points)
        k = np.ones((X.shape[0],1))
        a = np.dot(np.linalg.inv(X), k)
        return a.T[0]


    def _generate_hyperplanes(self) -> list[np.ndarray, np.ndarray]:
        """
        Generate hyperplanes for a single hash table.

        -- Return --
        np.array of shape (hash_size, n_dim, n_dim) 
            Points which define the generated hyperplanes.
        np.array of shape (1, n_dim)
            a values of the hyperplane equations
        """
        points = self.rng.uniform(
            size=(self.hash_size, self.n_dim, self.n_dim))
        equations = np.array([self._generate_hyperplane_equation(pts) \
            for pts in points])
        return points, equations

    
    def index(self, p : structs.Point):
        """
        Index a point @p
        """
        [gen.index(p) for gen in self.hash_generators]
        return

    def get(self, p : structs.Point):
        """
        Get nearby points, from all hash generators.
        """
        return [gen.table[gen.hash(p.array)] for gen in self.hash_generators]



#  Distance functions
# def hamming_dist(bitarray1, bitarray2):
#     xor_result = bitarray(bitarray1) ^ bitarray(bitarray2)
#     return xor_result.count()

def euclidean_dist(x, y):
    diff = np.array(x) - y
    return np.sqrt(np.dot(diff, diff))

def euclidean_dist_square(x, y):
    diff = np.array(x) - y
    return np.dot(diff, diff)

def euclidean_dist_centred(x, y):
    diff = np.mean(x) - np.mean(y)
    return np.dot(diff, diff)

def l1norm_dist(x, y):
    return sum(abs(x - y))

def cosine_dist(x, y):
    return 1 - float(np.dot(x, y)) / ((np.dot(x, x) * np.dot(y, y)) ** 0.5)