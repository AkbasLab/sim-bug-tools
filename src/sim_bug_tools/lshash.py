import numpy as np

class LSHash:
    def __init__(self, 
        hash_size : int, 
        n_dim : int, 
        n_hashtables : int = 1,
        seed : int = 0):
        """ LSHash implments locality sensitive hashing using random projection for
        input vectors of dimension `input_dim`.
    
        -- Parameters --
        hash_size : int
            Number of bits in the hash
        n_dim : int
            Number of dimensions of the input.
        (optional) n_hashtables : int
            Number of hashtables >= 1
        (optional) seed : int
            Seed for random number generator
        """
        self._hash_size = hash_size
        self._n_dim = n_dim
        self._n_hash_tables = n_hashtables
        assert self._n_hash_tables >= 1
        self._seed = int(seed)
        self._rng = np.random.default_rng(self.seed)

        # Hyperplanes
        self._generate_hyperplanes()
        
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
    def n_hash_tables(self) -> int:
        """
        Number of hashtables >= 1
        """
        return 

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

    def _generate_hyperplane_equation(self, points : np.ndarray) -> np.ndarray:
        """
        Generate an affine hyperplanes from N-d points.
        """
        X = np.array(points)
        k = np.ones((X.shape[0],1))
        a = np.dot(np.linalg.inv(X), k)
        return a.T

    def _generate_hyperplanes(self) -> np.ndarray:
        """
        Generate hyperplanes for a single hash table.
        """
        points = self.rng.uniform(
            size=(self.hash_size, self.n_dim, self.n_dim))
        equations = np.array([self._generate_hyperplane_equation(pts) \
            for pts in points])
        for i, pts in enumerate(points):
            print()
            print("Points")
            print(pts)
            print("Equation")
            print(equations[i])
        return

    # def _generate_hash_tables(self):
    #     return
