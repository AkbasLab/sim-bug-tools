import unittest
from sim_bug_tools.rng.lds.sequences import (
    Sequence,
    HaltonSequence,
    SobolSequence,
    FaureSequence,
    RandomSequence,
    LatticeSequence
)
from sim_bug_tools.structs import Domain, Point, Grid
from unittest import TestCase
import matplotlib.pyplot as plt
import sim_bug_tools.utils as utils
import numpy as np

NUM_DIMENSIONS = 5
NUM_POINTS = 100

# Add your sequence here to add it to the tests
SEQUENCES: dict[str, Sequence] = {
    "Halton": HaltonSequence,
    "Sobol": SobolSequence,
    "Faure": FaureSequence,
}

# class TestLattice(TestCase):
#     def test_lattice(self):
#         n_dim = 2
#         domain = Domain.normalized(n_dim)
#         axes_names = ["x", "y"]
#         n_pts = 1000
#         skip = utils.prime(40)

#         seq = LatticeSequence(domain, axes_names)
#         print("\n\n")

#         print("\n\n")
#         quit()
#         return

    

class TestSequences(TestCase):
    def setUp(self) -> None:
        self.domain = Domain.normalized(NUM_DIMENSIONS)
        self.axes_names = [f"Axis#{x}" for x in range(NUM_DIMENSIONS)]

        self.sequences: dict[str, Sequence] = {
            name: Seq(self.domain, self.axes_names) for name, Seq in SEQUENCES.items()
        }

        return super().setUp()

    def test_sequence_points_are_within_domain(self):
        for seq_name, seq in self.sequences.items():
            points = seq.get_points(NUM_POINTS)
            self.assertTrue(
                all(map((lambda p: p in self.domain), points)),
                f"Not all points are within a normalized domain! Sequence: {seq_name}.",
            )

    def test_sequences_have_correct_dimensions(self):
        for seq_name, seq in self.sequences.items():
            points = seq.get_points(NUM_POINTS)
            self.assertTrue(
                all(map((lambda p: len(p) == len(self.domain)), points)),
                f"Dimension Mismatch between points in sequence and their domain.",
            )


class TestSequenceGetSample(TestCase):
    def setUp(self) -> None:
        self.domain = Domain.normalized(NUM_DIMENSIONS)
        self.axes_names = [f"Axis#{x}" for x in range(NUM_DIMENSIONS)]

        self.sequences: dict[str, Sequence] = {
            name: Seq(self.domain, self.axes_names) for name, Seq in SEQUENCES.items()
        }

        return super().setUp()

    def test_skip_interval(self):
        skip = 3
        num_points = 10

        for name, Seq_class in SEQUENCES.items():
            full_sequence = Seq_class(self.domain, self.axes_names).get_points(
                NUM_POINTS
            )
            seq = Seq_class(self.domain, self.axes_names)

            sample = seq.get_sample(num_points, skip)

            for i, point in enumerate(sample.points):
                self.assertTupleEqual(
                    tuple(point.array),
                    tuple(full_sequence[i * skip].array),
                    f"Skip interval is inaccurate for sequence {name}.",
                )


def scatterplot(points, title: str):
    x = [p[0] for p in points]
    y = [p[1] for p in points]
    plt.scatter(x, y, marker=".", color="black")
    plt.title(title)
    plt.savefig("sim_bug_tools/tests/rng/lds/graphs/%s.png" % title)
    plt.clf()
    return


class TestSequence2dPlot(TestCase):
    def test_plots(self):
        n_dim = 2
        domain = Domain.normalized(n_dim)
        axes_names = ["x", "y"]
        n_pts = 1000
        skip = utils.prime(40)

        halton = HaltonSequence(domain, axes_names)
        points = halton.get_sample(n_pts, skip).points
        scatterplot(points, "Halton")

        sobol = SobolSequence(domain, axes_names)
        points = sobol.get_sample(n_pts, skip).points
        scatterplot(points, "Sobol")

        faure = FaureSequence(domain, axes_names)
        points = faure.get_sample(n_pts, skip).points
        scatterplot(points, "Faure")

        rand = RandomSequence(domain, axes_names)
        rand.seed = 555
        points = rand.get_sample(n_pts, skip).points
        scatterplot(points, "Random")
        return


class TestRandomSequence(TestCase):
    def test_random_sequence(self):
        n_dim = 2
        domain = Domain.normalized(n_dim)
        axes_names = ["x", "y"]
        rand = RandomSequence(domain, axes_names)
        rand.seed = 555  # Seed MUST be 555
        point_value = np.array([point.array for point in rand.get_points(3)])
        expected_value = np.array(  # for seed 555
            [
                [0.71783409, 0.04785513],
                [0.94447198, 0.68638004],
                [0.58120733, 0.14267862],
            ]
        )

        for i_point, point in enumerate(point_value):
            for i_val, val in enumerate(point):
                self.assertAlmostEqual(
                    point_value[i_point, i_val],
                    expected_value[i_point, i_val],
                    places=8,
                )
                continue
            continue

        return


def main():
    unittest.main()


if __name__ == "__main__":
    main()
