import sim_bug_tools.structs as structs, sim_bug_tools.utils as utils
import unittest, random
# from sim_bug_tools.tests._testdata import PointTestData, DomainTestData
from unittest import TestCase
import math
import numpy as np

from sim_bug_tools.utils import convert_to_tuples


class TestPoint(TestCase):
    # Test data settings
    NUM_DIMENSIONS = (4, 10)
    ELEMENT_RANGE = (0, 100)
    NUM_POINTS = 8

    @classmethod
    def setUpClass(cls) -> None:
        cls.seed = 0
        cls.msg = f"[TestPoint] Seed: {TestPoint.seed}"
        return super().setUpClass()

    def setUp(self) -> None:
        self.pointData = PointTestData(TestPoint.seed)
        self.pointData.setup_uniform_arrays(
            TestPoint.NUM_POINTS, TestPoint.NUM_DIMENSIONS[1], TestPoint.ELEMENT_RANGE
        )
        self.pointData.setup_invalid_nonuniform_arrays(
            TestPoint.NUM_POINTS, TestPoint.ELEMENT_RANGE, TestPoint.NUM_DIMENSIONS
        )
        self.pointData.setup_nonuniform_arrays(
            TestPoint.NUM_POINTS, TestPoint.ELEMENT_RANGE, TestPoint.NUM_DIMENSIONS
        )
        TestPoint.seed += 1
        return super().setUp()

    def test_point_array_matches_original_array(self):
        for array in self.pointData.nonuniform_arrays:
            point = structs.Point(array)
            point_array = [axis for axis in point]
            self.assertEqual(array, point_array)

    def test_point_len_matches_array_len(self):
        for array in self.pointData.nonuniform_arrays:
            point = structs.Point(array)
            self.assertEqual(len(array), len(point))

    def test_zeros_point(self):
        num_dimensions = 5
        point_zeros = structs.Point.zeros(num_dimensions)

        self.assertEqual(len(point_zeros), num_dimensions)
        self.assertTrue(
            all(map(lambda axis: axis == 0, point_zeros)), "Not all axes equals zero!"
        )

    def test_point_subtraction(self):
        expected_results = []
        minuends, subtrahends = self.split_array_in_two(self.pointData.uniform_points)

        for i in range(len(minuends)):
            minuend = minuends[i]
            subtrahend = subtrahends[i]

            subresult = tuple([minuend[j] - subtrahend[j] for j in range(len(minuend))])
            expected_results += [subresult]

        results = [
            tuple((minuends[i] - subtrahends[i]).array) for i in range(len(minuends))
        ]

        for i in range(len(results)):
            self.assertTupleEqual(results[i], expected_results[i])

    def test_point_subtraction(self):
        expected_results = []
        addend_setA, addend_setB = self.split_array_in_two(
            self.pointData.uniform_points
        )

        for i in range(len(addend_setA)):
            addend_A = addend_setA[i]
            addend_B = addend_setB[i]

            subresult = tuple([addend_A[j] - addend_B[j] for j in range(len(addend_A))])

            expected_results += [subresult]

        results = [
            tuple((addend_setA[i] - addend_setB[i]).array)
            for i in range(len(addend_setA))
        ]

        for i in range(len(results)):
            self.assertTupleEqual(results[i], expected_results[i])

    def split_array_in_two(self, array):
        "Splits the array in two halves. If uneven, the first half will be shorter than the second."
        return (array[: len(array) // 2], array[len(array) // 2 :])


class TestDomain(TestCase):
    NUM_DIMENSIONS = (4, 10)
    ELEMENT_RANGE = (0, 100)
    NUM_DOMAINS = 8

    @classmethod
    def setUpClass(cls) -> None:
        cls.seed = 0
        cls.msg = f"[TestDomain] Seed: {TestDomain.seed}"
        return super().setUpClass()

    def setUp(self) -> None:
        self.domainData = DomainTestData(TestDomain.seed)
        self.domainData.setup_valid_domain_arrays(
            TestDomain.NUM_DOMAINS, TestDomain.ELEMENT_RANGE, TestDomain.NUM_DIMENSIONS
        )

        self.domainData.setup_invalid_domain_arrays(
            TestDomain.NUM_DOMAINS, TestDomain.ELEMENT_RANGE, TestDomain.NUM_DIMENSIONS
        )

        TestDomain.seed += 1
        return super().setUp()

    def test_array_is_domain(self):
        for valid_array_domain in self.domainData.domain_arrays:
            self.assertTrue(structs.Domain.is_domain(valid_array_domain))

        for invalid_array_domain in self.domainData.invalid_domain_arrays:
            self.assertFalse(structs.Domain.is_domain(invalid_array_domain))

    def test_bounding_points_equal_transposed_array(self):
        for array_domain in self.domainData.domain_arrays:
            domain = structs.Domain(array_domain)
            bounding_points = convert_to_tuples(domain.bounding_points)
            transposed_array = convert_to_tuples(utils.transposeList(array_domain))
            self.assertTupleEqual(transposed_array, bounding_points)

    def test_num_dimensions_matches_num_limits(self):
        for array_domain in self.domainData.domain_arrays:
            domain = structs.Domain(array_domain)
            self.assertEqual(len(array_domain), len(domain))

    def test_array_of_limits_matches_domains_array(self):
        for array_domain in self.domainData.domain_arrays:
            domain = structs.Domain(array_domain)
            self.assertTupleEqual(
                tuple(array_domain), tuple(convert_to_tuples(domain.array))
            )

    def test_scaling_domain(self):
        scalar = 3.5

        for array_domain in self.domainData.domain_arrays:
            domain = structs.Domain(array_domain)
            scaled_domain = domain * scalar

            for i in range(len(array_domain)):
                lower, upper = array_domain[i]
                self.assertAlmostEqual(scaled_domain[i][0], lower * scalar)
                self.assertAlmostEqual(scaled_domain[i][1], upper * scalar)

    def test_factory_from_bounding_points(self):
        for array_domain in self.domainData.domain_arrays:
            array_bounding_points = utils.transposeList(array_domain)
            domain = structs.Domain.from_bounding_points(
                array_bounding_points[0], array_bounding_points[1]
            )

            self.assertTupleEqual(
                convert_to_tuples(array_bounding_points),
                convert_to_tuples(domain.bounding_points),
                "The domain.bounding_points does not match the array that the domain was constructed from.",
            )
            self.assertTupleEqual(
                convert_to_tuples(array_domain),
                convert_to_tuples(domain.array),
                "The domain.array does not match the array that the domain was constructed from.",
            )


class TestGridIndexCalculations(TestCase):
    def test_valid_index_domain_from_normalized_domain(self):
        num_dimensions = 8
        domain = structs.Domain.normalized(num_dimensions)

        NUM_SLICES = 35
        RESOLUTION = [1 / NUM_SLICES for i in range(num_dimensions)]

        grid = structs.Grid(RESOLUTION)

        index_domain = grid.calculate_index_domain(domain)

        lower_bound, upper_bound = index_domain.bounding_points

        expected_upper_bound = tuple([NUM_SLICES - 1.0 for i in range(num_dimensions)])

        self.assertTupleEqual(
            tuple(lower_bound),
            tuple(structs.Point.zeros(num_dimensions)),
        )

        self.assertTupleEqual(tuple(upper_bound), expected_upper_bound)

    def test_distance_to(self):
        p1 = structs.Point([.1,.2,.3,.4])
        p2 = structs.Point([.2,.2,.3,.4])
        self.assertEqual( p1.distance_to(p2), .1 )

        self.assertEqual(
            p1.project_towards_point(p2, .01).array.all(),
            structs.Point([.11,.2,.3,.4]).array.all()
        )
        
        return

class TestPolyline(TestCase):

    def test_init(self):
        # print("\n\n")
        coords = [  [ 0.07394661, -0.45298046],
                    [ 0.06390216, -0.5145072 ],
                    [ 0.03358327,  0.07935462],
                    [ 0.12202018,  0.07722148],
                    [ 0.12887313,  0.06387744],
                    [ 0.07394661, -0.45298046] ]

        polyline = structs.PolyLine(
            points = [structs.Point(xy) for xy in coords]
        )

        
        polyline2 = polyline.copy()
        polyline2._shape = (5,3)
        try:
            polyline2.plot()
        except NotImplementedError as e:
            self.assertTrue(
                "not yet implemented" in str(e)
            )


        polyline.plot()

        # print("\n\n")
        return

def quick_domain(a: float, b : float, n_dim : int) -> structs.Domain:
    return structs.Domain( [(a,b) for i in range(n_dim)] )

class TestDomainEdgePoints(unittest.TestCase):
    def test_edge_points(self):
        print("\n\n")
        
        domain = structs.Domain( [(.1,.2), (.15,.21), (.17,.22) ] )


        # for i in range(domain.n_buckets[0]):
        #     print(0.1 + 0.01*i, end=" ")
        # print(domain.x)
        print("\n\n")
        return

def main():
    unittest.main()


if __name__ == "__main__":
    main()
