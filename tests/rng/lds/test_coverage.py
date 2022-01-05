from sim_bug_tools.rng.lds.coverage import Heatmap
from sim_bug_tools.rng.lds.sequences import Sample
from sim_bug_tools import utils
from sim_bug_tools.structs import Point, Domain, Grid

import unittest, random
from unittest import TestCase

from sim_bug_tools.tests._testdata import PointTestData


class TestHeatmap(TestCase):
    NUM_POINTS = 1000
    NUM_DIMENSIONS = 8
    RES_STEP_SIZE = 0.1

    @classmethod
    def setUpClass(cls) -> None:
        cls.seed = 0
        return super().setUpClass()

    def setUp(self) -> None:
        # Testing normalized domain, with resolution steps of 0.1
        self.pointData = PointTestData(TestHeatmap.seed)
        self.pointData.setup_uniform_arrays(
            TestHeatmap.NUM_POINTS, TestHeatmap.NUM_DIMENSIONS, (0, 1)
        )

        self.domain = Domain.normalized(TestHeatmap.NUM_DIMENSIONS)

        self.axes_names = [f"Axis#{x+1}" for x in range(TestHeatmap.NUM_DIMENSIONS)]
        self.sample = Sample(
            self.pointData.uniform_points, self.axes_names, self.domain
        )

        resolution = [
            TestHeatmap.RES_STEP_SIZE for i in range(TestHeatmap.NUM_DIMENSIONS)
        ]

        self.grid = Grid(resolution)
        self.heatmap: Heatmap = Heatmap(self.sample, self.grid)

        TestHeatmap.seed += 1
        return super().setUp()

    def test_swapping_active_axes_changes_active_axes(self):
        x, y = self.heatmap.active_axes
        x_index = self.heatmap.axes[x]
        y_index = self.heatmap.axes[y]

        self.heatmap.swap_axes(x, y)

        new_x, new_y = self.heatmap.active_axes
        new_x_index = self.heatmap.axes[new_x]
        new_y_index = self.heatmap.axes[new_y]

        self.assertEqual([y, x], self.heatmap.active_axes)
        self.assertEqual((x_index, y_index), (new_y_index, new_x_index))

    def test_swapping_active_with_constant_axes_changes_both(self):
        x, y = self.heatmap.active_axes

        constant_axes = self.heatmap.constant_axes
        i = random.randint(0, len(constant_axes) - 1)
        axis = tuple(constant_axes.keys())[i]

        self.heatmap.swap_axes(x, axis)

        self.assertEqual([axis, y], self.heatmap.active_axes)

        self.heatmap.swap_axes(x, y)

        self.assertEqual([axis, x], self.heatmap.active_axes)

    def test_swapping_constant_axes_does_not_change_active(self):
        x, y = self.heatmap.active_axes

        constant_axes = self.heatmap.constant_axes
        i = random.randint(0, len(constant_axes) - 1)
        if i == 0:
            j = random.randint(1, len(constant_axes) - 1)
        else:
            j = random.randint(0, i)

        axis1 = tuple(constant_axes.keys())[i]
        axis2 = tuple(constant_axes.keys())[j]
        axis1_index = self.heatmap.axes[axis1]
        axis2_index = self.heatmap.axes[axis2]

        self.heatmap.swap_axes(axis1, axis2)

        new_axis1_index = self.heatmap.axes[axis1]
        new_axis2_index = self.heatmap.axes[axis2]

        self.assertEqual([x, y], self.heatmap.active_axes)
        self.assertEqual((new_axis2_index, new_axis1_index), (axis1_index, axis2_index))

    def test_heatmap_axes_match_original_axes():
        # When converting a sample into a heatmap, do the points get converted accurately?
        print("Test")

        axes_names = ["speed", "distance", "angle"]
        points = [Point(5, 7, 13), Point(17, 19, 23), Point(1, 2, 3)]

        axes_values = {
            axes_names[0]: [5, 17, 1],
            axes_names[1]: [7, 19, 23],
            axes_names[2]: [13, 23, 3],
        }

        domain = Domain.from_dimensions([30, 30, 30])

        sample = Sample(points, axes_names, domain)
        grid = Grid([1 for i in range(3)])
        heatmap = Heatmap(sample, grid)

        print(heatmap._matrix_domain)

        frame = heatmap.frame
        print(row)
        # self.assertTrue(all(map(lambda element: element == 0, row)))

        heatmap.translate_frame({"angle": 1})
        frame = heatmap.frame
        print(frame)

        heatmap.translate_frame({"angle": 23})
        frame = heatmap.frame
        print(frame)

        heatmap.translate_frame({"angle": 3})
        frame = heatmap.frame
        print(frame)


def main():
    unittest.main()
    # suite = unittest.TestSuite()
    # suite.addTest(TestHeatmap("test_heatmap_axes_match_original_axes"))
    # runner = unittest.TextTestRunner()
    # runner.run(suite)


if __name__ == "__main__":
    main()
