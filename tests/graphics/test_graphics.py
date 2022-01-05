import unittest
from unittest import TestCase
import sim_bug_tools.graphics as graphics
import sim_bug_tools.rng.lds.sequences as sequences
import sim_bug_tools.structs as structs
import sim_bug_tools.utils as utils
import sim_bug_tools.scenario.clients as clients
import random
import matplotlib.pyplot as plt
import networkx as nx


class TestGraphics(TestCase):


    def test_voronoi_area(self):
        print("\n\n")

        print("\n\n")
        return

    def _test_bugs(self):
        print("\n\n")

        # Load in the test dataset
        fn = "sim_bug_tools/tests/graphics/bugs/p4r1000.pickle"
        points, statuses = utils.load(fn)
        bugs = [clients.Scenario.YIELD == status for status in statuses]

        bugs_int = [int(i) for i in bugs]
        # print("Expected bugs:", sum(bugs_int))

        # n-d voronoi
        voronoi = graphics.Voronoi(points,bugs)
        self.assertEqual(
            sum(bugs_int),
            len(voronoi.bug_indices)
        )

        # Top 2 PC      
        points_pc2 = graphics.top2pca(points)
        self.assertEqual(
            points_pc2[0].size, 2
        )
        self.assertEqual(
            len(points_pc2),
            len(points)
        )


        # Plotting buts in top 2 PC
        fig_path = "sim_bug_tools/tests/graphics/figures"
        ax = graphics.plot_bugs(points, bugs)
        ax.set_title("Yield moments (bugs) after 1000 tests.")
        plt.savefig("%s/test" % fig_path)

        
        # Plotting the voronoi in top 2 PC
        voronoi_pc2 = graphics.Voronoi(points_pc2, bugs) 
        ax = graphics.plot_voronoi_only(voronoi_pc2) 
        ax.set_title("Voronoi only. 1000 tests.")
        plt.savefig("%s/voronoi_only" % fig_path)

        self.assertRaises(
            AssertionError,
            graphics.plot_voronoi_only,
            voronoi
        )
         

        # Plotting the voronoi with bug envelope
        ax = graphics.plot_voronoi_bug_envelope(voronoi_pc2)
        ax.set_title("Voronoi w/ bug envelope. 1000 test.")        
        plt.savefig("%s/voronoi_bug_envelope" % fig_path)

        print("\n\n")
        return



def main():
    unittest.main()


if __name__ == "__main__":
    main()