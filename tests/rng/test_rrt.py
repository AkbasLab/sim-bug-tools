import unittest
import sim_bug_tools.structs as structs
import sim_bug_tools.rng.lds.sequences as sequences
from sim_bug_tools.rng.rrt import RapidlyExploringRandomTree
import matplotlib.pyplot as plt

def plot_rrt(rrt : RapidlyExploringRandomTree, name : str):
    rrt1 = rrt

    # Initialize the graph as a dynamic graph
    plt.ion()
    fig, ax = plt.subplots()
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.title(name)

    plt.draw()


    delay = 0.1

    # Root
    plt.plot(
        rrt1.root.array[0],
        rrt1.root.array[1],
        marker = ".",
        color="red"
    )
    plt.pause(delay)
    
    ax = plt.gca()

    for i in range(100):

        # RRT step
        random_point, nearest_point_in_contents, projected_point = rrt1.step()

        # Plot the random point
        plt.plot(
            random_point.array[0],
            random_point.array[1],
            marker = "+",
            color = "grey"
        )
        plt.pause(delay)


        # Draw a line betwee nearest point and projected point
        xy0 = nearest_point_in_contents.array[0],projected_point.array[0]
        xy1 = nearest_point_in_contents.array[1],projected_point.array[1]
        # print(xy0,xy1)
        plt.plot(
            xy0,xy1,
            "-",
            color="black"
        )

        # Plot the projected point
        plt.plot(
            projected_point.array[0], 
            projected_point.array[1],
            marker = ".",
            color="black",
            linewidth=0.1
        )

        # Root
        plt.plot(
            rrt1.root.array[0],
            rrt1.root.array[1],
            marker = ".",
            color="red"
        )
        plt.pause(delay)
        continue


        # break

    plt.savefig("sim_bug_tools/tests/rng/graphs/rrt_%s.png" % name)
    plt.clf()
    return


class TestRRT(unittest.TestCase):

    def test_rrt(self):

        # print("\n\n\n")


        step_size = 0.01

        rrt = RapidlyExploringRandomTree(
            # root = structs.Point([0.5,0.5]),
            seq = sequences.RandomSequence(
                domain = structs.Domain.normalized(2),
                axes_names = ["x","y"]
            ),
            step_size = step_size,
            seed = 555,
            exploration_radius = 0.1
        )
        rrt.reset(structs.Point([0.5,0.5]))



        random_point, nearest_point_in_contents, projected_point = rrt.step()
        self.assertAlmostEqual(
            nearest_point_in_contents.distance_to(projected_point),
            step_size,
            places = 10
        )

        p = structs.Point([0.4,0.4])
        rrt.reset(p)
        self.assertTrue(
            (rrt.root.array == p.array).all()
        )
        self.assertEqual( rrt.size, 1 )


        # print("\n\n\n")
        return

    def test_rrt_plot(self): 

        SEQUENCES = {
            "Sobol" : sequences.SobolSequence,
            "Halton" : sequences.HaltonSequence,
            "Faure" : sequences.FaureSequence,
            "Random" : sequences.RandomSequence
        }

        for name, seq in SEQUENCES.items():
            rrt = RapidlyExploringRandomTree(
                seq = seq(
                    structs.Domain.normalized(2),
                    axes_names = ["x","y"]
                ),
                step_size = .05,
                seed = 444,
                exploration_radius = .5
            )
            rrt.reset( structs.Point([0.8,0.8]) )
            plot_rrt(rrt,name)
            continue

        rrt.reset(structs.Point(0.2,0.2))
        plot_rrt(rrt, "Random-Reset")
        return



        



def main():
    unittest.main()

if __name__ == "__main__":
    main()