import unittest
import sim_bug_tools.rng.bugger as bugger
import sim_bug_tools.rng.lds.sequences as sequences
import sim_bug_tools.structs as structs

class TestBugger(unittest.TestCase):
    def test_bugger(self):
        # print("\n\n")
        loc = structs.Domain([(0, 1), (0, 1), (0, 1)])
        size = structs.Domain([(0, 0.1), (0, 0.1), (0, 0.1)])
        sobol = sequences.SobolSequence(loc, ["x", "y", "z"])

        b = bugger.BugBuilder(loc, size, sobol)

        for i in range(10):
            # print(f"----- Bug #{i} -----")
            bug = b.build_bug()
            # print(type(bug))
            # print("Bounds:", bug.bounding_points, "Dimensions:", bug.dimensions)
            # break
        # print("\n\n")
        return

    def test_load_profile(self):
        fn = "sim_bug_tools/tests/rng/data/bug_profile.json"
        profile = bugger.profiles_from_json(fn)
        self.assertTrue(isinstance(profile[0][0], structs.Domain))
        return

def main():
    unittest.main()

if __name__ == "__main__":
    main()