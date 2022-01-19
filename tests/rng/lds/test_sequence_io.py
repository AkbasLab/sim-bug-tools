from random import Random
from sim_bug_tools.rng.lds.sequences import (
    Sequence,
    HaltonSequence,
    SobolSequence,
    FaureSequence,
    RandomSequence,
    LatticeSequence
)
import sim_bug_tools.rng.lds.sequences as sequences
from sim_bug_tools.structs import Domain
from unittest import TestCase
import unittest

class TestSequencesIO(TestCase):

    def test_io(self):
        print("\n\n")

        n_dim = 4
        domain = Domain([(0,1) for _ in range(n_dim)])
        axes_names = ["dim%d" % n for n in range(n_dim)]
        seed = 555
        
        SEQUENCES = [HaltonSequence, SobolSequence, FaureSequence,
            RandomSequence, LatticeSequence]

        seq_dicts = []
        for seq_obj in SEQUENCES:
            seq : Sequence = seq_obj(domain, axes_names)
            seq.seed = seed
            seq.get_points(100)
            seq_dicts.append(seq.as_dict())
            continue
        
        imported_sequences = [sequences.from_dict(seq_dict) \
            for seq_dict in seq_dicts]
        
        for i, seq in enumerate(imported_sequences):
            self.assertEqual(
                seq.__class__.__name__,
                SEQUENCES[i].__name__
            )
            continue
        
        self.assertRaises(KeyError, sequences.from_dict, {})
        self.assertRaises(KeyError, sequences.from_dict, {"class" : "FakeClass"})

        [seq.as_json() for seq in imported_sequences]
        return

def main():
    unittest.main()