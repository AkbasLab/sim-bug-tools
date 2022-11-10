import unittest
import os, sys
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(FILE_DIR))

import envelope

import sim_bug_tools.rng.lds.sequences as sequences
import sim_bug_tools.structs as structs
import sim_bug_tools.utils as utils

class TestEnvelope(unittest.TestCase):

    def test_envelope(self):

        e = envelope.Envelope()
        return