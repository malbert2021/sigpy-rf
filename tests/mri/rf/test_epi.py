import unittest
import numpy as np
import sigpy as sp
import sigpy.mri.rf as rf

from sigpy.mri.rf import epi

if __name__ == '__main__':
    unittest.main()

class TestEpi(unittest.TestCase):

    def test_dz_shutters(self):
        epi.dz_shutters(4)