import unittest
import numpy as np
import sigpy as sp
import sigpy.mri.rf as rf

from sigpy.mri.rf import ep

if __name__ == '__main__':
    unittest.main()

class TestEpi(unittest.TestCase):

    def test_dz_shutters(self):
        ep.dz_shutters(4)