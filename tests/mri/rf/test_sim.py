import unittest

import numpy as np
import numpy.testing as npt

import sigpy.mri.rf as rf

if __name__ == '__main__':
    unittest.main()


class TestSim(unittest.TestCase):

    def test_arb_phase_b1sel(self):
        b1 = np.linspace(0, 3, 121)  # Gauss
        nb1 = len(b1)

        #TODO: generate rfp and corresponding magnetization

        Mxd = np.zeros(nb1, 1)
        Myd = np.zeros(nb1, 1)
        Mzd = np.zeros(nb1, 1)
        Mx0 = 0
        My0 = 0
        Mz0 = 1.0
        nt = len(rfp_abs)

        for ii in range(nb1):
            [Mxd[ii], Myd[ii], Mzd[ii]] = rf.sim.arb_phase_b1sel(b1[ii], rfp_abs, rfp_angle, Mx0,
                                                                 My0, Mz0)

        # plot(abs.(Complex.(Mxd, Myd)), label="|Mxy|")
        # plot!(Mxd, label="Mx")
        # plot!(Myd, label="My")
        # plot!(Mzd, label="Mz")

    def test_abrm(self):
        #  also provides testing of SLR excitation. Check ex profile sim.
        tb = 8
        N = 128
        d1 = 0.01
        d2 = 0.01
        ptype = 'ex'
        ftype = 'ls'

        pulse = rf.slr.dzrf(N, tb, ptype, ftype, d1, d2, False)
        [a, b] = rf.sim.abrm(pulse, np.arange(-2 * tb, 2 * tb, 0.01), True)
        Mxy = 2 * np.multiply(np.conj(a), b)

        pts = np.array([Mxy[int(len(Mxy) / 2 - len(Mxy) / 3)],
                        Mxy[int(len(Mxy) / 2)],
                        Mxy[int(len(Mxy) / 2 + len(Mxy) / 3)]])

        npt.assert_almost_equal(abs(pts), np.array([0, 1, 0]), decimal=2)
