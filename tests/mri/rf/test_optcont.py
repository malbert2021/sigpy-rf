import unittest
import numpy as np
import numpy.testing as npt
from sigpy.mri.rf import optcont

if __name__ == '__main__':
    unittest.main()


class TestOptcont(unittest.TestCase):

    def test_rf_autodiff(self):
        rfp = [1, 2, 3, 4, 5]
        b1 = [1,2,3]
        mxd = [0.5, 0.5, 0.5]
        myd = [0.5, 0.5, 0.5]
        mzd = [0.5, 0.5, 0.5]
        w = [1,1,1]
        optcont.rf_autodiff(rfp, b1, mxd, myd, mzd, w, niters=5, step=0.00001, mx0=0, my0=0,
                           mz0=1.0)

    def test_optcont1d(self):
        print('Test not fully implemented')

        try:
            gamgdt, pulse = optcont.optcont1d(4, 256, 2, 8)
        finally:
            dt = 4e-6
            gambar = 4257  # gamma/2/pi, Hz/g
            [a, b] = rf.optcont.blochsim(pulse, x / (gambar * dt * gmag), gamgdt)
            Mxy = 2 * np.conj(a) * b

            pyplot.figure()
            pyplot.figure(np.abs(Mxy))
            pyplot.show()

        # TODO: compare with target Mxy, take integration

            alpha = rf.b2a(db)

    def test_blochsim(self):
        print('Test not implemented')
        # TODO: insert testing

    def test_deriv(self):
        print('Test not implemented')
        # TODO: insert testing
