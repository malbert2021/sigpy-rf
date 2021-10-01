import time
import unittest
import numpy as np
import numpy.testing as npt
import jax.numpy as jnp
from jax import jit
from sigpy.mri.rf import optcont
import sigpy.mri.rf as rf

if __name__ == '__main__':
    unittest.main()


class TestOptcont(unittest.TestCase):

    def test_rf_autodiff(self):
        t0 = time.time()
        print('Tests start.')
        # test parameters (can be changed)
        dt = 1e-6
        b1 = np.arange(0, 2, 0.1)  # gauss, b1 range to sim over
        nb1 = np.size(b1)
        pbc = 1.5  # b1 (Gauss)
        pbw = 0.4  # b1 (Gauss)

        # generate rf pulse

        rfp_bs, rfp_ss, _ = rf.dz_bssel_rf(dt=dt, tb=2, ndes=256, ptype='ex', flip=np.pi / 2,
                                           pbw=pbw,
                                           pbc=[pbc], d1e=0.01, d2e=0.01,
                                           rampfilt=True, bs_offset=2500)
        full_pulse = (rfp_bs + rfp_ss) * 2 * np.pi * 4258 * dt  # scaled
        print('Finish Generate rf pulse. Time: {:f}'.format(time.time()-t0))

        # simulate with target function to generate magnetization profile
        rfp_abs = abs(full_pulse)
        rfp_angle = np.angle(full_pulse)
        nt = np.size(rfp_abs)
        rf_op = np.append(rfp_abs, rfp_angle)

        w = np.ones(nb1)  # weight

        Mxd = np.zeros(nb1)
        Myd = np.zeros(nb1)
        Mzd = np.zeros(nb1)

        for ii in range(nb1):
            Mxd[ii], Myd[ii], Mzd[ii] = rf.sim.arb_phase_b1sel_np(rf_op, b1[ii], 0, 0, 1.0, nt)

        print('Finish Simulate magnetization profile. Time: {:f}'.format(time.time()-t0))

        # Experiment 1: optimize the pulse with its original target profile as sanity check
        Mxd = np.array(Mxd)
        Myd = np.array(Myd)
        Mzd = np.array(Mzd)

        # huge step size to show difference
        rf_test_1 = optcont.rf_autodiff(full_pulse, b1, Mxd, Myd, Mzd, w, niters=1, step=0.1,
                                        mx0=0, my0=0, mz0=1.0)
        print('Finish sanity check autodiff. Time: {:f}'.format(time.time()-t0))

        # compare results
        npt.assert_almost_equal(full_pulse, rf_test_1, decimal=2)

        # # Experiment 2: generate target profile
        # pbc = 1  # b1 (Gauss)
        # pbw = 0.4  # b1 (Gauss)
        # rfp_bs, rfp_ss, _ = rf.dz_bssel_rf(dt=dt, tb=2, ndes=256, ptype='ex', flip=np.pi / 2,
        #                                    pbw=pbw,
        #                                    pbc=[pbc], d1e=0.01, d2e=0.01,
        #                                    rampfilt=True, bs_offset=2500)
        # target_1 = (rfp_bs + rfp_ss) * 2 * np.pi * 4258 * dt  # scaled
        #
        # # simulate with target function to generate magnetization profile
        # rfp_abs = abs(target_1)
        # rfp_angle = np.angle(target_1)
        # nt = np.size(rfp_abs)
        # rf_op = np.append(rfp_abs, rfp_angle)
        #
        # w = np.ones(nb1)  # weight
        #
        # Mxd_2 = np.zeros(nb1)
        # Myd_2 = np.zeros(nb1)
        # Mzd_2 = np.zeros(nb1)
        #
        # for ii in range(nb1):
        #     Mxd_2[ii], Myd_2[ii], Mzd_2[ii] = rf.sim.arb_phase_b1sel_np(rf_op, b1[ii], 0, 0, 1.0,
        #                                                                 nt)
        #
        # print('Finish Simulate magnetization profile. Time: {:f}'.format(time.time() - t0))

        # # optimize the pulse with the new target profile
        # rf_test_2 = optcont.rf_autodiff(full_pulse, b1, Mxd_2, Myd_2, Mzd_2, w, niters=100,
        #                                 step=0.00001,
        #                                 mx0=0, my0=0, mz0=1.0)
        #
        # # generate new magnetization
        # Mxf_2 = np.zeros(nb1)
        # Myf_2 = np.zeros(nb1)
        # Mzf_2 = np.zeros(nb1)
        #
        # for ii in range(nb1):
        #     Mxf_2[ii], Myf_2[ii], Mzf_2[ii] = rf.sim.arb_phase_b1sel_np(rf_op, b1[ii], 0, 0, 1.0,
        #                                                                 nt)
        #
        # # compare results
        # npt.assert_almost_equal(Mxf_2, Mxd_2, decimal=2)

        # Experiment 2: Generate rf pulse from a pulse with different pass bandwidth and center
        print("Modify rf pulse start.")
        pbc = 1.5  # b1 (Gauss)
        pbw = 0.2  # b1 (Gauss)
        rfp_bs, rfp_ss, _ = rf.dz_bssel_rf(dt=dt, tb=2, ndes=256, ptype='ex', flip=np.pi / 2,
                                           pbw=pbw,
                                           pbc=[pbc], d1e=0.01, d2e=0.01,
                                           rampfilt=True, bs_offset=5000)
        full_pulse_shifted = (rfp_bs + rfp_ss) * 2 * np.pi * 4258 * dt  # scaled
        nt = np.size(full_pulse_shifted)

        # optimize test pulse
        rf_test_2 = optcont.rf_autodiff(full_pulse_shifted, b1, Mxd, Myd, Mzd, w, niters=100,
                                        step=0.001,
                                        mx0=0, my0=0, mz0=1.0)

        # generate magnetization profile with acquired pulse
        Mxi = np.zeros(nb1)
        Myi = np.zeros(nb1)
        Mzi = np.zeros(nb1)
        for ii in range(nb1):
            Mxd[ii], Myd[ii], Mzd[ii] = rf.sim.arb_phase_b1sel_np(rf_test_1, b1[ii], 0, 0, 1.0, nt)
            Mxi[ii], Myi[ii], Mzi[ii] = rf.sim.arb_phase_b1sel_np(rf_test_2, b1[ii], 0, 0, 1.0, nt)

        # graphs (temp)
        pyplot.figure()
        pyplot.plot(np.sqrt(Mxi ** 2 + Myi ** 2))
        pyplot.plot(np.sqrt(Mxd ** 2 + Myd ** 2))
        pyplot.show()

        # compare results
        npt.assert_almost_equal(rf_op, rf_test_1, decimal=2)
        npt.assert_almost_equal(Mxi, Mxd, decimal=2)
        npt.assert_almost_equal(Myi, Myd, decimal=2)
        npt.assert_almost_equal(Mzi, Mzd, decimal=2)
        print("Test passed.")




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
