import time
import unittest
import numpy as np
import numpy.testing as npt
from sigpy.mri.rf import optcont
import sigpy.mri.rf as rf
from sigpy import config
if config.pytorch_enabled:
    import torch


if __name__ == '__main__':
    unittest.main()


class TestOptcont(unittest.TestCase):

    def test_optcont1dLBFGS(self):
        t0 = time.time()
        print('Tests start.')
        # test parameters (can be changed)
        device = torch.device("cpu")
        N = 256
        os = 2
        tb = 8
        d1 = 0.01
        d2 = 0.01
        d1 = np.sqrt(d1 / 2)  # Mxy -> beta ripple for ex pulse
        d2 = d2 / np.sqrt(2)
        dt = 4e-6
        dthick = 4

        # generate rf pulse
        rfSLR = torch.zeros(N, 2)
        rfSLR[:, 1] = -1 * torch.tensor(rf.slr.dzrf(N, tb, 'ex', 'ls', 0.01, 0.01))
        print('Finish Generate rf pulse. Time: {:f}'.format(time.time()-t0))

        # Experiment 1: optimize a pulse with same target profile
        _, rf_opt = optcont.optcont1dLBFGS(dthick, N, os, tb, max_iters = 25)
        rf_test_1 = np.zeros((N, 2))
        rf_test_1[:, 0] = np.real(rf_opt)
        rf_test_1[:, 1] = np.imag(rf_opt)
        print('Finish optimize pulse. Time: {:f}'.format(time.time()-t0))

        # compare final profiles
        gambar = 4257  # gamma/2/pi, Hz/g
        gmag = tb / (N * dt) / dthick / gambar

        # get spatial locations + gradient
        x = (torch.arange(0, N * os, 1) / N / os - 1 / 2)
        gamgdt = 2 * np.pi * gambar * gmag * dt * torch.ones((N,1))  

        dib = rf.slr.dinf(d1, d2)
        ftwb = dib / tb

        # freq edges, normalized to 2*nyquist
        fb = torch.tensor([0, (1 - ftwb) * (tb / 2),
                        (1 + ftwb) * (tb / 2), N / 2], device=device) / N

        dpass = torch.abs(x) < fb[1]  # passband mask
        dstop = torch.abs(x) > fb[2]  # stopband mask
        w = dpass + dstop  # zero out transition band

        x = x.reshape(N*os, 1)    
  
        _, _, brSLR, biSLR = optcont.blochsimAD(rfSLR, x/(gambar*dt*gmag), 
                                                gamgdt, device)
        _, _, brTest, biTest = optcont.blochsimAD(torch.tensor(rf_test_1), x/(gambar*dt*gmag), 
                                                  gamgdt, device)

        # convert to numpy and zero out transiiton band of profiles
        rfSLR = rfSLR.numpy()
        brSLR = (w * brSLR).numpy()    
        biSLR = (w * biSLR).numpy()    
        brTest = (w * brTest).detach().numpy()    
        biTest = (w * biTest).detach().numpy()
        print('Finish generate profiles. Time: {:f}'.format(time.time()-t0))

        # compare results
        npt.assert_almost_equal(rfSLR, rf_test_1, decimal=2)
        npt.assert_almost_equal(brSLR, brTest, decimal = 2)
        npt.assert_almost_equal(biSLR, biTest, decimal = 2)
        print('Finish test 1. Time: {:f}'.format(time.time()-t0)) 


    def test_blochsimAD(self):
        print('Test not implemented')

    
    def test_blochsim_errAD(self):
        print('Test not implemented')


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
