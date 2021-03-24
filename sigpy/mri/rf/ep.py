# -*- coding: utf-8 -*-
"""Pulse Designers for Echo Planar Imaging Excitations

"""
import numpy as np
import sigpy.mri.rf.trajgrad as trajgrad
import sigpy.mri.rf.slr as slr
import sigpy.plot as pl

__all__ = ['dz_shutters']


def dz_shutters(Nshots, dt=6.4e-6, extraShotsForOverlap=0, cancelAlphaPhs=0, R=2,
                inPlaneSimDim=None, flip=90, flyback=0, delayTolerance=0, tbw=None, gzmax=4,
                gymax=4, gslew=20000):
    # set up variables
    if tbw is None:
        tbw = np.array([3, 3])
    if inPlaneSimDim is None:
        inPlaneSimDim = np.array([85, 96])

    imFOV = 0.2 * inPlaneSimDim[1]
    # cm, imaging FOV in shuttered dim.0.2 comes from res of B1 + maps
    dthick = [0.5, imFOV / (R * Nshots)]  # slice thickness, shutter width (cm)
    kw = tbw / dthick  # width of k-space coverage in each dimension (1/cm)
    gz_area = kw[0] / 4257  # z(slice)-gradient area (g-s/cm)

    # design trapezoidal gradient
    [gpos, ramppts] = trajgrad.min_trap_grad(gz_area * (1 + delayTolerance), gzmax, gslew, dt)
    # TODO: difference in final gpos area

    # plateau sums to desired area remove last point since it is zero and will give two
    # consecutive zeros in total waveform
    gpos = np.delete(gpos, -1, 1)
    nFlyback = 0
    if flyback:  # TODO: test flyback
        gzFlyback = trajgrad.trap_grad(sum(gpos) * dt, gzmax, gslew, dt)
        gzFlyback = np.delete(gzFlyback, -1, 1)
        gpos = gpos - 1 * gzFlyback
        nFlyback = gzFlyback.size

    Ntz = gpos.size

    # design slice-selective subpulse
    rfSl = np.real(slr.dzrf(np.rint((Ntz - 2 * ramppts + 1) / (1 + delayTolerance)).astype(int)
                            - nFlyback, tbw[0], 'st', 'ls', 0.01, 0.01))  # arb units
    # zero pad rf back to length of plateau if delayTolerance > 0
    if delayTolerance > 0:
        nPad = np.floor(((Ntz - 2 * ramppts + 1) - rfSl.size) / 2)
        rfSl = np.append(np.zeros((1, nPad)), rfSl, np.zeros((1, nPad)), 1)
        if rfSl.size < Ntz - 2 * ramppts + 1:
            rfSl = np.append(rfSl, 0)

    # normalize to one radian flip
    rfSl = rfSl / np.sum(rfSl)
    # TODO: small difference in value but generally the same shape

'''
    # design the shutter envelope
    if flip == 90:
        if ~cancelAlphaPhs:
            rfShut = np.real(slr.dzrf(np.rint(kw[1] * Nshots * dthick[1])).astype(int),
                             tbw[1], 'ex', 'ls', 0.01, 0.01) # radians
        else:
            [_, bShut] = np.dzrf(np.rint(kw[1] * Nshots * dthick[1]).astype(int), tbw[1], 'ex', 'ls', 0.01, 0.01)
            Bshut = ft(bShut)   #TODO: what is this
            Bshut = Bshut * np.exp(-1j * 2 * np.pi / np.round(kw[1] * Nshots * dthick[1]) * 1 *
                                   (-(np.round(kw[1] * Nshots * dthick[1])) / 2:round(kw[1] *
                                   Nshots * dthick(
                2)) / 2 - 1))
            bShut = ift(Bshut);
            aShut = b2a(bShut);
            bShut = ifft(fft(bShut). * exp(1
            i * angle(fft(aShut))));
            rfShut = real(b2rf(bShut));
    elif flip == 180:
    rfShut = real(
        dzrf(round(kw(2) * Nshots * dthick(2)), tbw(2), 'se', 'ls', 0.01, 0.01));  # radians

    else  # small-tip
        if ~cancelAlphaPhs
            rfShut = real(
            dzrf(round(kw(2) * Nshots * dthick(2)), tbw(2), 'st', 'ls', 0.01, 0.01));  # arb units
            # scale to target flip
            rfShut = rfShut. / sum(rfShut) * flip * pi / 180;  # radians
        else
            bShut = dzrf(round(kw(2) * Nshots * dthick(2)), tbw(2), 'st', 'ls', 0.01, 0.01);  # arb units
            Bshut = ft(bShut);
            Bshut = Bshut. * exp(-1
            i * 2 * pi / round(kw(2) * Nshots * dthick(2)) * 1 * (-round(kw(2) * Nshots * dthick(2)) / 2
                                                          :round(kw(2) * Nshots * dthick(
            2)) / 2 - 1));
            bShut = ift(Bshut);
            bShut = bShut * sind(flip / 2);
            aShut = b2a(bShut);
            bShut = ifft(fft(bShut). * exp(1
            i * angle(fft(aShut))));
            rfShut = real(b2rf(bShut));  # radians
'''