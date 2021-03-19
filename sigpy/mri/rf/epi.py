# -*- coding: utf-8 -*-
"""Pulse Designers for Echo Planar Imaging Excitations

"""
import numpy as np
import sigpy.mri.rf.trajgrad as trajgrad

__all__ = ['dz_shutters']


def dz_shutters(n_shots, dt=6.4e-6, extraShotsForOverlap=0, cancelAlphaPhs=0, R=2,
                inPlaneSimDim=None, flip=90, flyback=0, delayTolerance=0, tbw=None, gzmax=4,
                gymax=4, gslew=20000):

    # set up variables
    if tbw is None:
        tbw = np.array([3, 3])
    if inPlaneSimDim is None:
        inPlaneSimDim = np.array([85, 96])

    imFOV = 0.2 * inPlaneSimDim(2)
    # cm, imaging FOV in shuttered dim.0.2 comes from res of B1 + maps
    dthick = 0.5 * imFOV / (R * n_shots)  # slice thickness, shutter width (cm)
    kw = tbw / dthick  # width of k-space coverage in each dimension (1/cm)
    gz_area = kw(1) / 4257  # z(slice)-gradient area (g-s/cm)

    # design trapezoidal gradient
    [gpos, ramppts] = trajgrad.trap_grad(gz_area * (1 + delayTolerance), gzmax, gslew, dt)
    #TODO: find or write another trap function

    # plateau sums to desired area remove last point since it is zero and will give two
    # consecutive zeros in total waveform
    gpos = gpos[:-1]
    nFlyback = 0
    if flyback:
        gzFlyback = trajgrad.trap_grad(sum(gpos) * dt, gzmax, gslew, dt)
        gzFlyback = gzFlyback[:-1]
        gpos = gpos - 1 * gzFlyback
        nFlyback = gzFlyback.size

    Ntz = gpos.size



    raise NotImplementedError
