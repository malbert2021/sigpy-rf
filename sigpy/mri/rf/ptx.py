# -*- coding: utf-8 -*-
"""MRI RF excitation pulse design functions,
    including SLR and small tip spatial design
"""

import sigpy as sp
import numpy as np

from sigpy.mri import rf as rf
from sigpy import backend
from scipy.interpolate import interp1d


__all__ = ['stspa', 'stspk']


def stspa(target, sens, coord, dt, roi=None, alpha=0, b0=None, tseg=None,
          st=None, phase_update_interval=float('inf'), explicit=False,
          max_iter=1000, tol=1E-6):
    """Small tip spatial domain method for multicoil parallel excitation.
       Allows for constrained or unconstrained designs.

    Args:
        target (array): desired magnetization profile. [dim dim]
        sens (array): sensitivity maps. [Nc dim dim]
        coord (array): coordinates for noncartesian trajectories. [Nt 2]
        dt (float): hardware sampling dwell time.
        roi (array): array for error weighting, specify spatial ROI. [dim dim]
        alpha (float): regularization term, if unconstrained.
        b0 (array): B0 inhomogeneity map [dim dim]. For explicit matrix
            building.
        tseg (None or Dictionary): parameters for time-segmented off-resonance
            correction. Parameters are 'b0' (array), 'dt' (float),
            'lseg' (int), and 'n_bins' (int). Lseg is the number of
            time segments used, and n_bins is the number of histogram bins.
        st (None or Dictionary): 'subject to' constraint parameters. Parameters
            are avg power 'cNorm' (float), peak power 'cMax' (float),
            'mu' (float), 'rhoNorm' (float), 'rhoMax' (float), 'cgiter' (int),
            'max_iter' (int), 'L' (list of arrays), 'c' (float), 'rho' (float),
            and 'lam' (float). These parameters are explained in detail in the
            SDMM documentation.
        phase_update_interval (int): number of iters between exclusive phase
            updates. If 0, no phase updates performed.
        explicit (bool): Use explicit matrix.
        max_iter (int): max number of iterations.
        tol (float): allowable error.

    Returns:
        array: pulses out.

    References:
        Grissom, W., Yip, C., Zhang, Z., Stenger, V. A., Fessler, J. A.
        & Noll, D. C.(2006).
        Spatial Domain Method for the Design of RF Pulses in Multicoil
        Parallel Excitation. Magnetic resonance in medicine, 56, 620-629.
    """
    Nc = sens.shape[0]
    Nt = coord.shape[0]
    device = backend.get_device(target)
    xp = device.xp
    with device:
        pulses = xp.zeros((Nc, Nt), xp.complex)

        # set up the system matrix
        if explicit:
            A = rf.linop.PtxSpatialExplicit(sens, coord, dt,
                                            target.shape, b0)
        else:
            A = sp.mri.linop.Sense(sens, coord, weights=None, tseg=tseg,
                                   ishape=target.shape).H

        # handle the Ns * Ns error weighting ROI matrix
        W = sp.linop.Multiply(A.oshape, xp.ones(target.shape))
        if roi is not None:
            W = sp.linop.Multiply(A.oshape, roi)

        # apply ROI
        A = W * A

        # Unconstrained, use conjugate gradient
        if st is None:
            I = sp.linop.Identity((Nc, coord.shape[0]))
            b = A.H * W * target

            alg_method = sp.alg.ConjugateGradient(A.H * A + alpha * I,
                                                  b, pulses, P=None,
                                                  max_iter=max_iter, tol=tol)

        # Constrained case, use SDMM
        else:
            # vectorize target for SDMM
            target = W * target
            d = xp.expand_dims(target.flatten(), axis=0)
            alg_method = sp.alg.SDMM(A, d, st['lam'], st['L'], st['c'],
                                     st['mu'], st['rho'], st['rhoMax'],
                                     st['rhoNorm'], 10**-5, 10**-2, st['cMax'],
                                     st['cNorm'], st['cgiter'], st['max_iter'])

        # perform the design: apply optimization method to find solution pulse
        while not alg_method.done():

            # phase_update switch
            if (alg_method.iter > 0) and \
                    (alg_method.iter % phase_update_interval == 0):
                target = xp.abs(target) * xp.exp(
                    1j * xp.angle(
                        xp.reshape(A * alg_method.x, target.shape)))
                b = A.H * target
                alg_method.b = b

            alg_method.update()

        if st is not None:
            pulses = xp.reshape(alg_method.x, [Nc, Nt])
        return pulses


def stspk(mask, sens, n_spokes, fov, dx_max, gts, sl_thick, tbw, dgdtmax, gmax,
         alpha=0, iter_dif=0.0001):
    """Small tip spokes and k-t points parallel transmit pulse designer.

       Args:
           mask (ndarray): region in which to optimize flip angle uniformity
               in slice. [dim dim]
           sens (ndarray): sensitivity maps. [nc dim dim]
           n_spokes (int): number of spokes to be created in the design.
           fov (float): excitation FOV (cm).
           dx_max (float): max. resolution of the trajectory (cm).
           gts (float): hardware sampling dwell time (s).
           sl_thick (float): slice thickness (cm).
           tbw (int): time-bandwidth product.
           dgdtmax (float): max gradient slew (g/cm/s).
           gmax (float): max gradient amplitude (g/cm).
           alpha (float): regularization parameter.
           iter_dif (float): for each spoke, the difference in cost btwn.
              successive iterations at which to terminate MLS iterations.

    Returns:
        2-element tuple containing

        - **pulses** (*array*): RF waveform out.
        - **g** (*array*): corresponding gradient, in g/cm.

       References:
           Grissom, W., Khalighi, M., Sacolick, L., Rutt, B. & Vogel, M (2012).
           Small-tip-angle spokes pulse design using interleaved greedy and
           local optimization methods. Magnetic Resonance in Medicine, 68(5),
           1553-62.
       """

    nc = sens.shape[0]

    kmax = 1 / dx_max  # /cm, max spatial freq of trajectory
    # greedy kx, ky grid
    kxs, kys = np.meshgrid(np.linspace(-kmax / 2, kmax / 2 - 1 / fov,
                                       fov * kmax),
                           np.linspace(-kmax / 2, kmax / 2 - 1 / fov,
                                       fov * kmax))
    # vectorize the grid
    kxs = kxs.flatten()
    kys = kys.flatten()

    dc = np.intersect1d(np.where((kxs == 0)), np.where((kys == 0)))[0]

    # remove DC
    kxs = np.concatenate([kxs[:dc], kxs[dc+1:]])
    kys = np.concatenate([kys[:dc], kys[dc+1:]])

    # step 2: design the weights
    kx, ky = np.zeros(1), np.zeros(1)
    k = np.expand_dims(np.concatenate((kx, ky)), 0)

    # initial target phase
    phs = np.zeros((np.size(mask), 1))

    for ii in range(n_spokes):

        # build Afull
        Anum = rf.PtxSpatialExplicit(sens, k, gts, mask.shape, fov=fov,
                                     ret_array=True)

        # design wfull using MLS
        # initialize wfull
        w_full = np.linalg.inv(Anum.T @ Anum + alpha * np.eye((ii+1)*nc)) @ \
                 (Anum.T @ np.exp(1j*phs))

        err = Anum @ w_full - np.exp(1j * phs)
        cost = np.real(err.T @ err + alpha * w_full.T @ w_full)
        cost_old = 10 * cost  # to get the loop going
        while np.absolute(cost - cost_old) > iter_dif * cost_old:
            cost_old = cost
            phs = np.angle(Anum @ w_full)
            w_full = np.linalg.pinv(
                Anum.T @ Anum + alpha * np.eye((ii + 1) * nc)) @ (
                                 Anum.T @ np.exp(1j * phs))
            err = Anum @ w_full - np.exp(1j * phs)
            cost = np.real(err.T @ err + alpha * w_full.T @ w_full)

        # add a spoke using greedy method
        if ii < n_spokes - 1:

            r = np.exp(1j * phs) - Anum @ w_full

            rfnorm = np.zeros(kxs.shape)
            for jj in range(kxs.size):
                inv = np.linalg.pinv((Anum.T @ Anum) @ (Anum.T @ r))
                rfnorm[jj] = np.linalg.norm(inv)

            ind = np.argmax(rfnorm)
            k_new = np.zeros((1, 2))
            k_new[0, 0] = kxs[ind]
            k_new[0, 1] = kys[ind]
            if ii % 2 != 0:  # add to end of pulse
                k = np.concatenate((k, k_new))
            else:  # add to beginning of pulse
                k = np.concatenate((k_new, k))

            # remove chosen point from candidates
            kxs = np.concatenate([kxs[:ind], kxs[ind + 1:]])
            kys = np.concatenate([kys[:ind], kys[ind + 1:]])

    # from our spoke selections, build the whole waveforms
    area = tbw / (sl_thick / 10) / 4257  # thick * kwid = twb, kwid = gam*area
    [subgz, nramp] = rf.min_trap_grad(area, gmax, dgdtmax, gts)
    npts = 128
    subrf = rf.dzrf(npts, tbw, 'st')

    n_plat = subgz.size - 2 * nramp  # time points on trap plateau
    # interpolate to stretch out waveform to appropriate length
    f = interp1d(np.arange(0, npts, 1)/npts, subrf, fill_value='extrapolate')
    subrf = f(np.arange(0, n_plat, 1)/n_plat)
    subrf = np.concatenate((np.zeros(nramp), subrf, np.zeros(nramp)))

    # calc gradient, add extra 0 location at end for return to (0, 0)
    gxarea = np.diff(np.concatenate((k[:, 0], np.zeros(1)))) / 4257
    gyarea = np.diff(np.concatenate((k[:, 1], np.zeros(1)))) / 4257

    gx, gy, gz = [], [], []
    gz_sign = -1
    for ii in range(n_spokes):
        gz_sign *= -1
        gz.extend(np.squeeze(gz_sign * subgz).tolist())  # alt sign of gz

        gx.extend([0] * np.size(subgz))  # zeros for gz duration
        if np.absolute(gxarea[ii]) > 0:
            [gblip, _] = rf.trap_grad(abs(gxarea[ii]), gmax, dgdtmax, gts)
            gxblip = np.int(np.sign(gxarea[ii])) * gblip
            gx = gx[:len(gx) - len(gxblip.T)]
            gx.extend(np.squeeze(gxblip).tolist())

        gy.extend([0] * np.size(subgz))
        if np.absolute(gyarea[ii]) > 0:
            [gblip, _] = rf.trap_grad(abs(gyarea[ii]), gmax, dgdtmax, gts)
            gyblip = np.int(np.sign(gyarea[ii])) * gblip
            gy = gy[:len(gy) - len(gyblip.T)]
            gy.extend(np.squeeze(gyblip).tolist())

    [gref, _] = rf.trap_grad(gts * np.sum(subgz) / 2, gmax, dgdtmax, gts)
    gzref = - gref
    gz.extend(np.squeeze(gzref).tolist())
    gx.extend([0] * np.size(gzref))
    gy.extend([0] * np.size(gzref))
    pulses = np.kron(np.reshape(w_full, (nc, n_spokes)), subrf)

    # add zeros for gzref
    rf_ref = np.zeros((nc, len(gzref.T)))
    pulses = np.concatenate((pulses, rf_ref), 1)

    # combine gradient waveforms
    gx = np.array(gx)
    g = np.vstack((np.array(gx), np.array(gy), np.array(gz)))

    return pulses, g
