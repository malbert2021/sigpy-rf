# -*- coding: utf-8 -*-
"""Optimal Control Pulse Design functions.
"""
from sigpy import backend
from sigpy import config
from sigpy.mri.rf import slr
import numpy as np
if config.pytorch_enabled:
    import torch
if config.cupy_enabled:
    import cupy as cp


__all__ = ['blochsimAD', 'blochsim_errAD', 'optcont1dLBFGS', 'optcont1d', 
           'blochsim', 'deriv']


def blochsimAD(rf, x, g, device):
    r"""RF pulse simulation, with simultaneous RF + gradient rotations.
    Assume x has inverse spatial units of g, and g has gamma*dt applied, and 
    assume x = [..., Ndim], g = [Nt, Ndim], and rf = [Nt, 2] with real 
    components in the first column and imag components in the second column.

    Args:
        rf (tensor): rf waveform input.
        x (tensor): spatial locations.
        g (tensor): gradient waveform.
        device (pytorch device): device to run function on

    Returns:
        tensor: real component of SLR alpha parameter
        tensor: imag component of SLR alpha parameter
        tensor: real component of SLR beta parameter
        tensor: imag component of SLR beta parameter
    """
    ar = torch.ones((x.size()[0], ), device=device)
    ai = torch.zeros((x.size()[0],), device=device)
    br = torch.zeros((x.size()[0],), device=device)
    bi = torch.zeros((x.size()[0],), device=device)
    
    for mm in range(0, rf.size()[0], 1):  # loop over time

        pt = rf[mm, :]
        mag = torch.sqrt(torch.sum(pt**2))
        ang = torch.arctan2(pt[1], pt[0])
        
        # apply RF
        cr = torch.cos(mag / 2)
        ci = 0
        sr = -1 * torch.sin(ang) * torch.sin(mag / 2)
        si = torch.cos(ang) * torch.sin(mag / 2)
        
        art = (ar * cr - ai * ci) - (br * sr + bi * si)
        ait = (ar * ci + ai * cr) - (bi * sr - br * si)
        brt = (ar * sr - ai * si) + (br * cr - bi * ci)
        bit = (ar * si + ai * sr) + (br * ci + bi * cr)
        
        ar = art
        ai = ait
        br = brt
        bi = bit

        # apply gradient
        zr = torch.cos(-1 * x @ g[mm, :])
        zi = torch.sin(-1 * x @ g[mm, :])
    
        brt = br * zr - bi * zi
        bit = br * zi + bi * zr
        
        br = brt
        bi = bit
    
    # apply total phase accrual
    zr = torch.cos(x @ sum(g, 0) / 2)
    zi = torch.sin(x @ sum(g, 0) / 2)

    art = ar * zr - ai * zi
    ait = ar * zi + ai * zr
    brt = br * zr - bi * zi
    bit = br * zi + bi * zr
    
    ar = art
    ai = ait        
    br = brt
    bi = bit
    
    return ar, ai, br, bi
    

def blochsim_errAD(rfp, x, g, device, w, db, da=None):
    r"""Loss function for 1D optimal control pulse designer using autodiff.
    Assume x has inverse spatial units of g, and g has gamma*dt applied, and 
    assume x = [..., Ndim], g = [Nt, Ndim], rf = [Nt, 2] with real components
    in the first column and imag components in the second column, and db = 
    [..., 2 and da = [..., 2] with real components in the first column and 
    imag components in the second column.

    Args:
        rfp (tensor): rf waveform input.
        x (tensor): spatial locations.
        g (tensor): gradient waveform.
        device (pytorch device): device to run function on
        w (tensor): weights on profile locations for error calculation. 
        db (tensor): target SLR beta parameter
        da (tensor): target SLR alpha parameter

    Returns:
        float: weighted error between pulse's SLR parameter profile and target
    """
    ar, ai, br, bi = blochsimAD(rfp, x, g, device)
    err = torch.sum(w * ((db[:, 0] - br) ** 2 + (db[:, 1] - bi) ** 2))

    if da:
        err += torch.sum(w * ((da[:, 0] - ar) ** 2 + (da[:, 1] - ai) ** 2))
    
    return err


def optcont1dLBFGS(dthick, N, os, tb, max_iters=100, d1=0.01,
              d2=0.01, dt=4e-6, conv_tolerance=1e-9, dev="cpu"):
    r"""1D optimal control pulse designer using autodiff

    Args:
        dthick: thickness of the slice (cm)
        N: number of points in pulse
        os: matrix scaling factor
        tb: time bandwidth product, unitless
        stepsize: optimization step size
        max_iters: max number of iterations
        d1: ripple level in passband
        d2: ripple level in stopband
        dt: dwell time (s)
        conv_tolerance: max change between iterations, convergence tolerance
        dev: device to run function on ("gpu" to run on CUDA, otherwise defaults
          to running on cpu)

    Returns:
        gamgdt: scaled gradient
        pulse: pulse of interest, complex RF waveform

    """
    device = torch.device("cuda:0" if dev == "gpu" and
                          torch.cuda.is_available() else "cpu")

    # set mag of gamgdt according to tb + dthick        
    gambar = 4257  # gamma/2/pi, Hz/g
    gmag = tb / (N * dt) / dthick / gambar

    # get spatial locations + gradient
    x = (torch.arange(0, N * os, 1, device=device) / N / os - 1 / 2)
    gamgdt = 2 * np.pi * gambar * gmag * dt * torch.ones((N,1), device=device)    

    # set up target beta pattern
    d1 = np.sqrt(d1 / 2)  # Mxy -> beta ripple for ex pulse
    d2 = d2 / np.sqrt(2)
    dib = slr.dinf(d1, d2)
    ftwb = dib / tb

    # freq edges, normalized to 2*nyquist
    fb = torch.tensor([0, (1 - ftwb) * (tb / 2),
                     (1 + ftwb) * (tb / 2), N / 2], device=device) / N

    dpass = torch.abs(x) < fb[1]  # passband mask
    dstop = torch.abs(x) > fb[2]  # stopband mask
    wb = [1, d1 / d2]
    w = dpass + wb[1] / wb[0] * dstop  # 'points we care about' mask

    # target beta pattern
    db = torch.zeros(N*os, 2, device=device)
    db[:, 0] = np.sqrt(1/2)*dpass*torch.cos(-1/2*x*2*np.pi)
    db[:, 1] = np.sqrt(1/2)*dpass*torch.sin(-1/2*x*2*np.pi)
    db[0, 0] = 0
    
    #reshape x for Ndim=1
    x = x.reshape(N*os, 1)

    # initialize optimization
    pulse = torch.full((N,2), 1e-7, requires_grad = True, device = device)
    lbfgs = torch.optim.LBFGS([pulse], lr = .1, tolerance_change=conv_tolerance)
    
    def closure():
        lbfgs.zero_grad()
        loss = blochsim_errAD(pulse, x/(gambar*dt*gmag), gamgdt, device, w, db)
        loss.backward()
        return loss
    
    # perform optimization using LBFGS
    cost = np.zeros(max_iters)    
    for ii in range(0, max_iters, 1):
        loss = blochsim_errAD(pulse, x/(gambar*dt*gmag), gamgdt, device, w, db)
        cost[ii] = loss.item()
        lbfgs.step(closure)

    pulse = pulse[:, 0] + 1j * pulse[:, 1] # pulse in complex form
    
    # convert to np type
    gamgdt = gamgdt.detach().numpy()
    pulse = pulse.detach().numpy()

    # put on correct device
    if dev == "gpu" and config.cupy_enabled:
        device = backend.Device(0)
    else:
        device = backend.Device(-1)
    xp = device.xp
    pulse = xp.array(pulse)
    gamgdt = xp.array(pulse)

    return gamgdt, pulse


def optcont1d(dthick, N, os, tb, stepsize=0.001, max_iters=1000, d1=0.01,
              d2=0.01, dt=4e-6, conv_tolerance=1e-5):
    r"""1D optimal control pulse designer

    Args:
        dthick: thickness of the slice (cm)
        N: number of points in pulse
        os: matrix scaling factor
        tb: time bandwidth product, unitless
        stepsize: optimization step size
        max_iters: max number of iterations
        d1: ripple level in passband
        d2: ripple level in stopband
        dt: dwell time (s)
        conv_tolerance: max change between iterations, convergence tolerance

    Returns:
        gamgdt: scaled gradient
        pulse: pulse of interest, complex RF waveform

    """

    # set mag of gamgdt according to tb + dthick
    gambar = 4257  # gamma/2/pi, Hz/g
    gmag = tb / (N * dt) / dthick / gambar

    # get spatial locations + gradient
    x = np.arange(0, N * os, 1) / N / os - 1 / 2
    gamgdt = 2 * np.pi * gambar * gmag * dt * np.ones(N)

    # set up target beta pattern
    d1 = np.sqrt(d1 / 2)  # Mxy -> beta ripple for ex pulse
    d2 = d2 / np.sqrt(2)
    dib = slr.dinf(d1, d2)
    ftwb = dib / tb
    # freq edges, normalized to 2*nyquist
    fb = np.asarray([0, (1 - ftwb) * (tb / 2),
                     (1 + ftwb) * (tb / 2), N / 2]) / N

    dpass = np.abs(x) < fb[1]  # passband mask
    dstop = np.abs(x) > fb[2]  # stopband mask
    wb = [1, d1 / d2]
    w = dpass + wb[1] / wb[0] * dstop  # 'points we care about' mask

    # target beta pattern
    db = np.sqrt(1 / 2) * dpass * np.exp(-1j / 2 * x * 2 * np.pi)

    pulse = np.zeros(N, dtype=complex)

    a = np.exp(1j / 2 * x / (gambar * dt * gmag) * np.sum(gamgdt))
    b = np.zeros(a.shape, dtype=complex)

    eb = b - db
    cost = np.zeros(max_iters + 1)
    cost[0] = np.real(np.sum(w * np.abs(eb) ** 2))

    for ii in range(0, max_iters, 1):
        # calculate search direction
        auxb = w * (b - db)
        drf = deriv(pulse, x / (gambar * dt * gmag), gamgdt, None,
                    auxb, a, b)
        drf = 1j * np.imag(drf)

        # get test point
        pulse -= stepsize * drf

        # simulate test point
        [a, b] = blochsim(pulse, x / (gambar * dt * gmag), gamgdt)

        # calculate cost
        eb = b - db
        cost[ii + 1] = np.sum(w * np.abs(eb) ** 2)

        # check cost with tolerance
        if (cost[ii] - cost[ii + 1]) / cost[ii] < conv_tolerance:
            break

    return gamgdt, pulse


def blochsim(rf, x, g):
    r"""1D RF pulse simulation, with simultaneous RF + gradient rotations.
    Assume x has inverse spatial units of g, and g has gamma*dt applied and
    assume x = [...,Ndim], g = [Ndim,Nt].

     Args:
         rf (array): rf waveform input.
         x (array): spatial locations.
         g (array): gradient waveform.

     Returns:
         array: SLR alpha parameter
         array: SLR beta parameter
     """

    device = backend.get_device(rf)
    xp = device.xp
    with device:
        a = xp.ones(xp.shape(x)[0], dtype=complex)
        b = xp.zeros(xp.shape(x)[0], dtype=complex)
        for mm in range(0, xp.size(rf), 1):  # loop over time

            # apply RF
            c = xp.cos(xp.abs(rf[mm]) / 2)
            s = 1j * xp.exp(1j * xp.angle(rf[mm])) * xp.sin(xp.abs(rf[mm]) / 2)
            at = a * c - b * xp.conj(s)
            bt = a * s + b * c
            a = at
            b = bt

            # apply gradient
            if g.ndim > 1:
                z = xp.exp(-1j * x @ g[mm, :])
            else:
                z = xp.exp(-1j * x * g[mm])
            b = b * z

        # apply total phase accrual
        if g.ndim > 1:
            z = xp.exp(1j / 2 * x @ xp.sum(g, 0))
        else:
            z = xp.exp(1j / 2 * x * xp.sum(g))
        a = a * z
        b = b * z

        return a, b


def deriv(rf, x, g, auxa, auxb, af, bf):
    r"""1D RF pulse simulation, with simultaneous RF + gradient rotations.

    'rf', 'g', and 'x' should have consistent units.

     Args:
         rf (array): rf waveform input.
         x (array): spatial locations.
         g (array): gradient waveform.
         auxa (None or array): auxa
         auxb (array): auxb
         af (array): forward sim a.
         bf( array): forward sim b.

     Returns:
         array: SLR alpha parameter
         array: SLR beta parameter
     """

    device = backend.get_device(rf)
    xp = device.xp
    with device:
        drf = xp.zeros(xp.shape(rf), dtype=complex)
        ar = xp.ones(xp.shape(af), dtype=complex)
        br = xp.zeros(xp.shape(bf), dtype=complex)

        for mm in range(xp.size(rf) - 1, -1, -1):

            # calculate gradient blip phase
            if g.ndim > 1:
                z = xp.exp(1j / 2 * x @ g[mm, :])
            else:
                z = xp.exp(1j / 2 * x * g[mm])

            # strip off gradient blip from forward sim
            af = af * xp.conj(z)
            bf = bf * z

            # add gradient blip to backward sim
            ar = ar * z
            br = br * z

            # strip off the curent rf rotation from forward sim
            c = xp.cos(xp.abs(rf[mm]) / 2)
            s = 1j * xp.exp(1j * xp.angle(rf[mm])) * xp.sin(xp.abs(rf[mm]) / 2)
            at = af * c + bf * xp.conj(s)
            bt = -af * s + bf * c
            af = at
            bf = bt

            # calculate derivatives wrt rf[mm]
            db1 = xp.conj(1j / 2 * br * bf) * auxb
            db2 = xp.conj(1j / 2 * af) * ar * auxb
            drf[mm] = xp.sum(db2 + xp.conj(db1))
            if auxa is not None:
                da1 = xp.conj(1j / 2 * bf * ar) * auxa
                da2 = 1j / 2 * xp.conj(af) * br * auxa
                drf[mm] += xp.sum(da2 + xp.conj(da1))

            # add current rf rotation to backward sim
            art = ar * c - xp.conj(br) * s
            brt = br * c + xp.conj(ar) * s
            ar = art
            br = brt

        return drf
