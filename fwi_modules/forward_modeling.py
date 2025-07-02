# Modified from https://www.kaggle.com/code/manatoyo/improved-vel-to-seis

# https://arxiv.org/pdf/2111.02926
# https://csim.kaust.edu.sa/files/SeismicInversion/Chapter.FD/lab.FD2.8/lab.html
# ruff: noqa
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm


# Modified to use torch
def ricker(f, dt, nt=None, device="cpu"):
    nw = int(2.2 / f / dt)
    nw = 2 * (nw // 2) + 1
    nc = nw // 2 + 1  # 중심 인덱스를 1-based 기준으로 설정

    k = torch.arange(1, nw + 1, dtype=torch.float32, device=device)  # 1-based index
    alpha = (nc - k) * f * dt * torch.pi
    beta = alpha**2
    w0 = (1.0 - 2.0 * beta) * torch.exp(-beta)

    # 1-based wavelet 생성
    if nt is not None:
        if nt < len(w0):
            raise ValueError("nt is smaller than condition!")
        w = torch.zeros(nt + 1, dtype=torch.float32, device=device)  # dummy 포함
        w[1 : len(w0) + 1] = w0
    else:
        w = torch.zeros(len(w0) + 1, dtype=torch.float32, device=device)
        w[1:] = w0

    # 1-based time axis 생성
    if nt is not None:
        tw = torch.arange(1, len(w), dtype=torch.float32, device=device) * dt
    else:
        tw = torch.arange(1, len(w), dtype=torch.float32, device=device) * dt

    return w, tw


def AbcCoef2D(vel, nbc, dx):
    """
    Calculates coefficients for a 2D Absorbing Boundary Condition (ABC).
    Now uses torch instead of numpy.

    Args:
        vel (torch.Tensor): The padded 2D velocity model.
        nbc (int): The number of padding cells (boundary width).
        dx (float): The spatial grid interval.

    Returns:
        torch.Tensor: A 2D tensor of damping coefficients.
    """
    nb, nzbc, nxbc = vel.shape
    velmin = vel.amin((1, 2))
    nz = nzbc - 2 * nbc
    nx = nxbc - 2 * nbc

    if nbc <= 1:
        return torch.zeros_like(vel)

    a = (nbc - 1) * dx
    kappa = 3.0 * velmin * torch.log(torch.tensor(1e7, device=vel.device)) / (2.0 * a)

    damp1d = kappa.unsqueeze(-1) * (
        (torch.arange(nbc, dtype=torch.float32, device=vel.device) * dx / a) ** 2
    )
    damp = torch.zeros((nb, nzbc, nxbc), dtype=torch.float32, device=vel.device)

    # Fill left and right damping zones
    damp[:, :, 0:nbc] = torch.flip(damp1d, [1]).unsqueeze(1)
    damp[:, :, nx + nbc : nx + 2 * nbc] = damp1d.unsqueeze(1)

    # Fill top and bottom damping zones
    damp[:, 0:nbc, nbc : nbc + nx] = torch.flip(damp1d, [1]).unsqueeze(-1)
    damp[:, nbc + nz : nz + 2 * nbc, nbc : nbc + nx] = damp1d.unsqueeze(-1)

    return damp


def padvel(v0, nbc):
    """
    Pads the velocity model by extending the edge values outward.
    Now uses torch instead of numpy.
    """
    v0 = v0.unsqueeze(0)
    ret = F.pad(v0, (nbc, nbc, nbc, nbc), mode="replicate")
    return ret.squeeze(0)


def expand_source(s0, nt):
    """
    Ensures the source time function has length 'nt'.
    Now uses torch instead of numpy.
    """
    nt0 = s0.size(0)
    if nt0 < nt:
        s = torch.zeros(nt, dtype=torch.float32, device=s0.device)
        s[:nt0] = s0
        return s
    else:
        return s0[:nt]


def adjust_sr(coord, dx, nbc, device="cpu"):
    """
    Converts physical source/receiver coordinates to grid indices.
    Now uses torch instead of numpy.
    """
    # MATLAB's round(x.5) rounds away from zero. Using torch.floor(x + 0.5) for
    # positive numbers emulates MATLAB's behavior.

    isx = torch.floor(coord["sx"] / dx + 0.5).long() + nbc
    isz = torch.floor(coord["sz"] / dx + 0.5).long() + nbc
    igx = torch.floor(coord["gx"] / dx + 0.5).long() + nbc
    igz = torch.floor(coord["gz"] / dx + 0.5).long() + nbc

    # if coord["sz"] < 0.5:
    #     isz += 1

    # igz += (torch.abs(torch.tensor(coord["gz"], device=device)) < 0.5).long()

    return isx, isz, igx, igz


def create_laplacian_kernel(device="cpu"):
    """
    Creates a 5x5 convolutional kernel equivalent to the finite difference Laplacian
    in the selected code.
    """
    c2 = 4.0 / 3.0
    c3 = -1.0 / 12.0

    # Create 5x5 kernel
    kernel = torch.zeros(1, 1, 5, 5, device=device)

    # Center position is (2, 2) in 0-indexed coordinates
    kernel[0, 0, 2, 1] = c2  # left (j-1)
    kernel[0, 0, 2, 3] = c2  # right (j+1)
    kernel[0, 0, 1, 2] = c2  # up (i-1)
    kernel[0, 0, 3, 2] = c2  # down (i+1)

    kernel[0, 0, 2, 0] = c3  # left by 2 (j-2)
    kernel[0, 0, 2, 4] = c3  # right by 2 (j+2)
    kernel[0, 0, 0, 2] = c3  # up by 2 (i-2)
    kernel[0, 0, 4, 2] = c3  # down by 2 (i+2)

    return kernel


def apply_laplacian_conv(p1, kernel):
    """
    Apply the Laplacian using convolution.

    Args:
        p1: Input tensor of shape (height, width)
        kernel: The Laplacian kernel
    """
    (nb, n, h, w) = p1.shape
    x = p1.reshape(nb * n, 1, h, w)
    x = F.pad(x, (2, 2, 2, 2), mode="circular")
    laplacian = F.conv2d(x, kernel)
    ret = laplacian.reshape(nb, n, h, w)

    return ret


def a2d_mod_abc24(v, nbc, dx, nt, dt, s, coord, isFS, device="cpu"):
    """
    Performs a 2D acoustic wave finite-difference simulation (4th order).
    Now uses torch instead of numpy.
    """
    n_receivers = (
        len(coord["gx"])
        if isinstance(coord["gx"], (list, tuple))
        else coord["gx"].shape[0]
    )
    nb = v.size(0)
    seis = torch.zeros((nb, 6, nt, n_receivers), dtype=torch.float32, device=device)

    c1 = torch.tensor(-2.5, device=device, dtype=torch.float32)
    c2 = torch.tensor(4.0 / 3.0, device=device, dtype=torch.float32)
    c3 = torch.tensor(-1.0 / 12.0, device=device, dtype=torch.float32)

    # Convert v to tensor if it's numpy
    if isinstance(v, np.ndarray):
        v = torch.from_numpy(v).float().to(device)

    v_padded = padvel(v, nbc)
    abc = AbcCoef2D(v_padded, nbc, dx)

    alpha = (v_padded * dt / dx) ** 2
    kappa = abc * dt
    temp1 = 2 + 2 * c1 * alpha - kappa
    temp2 = 1 - kappa
    beta_dt = (v_padded * dt) ** 2
    temp1 = temp1.unsqueeze(1)
    temp2 = temp2.unsqueeze(1)
    alpha = alpha.unsqueeze(1)

    # Convert s to tensor if it's numpy
    if isinstance(s, np.ndarray):
        s = torch.from_numpy(s).float().to(device)
    s = expand_source(s, nt)

    # isx, isz, igx, igz = adjust_sr(coord, dx, nbc, device)

    p0 = torch.zeros((nb, 6, 310, 310), dtype=torch.float32, device=device)
    p1 = torch.zeros((nb, 6, 310, 310), dtype=torch.float32, device=device)
    kernel = create_laplacian_kernel(device)
    # print(f"isx = {isx}, isz = {isz}")
    source_inds = torch.arange(6, dtype=torch.long, device=device)
    # 6ch for flip aug
    isx = torch.tensor([120, 137, 154, 155, 172, 189], dtype=torch.long, device=device)
    igz = 121
    isz = 121

    for it in range(nt):
        # laplacian = c2 * (
        #     torch.roll(p1, 1, dims=1)
        #     + torch.roll(p1, -1, dims=1)
        #     + torch.roll(p1, 1, dims=0)
        #     + torch.roll(p1, -1, dims=0)
        # ) + c3 * (
        #     torch.roll(p1, 2, dims=1)
        #     + torch.roll(p1, -2, dims=1)
        #     + torch.roll(p1, 2, dims=0)
        #     + torch.roll(p1, -2, dims=0)
        # )
        laplacian = apply_laplacian_conv(p1, kernel)

        p = temp1 * p1 - temp2 * p0 + alpha * laplacian
        p[:, source_inds, isz, isx] += beta_dt[:, isz, isx] * s[it]
        seis[:, source_inds, it, :] = p[:, source_inds, igz, 120:190]

        p0, p1 = p1, p

    return seis


def vel_to_seis(vel, method="abc24", device="cpu"):
    """
    Runs the simulation for multiple sources and collects the seismograms.
    Now uses torch instead of numpy.

    Args:
        vel (torch.Tensor or np.ndarray): The (N, 70, 70) velocity model.
        method (str): The simulation function to use ('abc24').
        device (str): The device to run computation on ('cpu' or 'cuda').

    Returns:
        torch.Tensor: Stacked seismogram data of shape (5, 1001, 70).
    """
    # Convert vel to tensor if it's numpy
    if isinstance(vel, np.ndarray):
        vel = torch.from_numpy(vel).float().to(device)
    elif not isinstance(vel, torch.Tensor):
        vel = torch.tensor(vel, dtype=torch.float32, device=device)
    else:
        vel = vel.to(device)

    # 1. Model and Simulation Parameters
    nb, nz, nx = vel.shape
    dx = 10.0
    nbc = 120
    nt = 1001
    dt = 1e-3
    freq = 15.0
    isFS = False  # Use free surface condition or not

    # 2. Generate Ricker wavelet source
    s, _ = ricker(freq, dt, device=device)

    # 3. Setup Receiver Coordinates
    # Receivers are placed at every grid point horizontally at a fixed depth.
    coord = {}
    coord["sz"] = torch.tensor(1 * dx, dtype=torch.float32, device=device)
    coord["gx"] = torch.arange(nx, dtype=torch.float32, device=device) * dx
    coord["gz"] = torch.ones(nx, dtype=torch.float32, device=device) * dx

    # 4. Loop over source positions and run simulation
    source_x_locations = [0, 17, 34, 52, 69]  # Using 0-based indices now

    seis_data = a2d_mod_abc24(vel, nbc, dx, nt, dt, s, coord, isFS, device)

    return seis_data
