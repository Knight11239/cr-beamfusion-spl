"""STFT（FFT 512、窗长 512 sqrt-Hann、hop 128，F = 257）与阵元增益/相位失配（CLAUDE.md 1、3.6 节）。"""
from __future__ import annotations

import math

import torch

N_FFT = 512
HOP = 128


def sqrt_hann(n=N_FFT, dtype=torch.float32, device=None):
    return torch.hann_window(n, periodic=True, dtype=dtype, device=device).sqrt()


def stft(x: torch.Tensor, n_fft=N_FFT, hop=HOP) -> torch.Tensor:
    """x: [..., N] 实信号 -> [..., F, T] 复数谱。"""
    shape = x.shape
    win = sqrt_hann(n_fft, dtype=x.dtype, device=x.device)
    X = torch.stft(x.reshape(-1, shape[-1]), n_fft, hop, n_fft, win, center=True, return_complex=True)
    return X.reshape(*shape[:-1], *X.shape[-2:])


def istft(X: torch.Tensor, length: int, n_fft=N_FFT, hop=HOP) -> torch.Tensor:
    shape = X.shape
    win = sqrt_hann(n_fft, dtype=X.real.dtype, device=X.device)
    x = torch.istft(X.reshape(-1, *shape[-2:]), n_fft, hop, n_fft, win, center=True, length=length)
    return x.reshape(*shape[:-2], length)


def mismatch_factors(gain_db: torch.Tensor, phase_deg: torch.Tensor) -> torch.Tensor:
    """每个麦克风的复数失配因子 10^(g/20) exp(j phi)，形状同 gain_db（[..., M]）。"""
    g = torch.pow(10.0, gain_db / 20.0)
    ph = phase_deg * (math.pi / 180.0)
    return torch.polar(g.to(torch.float32), ph.to(torch.float32))


def apply_mismatch(Y: torch.Tensor, gain_db: torch.Tensor, phase_deg: torch.Tensor) -> torch.Tensor:
    """在 STFT 域对阵列观测施加增益/相位失配（所有频点相同）。

    Y: [B, M, F, T] 或 [M, F, T]；gain_db、phase_deg: [B, M] 或 [M]。
    失配是逐通道线性运算，因此可分别作用于目标/干扰/噪声分量，保证 shadow filtering 的分解仍然精确。
    """
    a = mismatch_factors(gain_db, phase_deg).to(Y.dtype)
    return Y * a[..., None, None]
