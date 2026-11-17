"""WNG 投影层（CLAUDE.md 6.2 节）。

仿射融合 h = h_D + sum_p u_p g_p（g_p = h_p - h_D，g_p^H d = 0），c = sum_p u_p g_p，||c||^2 = u^H G u，
W[h] = 1 / (1/M + ||c||^2)。给定 W0（线性，W0 <= M）：

  r = sqrt(1/W0 - 1/M)，n = sqrt(u^H G u + 1e-12)
  hard：  s = min(1, r / n)
  smooth：s = r / smax(r, n)，smax(a, b) = b + softplus(beta (a - b)) / beta，beta = 20 / r

投影后 h_hat = h_D + s c：无失真不变（g_p^H d = 0），||s c|| <= r，即 WNG >= W0。
smooth 版本恒有 smax(r, n) >= max(r, n)（softplus(x) >= max(0, x)），所以 s <= min(1, r/n)，约束不会被放松；
代价是 WNG 已满足时 s 略小于 1（hard 版本此时 s = 1 精确成立）。
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as Fn


def quad_norm(u: torch.Tensor, G: torch.Tensor) -> torch.Tensor:
    """u^H G u（实数）。u: [B, P, F, T] complex，G: [F, P, P] complex -> [B, F, T]。"""
    G = G.to(u.dtype)
    return torch.einsum("bpft,fpq,bqft->bft", u.conj(), G, u).real.clamp_min(0.0)


def radius(w0_lin, n_mics: int):
    """r = sqrt(1/W0 - 1/M)。w0_lin 可为标量或张量（如 [B, F, 1]）。"""
    if torch.is_tensor(w0_lin):
        if torch.any(w0_lin > n_mics):
            raise ValueError("W0 must not exceed M")
        return torch.sqrt(1.0 / w0_lin - 1.0 / n_mics)
    if w0_lin > n_mics:
        raise ValueError("W0 must not exceed M")
    return math.sqrt(1.0 / w0_lin - 1.0 / n_mics)


def scale_factor(n: torch.Tensor, r, mode: str = "hard") -> torch.Tensor:
    """投影系数 s（与 n 同形状）。n = sqrt(u^H G u + 1e-12)，r 为标量或可广播的张量。"""
    if mode == "hard":
        return torch.clamp(r / n, max=1.0)
    if mode == "smooth":
        beta = 20.0 / r
        smax = n + Fn.softplus(beta * (r - n)) / beta
        return r / smax
    raise ValueError(mode)


def project(u: torch.Tensor, G: torch.Tensor, w0_lin, n_mics: int, mode: str = "hard"):
    """返回 (s [B, F, T], n [B, F, T], wng_lin [B, F, T])；wng_lin 为投影后融合滤波器的 WNG（线性）。"""
    q = quad_norm(u, G)
    n = torch.sqrt(q + 1e-12)
    s = scale_factor(n, radius(w0_lin, n_mics), mode)
    wng_lin = 1.0 / (1.0 / n_mics + s.pow(2) * q)
    return s, n, wng_lin
