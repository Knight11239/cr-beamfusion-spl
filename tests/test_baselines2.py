"""第二批基线的单元测试：AF-DMA 的选择规则，DMA-GSC-MPDR 的无失真、WNG 约束与因果性（在滤波器空间验证）。"""
import numpy as np
import torch

from src.baselines2 import afdma_fuse, gsc_mpdr_fuse, subspace_gram
from src.beamformers import build_dictionary, steer, stft_freqs

M, DELTA, C = 8, 0.01, 343.0


def _dict():
    F = stft_freqs(16000, 512)
    H = build_dictionary(F, n_mics=M, spacing=DELTA, c=C)["H"]  # [P, F, M]
    d = steer(F, 0.0, M, DELTA, C)  # [F, M]
    return torch.as_tensor(H), torch.as_tensor(d)


def _beams(H, Y):
    """Z_p = h_pᴴ y：H [P, F, M]，Y [B, M, F, T] → [B, P, F, T]。"""
    return torch.einsum("pfm,bmft->bpft", H.conj(), Y)


def _rand_Y(B=2, T=40, seed=0):
    g = torch.Generator().manual_seed(seed)
    F = 257
    return torch.complex(torch.randn(B, M, F, T, generator=g, dtype=torch.float64),
                         torch.randn(B, M, F, T, generator=g, dtype=torch.float64))


def test_afdma_selects_min_power_and_ties():
    Z = torch.zeros(1, 5, 2, 3, dtype=torch.complex64)
    Z[0, :, 0, 0] = torch.tensor([3, 1, 2, 5, 4], dtype=torch.complex64)  # 最小：第 1 个
    Z[0, :, 1, 0] = torch.tensor([1, 2, 1, 3, 1], dtype=torch.complex64)  # 并列：0、2、4
    a = afdma_fuse(Z)
    assert torch.allclose(a[0, :, 0, 0], torch.tensor([0, 1, 0, 0, 0.]))
    assert torch.allclose(a[0, :, 1, 0], torch.tensor([1 / 3, 0, 1 / 3, 0, 1 / 3]))
    assert torch.allclose(a.sum(1), torch.ones(1, 2, 3))
    a4 = afdma_fuse(Z, beams=[0, 1, 2, 4])  # 原文 4 波束字典：null-150 不参与
    assert (a4[:, 3] == 0).all()
    assert torch.allclose(a4[0, :, 1, 0], torch.tensor([1 / 3, 0, 1 / 3, 0, 1 / 3]))


def test_gsc_mpdr_distortionless_and_wng():
    H, d = _dict()
    Z = _beams(H, _rand_Y())
    K = torch.einsum("pfm,qfm->fpq", H.conj(), H)
    G = subspace_gram(K)
    for w0 in (None, -5.0, 0.0):
        coef, s = gsc_mpdr_fuse(Z, lam=0.9, delta=1e-3, G=G, w0_db=w0, n_mics=M)
        assert torch.allclose(coef.sum(1), torch.ones_like(coef.sum(1)))
        # 在滤波器空间显式重建 h = Σ conj(c_p) h_p，并核对 hᴴd = 1 与 WNG ≥ W0（f > 0）
        h = torch.einsum("bpft,pfm->bftm", coef.conj(), H)
        resp = torch.einsum("bftm,fm->bft", h.conj(), d)
        assert (resp[:, 1:] - 1).abs().max() < 1e-9
        wng = 10 * torch.log10(resp.abs() ** 2 / (h.abs() ** 2).sum(-1))
        if w0 is not None:
            assert wng[:, 1:].min() >= w0 - 1e-6
            assert (s <= 1 + 1e-12).all()


def test_gsc_mpdr_causal():
    H, _ = _dict()
    Y = _rand_Y(B=1, T=30, seed=1)
    Z1 = _beams(H, Y)
    Y2 = Y.clone()
    Y2[..., 20:] = _rand_Y(B=1, T=10, seed=2)
    Z2 = _beams(H, Y2)
    c1, _ = gsc_mpdr_fuse(Z1, 0.95, 1e-3)
    c2, _ = gsc_mpdr_fuse(Z2, 0.95, 1e-3)
    assert torch.equal(c1[..., :20], c2[..., :20])  # 第 20 帧之后的输入不影响之前的系数
    assert not torch.equal(c1[..., 20:], c2[..., 20:])


def test_gsc_mpdr_minimizes_recursive_power():
    """在最后一帧：解使 J(u) = uᴴRu + 2Re(uᴴr) 在加载后的 R 下最小（与随机扰动比较）。"""
    H, _ = _dict()
    Z = _beams(H, _rand_Y(B=1, T=50, seed=3)).to(torch.complex128)
    lam, delta = 0.9, 1e-3
    coef, _ = gsc_mpdr_fuse(Z, lam, delta)
    e = Z[:, 1:] - Z[:, :1]
    zD = Z[:, 0]
    f = 100
    R = torch.zeros(4, 4, dtype=torch.complex128)
    r = torch.zeros(4, dtype=torch.complex128)
    for t in range(Z.shape[-1]):
        et = e[0, :, f, t]
        R = lam * R + (1 - lam) * torch.outer(et, et.conj())
        r = lam * r + (1 - lam) * et * zD[0, f, t].conj()
    A = R + delta * torch.trace(R).real / 4 * torch.eye(4)
    u = coef[0, 1:, f, -1].conj()
    J = lambda v: (v.conj() @ A @ v).real + 2 * (v.conj() @ r).real  # noqa: E731
    g = torch.Generator().manual_seed(0)
    for _ in range(20):
        dv = torch.complex(torch.randn(4, generator=g, dtype=torch.float64), torch.randn(4, generator=g, dtype=torch.float64))
        assert J(u) <= J(u + 1e-3 * dv) + 1e-12


def _mpdr_loop(Z, lam, delta, G=None, w0_db=None, n_mics=None):
    """逐帧递归的参考实现（2026-11-08 版本的 gsc_mpdr_fuse；λ = 1 时为累积平均）。"""
    import math
    B, P, F, T = Z.shape
    Zd = Z.to(torch.complex128)
    e, zD, Pp = Zd[:, 1:] - Zd[:, :1], Zd[:, 0], P - 1
    R = torch.zeros(B, F, Pp, Pp, dtype=torch.complex128)
    r = torch.zeros(B, F, Pp, dtype=torch.complex128)
    coef = torch.zeros(B, P, F, T, dtype=torch.complex128)
    for t in range(T):
        et = e[..., t].permute(0, 2, 1)
        a, b = (lam, 1 - lam) if lam < 1 else (t / (t + 1), 1 / (t + 1))
        R = a * R + b * et[..., :, None] * et.conj()[..., None, :]
        r = a * r + b * et * zD[..., t, None].conj()
        tr = torch.diagonal(R, dim1=-2, dim2=-1).real.sum(-1)
        u = -torch.linalg.solve(R + (delta * tr / Pp + 1e-30)[..., None, None] * torch.eye(Pp), r[..., None])[..., 0]
        if w0_db is not None:
            rad = math.sqrt(1 / 10 ** (w0_db / 10) - 1 / n_mics)
            n = torch.sqrt(torch.einsum("bfp,fpq,bfq->bf", u.conj(), G, u).real.clamp_min(0) + 1e-24)
            u = u * torch.clamp(rad / n, max=1.0)[..., None]
        c = u.conj()
        coef[:, 1:, :, t] = c.permute(0, 2, 1)
        coef[:, 0, :, t] = 1 - c.sum(-1)
    return coef


def test_gsc_mpdr_block_matches_loop():
    """按块并行的递归与逐帧递归一致（λ < 1 与 λ = 1 累积平均；含 WNG 投影）。"""
    H, _ = _dict()
    Z = _beams(H, _rand_Y(B=1, T=300, seed=4))
    G = subspace_gram(torch.einsum("pfm,qfm->fpq", H.conj(), H))
    for lam, delta, w0 in ((0.9, 1e-3, None), (0.998, 1e-5, None), (0.9995, 1e-6, -10.0), (1.0, 1e-5, None), (1.0, 1e-6, -5.0)):
        c1, _ = gsc_mpdr_fuse(Z, lam, delta, G, w0, M)
        c2 = _mpdr_loop(Z, lam, delta, G, w0, M)
        err = ((c1 - c2).abs() / (c2.abs() + 1e-3)).max()
        assert err < 1e-6, (lam, delta, w0, float(err))
