"""第二批基线（用户 2026-11-08；只用 CPU、不需训练）。

AF-DMA（Zhao et al., IEEE SPL 32: 3794–3798, 2025, DOI 10.1109/LSP.2025.3612336），按原文式 (10)–(24) 复现：
  候选为 K 个零点 DMA 加 MWNG（= h_D）；权重为实数、非负、和为 1；准则为瞬时输出功率最小，经 Jensen 松弛后的解为
  在每个时频点选 |Z_k|^2 最小的波束（权重 1）；若有多个并列最小值，对它们取均匀权重。没有平滑，没有可调超参数。
  （与原文配置的差异——STFT、数据、房间等——见 results/baselines2/README.md。）

DMA-GSC-MPDR（传统复数仿射基线，与 AF-DMA 不同）：与本方法使用同一个 DMA 差分子空间 h = h_D + Σ_p u_p g_p
  （g_p = h_p − h_D，故对任意复数 u 都无失真）。在波束输出空间：Ẑ = Z_D + Σ_p c_p e_p，e_p = Z_p − Z_D，c_p = conj(u_p)。
  逐频点用递归平均估计 R_t = λ R_{t−1} + (1 − λ) e_t e_tᴴ 与 r_t = λ r_{t−1} + (1 − λ) e_t Z_D,t*（只用当前及过去帧，严格因果），
  求使输出功率最小的复数 u：u_t = −(R_t + δ·tr(R_t)/P′·I)^{-1} r_t（闭式解，相对对角加载 δ）。
  可选：再施加 Cox 的 WNG 约束（scaled projection，与本方法的投影公式相同）：n = sqrt(uᴴ G u)，r = sqrt(1/W0 − 1/M)，
  u ← min(1, r/n)·u。超参数（λ、δ、W0）只在验证集上选择。
"""
from __future__ import annotations

import math

import torch


@torch.no_grad()
def afdma_fuse(Z: torch.Tensor, beams=None):
    """Z: [B, P, F, T] complex → alpha [B, P, F, T]（实数、非负、和为 1）。beams：参与选择的波束下标（默认全部）。"""
    B, P, F, T = Z.shape
    idx = list(range(P)) if beams is None else list(beams)
    pw = (Z[:, idx].abs() ** 2).to(torch.float64)  # [B, K, F, T]
    mn = pw.min(1, keepdim=True).values
    mask = (pw == mn).to(torch.float64)  # 并列最小值都为 1
    a = mask / mask.sum(1, keepdim=True)
    alpha = torch.zeros(B, P, F, T, dtype=torch.float32, device=Z.device)
    alpha[:, idx] = a.to(torch.float32)
    return alpha


def subspace_gram(K: torch.Tensor) -> torch.Tensor:
    """由含 h_D 的字典 Gram 矩阵 K_pq = h_pᴴ h_q（[F, P, P]）得到差分子空间的 G_pq = g_pᴴ g_q（[F, P−1, P−1]）。"""
    K = K.to(torch.complex128)
    return K[:, 1:, 1:] - K[:, 1:, :1] - K[:, :1, 1:] + K[:, :1, :1]


@torch.no_grad()
def gsc_mpdr_fuse(Z: torch.Tensor, lam: float, delta: float, G: torch.Tensor | None = None, w0_db=None,
                  n_mics: int | None = None, block: int | None = None):
    return mpdr_coef(mpdr_u(Z, lam, delta, block), G, w0_db, n_mics)


@torch.no_grad()
def mpdr_u(Z: torch.Tensor, lam: float, delta: float, block: int | None = None):
    """Z: [B, P, F, T]（第 0 个为 h_D）→ 未投影的 u [B, F, T, P′]（gsc_mpdr_fuse = mpdr_coef(mpdr_u(...))，
    返回 (coef [B, P, F, T] complex128，s [B, F, T] 投影缩放)）。coef 与 evaluate 的 apply 一致：Ẑ = Σ_p coef_p Z_p，coef_0 = 1 − Σ_{p≥1} coef_p，coef_p = s·conj(u_p)。

    λ < 1：R_t = λ R_{t−1} + (1 − λ) e_t e_tᴴ（r 同理），R_{−1} = 0。
    λ = 1（用户 2026-11-09）：从句首开始的累积平均 R_t = (1/(t+1)) Σ_{k≤t} e_k e_kᴴ（仍然因果，是 λ 的上限）。
      实现为累积和：u = −(R + δ·tr(R)/P′·I)^{-1} r 对 R、r 的共同缩放不变（加载与迹成比例），故累积和与累积平均给出同一个 u。
    计算（2026-11-09 起）：递归按块并行（块内 λ^{−k} 加权累积和，块长使 λ^{−块长} ≤ 1e4，float64），再对所有帧批量求解；
    与逐帧递归在数值上一致（tests/test_baselines2.py 与逐帧参考实现核对）。"""
    B, P, F, T = Z.shape
    lam = float(lam)
    assert 0 < lam <= 1
    Zd = Z.to(torch.complex128)
    e = (Zd[:, 1:] - Zd[:, :1]).permute(0, 2, 3, 1)  # [B, F, T, P']
    zD = Zd[:, 0]  # [B, F, T]
    Pp = P - 1
    X = e[..., :, None] * e.conj()[..., None, :]  # e_t e_tᴴ [B, F, T, P', P']
    x = e * zD[..., None].conj()  # e_t Z_D,t* [B, F, T, P']
    w = 1.0 if lam == 1.0 else 1.0 - lam
    if block is None:
        block = T if lam == 1.0 else max(1, min(T, int(math.log(1e4) / -math.log(lam))))
    Rs, rs = [], []
    R0 = torch.zeros(B, F, Pp, Pp, dtype=torch.complex128, device=Z.device)
    r0 = torch.zeros(B, F, Pp, dtype=torch.complex128, device=Z.device)
    for t0 in range(0, T, block):
        n = min(block, T - t0)
        j = torch.arange(n, dtype=torch.float64, device=Z.device)
        up, dn = lam ** (j + 1), lam ** (-j)  # λ^{j+1}，λ^{−k}
        Rb = up[:, None, None] / lam * (torch.cumsum(X[:, :, t0:t0 + n] * dn[:, None, None], 2) * w) + up[:, None, None] * R0[:, :, None]
        rb = up[:, None] / lam * (torch.cumsum(x[:, :, t0:t0 + n] * dn[:, None], 2) * w) + up[:, None] * r0[:, :, None]
        Rs.append(Rb)
        rs.append(rb)
        R0, r0 = Rb[:, :, -1], rb[:, :, -1]
    R, r = torch.cat(Rs, 2), torch.cat(rs, 2)  # [B, F, T, P', P']，[B, F, T, P']
    eye = torch.eye(Pp, dtype=torch.complex128, device=Z.device)
    tr = torch.diagonal(R, dim1=-2, dim2=-1).real.sum(-1)  # [B, F, T]
    A = R + (delta * tr / Pp + 1e-30)[..., None, None] * eye
    return -torch.linalg.solve(A, r[..., None])[..., 0]  # u [B, F, T, P']


@torch.no_grad()
def mpdr_coef(u: torch.Tensor, G: torch.Tensor | None = None, w0_db=None, n_mics: int | None = None):
    """u [B, F, T, P'] → (coef [B, P, F, T]，s [B, F, T])；可选 WNG 投影（同一个 u 可用于多个 W0）。"""
    B, F, T, _ = u.shape
    S = torch.ones(B, F, T, dtype=torch.float64, device=u.device)
    if w0_db is not None:
        w0 = 10 ** (float(w0_db) / 10)
        assert w0 <= n_mics, "W0 must not exceed M"
        rad = math.sqrt(max(1.0 / w0 - 1.0 / n_mics, 0.0))
        Gc = G.to(torch.complex128).to(u.device)  # [F, P', P']
        nrm = torch.sqrt(torch.einsum("bftp,fpq,bftq->bft", u.conj(), Gc, u).real.clamp_min(0) + 1e-24)
        S = torch.clamp(rad / nrm, max=1.0)
        u = u * S[..., None]
    c = u.conj().permute(0, 3, 1, 2)  # c_p = conj(u_p) [B, P', F, T]
    coef = torch.cat([1 - c.sum(1, keepdim=True), c], 1)
    return coef, S
