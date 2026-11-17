"""ACC（自适应凸组合）基线，按 CLAUDE.md 5.2 节的描述重新实现（未获得 Jin et al., ICASSP 2024 原文细节，
可能与原文存在差异）。

每个频点维护凸权重 alpha(f)（初始为均匀分布），逐帧：
    S_t = sum_p alpha_p Z_p,t                         （用更新前的权重输出，在线、因果）
    P_hat <- lam * P_hat + (1 - lam) |S_t|^2          （输出功率的递归平滑估计）
    grad_p = 2 Re(conj(S_t) Z_p,t)                    （J = |S|^2 对 alpha_p 的梯度）
    a_p = alpha_p * exp(-mu * grad_p / P_hat)，alpha = a / sum(a)   （指数梯度）
mu 在验证集上从 {0.01, 0.03, 0.1, 0.3} 中选择；lam 由 CLAUDE.md 未规定，取 0.9（配置可改）。
由于字典成员都是无失真的，任意凸组合仍无失真。
"""
from __future__ import annotations

import torch


@torch.no_grad()
def acc_fuse(Z: torch.Tensor, mu: float, lam: float = 0.9, max_step: float = 30.0):
    """Z: [B, P, F, T] complex -> (S [B, F, T], alpha [B, P, F, T])，alpha 为每帧实际使用的权重。"""
    B, P, F, T = Z.shape
    alpha = torch.full((B, P, F), 1.0 / P, dtype=torch.float64, device=Z.device)
    Zd = Z.to(torch.complex128)
    p_floor = (Zd.abs() ** 2).mean(dim=(1, 3)) * 1e-6 + 1e-20  # [B, F]，避免静音帧除零
    p_hat = None
    S = torch.empty(B, F, T, dtype=torch.complex128, device=Z.device)
    A = torch.empty(B, P, F, T, dtype=torch.float64, device=Z.device)
    for t in range(T):
        zt = Zd[..., t]  # [B, P, F]
        st = (alpha * zt).sum(1)
        S[..., t] = st
        A[..., t] = alpha
        pw = st.abs() ** 2
        p_hat = pw if p_hat is None else lam * p_hat + (1 - lam) * pw
        grad = 2 * (st.conj()[:, None] * zt).real
        step = (-mu * grad / (p_hat + p_floor)[:, None]).clamp(-max_step, max_step)
        a = alpha * torch.exp(step)
        alpha = a / a.sum(1, keepdim=True)
    return S.to(Z.dtype), A.to(torch.float32)
