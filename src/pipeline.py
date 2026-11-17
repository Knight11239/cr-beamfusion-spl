"""训练/评测共用的前端：时域阵列信号 -> STFT（可选失配）-> 固定波束输出。"""
from __future__ import annotations

import torch

from .stft import apply_mismatch, istft, stft


def array_stft(x: torch.Tensor, gain_db: torch.Tensor | None = None, phase_deg: torch.Tensor | None = None):
    """x: [B, M, N] -> Y: [B, M, F, T]；若给出失配参数（[B, M]），在 STFT 域对每个通道施加。"""
    Y = stft(x)
    if gain_db is not None and (gain_db.abs().max() > 0 or phase_deg.abs().max() > 0):
        Y = apply_mismatch(Y, gain_db, phase_deg)
    return Y


def prepare(batch: dict, bank, use_mismatch: bool = True, target_key: str = "ref"):
    """返回 (Y_mix, Z_mix, X)，X 为训练目标的 STFT（target_key：ref_direct = 直达声，ref = x_ref）。"""
    g, p = (batch["mm_gain_db"], batch["mm_phase_deg"]) if use_mismatch else (None, None)
    Y = array_stft(batch["mix"], g, p)
    return Y, bank(Y), stft(batch[target_key])


def stft_mse(S: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    """L = mean |S - X_ref|^2（STFT 域，与 BeamFusion 一致）。"""
    return (S - X).abs().pow(2).mean()


def to_time(S: torch.Tensor, length: int) -> torch.Tensor:
    return istft(S, length)
