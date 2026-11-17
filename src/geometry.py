"""阵列几何与导向向量（CLAUDE.md 第 1 节）。

约定
----
- 均匀线阵，M 元，间距 delta；阵元坐标（沿阵列轴，以阵列中心为参考）x_m = (m - (M-1)/2) * delta。
- 方位角 theta 从 +x 轴（端射）起算，目标 theta_s = 0；声源物理方向单位向量 u(theta) = (cos, sin, 0)。
- 导向向量 d_m(w, theta) = exp(-j w x_m cos(theta) / c)，STFT 为标准 exp(-j w t) 约定（torch.stft）。

符号一致性：在 exp(-j w t) 的傅里叶约定下，时延 tau 对应因子 exp(-j w tau)。若第 m 个麦克风真的放在
center + x_m * e_x，那么 theta=0 的声源先到达 x_m > 0 的麦克风（tau_m = -x_m cos(theta)/c），实际导向向量是
exp(+j w x_m cos(theta)/c)，即 CLAUDE.md 公式的共轭。为使 CLAUDE.md 的 d 与物理仿真严格一致，默认
（mic_order = "target_first"）把第 m 个麦克风放在 center - x_m * e_x：m = 0 离目标最近，此时
tau_m = +x_m cos(theta)/c，d_m = exp(-j w tau_m) 与 CLAUDE.md 完全一致（tests/test_stage1.py 有数值验证）。
"literal" 选项按字面把麦克风放在 center + x_m * e_x。两种约定只差通道顺序翻转。
"""
from __future__ import annotations

import numpy as np

C_SOUND = 343.0


def mic_coords(n_mics: int = 8, spacing: float = 0.01) -> np.ndarray:
    """阵列轴上的坐标 x_m（float64，中心参考）。"""
    m = np.arange(n_mics, dtype=np.float64)
    return (m - (n_mics - 1) / 2.0) * spacing


def axis_sign(mic_order: str = "target_first") -> float:
    if mic_order == "target_first":
        return -1.0
    if mic_order == "literal":
        return 1.0
    raise ValueError(f"unknown mic_order {mic_order}")


def mic_positions(center, n_mics=8, spacing=0.01, mic_order="target_first", dx=None) -> np.ndarray:
    """房间坐标下的麦克风位置，形状 [3, M]。dx（可选）为各麦克风 x 坐标的位置误差（米，物理 x 轴）。"""
    center = np.asarray(center, dtype=np.float64)
    x = axis_sign(mic_order) * mic_coords(n_mics, spacing)
    pos = np.tile(center[:, None], (1, n_mics))
    pos[0] += x
    if dx is not None:
        pos[0] += np.asarray(dx, dtype=np.float64)
    return pos


def source_direction(theta_deg) -> np.ndarray:
    th = np.deg2rad(np.asarray(theta_deg, dtype=np.float64))
    return np.stack([np.cos(th), np.sin(th), np.zeros_like(th)], axis=-1)


def source_position(center, theta_deg, dist) -> np.ndarray:
    return np.asarray(center, dtype=np.float64) + dist * source_direction(theta_deg)


def steering_vector(freqs, theta_deg, n_mics=8, spacing=0.01, c=C_SOUND) -> np.ndarray:
    """d(f, theta)，complex128，形状 [F, n_theta, M]（freqs 与 theta 可为标量或 1-D）。"""
    f = np.atleast_1d(np.asarray(freqs, dtype=np.float64))
    th = np.atleast_1d(np.deg2rad(np.asarray(theta_deg, dtype=np.float64)))
    x = mic_coords(n_mics, spacing)
    w = 2 * np.pi * f
    phase = -w[:, None, None] * x[None, None, :] * np.cos(th)[None, :, None] / c
    return np.exp(1j * phase)
