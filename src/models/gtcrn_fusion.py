"""波束融合模型（CLAUDE.md 5.1 节；仿射与投影头在阶段 4 加入）。

数据流：阵列 STFT Y [B, M, F, T] -> 固定波束 Z_p = h_p^H y（BeamBank）-> 特征 [实部, 虚部, 幅度]
[B, 3P, T, F] -> GTCRN 骨干 -> 融合头。

- convex（BeamFusion）：P 个 logits，在 P 维 softmax 得 alpha_p(t, f)，S = sum_p alpha_p Z_p。
- affine（本方法，CLAUDE.md 6.1–6.3）：以 h_D 为锚点，2P' 个通道（P' = P_total - 1 个 g_p 的复系数 w_p 的实部、虚部），
  S = Z_D + s * sum_p w_p (Z_p - Z_D)；s 由 WNG 投影层给出（proj = none 时 s = 1）。见 AffineFusionNet。
- affine，weights = real（实系数仿射消融，用户 2026-11-09）：P' 个通道直接作为实数 w_p（不经 softmax、可为负），
  其余（骨干、锚点、初始化、损失、投影）与复数版相同。
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn

from ..beamformers import build_dictionary, gram, stft_freqs
from ..stft import HOP
from .gtcrn import GTCRNBackbone
from .projection import project


class BeamBank(nn.Module):
    """固定波束形成器字典（complex128 设计后转 complex64）。Z_p = h_p^H y。"""

    def __init__(self, H: np.ndarray, names=None):
        super().__init__()
        self.register_buffer("H", torch.as_tensor(H).to(torch.complex64))  # [P, F, M]
        self.names = list(names) if names is not None else [f"beam{p}" for p in range(H.shape[0])]

    @classmethod
    def from_config(cls, cfg, null_angles=(90.0, 120.0, 150.0, 180.0)):
        F = stft_freqs(cfg["signal"]["fs"], 512)
        d = build_dictionary(F, null_angles, n_mics=cfg["array"]["n_mics"], spacing=cfg["array"]["spacing"],
                             c=cfg["signal"]["c"])
        return cls(d["H"], d["names"])

    @property
    def n_beams(self):
        return self.H.shape[0]

    def forward(self, Y):
        """Y: [B, M, F, T] complex -> Z: [B, P, F, T]"""
        return torch.einsum("pfm,bmft->bpft", self.H.conj(), Y)


def beam_features(Z):
    """Z: [B, P, F, T] -> [B, 3P, T, F]（实部、虚部、幅度在通道维拼接，与 BeamFusion 论文一致）。"""
    Zt = Z.transpose(-1, -2)
    return torch.cat([Zt.real, Zt.imag, Zt.abs()], dim=1)


class BeamFusionNet(nn.Module):
    """BeamFusion（凸组合）。forward(Z) -> {"S": [B,F,T], "alpha": [B,P,F,T]}"""

    def __init__(self, n_beams=5, width=32, hidden=32, intra_bidirectional=True):
        super().__init__()
        self.n_beams = n_beams
        self.backbone = GTCRNBackbone(3 * n_beams, n_beams, width, hidden, intra_bidirectional)

    def weights(self, Z):
        logits = self.backbone(beam_features(Z))  # [B, P, T, F]
        return torch.softmax(logits, dim=1).transpose(-1, -2)  # [B, P, F, T]

    def forward(self, Z):
        alpha = self.weights(Z)
        return {"S": (alpha * Z).sum(1), "alpha": alpha}


class AffineFusionNet(nn.Module):
    """仿射融合 + 可选 WNG 投影（CLAUDE.md 6.1–6.3）。forward(Z) -> {"S", "w", "u", "s", "wng_db", "w0_db"}。

    - 锚点为字典第 0 个成员 h_D；g_p = h_p - h_D（p = 1..P'）；Gram 矩阵 G [F, P', P'] 由 complex128 的字典计算。
    - 网络输出 2P'（+1，可学习 W0 时）个通道；w_p = out[p] + j out[P' + p]，u_p = conj(w_p)。
    - 输出层初始化：权重乘以 0.01、偏置置 0（输出层不接 BatchNorm，否则缩放会被 BN 抵消），训练开始时 w ≈ 0，
      即近似为延迟求和滤波器。
    - proj ∈ {none, hard, smooth}；w0_db 为固定值（dB），或 "learn"：额外通道每帧给出
      W0_raw(t, f) = -15 + (W0max + 15) sigmoid(a(t, f)) ∈ [-15, W0max] dB（W0max = floor(10 log10 M)，M = 8 时 9 dB），再沿时间做因果递归平均（时间常数 w0_tau_s，默认 0.5 s；
      用户 2026-10-31 决定，替代原来的整句平均）：W0(t) = lam W0(t-1) + (1 - lam) W0_raw(t)，W0(0) = W0_raw(0)，
      lam = exp(-hop / (tau fs))。凸组合保持范围 [-15, 9] dB，模型整体保持因果。
    - 可学习 W0 的两阶段训练：set_epoch(epoch) 时，epoch < w0_freeze_epochs 用固定 W0 = w0_init_db。
    - f = 0 Hz：字典各成员均为 h_D，g_p = 0，G = 0，输出恒为 Z_D。
    """

    def __init__(self, n_beams, G: np.ndarray, n_mics: int, width=32, hidden=32, intra_bidirectional=True,
                 proj="none", w0_db=0.0, w0_freeze_epochs=0, w0_init_db=0.0, out_init_scale=0.01,
                 w0_tau_s=0.5, frame_rate=125.0, weights="complex"):
        super().__init__()
        self.P = n_beams - 1
        if weights not in ("complex", "real"):
            raise ValueError(weights)
        self.real_weights = weights == "real"
        self.nw = self.P if self.real_weights else 2 * self.P  # 权重通道数
        self.M = n_mics
        self.proj = proj
        self.learn_w0 = w0_db == "learn"
        if not self.learn_w0 and proj != "none" and 10 ** (float(w0_db) / 10) > n_mics:
            raise ValueError("W0 must not exceed 10 log10(M)")
        self.w0_db = None if self.learn_w0 else float(w0_db)
        self.w0_freeze_epochs, self.w0_init_db = int(w0_freeze_epochs), float(w0_init_db)
        self.w0_frozen = self.learn_w0 and self.w0_freeze_epochs > 0
        self.w0_lam = math.exp(-1.0 / (w0_tau_s * frame_rate))  # 每帧的递归平均系数
        # 可学习 W0 的上限：10 log10(M) 向下取整到整数 dB（用户 2026-11-04 确认；M = 8 时为 9 dB，与原定义相同；
        # M = 4 时为 6 dB）。范围 [-15, w0_max_db]。
        self.w0_max_db = float(math.floor(10 * math.log10(n_mics)))
        n_out = self.nw + (1 if self.learn_w0 else 0)
        self.backbone = GTCRNBackbone(3 * n_beams, n_out, width, hidden, intra_bidirectional, last_bn=False)
        conv = self.backbone.output_layer.conv
        with torch.no_grad():
            conv.weight.mul_(out_init_scale)
            conv.bias.zero_()
        self.register_buffer("G", torch.as_tensor(G).to(torch.complex128))  # [F, P', P']
        # 推理 / 评测时以 float64 计算投影（用户 2026-10-31 决定）：u^H G u、s 与输出在 complex128 / float64 中计算，
        # 使“每个时频点 WNG >= W0”在 float64 精度内成立（float32 下 w 较大时有 <= 0.004 dB 的舍入越界）。
        # 训练不变（默认 False，不进入 state_dict）；由 src/evaluate.py 的 ModelMethod 打开。
        self.proj_fp64 = False

    def set_epoch(self, epoch: int):
        self.w0_frozen = self.learn_w0 and epoch < self.w0_freeze_epochs

    def w0(self, out):
        """返回 W0_dB，形状 [B, F, T]（固定时为标量）。"""
        if not self.learn_w0:
            return self.w0_db
        B, _, T, F = out.shape
        if self.w0_frozen:
            return torch.full((B, F, T), self.w0_init_db, dtype=out.dtype, device=out.device)
        raw = -15.0 + (self.w0_max_db + 15.0) * torch.sigmoid(out[:, self.nw].transpose(-1, -2))  # [B, F, T]，每帧
        return causal_ema(raw, self.w0_lam)

    def forward(self, Z):
        out = self.backbone(beam_features(Z))  # [B, n_out, T, F]
        o = out.transpose(-1, -2)  # [B, n_out, F, T]
        if self.real_weights:  # 实系数：w 为实数（以零虚部的复数表示，后续计算与复数版相同）
            w = torch.complex(o[:, :self.P], torch.zeros_like(o[:, :self.P]))
        else:
            w = torch.complex(o[:, :self.P], o[:, self.P:2 * self.P])  # [B, P', F, T]
        u = w.conj()
        D = Z[:, 1:] - Z[:, :1]  # Z_p - Z_D
        res = {"w": w, "u": u}
        if self.proj == "none":
            q = torch.einsum("bpft,fpq,bqft->bft", u.conj(), self.G.to(u.dtype), u).real
            s = torch.ones_like(q)
            res["wng_db"] = 10 * torch.log10(1.0 / (1.0 / self.M + q.clamp_min(0)))
        else:
            w0_db = self.w0(out)
            if self.proj_fp64:  # 推理 / 评测：投影在 float64 中计算（训练路径不变）
                u, w, D = u.to(torch.complex128), w.to(torch.complex128), D.to(torch.complex128)
                res.update(w=w, u=u)
                if torch.is_tensor(w0_db):
                    w0_db = w0_db.double()
            w0_lin = 10 ** (w0_db / 10) if torch.is_tensor(w0_db) else 10 ** (w0_db / 10)
            s, _, wng_lin = project(u, self.G, w0_lin, self.M, self.proj)
            res["wng_db"] = 10 * torch.log10(wng_lin)
            res["w0_db"] = w0_db
        res["s"] = s
        res["S"] = (Z[:, 0] + s * (w * D).sum(1)).to(Z.dtype)
        return res


class SameHeadFusionNet(nn.Module):
    """新对照（用户 2026-11-13；research/analysis_plan_controls.md §1 的 C 与 F）：与 AffineFusionNet 完全相同的骨干与输出层
    （最后一层 ConvTranspose2d(32 → n_out, (1, 5), stride (1, 2))、无 BatchNorm、恒等激活、ERB 合成；权重乘以 out_init_scale），
    只改最后的映射：

    - mode = "convex"（convex_samehead）：n_out = P + 1 个实数 logits，在 P + 1 维 softmax 得凸权重 α_p；Ẑ = Σ_p α_p Z_p。
      偏置初始化：b_0 = 0，b_p = −β（p ≥ 1），β = ln(P (α_D0 / (1 − α_D0)))，使初始 α_D ≈ α_D0（默认 0.95），初始滤波器接近 h_D。
    - mode = "free"（free_complex）：n_out = 2(P + 1) 个通道，w_p = out[p] + j out[P + 1 + p]（p = 0..P，第 0 个为 h_D）；
      Ẑ = Σ_p w_p Z_p，不约束 Σ w = 1（不保证标称无失真）；有效滤波器 h = Σ_p conj(w_p) h_p。偏置初始化：w_D 实部为 1，
      其余为 0，即初始 w ≈ e_D（≈ h_D）。不加投影、输出重标定或后滤波。
    forward(Z) → {"S", "coef" [B, P + 1, F, T]（Ẑ = Σ coef_p Z_p）, "alpha" 或 "w"}。
    （常数偏置经 ERB 合成后在所有频点保持不变，已核对。）
    """

    def __init__(self, n_beams, mode, width=32, hidden=32, intra_bidirectional=True, out_init_scale=0.01,
                 alpha_d0=0.95):
        super().__init__()
        if mode not in ("convex", "free"):
            raise ValueError(mode)
        self.mode, self.Pt = mode, n_beams
        n_out = n_beams if mode == "convex" else 2 * n_beams
        self.backbone = GTCRNBackbone(3 * n_beams, n_out, width, hidden, intra_bidirectional, last_bn=False)
        conv = self.backbone.output_layer.conv
        with torch.no_grad():
            conv.weight.mul_(out_init_scale)
            conv.bias.zero_()
            if mode == "convex":
                beta = math.log((n_beams - 1) * alpha_d0 / (1.0 - alpha_d0))
                conv.bias[1:] = -beta
            else:
                conv.bias[0] = 1.0

    def forward(self, Z):
        o = self.backbone(beam_features(Z)).transpose(-1, -2)  # [B, n_out, F, T]
        if self.mode == "convex":
            alpha = torch.softmax(o, dim=1)
            return {"S": (alpha * Z).sum(1), "alpha": alpha, "coef": alpha}
        w = torch.complex(o[:, :self.Pt], o[:, self.Pt:])
        return {"S": (w * Z).sum(1), "w": w, "coef": w}


def causal_ema(x: torch.Tensor, lam: float, chunk: int = 128) -> torch.Tensor:
    """沿最后一维的因果递归平均 y_t = lam y_{t-1} + (1 - lam) x_t，y_0 = x_0（可求导）。

    分块并行计算：块内 y_j = lam^(j+1) [s + (1 - lam) sum_{k<=j} lam^-(k+1) x_k]，s 为上一块末的 y；
    块长 128 时 lam^-128 ≈ e^2（tau = 0.5 s），在 float64 中计算，数值上与逐帧递归一致（单元测试核对）。
    """
    xd = x.double()
    T = xd.shape[-1]
    s = xd[..., :1]  # y_{-1} := x_0，使 y_0 = x_0
    outs = []
    for t0 in range(0, T, chunk):
        xc = xd[..., t0:t0 + chunk]
        n = xc.shape[-1]
        k = torch.arange(1, n + 1, dtype=torch.float64, device=x.device)
        y = lam ** k * (s + (1 - lam) * torch.cumsum(xc * lam ** (-k), dim=-1))
        outs.append(y)
        s = y[..., -1:]
    return torch.cat(outs, -1).to(x.dtype)


class MaskMVDR(nn.Module):
    """掩码式 MVDR 基线（CLAUDE.md 5.3 节；BeamFusion 论文中没有该基线，具体做法为本项目设定）。

    同一 GTCRN 骨干、同样的输入特征（P 个波束输出），输出 1 个通道，sigmoid 得到实值语音掩码 m(t, f)。
    噪声（含干扰）协方差按整句平均：Phi_v(f) = sum_t (1 - m) y y^H / sum_t (1 - m)；
    对角加载：Phi_v + lam * tr(Phi_v)/M * I；MVDR：h = Phi^-1 d / (d^H Phi^-1 d)，d 为标称导向向量。
    训练与 BeamFusion 相同（STFT 域对 x_ref 的 MSE，端到端穿过 MVDR）；lam 在验证集上调。
    注意：整句平均使该基线是非因果（离线）的。
    """

    def __init__(self, n_beams, steering: np.ndarray, width=32, hidden=32, intra_bidirectional=True,
                 load=1e-3):
        super().__init__()
        self.backbone = GTCRNBackbone(3 * n_beams, 1, width, hidden, intra_bidirectional)
        self.register_buffer("d", torch.as_tensor(steering).to(torch.complex64))  # [F, M]
        self.load = load

    def mask(self, Z):
        return torch.sigmoid(self.backbone(beam_features(Z)))[:, 0].transpose(-1, -2)  # [B, F, T]

    def filters(self, Y, m, load=None):
        """Y: [B, M, F, T]，m: [B, F, T] -> h: [B, F, M]"""
        load = self.load if load is None else load
        w = (1.0 - m)  # 噪声掩码
        Yf = Y.permute(0, 2, 1, 3)  # [B, F, M, T]
        Phi = torch.einsum("bfmt,bfnt->bfmn", Yf * w[:, :, None, :], Yf.conj()) / (w.sum(-1)[..., None, None] + 1e-8)
        M = Phi.shape[-1]
        tr = torch.diagonal(Phi, dim1=-2, dim2=-1).real.sum(-1)  # [B, F]
        eye = torch.eye(M, dtype=Phi.dtype, device=Phi.device)
        Phi = Phi + (load * tr / M + 1e-10)[..., None, None] * eye
        # 8x8 求解用 complex128（低频/强干扰时 Phi 病态）；h = x / (d^H x) 使 h^H d = 1 精确成立
        Phi = Phi.to(torch.complex128)
        d = self.d.to(torch.complex128)[None].expand(Phi.shape[0], -1, -1)  # [B, F, M]
        x = torch.linalg.solve(Phi, d[..., None])[..., 0]  # Phi^-1 d
        denom = torch.einsum("bfm,bfm->bf", d.conj(), x)
        return (x / denom[..., None]).to(torch.complex64)

    def forward(self, Z, Y, load=None):
        m = self.mask(Z)
        h = self.filters(Y, m, load)
        S = torch.einsum("bfm,bmft->bft", h.conj(), Y)
        return {"S": S, "mask": m, "h": h}


def build_model(model_cfg, cfg, bank: BeamBank):
    kind = model_cfg["kind"]
    common = dict(width=model_cfg.get("width", 32), hidden=model_cfg.get("hidden", 32),
                  intra_bidirectional=model_cfg.get("intra_bidirectional", True))
    if kind == "beamfusion":
        return BeamFusionNet(bank.n_beams, **common)
    if kind == "mask_mvdr":
        from ..beamformers import steer
        F = stft_freqs(cfg["signal"]["fs"], 512)
        d = steer(F, 0.0, cfg["array"]["n_mics"], cfg["array"]["spacing"], cfg["signal"]["c"])
        return MaskMVDR(bank.n_beams, d, load=model_cfg.get("mvdr_load", 1e-3), **common)
    if kind == "affine":
        F = stft_freqs(cfg["signal"]["fs"], 512)
        d = build_dictionary(F, n_mics=cfg["array"]["n_mics"], spacing=cfg["array"]["spacing"], c=cfg["signal"]["c"])
        G, _ = gram(d["H"], anchor=0)  # complex128
        return AffineFusionNet(bank.n_beams, G, cfg["array"]["n_mics"], proj=model_cfg.get("proj", "none"),
                               w0_db=model_cfg.get("w0_db", 0.0), w0_freeze_epochs=model_cfg.get("w0_freeze_epochs", 0),
                               w0_init_db=model_cfg.get("w0_init_db", 0.0),
                               out_init_scale=model_cfg.get("out_init_scale", 0.01),
                               w0_tau_s=model_cfg.get("w0_tau_s", 0.5),
                               frame_rate=cfg["signal"]["fs"] / HOP, weights=model_cfg.get("weights", "complex"), **common)
    if kind == "samehead":  # convex_samehead / free_complex（用户 2026-11-13）
        return SameHeadFusionNet(bank.n_beams, model_cfg["mode"], out_init_scale=model_cfg.get("out_init_scale", 0.01),
                                 alpha_d0=model_cfg.get("alpha_d0", 0.95), **common)
    raise ValueError(kind)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
