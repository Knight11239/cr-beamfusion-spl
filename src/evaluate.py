"""评测：指标 + shadow filtering（CLAUDE.md 7.1、7.2 节；阶段 3 用于基线）。

所有方法都是“逐时频点的线性（时变）滤波器”：
  - 融合类（BeamFusion、ACC、单个固定波束）：系数为 alpha [B, P, F, T]，S = sum_p alpha_p Z_p；
  - 仿射融合（阶段 4）：Z_D + s sum_p w_p (Z_p - Z_D) 写成同样的形式，复系数 c_0 = 1 - s sum_p w_p，c_p = s w_p；
  - 掩码式 MVDR：系数为每个频点的滤波器 h [B, F, M]，S = h^H y。
系数只由混合信号得到；再用同一组系数分别处理目标、干扰、噪声分量（shadow filtering），
因此输出 SIR / SNR 是精确分解，不需要 BSS-eval 的近似（另外也可报告 BSS-eval SIR 便于与原文对比）。

输入参考：阵列中心虚拟麦克风处的混合信号（center_mix）。Δ 指标 = 输出指标 − 输入指标。
两套参考信号（并列输出，互不覆盖）：
  - x_ref（CLAUDE.md 3.3，默认）：中心处“直达声 + 50 ms 早期反射”，列名 si_sdr_* / snr_* / stoi_* / pesq_*；
  - 直达声参考 x_dp：目标源经中心处直达声 RIR（max_order=0，即 dataset 的 ref_direct）卷积，
    与 BeamFusion 原文的“原始直达声分量”对应，列名带 _dp（si_sdr_dp_*、snr_dp_*、stoi_dp_*、pesq_dp_*）。
shadow SIR / SNR 与参考信号无关，只有一套。

方法写法（--methods）：
  fixed                         字典中每个固定波束各作为一个方法（h_D、null-90 …）
  acc:mu=0.1[:lam=0.9]          ACC
  ckpt:NAME=PATH[:load=1e-3]    训练好的模型（类型从检查点读取；load 只对 mask_mvdr 有效）

用法：
  python -m src.evaluate --data-config configs/data_dev.yaml --split test_s1 --methods fixed acc:mu=0.1 \
      ckpt:beamfusion=runs/bf/best.pt --out results/stage3/eval_test_s1 [--kappa 1.0] [--bss] [--max-items 20]
  调参（验证集）：--split val --methods acc:mu=0.01 acc:mu=0.03 acc:mu=0.1 acc:mu=0.3 --select-by d_si_sdr
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from . import metrics as mt
from .acc import acc_fuse
from .config import load_config, resolve_device
from .dataset import BeamFusionDataset, collate_mixed
from .models.gtcrn_fusion import BeamBank, build_model
from .pipeline import array_stft
from .stft import istft, stft

COMPONENTS = ("mix", "target", "interf", "noise")


# ------------------------------------------------------------------------------------------- methods
class FusionMethod:
    """系数为融合权重 alpha。"""
    coef_kind = "alpha"

    def apply(self, coef, Z, Y):
        return (coef * Z).sum(1)


class FixedBeam(FusionMethod):
    def __init__(self, p, name):
        self.p, self.name = p, name

    def coef(self, Z, Y):
        a = torch.zeros(Z.shape, dtype=torch.float32, device=Z.device)
        a[:, self.p] = 1.0
        return a


class ACCMethod(FusionMethod):
    def __init__(self, mu, lam=0.9):
        self.mu, self.lam = mu, lam
        self.name = f"ACC(mu={mu:g})"

    def coef(self, Z, Y):
        return acc_fuse(Z, self.mu, self.lam)[1]


class AFDMAMethod(FusionMethod):
    """AF-DMA（Zhao et al., IEEE SPL 2025）：每个时频点选 |Z_k|^2 最小的波束，并列时均匀加权（src/baselines2.py）。"""

    def __init__(self, beams=None, name="AF-DMA"):
        self.beams, self.name = beams, name

    def coef(self, Z, Y):
        from .baselines2 import afdma_fuse
        return afdma_fuse(Z, self.beams)


class GSCMPDRMethod(FusionMethod):
    """DMA-GSC-MPDR：DMA 差分子空间内的因果递归最小输出功率复数仿射融合，可选 WNG 投影（src/baselines2.py）。"""

    def __init__(self, cfg, lam, delta, w0_db=None):
        self.lam, self.delta, self.w0_db = float(lam), float(delta), w0_db
        self.n_mics = cfg["array"]["n_mics"]
        self.G = None
        if w0_db is not None:
            from .baselines2 import subspace_gram
            self.G = subspace_gram(dictionary_gram(cfg)[0])
        self.name = f"DMA-GSC-MPDR(lam={lam:g},delta={delta:g},W0={'none' if w0_db is None else f'{w0_db:g}'})"
        self.last_s = None

    def coef(self, Z, Y):
        from .baselines2 import gsc_mpdr_fuse
        c, s = gsc_mpdr_fuse(Z, self.lam, self.delta, self.G, self.w0_db, self.n_mics)
        self.last_s = s
        return c

    def apply(self, coef, Z, Y):
        return (coef * Z.to(coef.dtype)).sum(1).to(Z.dtype)


class PostProjMethod:
    """推理时径向投影的通用包装（事后探索，用户 2026-11-15）：对任意逐波束融合系数 c（sum_p c_p = 1，h = sum_p conj(c_p) h_p，
    第 0 个为 h_D）施加 hard 投影，可只作用于一个频带。h = h_D + B u，u_p = conj(c_p)（p >= 1），n = sqrt(u^H G u + 1e-12)，
    s = min(1, r / n)，c'_p = s c_p，c'_0 = 1 - sum_p c'_p（与 projection.py 相同）。凸组合（实 alpha）同样换算后投影；
    频带外 s = 1。complex128 计算。last_s [B, F, T] 供激活率统计。"""

    def __init__(self, base, cfg, w0_db, band=None):
        from .beamformers import build_dictionary, stft_freqs
        a = cfg["array"]
        F = stft_freqs(cfg["signal"]["fs"], 512)
        H = torch.as_tensor(build_dictionary(F, n_mics=a["n_mics"], spacing=a["spacing"], c=cfg["signal"]["c"])["H"])
        g = H[1:] - H[:1]  # [P, F, M]
        self.G = torch.einsum("pfm,qfm->fpq", g.conj(), g)  # [F, P, P] complex128
        self.r = float(np.sqrt(1.0 / 10 ** (w0_db / 10) - 1.0 / a["n_mics"]))
        self.mask = torch.ones(len(F), dtype=torch.bool) if band is None else \
            torch.as_tensor((F > band[0]) & (F <= band[1]))
        self.base, self.w0_db, self.band = base, w0_db, band
        self.coef_kind = "alpha"
        tag = f"postW0={w0_db:g}" + ("" if band is None else f"@{band[0]:g}-{band[1]:g}Hz")
        self.name = f"{base.name}({tag})"
        self.last_s = None

    @torch.no_grad()
    def coef(self, Z, Y):
        c = self.base.coef(Z, Y).to(torch.complex128)  # [B, P+1, F, T]
        u = c[:, 1:].conj()
        q = torch.einsum("bpft,fpq,bqft->bft", u.conj(), self.G.to(c.device), u).real.clamp_min(0.0)
        s = torch.clamp(self.r / torch.sqrt(q + 1e-12), max=1.0)
        s = torch.where(self.mask.to(c.device)[None, :, None], s, torch.ones_like(s))
        self.last_s = s
        cp = c[:, 1:] * s[:, None]
        return torch.cat([1 - cp.sum(1, keepdim=True), cp], 1)

    def apply(self, coef, Z, Y):
        dt = torch.promote_types(coef.dtype, Z.dtype)
        return (coef.to(dt) * Z.to(dt)).sum(1).to(Z.dtype)


class ModelMethod:
    def __init__(self, name, path, cfg, bank, device, load=None, proj_fp64=True):
        ck = torch.load(path, map_location=device, weights_only=False)
        self.mcfg = ck["model_config"]
        self.kind = self.mcfg["kind"]
        self.model = build_model(self.mcfg, cfg, bank).to(device).eval()
        self.model.load_state_dict(ck["model"])
        if self.kind == "affine":
            self.model.proj_fp64 = proj_fp64  # 推理时以 float64 计算投影（默认开启；False 仅用于复现旧的 float32 结果）
        if hasattr(self.model, "set_epoch"):
            # 可学习 W0 的冻结状态与训练该检查点时相同（ck["epoch"] 为 0 起的 epoch 序号）。build_model 后默认冻结
            # （w0_freeze_epochs > 0 时 W0 固定为 w0_init_db），不调用 set_epoch 会把可学习 W0 当成 0 dB 评测。
            ep = ck.get("epoch")
            if ep is None and getattr(self.model, "learn_w0", False) and self.model.w0_freeze_epochs > 0:
                raise ValueError(f"{path}: 可学习 W0 的检查点缺少 epoch，无法确定 W0 是否已解冻")
            self.model.set_epoch(ep if ep is not None else 10 ** 9)
        self.learn_w0 = bool(getattr(self.model, "learn_w0", False))
        self.last_w0 = None  # 可学习 W0：最近一次 coef() 的 W0_dB [B, F, T]，供评测统计分布
        self.load = load
        self.name = name + (f"(load={load:g})" if (load is not None and self.kind == "mask_mvdr") else "")
        self.coef_kind = "h" if self.kind == "mask_mvdr" else "alpha"  # affine 的复系数也按 "alpha" 处理
        self.ckpt_epoch = ck.get("epoch")

    @torch.no_grad()
    def coef(self, Z, Y):
        if self.kind == "mask_mvdr":
            return self.model(Z, Y, load=self.load)["h"]
        o = self.model(Z)
        if self.learn_w0:
            self.last_w0 = o["w0_db"]
        if self.kind == "affine":  # 等价的逐波束复系数：c_0 = 1 - s sum w，c_p = s w_p（sum_p c_p = 1）
            sw = o["s"][:, None] * o["w"]  # proj_fp64 时为 complex128
            return torch.cat([1 - sw.sum(1, keepdim=True), sw], 1)
        if "coef" in o:  # samehead（convex_samehead / free_complex）：Ẑ = Σ_p coef_p Z_p
            return o["coef"]
        return o["alpha"]

    def apply(self, coef, Z, Y):
        if self.coef_kind == "h":
            return torch.einsum("bfm,bmft->bft", coef.conj(), Y)
        # 在 coef 与 Z 的公共类型中相乘（complex128 系数 -> complex128；凸组合的实值 alpha 与 complex64 的 Z -> complex64），
        # 输出转回 Z 的类型。不能用 Z.to(coef.dtype)：实值 alpha 会把 Z 转成实数、丢掉虚部（ec07666 引入的错误）。
        dt = torch.promote_types(coef.dtype, Z.dtype)
        return (coef.to(dt) * Z.to(dt)).sum(1).to(Z.dtype)


# ------------------------------------------------------------------------------------------- WNG（CLAUDE.md 7.2）
# dB；箱宽 0.25 dB，边界整体下移 0.005 dB，使恰好落在投影边界上的值（如 W0 - 1e-6 dB）与 W0 在同一箱；超出范围的值计入两端的箱
WNG_EDGES = np.arange(-80.0, 12.0001, 0.25) - 0.005
WNG_THRESH = (-20.0, -10.0, -5.0, 0.0)  # 统计“低于阈值”的比例（精确计数，容差 1e-3 dB）


def dictionary_gram(cfg):
    """K_pq(f) = h_p^H h_q（含 h_D），complex128，[F, P, P]；以及标称导向向量 d [F, M]。"""
    from .beamformers import build_dictionary, steer, stft_freqs
    F = stft_freqs(cfg["signal"]["fs"], 512)
    a = cfg["array"]
    H = build_dictionary(F, n_mics=a["n_mics"], spacing=a["spacing"], c=cfg["signal"]["c"])["H"]
    K = np.einsum("pfm,qfm->fpq", H.conj(), H)
    return torch.as_tensor(K), torch.as_tensor(steer(F, 0.0, a["n_mics"], a["spacing"], cfg["signal"]["c"]))


def coef_wng_db(m, coef, K, d):
    """每个时频点的实测 WNG(h) = |h^H d|^2 / h^H h（dB，标称导向向量），[B, F, T]（MVDR 为 [B, F, 1]）。

    融合类：h = sum_p conj(c_p) h_p，h^H d = sum_p c_p（各成员无失真），h^H h = v^H K v，v = conj(c)；
    MVDR：直接由 h [B, F, M] 计算。在 complex128 / float64 中计算。
    """
    if m.coef_kind == "h":
        h = coef.to(torch.complex128)
        num = torch.einsum("bfm,fm->bf", h.conj(), d.to(h.device)).abs() ** 2
        return (10 * torch.log10(num / (h.abs() ** 2).sum(-1)))[..., None]
    c = coef.to(torch.complex128)
    v = c.conj()
    q = torch.einsum("bpft,fpq,bqft->bft", v.conj(), K.to(c.device), v).real
    num = c.sum(1).abs() ** 2
    return 10 * torch.log10(num / q)


def accumulate_wng(stats, key, wdb, T):
    """wdb: [B, F, T'] (T' = T 或 1)；跳过 f = 0。累计总直方图与逐频点直方图（MVDR 每帧权重相同，乘以 T）。"""
    w = wdb[:, 1:].double().cpu().numpy()
    rep = T if w.shape[-1] == 1 else 1
    st = stats.setdefault(key, {"freq_hist": np.zeros((w.shape[1], len(WNG_EDGES) - 1)), "min": np.inf, "max": -np.inf,
                                "n": 0, "below": {t: 0 for t in WNG_THRESH}})
    for t in WNG_THRESH:
        st["below"][t] += int((w < t - 1e-3).sum()) * rep
    idx = np.clip(np.searchsorted(WNG_EDGES, w, side="right") - 1, 0, len(WNG_EDGES) - 2)  # [B, F-1, T']
    for f in range(w.shape[1]):
        st["freq_hist"][f] += np.bincount(idx[:, f].ravel(), minlength=len(WNG_EDGES) - 1) * rep
    st["min"], st["max"] = min(st["min"], float(w.min())), max(st["max"], float(w.max()))
    st["n"] += w.size * rep


def write_wng_stats(stats, prefix, fs=16000, n_fft=512):
    """<prefix>_wng_hist.csv（总直方图）、<prefix>_wng_freq.csv（逐频点分位数，用于箱线图）、<prefix>_wng_summary.json。"""
    names = list(stats)
    for n in names:
        stats[n]["hist"] = stats[n]["freq_hist"].sum(0)
    lo = WNG_EDGES[:-1] + 0.005  # 箱的名义下界（-80, -79.75, ...）
    with open(f"{prefix}_wng_hist.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["wng_db_bin_lo", "wng_db_bin_hi"] + names)
        for i in range(len(lo)):
            w.writerow([f"{lo[i]:g}", f"{lo[i] + 0.25:g}"] + [int(stats[n]["hist"][i]) for n in names])

    def pct(h, qs):
        """分位数所在箱的名义下界（保守值：真实分位数 >= 它 - 0.005 dB），分辨率 0.25 dB。"""
        c = np.cumsum(h) / max(h.sum(), 1)
        return [float(lo[min(np.searchsorted(c, q), len(lo) - 1)]) for q in qs]

    qs = (0.001, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)
    with open(f"{prefix}_wng_freq.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "freq_Hz"] + [f"p{100 * q:g}" for q in qs])
        for n in names:
            for k, h in enumerate(stats[n]["freq_hist"]):
                w.writerow([n, f"{(k + 1) * fs / n_fft:g}"] + [f"{v:.2f}" for v in pct(h, qs)])
    summ = {}
    for n in names:
        h = stats[n]["hist"]
        tot = h.sum()
        summ[n] = {"n_tf": int(stats[n]["n"]), "min_db": stats[n]["min"], "max_db": stats[n]["max"],
                   **{f"p{100 * q:g}_db": v for q, v in zip(qs, pct(h, qs))},
                   **{f"frac_below_{t:g}db": stats[n]["below"][t] / stats[n]["n"] for t in WNG_THRESH}}
    Path(f"{prefix}_wng_summary.json").write_text(json.dumps(summ, indent=1))
    return summ


W0_LO, W0_HI = -15.0, 9.0  # 可学习 W0 的范围（CLAUDE.md 6.3）
W0_EDGES = np.arange(W0_LO, W0_HI + 0.0001, 0.1) - 0.005  # 0.1 dB 箱；恰好等于下界的值落在第一个箱
W0_AT_LO = (0.01, 0.1, 1.0)  # “处于下界”：W0 <= -15 + eps（dB）；sigmoid 饱和时 float32 中的值即为 -15


def accumulate_w0(stats, key, w0db):
    """可学习 W0 的分布：w0db [B, F, T]，跳过 f = 0（该频点直接用 h_D，不投影）。逐频点直方图、处于下界的计数。"""
    w = w0db[:, 1:].double().cpu().numpy()
    st = stats.setdefault(key, {"freq_hist": np.zeros((w.shape[1], len(W0_EDGES) - 1)),
                                "freq_at_lo": {e: np.zeros(w.shape[1]) for e in W0_AT_LO},
                                "freq_n": np.zeros(w.shape[1]), "sum": 0.0, "min": np.inf, "max": -np.inf})
    idx = np.clip(np.searchsorted(W0_EDGES, w, side="right") - 1, 0, len(W0_EDGES) - 2)
    for f in range(w.shape[1]):
        st["freq_hist"][f] += np.bincount(idx[:, f].ravel(), minlength=len(W0_EDGES) - 1)
    for e in W0_AT_LO:
        st["freq_at_lo"][e] += (w <= W0_LO + e).sum(axis=(0, 2))
    st["freq_n"] += w.shape[0] * w.shape[2]
    st["sum"] += float(w.sum())
    st["min"], st["max"] = min(st["min"], float(w.min())), max(st["max"], float(w.max()))


def write_w0_stats(stats, prefix, fs=16000, n_fft=512):
    """<prefix>_w0_freq.csv（逐频点分位数与处于下界的比例）、<prefix>_w0_summary.json（总体）。分位数为所在 0.1 dB 箱的
    名义下界。"""
    lo = W0_EDGES[:-1] + 0.005
    qs = (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)

    def pct(h):
        c = np.cumsum(h) / max(h.sum(), 1)
        return [float(lo[min(np.searchsorted(c, q), len(lo) - 1)]) for q in qs]

    with open(f"{prefix}_w0_freq.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "freq_Hz", "n"] + [f"p{100 * q:g}" for q in qs] + [f"frac_at_lo_{e:g}db" for e in W0_AT_LO])
        for n, st in stats.items():
            for k, h in enumerate(st["freq_hist"]):
                w.writerow([n, f"{(k + 1) * fs / n_fft:g}", int(st["freq_n"][k])] + [f"{v:.2f}" for v in pct(h)] +
                           [f"{st['freq_at_lo'][e][k] / st['freq_n'][k]:.6f}" for e in W0_AT_LO])
    summ = {}
    for n, st in stats.items():
        N = st["freq_n"].sum()
        summ[n] = {"n_tf": int(N), "mean_db": st["sum"] / N, "min_db": st["min"], "max_db": st["max"],
                   **{f"p{100 * q:g}_db": v for q, v in zip(qs, pct(st["freq_hist"].sum(0)))},
                   **{f"frac_at_lo_{e:g}db": float(st["freq_at_lo"][e].sum() / N) for e in W0_AT_LO}}
    Path(f"{prefix}_w0_summary.json").write_text(json.dumps(summ, indent=1))
    return summ


def _sha256(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def write_provenance(out, cfg, data_config, specs, splits):
    """provenance.json：数据配置、各检查点、场景元数据与所用 RIR 文件的 sha256（事后探索第二轮补充的可追溯记录）。"""
    from .config import meta_path, rir_dir
    from .simulate import load_meta
    prov = {"data_config": {"path": data_config, "sha256": _sha256(data_config)}, "ckpt": {}, "meta": {}, "rir": {}}
    for sp in specs:
        if sp.startswith("ckpt:"):
            path = sp[5:].split(":")[0].split("=", 1)[1]
            prov["ckpt"][path] = _sha256(path)
    for split in splits:
        prov["meta"][split] = {"path": str(meta_path(cfg, split)), "sha256": _sha256(meta_path(cfg, split))}
        rd = rir_dir(cfg, split, 0.0)
        prov["rir"][split] = {f: _sha256(rd / f) for f in sorted({sc["rir_file"] for sc in load_meta(cfg, split)})}
    Path(out).mkdir(parents=True, exist_ok=True)
    (Path(out) / "provenance.json").write_text(json.dumps(prov, indent=1))
    return prov


def parse_methods(specs, cfg, bank, device):
    out = []
    for s in specs:
        head, _, rest = s.partition(":")
        if head == "fixed":
            out += [FixedBeam(p, n) for p, n in enumerate(bank.names)]
        elif head == "acc":
            kv = dict(x.split("=") for x in rest.split(":") if x)
            out.append(ACCMethod(float(kv["mu"]), float(kv.get("lam", 0.9))))
        elif head == "afdma":  # afdma 或 afdma:beams=0,1,2,4（原文的 4 波束字典：h_D + null-90/120/180）
            kv = dict(x.split("=") for x in rest.split(":") if x)
            if "beams" in kv:
                b = [int(v) for v in kv["beams"].split(",")]
                out.append(AFDMAMethod(b, f"AF-DMA({len(b)})"))
            else:
                out.append(AFDMAMethod(None, f"AF-DMA({len(bank.names)})"))
        elif head == "mpdr":  # mpdr:lam=0.95:delta=1e-3[:w0=-5][:sel=mmval]
            kv = dict(x.split("=") for x in rest.split(":") if x)
            w0 = kv.get("w0")
            m = GSCMPDRMethod(cfg, float(kv["lam"]), float(kv["delta"]), None if w0 in (None, "none") else float(w0))
            if kv.get("sel"):  # 选参方式标签（如 mmval = 含失配验证集选参），只改名称，便于与按无失配验证集选参的版本并列
                m.name = m.name.replace("DMA-GSC-MPDR(", f"DMA-GSC-MPDR-{kv['sel']}(")
            out.append(m)
        elif head == "ckpt":
            parts = rest.split(":")
            name, path = parts[0].split("=", 1)
            kv = dict(x.split("=") for x in parts[1:] if x)
            mm = ModelMethod(name, path, cfg, bank, device, float(kv["load"]) if "load" in kv else None,
                             proj_fp64=kv.get("proj", "fp64") != "fp32")  # ckpt:N=P:proj=fp32 复现旧的 float32 投影
            if "proj_w0" in kv:  # 推理时径向投影（事后探索，用户 2026-11-14）：对无投影训练的仿射模型施加固定 W0 的 hard 投影，不重新训练
                if mm.kind != "affine" or getattr(mm.model, "learn_w0", False):
                    raise ValueError("proj_w0 only applies to affine models with a fixed W0")
                mm.model.proj, mm.model.w0_db = "hard", float(kv["proj_w0"])
                mm.name = f"{name}(projW0={float(kv['proj_w0']):g})"
            if "post_w0" in kv:  # ckpt:N=P:post_w0=-5[:post_band=0-1000]：通用推理时投影（凸 / 仿射均可，可限频带；事后探索 2026-11-15）
                if mm.coef_kind != "alpha" or getattr(mm.model, "learn_w0", False) or getattr(mm.model, "proj", "none") != "none":
                    raise ValueError("post_w0 requires an unprojected fusion model")
                band = tuple(float(x) for x in kv["post_band"].split("-")) if "post_band" in kv else None
                mm = PostProjMethod(mm, cfg, float(kv["post_w0"]), band)
            mm.ckpt_sha256 = _sha256(path)
            out.append(mm)
        else:
            raise ValueError(s)
    return out


# ------------------------------------------------------------------------------------------- metrics
def _in_metrics(args):
    """输入（中心麦克风混合信号）的指标：每个样本只算一次，所有方法共用。"""
    inp, ref, ref_dp, icomp, do_bss, do_pq = args
    r = {"sir_shadow_in": mt.power_ratio_db(icomp["target"], icomp["interf"]),
         "snr_noise_shadow_in": mt.power_ratio_db(icomp["target"], icomp["noise"])}
    r.update(_ref_metrics(inp, ref, "", "in", do_pq))
    r.update(_ref_metrics(inp, ref_dp, "_dp", "in", do_pq))
    if do_bss:  # do_bss = True / "ref"：以 x_ref 为目标（列 bss_sir_*）；"dp"：以 x_dp 为目标（列 bss_sir_dp_*）
        key = "bss_sir_dp" if do_bss == "dp" else "bss_sir"
        r[f"{key}_in"] = mt.bss_sir(inp, ref_dp if do_bss == "dp" else ref, icomp["interf"])
    return r


def _out_metrics(args):
    """单个样本、单个方法的输出指标（在进程池中运行）。"""
    out, comp, ref, ref_dp, interf_ref, do_bss, do_pq = args
    r = {"sir_shadow_out": mt.power_ratio_db(comp["target"], comp["interf"]),
         "snr_noise_shadow_out": mt.power_ratio_db(comp["target"], comp["noise"]),
         "target_sdr_out": mt.snr(comp["target"], ref),  # 只看目标分量：与 x_ref 的偏差（失真 + 晚期混响）
         "target_sdr_dp_out": mt.snr(comp["target"], ref_dp)}  # 同上，相对直达声参考（含全部反射）
    r.update(_ref_metrics(out, ref, "", "out", do_pq))
    r.update(_ref_metrics(out, ref_dp, "_dp", "out", do_pq))
    if do_bss:
        key = "bss_sir_dp" if do_bss == "dp" else "bss_sir"
        r[f"{key}_out"] = mt.bss_sir(out, ref_dp if do_bss == "dp" else ref, interf_ref)
    return r


def _ref_metrics(x, ref, tag, io, do_pq):
    """依赖参考信号的指标；tag="" 为 x_ref，tag="_dp" 为直达声参考。"""
    r = {f"si_sdr{tag}_{io}": mt.si_sdr(x, ref), f"snr{tag}_{io}": mt.snr(x, ref)}
    if do_pq:
        r[f"stoi{tag}_{io}"], r[f"pesq{tag}_{io}"] = mt.stoi(x, ref), mt.pesq_wb(x, ref)
    return r


def _deltas(r):
    for k in ("si_sdr", "snr", "sir_shadow", "snr_noise_shadow", "stoi", "pesq", "bss_sir", "bss_sir_dp",
              "si_sdr_dp", "snr_dp", "stoi_dp", "pesq_dp"):
        if f"{k}_in" in r and f"{k}_out" in r:
            r[f"d_{k}"] = r[f"{k}_out"] - r[f"{k}_in"]
    return r


def evaluate(cfg, split, methods, bank, device, kappa=0.0, max_items=None, batch_size=4, num_workers=2,
             metric_workers=4, do_bss=False, do_pq=True, log=print, freq_stats: dict | None = None,
             wng_stats: dict | None = None, val_mismatch: bool = False, w0_stats: dict | None = None):
    """返回逐样本、逐方法的指标行。freq_stats（可选）会被填入目标直达声失真的逐频点累积量：
    {(method, t60): [sum_t |S_d - X_d|^2 按频点求和, sum_t |X_d|^2 按频点求和]}（各样本求和，即能量加权）。
    wng_stats（可选）：{method: 逐频点 WNG 直方图等}，统计所有样本、所有时频点（f > 0）的实测 WNG（7.2 节）。"""
    K, d_nom = dictionary_gram(cfg) if wng_stats is not None else (None, None)
    # val_mismatch：含失配的验证集（增益 / 相位，kappa ~ U[train_mismatch.kappa]，每条样本固定种子），用于“失配验证集选加载”的 MVDR
    ds = BeamFusionDataset(cfg, split, kappa=kappa, mix_on="worker", return_components=True, max_items=max_items,
                           val_mismatch=val_mismatch)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_mixed)
    by_index = {s["index"]: s for s in ds.scenes}
    rows = []
    pool = ProcessPoolExecutor(metric_workers) if metric_workers > 1 else None
    t0 = time.time()
    for bi, b in enumerate(dl):
        b = {k: v.to(device) for k, v in b.items()}
        N = b["mix"].shape[-1]
        comps = COMPONENTS + (("target_direct",) if "target_direct" in b else ())
        Ys = {c: array_stft(b[c], b["mm_gain_db"], b["mm_phase_deg"]) for c in comps}
        Zs = {c: bank(Y) for c, Y in Ys.items()}
        inp = {"target": b["center_target"], "interf": b["center_interf"],
               "noise": b["center_mix"] - b["center_target"] - b["center_interf"]}
        B = b["mix"].shape[0]
        ref = b["ref"].cpu().numpy()
        ref_dp = b["ref_direct"].cpu().numpy()
        icomp = {c: v.cpu().numpy() for c, v in inp.items()}
        cmix = b["center_mix"].cpu().numpy()
        in_jobs = [(cmix[i], ref[i], ref_dp[i], {c: v[i] for c, v in icomp.items()}, do_bss, do_pq)
                   for i in range(B)]
        jobs, meta = [], []
        for m in methods:
            coef = m.coef(Zs["mix"], Ys["mix"])
            if wng_stats is not None:
                accumulate_wng(wng_stats, m.name, coef_wng_db(m, coef, K, d_nom), Zs["mix"].shape[-1])
            if w0_stats is not None and getattr(m, "learn_w0", False):
                accumulate_w0(w0_stats, m.name, m.last_w0)
            outs = {c: istft(m.apply(coef, Zs[c], Ys[c]), N).cpu().numpy() for c in COMPONENTS}
            td_sdr = [None] * B
            if "target_direct" in Zs:  # 目标失真：只含目标直达声分量时，输出与 x_ref 直达声部分的误差
                Sd = m.apply(coef, Zs["target_direct"], Ys["target_direct"])
                Xd = stft(b["ref_direct"])
                rd = b["ref_direct"].cpu().numpy()
                od = istft(Sd, N).cpu().numpy()
                td_sdr = [mt.snr(od[i], rd[i]) for i in range(B)]
                if freq_stats is not None:
                    e = (Sd - Xd).abs().pow(2).sum(-1).double().cpu().numpy()  # [B, F]
                    x = Xd.abs().pow(2).sum(-1).double().cpu().numpy()
                    for i in range(B):
                        t60 = by_index[int(b["index"][i])]["t60_target"]
                        for key in ((m.name, t60), (m.name, "all")):
                            acc = freq_stats.setdefault(key, [np.zeros(e.shape[1]), np.zeros(e.shape[1])])
                            acc[0] += e[i]
                            acc[1] += x[i]
            for i in range(B):
                sc = by_index[int(b["index"][i])]
                jobs.append((outs["mix"][i], {c: outs[c][i] for c in ("target", "interf", "noise")}, ref[i],
                             ref_dp[i], icomp["interf"][i], do_bss, do_pq))
                meta.append({"split": split, "kappa": kappa, "id": sc["id"], "scenario": sc["scenario"],
                             "t60": sc["t60_target"], "n_interf": len(sc["interferers"]),
                             "moving": any(it["moving"] for it in sc["interferers"]),
                             "snr_set": ds._snr_db(sc), "sir_set": sc["sir_db"], "method": m.name, "_i": i,
                             "target_direct_sdr_out": td_sdr[i],
                             "shadow_consistency_err": float(
                                 np.abs(outs["mix"][i] - outs["target"][i] - outs["interf"][i] - outs["noise"][i]).max()
                                 / (np.abs(outs["mix"][i]).max() + 1e-12))})
        mp = pool.map if pool else map
        res_in = list(mp(_in_metrics, in_jobs))
        res = mp(_out_metrics, jobs)
        for mrow, r in zip(meta, res):
            i = mrow.pop("_i")
            rows.append(_deltas({**mrow, **res_in[i], **r}))
        if (bi + 1) % 5 == 0:
            log(f"  [{split} kappa={kappa}] {min((bi + 1) * batch_size, len(ds))}/{len(ds)} "
                f"{(time.time() - t0) / 60:.1f} min")
    if pool:
        pool.shutdown()
    return rows


SUMMARY_KEYS = ("d_si_sdr", "d_snr", "si_sdr_out", "stoi_out", "pesq_out", "d_stoi", "d_pesq", "sir_shadow_out",
                "d_sir_shadow", "snr_noise_shadow_out", "target_sdr_out", "target_direct_sdr_out", "bss_sir_out", "bss_sir_dp_out", "d_bss_sir_dp",
                "d_si_sdr_dp", "d_snr_dp", "si_sdr_dp_out", "stoi_dp_out", "pesq_dp_out", "d_stoi_dp", "d_pesq_dp",
                "target_sdr_dp_out")


def summarize(rows, by=("method", "t60")):
    groups = {}
    for r in rows:
        groups.setdefault(tuple(r[k] for k in by), []).append(r)
    out = []
    for key, rs in groups.items():
        s = dict(zip(by, key))
        s["n"] = len(rs)
        for k in SUMMARY_KEYS:
            if k in rs[0]:
                s[k], s[k + "_ci95"] = mt.ci95([r[k] for r in rs])
        out.append(s)
    return out


def write_freq_stats(freq_stats, path, fs=16000, n_fft=512):
    """逐频点目标失真（dB）：10 log10(sum |S_d - X_d|^2 / sum |X_d|^2)，按 (方法, T60) 列出。"""
    keys = sorted(freq_stats, key=lambda k: (k[0], str(k[1])))
    F = len(next(iter(freq_stats.values()))[0]) if freq_stats else 0
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["freq_Hz"] + [f"{m}|T60={t}" for m, t in keys])
        for k in range(F):
            w.writerow([f"{k * fs / n_fft:g}"] + [
                f"{10 * np.log10((freq_stats[key][0][k] + 1e-30) / (freq_stats[key][1][k] + 1e-30)):.3f}"
                for key in keys])


def write_csv(rows, path):
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-config", default="configs/data_dev.yaml")
    ap.add_argument("--split", default="test_s1")
    ap.add_argument("--kappa", type=float, default=0.0)
    ap.add_argument("--methods", nargs="+", default=["fixed"])
    ap.add_argument("--out", required=True, help="输出前缀：<out>_samples.csv / <out>_summary.csv")
    ap.add_argument("--max-items", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--device", default=None)
    ap.add_argument("--metric-workers", type=int, default=4)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--bss", action="store_true", help="同时计算 BSS-eval SIR（较慢）")
    ap.add_argument("--no-pq", action="store_true", help="跳过 STOI / PESQ（调参时加速）")
    ap.add_argument("--select-by", default=None, help="在各方法中按该指标均值选最优，写入 <out>_selection.json")
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args(argv)
    cfg = load_config(a.data_config, a.set)
    device = resolve_device(a.device or cfg["device"])
    bank = BeamBank.from_config(cfg).to(device)
    methods = parse_methods(a.methods, cfg, bank, device)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    fstats = {}
    rows = evaluate(cfg, a.split, methods, bank, device, a.kappa, a.max_items, a.batch_size, a.num_workers,
                    a.metric_workers, a.bss, not a.no_pq, freq_stats=fstats)
    write_csv(rows, f"{out}_samples.csv")
    write_freq_stats(fstats, f"{out}_target_distortion_freq.csv")
    summ = summarize(rows, ("method", "t60")) + summarize(rows, ("method",))
    write_csv(summ, f"{out}_summary.csv")
    print(f"evaluated {len(rows)} rows in {(time.time() - t0) / 60:.1f} min")
    for s in summarize(rows, ("method",)):
        print(f"  {s['method']:<28} dSI-SDR {s['d_si_sdr']:6.2f}±{s['d_si_sdr_ci95']:.2f}  "
              f"dSI-SDR(direct) {s['d_si_sdr_dp']:6.2f}  dSNR {s['d_snr']:6.2f}  SIR_shadow {s['sir_shadow_out']:6.2f}"
              + (f"  STOI {s['stoi_out']:.3f}  PESQ {s['pesq_out']:.2f}" if "stoi_out" in s else ""))
    if a.select_by:
        summ_m = summarize(rows, ("method",))
        best = max(summ_m, key=lambda s: s[a.select_by])
        sel = {"split": a.split, "criterion": a.select_by, "best_method": best["method"],
               "candidates": {s["method"]: s[a.select_by] for s in summ_m}}
        with open(f"{out}_selection.json", "w") as f:
            json.dump(sel, f, indent=1)
        print("selected:", sel)


if __name__ == "__main__":
    main()
