"""在线混合数据集（CLAUDE.md 3.1、3.3、3.5、3.6 节）。

读语音 -> 与预生成的 RIR 做 FFT 卷积（torch，CPU 或 GPU）-> 按 SIR 缩放干扰 -> 加白噪声。

- 目标：静止，阵列像 = s * h_array；中心像 = s * h_center。
- 两套参考信号（CLAUDE.md 3.3，2026-10-30 修改）：
    ref_direct（x_dp，训练目标与主评测参考）= s * h_direct，h_direct 为中心处的直达声 RIR（max_order=0）；
    ref（x_ref，第二套评测参考）= s * h_ref，h_ref 为中心 RIR 在 [直达峰 - 1 ms, 直达峰 + 50 ms] 之外置零。
  两者共用同一时间轴；训练时用哪一个由 cfg['reference']['train_target'] 决定（见 train_target_key）。
- 移动干扰：分段静止 + 交叉淡化。源信号乘以权重 w_k(t)（每段 1 s，相邻段之间 50 ms 线性交叉淡化，
  sum_k w_k(t) = 1），第 k 段与第 k 个位置的 RIR 卷积后叠加（各段的混响尾自然延续到下一段）。
- SNR、SIR 都相对“目标直达声在参考点（阵列中心）的功率”定义：
    噪声方差 sigma^2 = P_td / 10^(SNR/10)（各阵元独立白噪声）；
    干扰：各干扰的直达声在中心处功率相等，总和为 P_td / 10^(SIR/10)。
- 失配（kappa > 0）：位置误差体现在重新仿真的 RIR 中；增益/相位误差以参数形式返回（mm_gain_db、mm_phase_deg），
  由 src.stft.apply_mismatch 在 STFT 域作用于阵列观测（各分量分别施加，保证 shadow filtering 精确）。

两种模式（loader.mix_on）：
  worker：__getitem__ 直接在 CPU 上完成混合，配合 collate_mixed；
  device：__getitem__ 只返回原始语音与 RIR，训练循环里调用 mix_batch(raws, cfg, device) 在 GPU 上混合。
噪声总是用 CPU 生成器产生（保证不同设备上测试集完全一致）。
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .config import derive_seed, resolve_device
from .simulate import direct_array_rir, load_meta, load_scene_rirs, mismatch_params
from .speech import load_index, read_clip

MIXED_KEYS = ("mix", "target", "interf", "noise", "ref", "ref_direct", "center_mix")
TRAIN_TARGETS = {"direct": "ref_direct", "early": "ref"}


def _draw_mismatch(cfg, rng, n_mics: int):
    lo, hi = cfg["train_mismatch"].get("kappa", [0.0, 1.0])
    kappa = float(rng.uniform(lo, hi))
    mc = cfg["mismatch"]
    gain = kappa * mc["gain_db"] * rng.uniform(-1.0, 1.0, n_mics)
    phase = kappa * mc["phase_deg"] * rng.uniform(-1.0, 1.0, n_mics)
    return {"kappa": kappa, "gain_db": gain.astype(np.float32), "phase_deg": phase.astype(np.float32)}


def train_mismatch_params(cfg, index: int, epoch: int, n_mics: int):
    """训练时在线施加的增益 / 相位失配（用户 2026-10-31 决定，第 3、4 批“含失配训练”）。

    每条训练样本、每个 epoch 独立抽取：kappa ~ U[kappa_lo, kappa_hi]（默认 [0, 1]），
    g_m = kappa * gain_db * U[-1, 1]（dB），phi_m = kappa * phase_deg * U[-1, 1]（度），范围同 CLAUDE.md 3.6；
    不加位置失配（不需要重新生成训练 RIR）。由全局种子、样本序号与 epoch 确定性派生，可复现、可续跑。
    未启用时返回 None。
    """
    tm = cfg.get("train_mismatch") or {}
    if not tm.get("enabled", False):
        return None
    return _draw_mismatch(cfg, np.random.default_rng(derive_seed(cfg["seed"], "train_mismatch", index, epoch)), n_mics)


def val_mismatch_enabled(cfg) -> bool:
    """含失配训练时，验证集（学习率调度与最佳检查点选择）也加失配（用户 2026-10-31 决定）。
    仅当 train_mismatch.enabled 且 train_mismatch.val 为真时成立；无失配训练的模型始终用无失配验证集。"""
    tm = cfg.get("train_mismatch") or {}
    return bool(tm.get("enabled", False) and tm.get("val", False))


def val_mismatch_params(cfg, index: int, n_mics: int):
    """含失配的验证集：分布与训练相同（kappa ~ U[kappa]，增益 / 相位，无位置失配），
    但每条样本固定（与 epoch 无关，种子由全局种子与样本序号派生，键 "val_mismatch"），所以各 epoch 的验证损失可比、可复现。"""
    return _draw_mismatch(cfg, np.random.default_rng(derive_seed(cfg["seed"], "val_mismatch", index)), n_mics)


def train_target_key(cfg) -> str:
    """训练目标在批次中的键名。配置缺省时为 "early"（2026-10-30 之前的训练均以 x_ref 为目标，其保存的
    data_config.yaml 没有该字段）；当前 configs/data*.yaml 显式设为 "direct"。"""
    return TRAIN_TARGETS[cfg["reference"].get("train_target", "early")]


def segment_weights(n: int, n_seg: int, seg_len: int, xfade: int, device=None) -> torch.Tensor:
    """[n_seg, n]，分段权重；段边界 b_k = k*seg_len，边界处以 xfade 个样本线性过渡，逐样本和为 1。"""
    t = torch.arange(n, dtype=torch.float64, device=device)
    rises = [torch.ones(n, dtype=torch.float64, device=device)]
    for k in range(1, n_seg):
        b = k * seg_len
        rises.append(((t - (b - xfade / 2)) / xfade).clamp(0, 1))
    rises.append(torch.zeros(n, dtype=torch.float64, device=device))
    return torch.stack([rises[k] - rises[k + 1] for k in range(n_seg)]).float()


def fft_conv_sum(sig: torch.Tensor, rir: torch.Tensor, n_out: int) -> torch.Tensor:
    """sum_k sig[k] * rir[k, c]（线性卷积，保留前 n_out 个样本）。sig: [K, N]，rir: [K, C, L] -> [C, n_out]。"""
    n = sig.shape[-1] + rir.shape[-1] - 1
    nfft = 1 << (n - 1).bit_length()
    S = torch.fft.rfft(sig, nfft)
    R = torch.fft.rfft(rir, nfft)
    Y = torch.einsum("kf,kcf->cf", S, R)
    return torch.fft.irfft(Y, nfft)[..., :n_out]


def conv_source(s: torch.Tensor, rir: torch.Tensor, seg_len: int, xfade: int) -> torch.Tensor:
    """单个声源（K 个位置）经 RIR 后的像，[C, N]。

    K = 1：静止源，整段卷积。K > 1：分段静止 + 交叉淡化；第 k 段的源信号 w_k(t) s(t) 只在
    [k*seg - xfade/2, (k+1)*seg + xfade/2) 内非零，因此逐段用短 FFT 卷积再重叠相加（与整段卷积数学上等价）。
    """
    N = s.shape[-1]
    K, C, L = rir.shape
    if K == 1:
        return fft_conv_sum(s[None], rir, N)
    W = segment_weights(N, K, seg_len, xfade, s.device)
    half = xfade // 2 + 1
    starts = [max(0, k * seg_len - half) for k in range(K)]
    ends = [min(N, (k + 1) * seg_len + half) if k < K - 1 else N for k in range(K)]
    T = max(e - b for b, e in zip(starts, ends))
    seg = torch.zeros(K, T, device=s.device)
    for k, (b, e) in enumerate(zip(starts, ends)):
        seg[k, : e - b] = W[k, b:e] * s[b:e]
    n = T + L - 1
    nfft = 1 << (n - 1).bit_length()
    Y = torch.fft.irfft(torch.fft.rfft(seg, nfft)[:, None] * torch.fft.rfft(rir, nfft), nfft)  # [K, C, nfft]
    out = torch.zeros(C, N + nfft, device=s.device)
    for k, b in enumerate(starts):
        out[:, b:b + nfft] += Y[k]
    return out[:, :N]


def reference_rir(center_rir: torch.Tensor, peak: int, fs: int, pre_ms: float, post_ms: float) -> torch.Tensor:
    h = torch.zeros_like(center_rir)
    lo = max(0, peak - int(round(pre_ms * fs / 1000)))
    hi = min(center_rir.shape[-1], peak + int(round(post_ms * fs / 1000)) + 1)
    h[..., lo:hi] = center_rir[..., lo:hi]
    return h


def mix_scene(raw: dict, cfg: dict, device="cpu", return_components=True) -> dict:
    """由一个场景的原始数据（语音 + RIR + 参数）生成阵列混合信号及各分量。所有输出为 float32。"""
    device = torch.device(device)
    fs = cfg["signal"]["fs"]
    src = cfg["sources"]
    N = raw["s_target"].shape[-1]
    seg_len = int(round(raw.get("segment_s", src["segment_s"]) * fs))
    xfade = int(round(src["crossfade_ms"] * fs / 1000))
    to = lambda a: torch.as_tensor(a, dtype=torch.float32, device=device)  # noqa: E731

    A, Cn, D = to(raw["array"]), to(raw["center"]), to(raw["direct"])
    peaks = raw["direct_peak"]
    M = A.shape[1]

    s0 = to(raw["s_target"])
    s0 = s0 * (src["target_rms"] / s0.pow(2).mean().sqrt().clamp_min(1e-8))
    # 目标：一次卷积得到 [8 阵元, 中心, 参考, 直达] 共 M+3 路
    L = max(A.shape[-1], Cn.shape[-1], D.shape[-1])
    pad = lambda x: torch.nn.functional.pad(x, (0, L - x.shape[-1]))  # noqa: E731
    if "ref_rir" in raw:  # 实测数据：参考 RIR 已在数据准备时按规格构造（麦克风 2、3 对齐平均）
        h_ref = to(raw["ref_rir"])
    else:
        h_ref = reference_rir(Cn[0], int(peaks[0]), fs, cfg["reference"]["pre_ms"], cfg["reference"]["post_ms"])
    R0 = torch.cat([pad(A[0]), pad(Cn[0:1]), pad(h_ref[None]), pad(D[0:1])])[None]
    y0 = conv_source(s0, R0, seg_len, xfade)
    target, center_t, ref, ref_direct = y0[:M], y0[M], y0[M + 1], y0[M + 2]
    p_td = ref_direct.double().pow(2).mean()

    interf = torch.zeros(M, N, device=device)
    center_i = torch.zeros(N, device=device)
    direct_i = torch.zeros(N, device=device)
    n_int = len(raw["s_interf"])
    for si, (a0, a1) in zip(raw["s_interf"], raw["interf_slices"]):
        si = to(si)
        si = si * (src["target_rms"] / si.pow(2).mean().sqrt().clamp_min(1e-8))
        Ri = torch.cat([pad(A[a0:a1]), pad(Cn[a0:a1, None]), pad(D[a0:a1, None])], dim=1)  # [K, M+2, L]
        yi = conv_source(si, Ri, seg_len, xfade)
        p_id = yi[M + 1].double().pow(2).mean()
        a = torch.sqrt(p_td * 10 ** (-raw["sir_db"] / 10) / n_int / p_id).float()
        interf += a * yi[:M]
        center_i += a * yi[M]
        direct_i += a * yi[M + 1]

    sigma = torch.sqrt(p_td * 10 ** (-raw["snr_db"] / 10)).float()
    g = torch.Generator().manual_seed(int(raw["noise_seed"]))
    noise_all = torch.randn(M + 1, N, generator=g).to(device) * sigma
    noise = noise_all[:M]

    out = {
        "mix": target + interf + noise,
        "ref": ref,
        "ref_direct": ref_direct,
        "mm_gain_db": to(raw["mm_gain_db"]),
        "mm_phase_deg": to(raw["mm_phase_deg"]),
        "snr_db": to(raw["snr_db"]),
        "sir_db": to(raw["sir_db"]),
        "t60": to(raw["t60"]),
        "n_interf": torch.tensor(n_int, device=device),
        "index": torch.tensor(raw["index"], device=device),
    }
    if return_components and "array_direct" in raw:
        # 目标直达声在各阵元处的分量（用于目标失真指标：输出 vs x_ref 的直达声部分）
        out["target_direct"] = conv_source(s0, to(raw["array_direct"])[None], seg_len, xfade)
    if return_components:
        out.update({
            "target": target,
            "interf": interf,
            "noise": noise,
            "center_mix": center_t + center_i + noise_all[M],
            "center_target": center_t,
            "center_interf": center_i,
            "center_interf_direct": direct_i,
        })
    return out


class BeamFusionDataset(Dataset):
    """split ∈ cfg['splits']（train / val / test_s1 / test_s2）；kappa > 0 时使用失配 RIR 与失配参数。"""

    def __init__(self, cfg, split, kappa: float = 0.0, mix_on=None, return_components=None, max_items=None,
                 val_mismatch: bool = False):
        """val_mismatch=True：验证集加固定的增益 / 相位失配（val_mismatch_params）；只由 train.py 在含失配训练时打开，
        评测与调参脚本不打开，因此它们使用的验证集不受训练配置影响。"""
        self.cfg, self.split, self.kappa = cfg, split, float(kappa)
        if val_mismatch and (split != "val" or self.kappa != 0):
            raise ValueError("val_mismatch only applies to the val split with kappa = 0")
        self.val_mismatch = val_mismatch
        self.scenes = load_meta(cfg, split)
        if max_items:
            self.scenes = self.scenes[:max_items]
        self.speech = {e["id"]: e for e in load_index(cfg, cfg["splits"][split]["speech"])}
        self.mix_on = mix_on or cfg["loader"]["mix_on"]
        self.return_components = cfg["loader"]["return_components"] if return_components is None else return_components
        self.epoch = 0
        self.is_train = split == "train"

    def set_epoch(self, epoch: int):
        """训练集每个 epoch 换一次噪声实现；验证/测试集噪声固定。"""
        self.epoch = epoch

    def __len__(self):
        return len(self.scenes)

    def _snr_db(self, sc):
        """白噪声 SNR：场景元数据中的值；测试集且 sources.test_snr_override_db 非 null 时用该固定值（自噪声扫描）。
        噪声实现的种子不变，因此不同 SNR 下是同一噪声实现的缩放。"""
        ov = self.cfg["sources"].get("test_snr_override_db")
        if ov is not None and self.split.startswith("test"):
            return float(ov)
        return sc["snr_db"]

    def load_raw(self, i: int) -> dict:
        sc = self.scenes[i]
        r = load_scene_rirs(self.cfg, sc, self.kappa)
        slices, k = [], 1
        for it in sc["interferers"]:
            n = len(it["trajectory_deg"])
            slices.append((k, k + n))
            k += n
        assert k == r["array"].shape[0]
        mm = mismatch_params(self.cfg, sc, self.kappa)
        ep = self.epoch if self.is_train else 0
        tm = None
        if self.is_train and self.kappa == 0:  # 训练时在线增益 / 相位失配（每个 epoch 重新抽取）
            tm = train_mismatch_params(self.cfg, sc["index"], ep, r["array"].shape[1])
        elif self.val_mismatch:  # 含失配训练的验证集（每条样本固定）
            tm = val_mismatch_params(self.cfg, sc["index"], r["array"].shape[1])
        if tm is not None:
            mm = {"gain_db": tm["gain_db"], "phase_deg": tm["phase_deg"]}
        return {
            "index": sc["index"],
            "s_target": read_clip(self.cfg, self.speech[sc["speech"]["target"]]),
            "s_interf": [read_clip(self.cfg, self.speech[s]) for s in sc["speech"]["interferers"]],
            "interf_slices": slices,
            "array": r["array"],
            "center": r["center"],
            "direct": r["direct"],
            "direct_peak": r["direct_peak"],
            "snr_db": self._snr_db(sc),
            "sir_db": sc["sir_db"],
            "t60": sc["t60_target"],
            "mm_gain_db": np.asarray(mm["gain_db"], np.float32),
            "mm_phase_deg": np.asarray(mm["phase_deg"], np.float32),
            "noise_seed": derive_seed(self.cfg["seed"], "noise", self.split, sc["index"], ep),
            **({"array_direct": r["array_direct"] if "array_direct" in r else direct_array_rir(self.cfg, sc, self.kappa)}
               if self.return_components else {}),
            # 实测数据（MIRD，R6）：RIR 文件自带 x_ref 的参考 RIR 与目标直达分量；场景自带分段长度（10/7 s）
            **({"ref_rir": r["ref_rir"]} if "ref_rir" in r else {}),
            **({"segment_s": sc["segment_s"]} if "segment_s" in sc else {}),
        }

    def __getitem__(self, i):
        raw = self.load_raw(i)
        if self.mix_on == "worker":
            with torch.no_grad():
                return mix_scene(raw, self.cfg, "cpu", self.return_components)
        return raw


def collate_mixed(batch: list[dict]) -> dict:
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}


def collate_raw(batch: list[dict]) -> list[dict]:
    return batch


def mix_batch(raws: list[dict], cfg, device, return_components=True) -> dict:
    with torch.no_grad():
        return collate_mixed([mix_scene(r, cfg, device, return_components) for r in raws])


def make_loader(cfg, split, batch_size, shuffle=None, kappa=0.0, num_workers=None, seed=None, **kw):
    """返回 (DataLoader, to_batch)。to_batch(b) 把 DataLoader 的输出变成放在 cfg['device'] 上的混合批次。"""
    ds = BeamFusionDataset(cfg, split, kappa, **kw)
    device = resolve_device(cfg)
    shuffle = (split == "train") if shuffle is None else shuffle
    g = torch.Generator().manual_seed(derive_seed(cfg["seed"], "loader", split) if seed is None else seed)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, generator=g,
                    num_workers=cfg["loader"]["num_workers"] if num_workers is None else num_workers,
                    collate_fn=collate_mixed if ds.mix_on == "worker" else collate_raw,
                    persistent_workers=False, pin_memory=device.type == "cuda")
    if ds.mix_on == "worker":
        def to_batch(b):
            return {k: v.to(device, non_blocking=True) for k, v in b.items()}
    else:
        def to_batch(b):
            return mix_batch(b, cfg, device, ds.return_components)
    return dl, to_batch
