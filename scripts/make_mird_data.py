"""R6：MIRD 实测测试集（规格见 CLAUDE.md §10.2；用户 2026-11-06 定稿）。只用 CPU。

  python scripts/make_mird_data.py --mird <MIRD v2 目录> --config configs/data_mird.yaml [--check-only] [--n-check 20]

输出（cfg paths.data_root 下，与仿真数据同一格式，评测代码不变）：
  meta/mird_s1.json、meta/mird_s2.json；rirs/mird_s1/*.npz、rirs/mird_s2/*.npz；reports/mird_ref_check.json（参考信号核验）。

处理（每个声源位置、每个子阵；48 kHz 原始 RIR）：
  1. 扣除系统延迟 int_delay = 624 样本；
  2. 直达声窗口：R3 的锥形窗（scripts/mird_r2r3.direct_window，窗长不含第一次反射），子阵 4 个麦克风公共一个窗口，阵列直达分量与 x_dp 共用；x_ref 窗口：各麦克风直达峰前 1 ms 到后 50 ms（矩形，同仿真）；
  3. 多相重采样到 16 kHz（scipy resample_poly，零相位）；
  4. 子阵麦克风 2、3（m = 1、2，子阵中心两侧）按标称几何（近场点声源，R2）的时延 τ_m = (|s − p_m| − |s − c|)/c 做分数时延对齐到子阵中心
     （频域相移），取平均：直达声窗口 → 中心“直达声” RIR（x_dp 的 RIR），x_ref 窗口 → x_ref 的 RIR，完整 RIR → 中心虚拟麦克风 RIR
     （输入混合信号的参考点）。
  5. 阵列输入：子阵 4 个麦克风的完整 RIR（m = 0 离目标最近：A 为 mic1–4，B 为 mic5–8）；目标直达声分量用各麦克风的直达声窗口。
场景（各 600 条 = 3 个 RT60 × 2 个子阵 × 100）：目标 MIRD +90°、2 m（θ = 0）。
  S1-实测：干扰 2 m，θ = 90°, 105°, …, 180°（7 个位置，每段 10/7 s，50 ms 交叉淡化）；SIR、SNR 同仿真。
  S2-实测：2–3 个干扰，15° 网格上 θ ∈ [30°, 180°]，至少一个静止干扰 ∈ [30°, 90°]，角度互不相同；一半样本有一个干扰按每段 15°
     移动（7 段，每段 10/7 s，方向随机，轨迹在 [30°, 180°] 内）；每个干扰的距离在 1 m / 2 m 中随机；SIR、SNR 同仿真。
  θ 为标称角度（相对阵列中心）；meta 中另记录按 R2 几何换算到子阵中心的实际角度与距离（报告用）。
"""
from __future__ import annotations

import argparse
import json
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import mird_r2r3 as R  # noqa: E402
from src.config import derive_seed, load_config  # noqa: E402

FS48, FS16 = 48000, 16000
L16 = 16000  # 1 s（RT60 ≤ 0.61 s）
RTS = ("0.160", "0.360", "0.610")
SUBS = ("A", "B")
SEG_S = 10.0 / 7.0
GRID = [float(a) for a in range(30, 181, 15)]  # θ 网格（标称，相对阵列中心）


def theta_to_mird(theta):
    """θ（本项目约定，e_x = u，目标 θ = 0 ↔ MIRD +90°）→ MIRD 方位文件编号。"""
    phi = 90.0 - theta  # [-90, 90]
    return int(round(phi)) % 360


@lru_cache(maxsize=None)
def load48(mird, rt, dist, ang):
    from scipy.io import loadmat
    pat = f"*Reverberation_{rt}s)_3-3-3-8-3-3-3/*Reverberation_{rt}s)_3-3-3-8-3-3-3_{dist}m_{ang:03d}.mat"
    fs = sorted(Path(mird).glob(pat))
    assert len(fs) == 1, (pat, fs)
    z = loadmat(fs[0], squeeze_me=True, struct_as_record=False)
    mp, sp = z["metapar"], z["simpar"]
    assert list(mp.mic_spacing) == [3, 3, 3, 8, 3, 3, 3] and abs(mp.reverberation - float(rt)) < 1e-9
    assert int(mp.distance) == dist and int(mp.azimuth) == ang and int(sp.fs) == FS48 and int(sp.int_delay) == R.LAT
    h = z["impulse_response"].T.astype(np.float64)  # [8, N]
    return h[:, R.LAT:R.LAT + int(1.25 * FS48)]


def frac_shift(x, tau):
    """x(t + tau)（tau 秒，分数时延，频域相移，零填充一倍长度）。"""
    n = x.shape[-1]
    nf = 2 * n
    X = np.fft.rfft(x, nf)
    f = np.fft.rfftfreq(nf, 1 / FS16)
    return np.fft.irfft(X * np.exp(2j * np.pi * f * tau), nf)[..., :n]


def to16(x):
    return resample_poly(x, 1, 3, axis=-1)[..., :L16]


@lru_cache(maxsize=None)
def position(mird, rt, sub, dist, theta):
    """一个声源位置在一个子阵上的处理结果（16 kHz）。"""
    ang = theta_to_mird(theta)
    h = load48(mird, rt, dist, ang)
    idx = R.SUB[sub]  # m = 0..3（u 方向：m = 0 在 MIRD +90° 一侧）
    # R3 的直达声窗口：子阵 4 个麦克风公共窗口（阵列直达分量），以及麦克风 2、3 的公共窗口（x_dp）
    # 同一个锥形窗（子阵 4 个麦克风的公共窗口，结束于其中最早的首次反射之前）用于阵列直达分量与 x_dp（取其中麦克风 2、3），
    # 使两者包含同样长度的扬声器响应
    w4, info4, _ = R.direct_window(h[:, :6000], idx)
    w23, info23 = w4, info4
    full = to16(h[idx])  # [4, L]
    dir4 = to16(np.pad(h[idx, :6000] * w4, ((0, 0), (0, h.shape[1] - 6000))))
    dir23 = to16(np.pad(h[[idx[1], idx[2]], :6000] * w23, ((0, 0), (0, h.shape[1] - 6000))))
    # x_ref 窗口：各麦克风直达峰前 1 ms 到后 50 ms
    pre, post = int(round(0.001 * FS48)), int(round(0.050 * FS48))
    ref48 = np.zeros((2, h.shape[1]))
    for j, k in enumerate((idx[1], idx[2])):
        p = int(np.argmax(np.abs(h[k, :6000])))
        ref48[j, max(0, p - pre):p + post + 1] = h[k, max(0, p - pre):p + post + 1]
    ref23 = to16(ref48)
    # 标称几何（R2）：声源相对阵列中心的位置、子阵中心、各麦克风到声源的距离
    s = R.source(dist, ang)
    cu = R.POS[idx].mean()
    rc = float(np.hypot(s[0] - cu, s[1]))
    tau = [(R.mic_dist(dist, ang, k) - rc) / R.C for k in (idx[1], idx[2])]
    align = lambda X: np.mean([frac_shift(X[j], tau[j]) for j in range(2)], 0)  # noqa: E731
    center = align(full[[1, 2]])
    direct = align(dir23)
    dir23_aligned = np.stack([frac_shift(dir23[j], tau[j]) for j in range(2)])  # 核验用：麦克风 2、3 各自对齐后的直达 RIR
    xref = align(ref23)
    _, th_act, r_act, _ = R.geometry(dist, ang, sub, False)
    return {"array": full.astype(np.float32), "array_direct": dir4.astype(np.float32),
            "center": center.astype(np.float32), "direct": direct.astype(np.float32), "ref": xref.astype(np.float32),
            "dir23_aligned": dir23_aligned, "theta_actual": float(th_act), "dist_actual": float(r_act),
            "window": {"x_dp_ms": info23["window_ms"], "x_dp_first_refl_ms": info23["first_refl_after_direct_ms"],
                       "array_ms": info4["window_ms"]}}


def sample_s2(rng, cfg):
    s2 = cfg["sources"]["s2"]
    n_int = int(rng.integers(s2["n_interferers"][0], s2["n_interferers"][1] + 1))
    return n_int


def make_scene(cfg, split, idx, speech_index):
    rng = np.random.default_rng(derive_seed(cfg["seed"], "mird_scene", split, idx))
    rt, sub = RTS[(idx // 2) % 3], SUBS[idx % 2]
    src = cfg["sources"]
    if split == "mird_s1":
        traj = [90.0 + 15.0 * k for k in range(7)]
        interferers = [{"moving": True, "trajectory_deg": traj, "dist": 2}]
        sir = float(rng.uniform(*src["s1"]["sir_db"]))
        scenario = "S1"
    else:
        s2 = src["s2"]
        n_int = int(rng.integers(s2["n_interferers"][0], s2["n_interferers"][1] + 1))
        moving = (idx // 6) % 2 == 1  # 每个 (RT60, 子阵) 组内一半有移动干扰
        n_static = n_int - 1 if moving else n_int
        while True:
            ang = rng.choice(GRID, n_static, replace=False)
            if np.any((ang >= 30) & (ang <= 90)):
                break
        interferers = [{"moving": False, "trajectory_deg": [float(a)], "dist": int(rng.choice([1, 2]))} for a in ang]
        if moving:
            d = 1.0 if rng.random() < 0.5 else -1.0
            starts = [a for a in GRID if 30 <= a + d * 90 <= 180]
            st = float(rng.choice(starts))
            interferers.append({"moving": True, "trajectory_deg": [st + d * 15.0 * k for k in range(7)],
                                "dist": int(rng.choice([1, 2]))})
        interferers = [interferers[i] for i in rng.permutation(len(interferers))]
        sir = float(rng.uniform(*src["s2"]["sir_db"]))
        scenario = "S2"
    spk = sorted({e["speaker"] for e in speech_index})
    chosen = rng.choice(len(spk), size=1 + len(interferers), replace=False)
    sp_ids = []
    for k in chosen:
        c = [e["id"] for e in speech_index if e["speaker"] == spk[k]]
        sp_ids.append(c[int(rng.integers(len(c)))])
    M = cfg["array"]["n_mics"]
    return {"id": f"{split}_{idx:05d}", "split": split, "index": idx, "scenario": scenario, "source": "MIRD v2",
            "rt60": rt, "subarray": sub, "t60_target": float(rt), "segment_s": SEG_S,
            "target": {"theta_deg": 0.0, "dist": 2.0, "mird_deg": 90},
            "interferers": interferers, "speech": {"target": sp_ids[0], "interferers": sp_ids[1:]},
            "snr_db": float(rng.uniform(*src["snr_db"])), "sir_db": sir,
            "mismatch_base": {"gain_u": [0.0] * M, "phase_u": [0.0] * M, "pos_z": [0.0] * M},
            "rir_file": f"{split}_{idx:05d}.npz"}


def scene_rirs(mird, sc):
    pos = [position(mird, sc["rt60"], sc["subarray"], 2, 0.0)]
    for it in sc["interferers"]:
        pos += [position(mird, sc["rt60"], sc["subarray"], it["dist"], a) for a in it["trajectory_deg"]]
    for it in sc["interferers"]:
        it["trajectory_actual_deg"] = [round(position(mird, sc["rt60"], sc["subarray"], it["dist"], a)["theta_actual"], 2)
                                       for a in it["trajectory_deg"]]
        it["dist_actual_m"] = round(position(mird, sc["rt60"], sc["subarray"], it["dist"],
                                             it["trajectory_deg"][0])["dist_actual"], 3)
    sc["target"]["theta_actual_deg"] = round(pos[0]["theta_actual"], 2)
    sc["target"]["dist_actual_m"] = round(pos[0]["dist_actual"], 3)
    out = {"array": np.stack([p["array"] for p in pos]), "center": np.stack([p["center"] for p in pos]),
           "direct": np.stack([p["direct"] for p in pos]),
           "direct_peak": np.array([int(np.argmax(np.abs(p["direct"]))) for p in pos]),
           "array_direct": pos[0]["array_direct"], "ref_rir": pos[0]["ref"]}
    return out, pos[0]["window"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mird", required=True)
    ap.add_argument("--config", "--data-config", dest="config", default="configs/data_mird.yaml")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--n-check", type=int, default=30, help="参考核验：每个 (RT60, 子阵) 用多少条目标语音")
    ap.add_argument("--report", default=None)
    ap.add_argument("--out-root", default=None, help="meta/ 与 rirs/ 的输出根目录（默认 cfg 的 data_root；语音仍从 data_root 读）")
    ap.add_argument("--verify-sha", default=None, help="results/mird/dataset_v2_files.json：使用前逐文件核对 sha256（Version 2）")
    a = ap.parse_args()
    cfg = load_config(a.config, a.set)
    from src.speech import load_index
    rep = {}
    if a.verify_sha:
        rep["sha_check"] = verify_sha(a.mird, a.verify_sha)
        print("sha256 check:", rep["sha_check"]["summary"])
        if rep["sha_check"]["summary"]["n_bad"]:
            raise SystemExit("MIRD files do not match Version 2 sha256 list")
    idx = load_index(cfg, cfg["splits"]["mird_s1"]["speech"])
    out_root = Path(a.out_root) if a.out_root else Path(cfg["paths"]["data_root"])
    rep["ref_check"] = ref_check(a.mird, cfg, idx, a.n_check)
    print(json.dumps(rep["ref_check"]["summary"], indent=1))
    if not a.check_only:
        for split in ("mird_s1", "mird_s2"):
            scenes = [make_scene(cfg, split, i, idx) for i in range(cfg["splits"][split]["n"])]
            d = out_root / "rirs" / split
            d.mkdir(parents=True, exist_ok=True)
            wins = []
            for sc in scenes:
                rr, win = scene_rirs(a.mird, sc)
                wins.append(win)
                np.savez(d / sc["rir_file"], info=json.dumps({"source": "MIRD v2", "rt60": sc["rt60"],
                                                               "subarray": sc["subarray"], "window": win}), **rr)
            p = out_root / "meta" / f"{split}.json"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps({"config": cfg, "scenes": scenes}, indent=1))
            if a.report:  # 元数据副本放到 reports/（回收时会下载，便于提交）
                Path(a.report).parent.mkdir(parents=True, exist_ok=True)
                (Path(a.report).parent / f"meta_{split}.json").write_text(json.dumps({"scenes": scenes}))
            rep[split] = {"n": len(scenes), "n_moving": sum(any(i["moving"] for i in s["interferers"]) for s in scenes),
                          "n_interf": {str(k): sum(len(s["interferers"]) == k for s in scenes) for k in (1, 2, 3)}}
            print(split, rep[split])
    out = Path(a.report) if a.report else out_root / "reports" / "mird_ref_check.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=1))


def verify_sha(mird, listing):
    import hashlib
    files = json.loads(Path(listing).read_text())["files"]
    bad, n = [], 0
    for rel, info in files.items():
        if not rel.endswith(".mat"):
            continue
        f = Path(mird) / rel
        if not f.exists():
            bad.append((rel, "missing"))
            continue
        h = hashlib.sha256()
        with open(f, "rb") as fo:
            for b in iter(lambda: fo.read(1 << 20), b""):
                h.update(b)
        n += 1
        if h.hexdigest() != info["sha256"]:
            bad.append((rel, "sha256 mismatch"))
    return {"summary": {"n_checked": n, "n_bad": len(bad)}, "bad": bad[:20]}


def ref_check(mird, cfg, speech_index, n):
    """只含目标直达声的样本经 h_D（标称导向矢量）后与 x_dp 的 SI-SDR（以及完整目标经 h_D 后与 x_ref 的 SI-SDR，作参照）。"""
    import torch
    from src import metrics as mt
    from src.models.gtcrn_fusion import BeamBank
    from src.speech import read_clip
    from src.stft import istft, stft
    bank = BeamBank.from_config(cfg)
    rng = np.random.default_rng(derive_seed(cfg["seed"], "mird_ref_check"))
    ents = [speech_index[i] for i in rng.choice(len(speech_index), n, replace=False)]
    rows = []
    for rt in RTS:
        for sub in SUBS:
            p = position(mird, rt, sub, 2, 0.0)
            for e in ents:
                s = read_clip(cfg, e).astype(np.float64)
                N = len(s)
                conv = lambda h: np.fft.irfft(np.fft.rfft(s, 2 * N) * np.fft.rfft(h, 2 * N), 2 * N)[..., :N]  # noqa: E731
                y = torch.as_tensor(conv(p["array_direct"]), dtype=torch.float32)  # [4, N]
                xdp = conv(p["direct"])
                Z = bank(stft(y)[None])  # [1, P, F, T]
                o = istft(Z[:, 0], N)[0].numpy().astype(np.float64)
                m2, m3 = conv(p["dir23_aligned"][0]), conv(p["dir23_aligned"][1])
                rows.append({"rt60": rt, "sub": sub, "si_sdr_hD_direct_vs_xdp": float(mt.si_sdr(o, xdp)),
                             "si_sdr_mic2_vs_mic3_aligned": float(mt.si_sdr(m2, m3))})
    v = np.array([r["si_sdr_hD_direct_vs_xdp"] for r in rows])
    summ = {"n": len(v), "median_db": float(np.median(v)), "p5_db": float(np.percentile(v, 5)),
            "min_db": float(v.min()), "max_db": float(v.max()),
            "by_group": {f"{rt}_{sub}": float(np.median([r["si_sdr_hD_direct_vs_xdp"] for r in rows
                                                         if r["rt60"] == rt and r["sub"] == sub]))
                         for rt in RTS for sub in SUBS},
            "mic2_vs_mic3_aligned_median_db": float(np.median([r["si_sdr_mic2_vs_mic3_aligned"] for r in rows])),
            "mic2_vs_mic3_by_group": {f"{rt}_{sub}": float(np.median([r["si_sdr_mic2_vs_mic3_aligned"] for r in rows
                                                                       if r["rt60"] == rt and r["sub"] == sub]))
                                      for rt in RTS for sub in SUBS},
            "windows": {f"{rt}_{sub}": position(mird, rt, sub, 2, 0.0)["window"] for rt in RTS for sub in SUBS}}
    return {"summary": summ, "rows": rows}


if __name__ == "__main__":
    main()
