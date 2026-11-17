"""MIRD（Hadad 等，IWAENC 2014；3-3-3-8-3-3-3 cm 阵列）的 R2（子阵与坐标约定）与 R3（真实阵列失配估计、等效 κ）。只用 CPU。

  python scripts/mird_r2r3.py --mird /path/to/MIRD --out results/mird

几何（MIRD .mat 的 metapar：mic_position = "mic1@90deg, mic8@270deg"，mic_spacing = [3 3 3 8 3 3 3] cm；论文：声源在以阵列
中心为圆心、半径 1 m / 2 m 的半圆上，方位 −90°…90°，步长 15°，文件名中 270…345 即 −90…−15）：
- 阵列轴单位向量 u 指向 MIRD 90°（mic1 一侧），法向 n 指向 MIRD 0°；麦克风 k 位于 (13 − cum_k) cm·u，
  cum = [0, 3, 6, 9, 17, 20, 23, 26] cm（mic1 +13 cm，mic8 −13 cm）；声源 s = r (sin φ · u + cos φ · n)。
- 子阵 A = mic1–4（中心 +8.5 cm），子阵 B = mic5–8（中心 −8.5 cm），各为 M = 4、δ = 3 cm 的均匀线阵（G2）。
- 与本项目约定（CLAUDE.md §1，target_first：物理位置 center − x_m e_x，x_m = (m − 1.5) δ，θ 自 e_x 起算，目标 θ = 0）
  的对应：取 e_x = u，则 m = 0 ↔ mic1（A）/ mic5（B），θ = ∠(s − c, u)，远场时 θ ≈ 90° − φ；MIRD 90° 为目标方向。
  镜像取法 e_x = −u（m = 0 ↔ mic4 / mic8，θ ≈ 90° + φ，MIRD 270° 为目标）也给出，作为 R3 的第二组目标位置。
- 冲激响应含固定的系统延迟 simpar.int_delay = 624 个样本（48 kHz），减去后为传播时间。

R2：每个文件、每个子阵，用直达声窗口的互相关（32 倍上采样）估计子阵内相对 mic(m = 0) 的时延，与标称时延（近场点声源，
c = 343 m/s；以及远场 −x_m cosθ/c）比较；用实测时延做远场最小二乘方位估计，与按实际几何换算的 θ 比较；绝对到达时间换算
距离，与几何距离比较。
R3：对目标位置（MIRD 90°，以及镜像 270°；1 m / 2 m；可用的 RT60），用锥形窗截取各麦克风的直达声（窗长不含第一次反射），
在本项目的 257 个 STFT 频点上计算 H_m(f)，与标称导向矢量比较：r_m = H_m / d_m，再除以子阵内的公共因子（几何平均幅度、
平均相位），得到逐麦克风的增益偏差（dB）与相位偏差（度）随频率的变化。参考模型两种：远场平面波（本方法使用的标称导向矢量）
与近场点声源（1/r 幅度 + 球面波时延；仿真数据本身包含这部分，故与 §3.6 可比的“硬件失配”以近场模型为准）。
相位偏差按 φ_m(f) = φ0_m − 2π f Δτ_m 拟合（500–7500 Hz）：φ0 对应 §3.6 的相位误差，Δτ 换算为轴向位置误差 Δx = c Δτ
（端射方向时延对 x 位置的灵敏度为 1）。等效 κ（§3.6：g ~ U[−κ, κ] dB，φ ~ U[−5κ, 5κ]°，Δx ~ N(0, (0.1κ cm)²)）：
κ_gain = √3·std(g)/1 dB，κ_phase = √3·std(φ0)/5°，κ_pos = std(Δx)/0.1 cm；以及逐频点的总等效 κ_eff(f)：
实测相对误差能量 mean_m |ã_m/d_m − 1|² 除以 κ = 1 时的期望值 (ln10/20)²/3 + (5° in rad)²/3 + (2π f · 0.1 cm / c)²。
"""
import argparse
import csv
import glob
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.io import loadmat  # noqa: E402
from scipy.signal import hilbert  # noqa: E402

FS = 48000
LAT = 624  # simpar.int_delay
C = 343.0
CUM = np.array([0, 3, 6, 9, 17, 20, 23, 26]) / 100.0
POS = 0.13 - CUM  # 沿 u 的坐标（m）
SUB = {"A": [0, 1, 2, 3], "B": [4, 5, 6, 7]}
DELTA = 0.03
FREQS = np.arange(257) * 16000 / 512
BAND = (FREQS >= 500) & (FREQS <= 7500)
LO = (FREQS >= 500) & (FREQS < 4000)
HI = (FREQS >= 4000) & (FREQS <= 7500)
ANGLES = [0, 15, 30, 45, 60, 75, 90, 270, 285, 300, 315, 330, 345]
COL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
REFL_DB, REFL_SKIP_MS = -25.0, 1.5  # 第一次反射：直达峰后 1.5 ms（扬声器自身响应之后）首个高于 −25 dB 的包络点
PRE_MS, RISE_MS, FALL_MS, GUARD_MS = 0.5, 0.25, 0.5, 0.1


def phi_signed(a):
    return a - 360 if a >= 180 else a


def load(mird, rt, dist, ang):
    pat = f"{mird}/*Reverberation_{rt}s)_3-3-3-8-3-3-3/*Reverberation_{rt}s)_3-3-3-8-3-3-3_{dist}m_{ang:03d}.mat"
    fs = glob.glob(pat)
    if not fs:
        return None, None
    z = loadmat(fs[0], squeeze_me=True, struct_as_record=False)
    mp = z["metapar"]
    assert list(mp.mic_spacing) == [3, 3, 3, 8, 3, 3, 3] and abs(mp.reverberation - float(rt)) < 1e-6
    assert int(mp.distance) == dist and int(mp.azimuth) == ang and int(z["simpar"].fs) == FS
    assert int(z["simpar"].int_delay) == LAT
    return z["impulse_response"][:6000].T.astype(np.float64), np.asarray(mp.mic_factors, float)


def source(dist, ang):
    p = np.deg2rad(phi_signed(ang))
    return dist * np.array([np.sin(p), np.cos(p)])  # (u, n)


def geometry(dist, ang, sub, flip):
    """子阵中心、本项目约定下的 θ（度）、距离、按 m 排列的麦克风索引与 x_m。"""
    idx = SUB[sub][::-1] if flip else SUB[sub]
    cu = POS[SUB[sub]].mean()
    s = source(dist, ang)
    v = s - np.array([cu, 0.0])
    ex = -1.0 if flip else 1.0
    th = np.degrees(np.arctan2(abs(v[1]), ex * v[0]))
    return cu, th, float(np.hypot(*v)), idx


def mic_dist(dist, ang, k):
    s = source(dist, ang)
    return float(np.hypot(s[0] - POS[k], s[1]))


def direct_window(h, idx):
    """公共锥形窗（同一绝对时间段作用于子阵内各麦克风）。返回窗、首次反射信息。"""
    env = np.abs(hilbert(h[idx], axis=-1))
    pk = env.argmax(-1)
    refl = []
    for e, p in zip(env, pk):
        e = e / e[p]
        st = p + int(REFL_SKIP_MS * FS / 1000)
        over = np.nonzero(20 * np.log10(e[st:] + 1e-12) > REFL_DB)[0]
        t = st + (over[0] if len(over) else len(e) - st - 1)
        seg = e[t:t + 24]
        refl.append((t, (t - p) / FS * 1000, 20 * np.log10(seg.max())))
    t_ref = min(r[0] for r in refl)
    a = pk.min() - int(PRE_MS * FS / 1000)
    b = t_ref - int(GUARD_MS * FS / 1000)
    n = np.arange(h.shape[-1])
    w = np.zeros(h.shape[-1])
    nr, nf = int(RISE_MS * FS / 1000), int(FALL_MS * FS / 1000)
    w[a:b] = 1.0
    w[a:a + nr] = 0.5 - 0.5 * np.cos(np.pi * np.arange(nr) / nr)
    w[b - nf:b] = 0.5 + 0.5 * np.cos(np.pi * np.arange(nf) / nf)
    info = {"direct_ms": float((pk.min() - LAT) / FS * 1000),
            "first_refl_after_direct_ms": float(min(r[1] for r in refl)),
            "first_refl_level_db": float(refl[int(np.argmin([r[1] for r in refl]))][2]),
            "window_ms": float((b - a) / FS * 1000), "window_flat_ms": float((b - a - nr - nf) / FS * 1000)}
    return w, info, pk


def tdoa(x, ref, up=32):
    n = 1 << int(np.ceil(np.log2(len(x) * 2)))
    X, R = np.fft.rfft(x, n), np.fft.rfft(ref, n)
    f = np.fft.rfftfreq(n, 1 / FS)
    G = X * np.conj(R) * (f <= 8000)
    cc = np.fft.irfft(G, n * up)
    lag = int(np.argmax(np.abs(cc)))
    if lag > n * up // 2:
        lag -= n * up
    return lag / (FS * up)


def spectrum(h, w):
    n = np.arange(h.shape[-1])
    E = np.exp(-2j * np.pi * np.outer(FREQS, n) / FS)  # [F, N]
    return (h * w) @ E.T  # [mics, F]


def r2(mird, rts):
    rows = []
    for rt in rts:
        for dist in (1, 2):
            for ang in ANGLES:
                h, _ = load(mird, rt, dist, ang)
                if h is None:
                    continue
                for sub in ("A", "B"):
                    cu, th, rr, idx = geometry(dist, ang, sub, False)
                    w, info, pk = direct_window(h, idx)
                    hw = h[idx] * w
                    meas = np.array([tdoa(hw[m], hw[0]) for m in range(4)]) * 1e6  # µs，相对 m = 0
                    rm = np.array([mic_dist(dist, ang, k) for k in idx])
                    nom_nf = (rm - rm[0]) / C * 1e6
                    xm = (np.arange(4) - 1.5) * DELTA
                    nom_ff = (xm - xm[0]) * np.cos(np.deg2rad(th)) / C * 1e6
                    grid = np.arange(0, 180.01, 0.1)
                    pred = (xm - xm[0])[None] * np.cos(np.deg2rad(grid))[:, None] / C * 1e6
                    th_hat = grid[np.argmin(((pred - meas[None]) ** 2).sum(1))]
                    toa = (pk - LAT) / FS  # 整样本峰值，用于距离核对
                    rows.append({"rt60": rt, "dist_m": dist, "mird_deg": phi_signed(ang), "sub": sub,
                                 "theta_ours_deg": round(th, 2), "theta_farfield_approx": 90 - phi_signed(ang),
                                 "r_sub_m": round(rr, 4), "theta_hat_deg": round(float(th_hat), 1),
                                 "tdoa_meas_us": np.round(meas, 2).tolist(), "tdoa_nom_nf_us": np.round(nom_nf, 2).tolist(),
                                 "tdoa_nom_ff_us": np.round(nom_ff, 2).tolist(),
                                 "tdoa_err_nf_rms_us": round(float(np.sqrt(np.mean((meas - nom_nf)[1:] ** 2))), 2),
                                 "tdoa_err_ff_rms_us": round(float(np.sqrt(np.mean((meas - nom_ff)[1:] ** 2))), 2),
                                 "dist_meas_minus_geom_cm": np.round((toa * C - rm) * 100, 1).tolist(), **info})
    return rows


def r3(mird, rts):
    out = []
    for rt in rts:
        for dist in (1, 2):
            for ang, flip in ((90, False), (270, True)):
                h, mf = load(mird, rt, dist, ang)
                if h is None:
                    continue
                for sub in ("A", "B"):
                    cu, th, rr, idx = geometry(dist, ang, sub, flip)
                    w, info, _ = direct_window(h, idx)
                    H = spectrum(h[idx], w)  # [4, F]
                    xm = (np.arange(4) - 1.5) * DELTA
                    d_ff = np.exp(-2j * np.pi * FREQS[None] * xm[:, None] * np.cos(np.deg2rad(th)) / C)
                    rm = np.array([mic_dist(dist, ang, k) for k in idx])
                    d_nf = (rr / rm)[:, None] * np.exp(-2j * np.pi * FREQS[None] * (rm - rr)[:, None] / C)
                    res = {"rt60": rt, "dist_m": dist, "mird_deg": phi_signed(ang), "sub": sub,
                           "mics": [k + 1 for k in idx], "theta_ours_deg": round(th, 2), **info,
                           "mic_factors_db": np.round(20 * np.log10(mf[idx]), 3).tolist()}
                    for name, d in (("nf", d_nf), ("ff", d_ff)):
                        r = H / d
                        lg = np.log(np.abs(r))
                        com = np.exp(lg.mean(0)) * np.exp(1j * np.angle(r.sum(0)))
                        a = r / com  # ã_m / d_m
                        g = 20 * np.log10(np.abs(a))
                        ph = np.degrees(np.angle(a))
                        A = np.stack([np.ones(BAND.sum()), -2 * np.pi * FREQS[BAND]], 1)
                        coef = np.linalg.lstsq(A, np.radians(ph[:, BAND]).T, rcond=None)[0]  # [2, 4]
                        phi0, dtau = np.degrees(coef[0]), coef[1]
                        dx_cm = C * (dtau - dtau.mean()) * 100
                        gb = g[:, BAND].mean(1)
                        err = np.mean(np.abs(a - 1) ** 2, 0)
                        unit = (np.log(10) / 20) ** 2 / 3 + np.radians(5) ** 2 / 3 + (2 * np.pi * FREQS * 0.001 / C) ** 2
                        keff = np.sqrt(err / unit)
                        res[name] = {"gain_db_band": np.round(gb, 3).tolist(), "phase0_deg": np.round(phi0, 2).tolist(),
                                     "dx_cm": np.round(dx_cm, 3).tolist(),
                                     "kappa_gain": round(float(np.sqrt(3) * gb.std()), 3),
                                     "kappa_phase": round(float(np.sqrt(3) * phi0.std() / 5), 3),
                                     "kappa_pos": round(float(dx_cm.std() / 0.1), 3),
                                     "kappa_eff_median_band": round(float(np.median(keff[BAND])), 3),
                                     "kappa_eff_p90_band": round(float(np.percentile(keff[BAND], 90)), 3),
                                     "kappa_eff_median_500_4000": round(float(np.median(keff[LO])), 3),
                                     "kappa_eff_median_4000_7500": round(float(np.median(keff[HI])), 3),
                                     "_g": g, "_ph": ph, "_keff": keff}
                        if name == "nf":
                            gcal = gb + 20 * np.log10(mf[idx])  # 乘以 mic_factors 之后
                            res["nf"]["gain_db_band_x_mic_factors"] = np.round(gcal - gcal.mean(), 3).tolist()
                            res["nf"]["kappa_gain_x_mic_factors"] = round(float(np.sqrt(3) * gcal.std()), 3)
                    out.append(res)
    return out


def plots(r2rows, r3rows, out):
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    for k, sub in enumerate("AB"):
        rr = [r for r in r2rows if r["sub"] == sub]
        ax[0].plot([r["theta_ours_deg"] for r in rr], [r["theta_hat_deg"] for r in rr], "o", ms=4, mfc="none",
                   color=COL[k], label=f"subarray {sub} (mics {'1–4' if sub == 'A' else '5–8'})")
        ax[1].plot([r["theta_ours_deg"] for r in rr], [r["tdoa_err_nf_rms_us"] for r in rr], "o", ms=4, mfc="none",
                   color=COL[k], label=f"{sub}: vs near-field nominal")
        ax[1].plot([r["theta_ours_deg"] for r in rr], [r["tdoa_err_ff_rms_us"] for r in rr], "x", ms=4,
                   color=COL[k], label=f"{sub}: vs far-field nominal")
    ax[0].plot([0, 180], [0, 180], color="#52514e", lw=0.8, ls="--")
    ax[0].set(xlabel="θ from geometry (deg, our convention)", ylabel="θ from measured TDOA (deg)",
              title="R2: measured vs geometric angle")
    ax[1].set(xlabel="θ (deg)", ylabel="TDOA error RMS (µs)", title="R2: TDOA error (1 sample @48 kHz = 20.8 µs)")
    for a in ax[:2]:
        a.legend(fontsize=7)
    rr = [r for r in r2rows]
    ax[2].hist([r["first_refl_after_direct_ms"] for r in rr if r["dist_m"] == 1], bins=20, color=COL[0], alpha=0.7,
               label="1 m")
    ax[2].hist([r["first_refl_after_direct_ms"] for r in rr if r["dist_m"] == 2], bins=20, color=COL[1], alpha=0.7,
               label="2 m")
    ax[2].set(xlabel="first reflection after direct (ms)", ylabel="count", title="first reflection (−25 dB rule)")
    ax[2].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "r2_angles_tdoa.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(2, 3, figsize=(15, 7), sharex=True)
    for col, model in enumerate(("nf", "ff")):
        for r in r3rows:
            for m in range(4):
                ls = "-" if r["sub"] == "A" else "--"
                lab = f"{r['sub']} m={m}" if (r["rt60"], r["dist_m"], r["mird_deg"]) == (r3rows[0]["rt60"], 2, 90) else None
                ax[0, col].plot(FREQS, r[model]["_g"][m], ls, color=COL[m], lw=0.5, alpha=0.6, label=lab)
                ax[1, col].plot(FREQS, r[model]["_ph"][m], ls, color=COL[m], lw=0.5, alpha=0.6)
        ax[0, col].set(ylabel="gain deviation (dB)", ylim=(-3, 3),
                       title=f"R3: vs {'near-field point-source' if model == 'nf' else 'far-field (method)'} model")
        ax[1, col].set(ylabel="phase deviation (deg)", xlabel="frequency (Hz)", ylim=(-30, 30))
        for a in ax[:, col]:
            a.axvspan(0, 500, color="#e4e3df", alpha=0.6)
    ax[0, 0].legend(fontsize=6, ncol=2)
    for r in r3rows:
        ax[0, 2].plot(FREQS, r["nf"]["_keff"], color=COL[0] if r["dist_m"] == 2 else COL[1], lw=0.6, alpha=0.6)
        ax[1, 2].plot(FREQS, r["ff"]["_keff"], color=COL[0] if r["dist_m"] == 2 else COL[1], lw=0.6, alpha=0.6)
    for a, t in zip(ax[:, 2], ("near-field", "far-field")):
        a.set(ylabel="equivalent κ_eff(f)", ylim=(0, 6), title=f"equivalent κ ({t}); blue 2 m, orange 1 m")
        a.axhline(1, color="#52514e", lw=0.8, ls="--")
        a.axvspan(0, 500, color="#e4e3df", alpha=0.6)
    ax[1, 2].set_xlabel("frequency (Hz)")
    fig.tight_layout()
    fig.savefig(out / "r3_mismatch.png", dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mird", required=True)
    ap.add_argument("--out", default="results/mird")
    ap.add_argument("--rt", nargs="*", default=None, help="只处理这些 RT60（如 0.610）；默认全部可用的")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rts = [rt for rt in (a.rt or ("0.160", "0.360", "0.610")) if load(a.mird, rt, 2, 0)[0] is not None]
    r2rows = r2(a.mird, rts)
    r3rows = r3(a.mird, rts)
    plots(r2rows, r3rows, out)
    with open(out / "r2_tdoa.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(r2rows[0]))
        wr.writeheader()
        wr.writerows(r2rows)
    clean = [{k: ({kk: vv for kk, vv in v.items() if not kk.startswith("_")} if isinstance(v, dict) else v)
              for k, v in r.items()} for r in r3rows]
    (out / "r3_mismatch.json").write_text(json.dumps(clean, indent=1))
    summ = {"rt60_available": rts,
            "r2": {"n": len(r2rows),
                   "theta_hat_minus_geom_abs_max_deg": float(max(abs(r["theta_hat_deg"] - r["theta_ours_deg"])
                                                                for r in r2rows)),
                   "theta_hat_minus_geom_abs_median_deg": float(np.median([abs(r["theta_hat_deg"] - r["theta_ours_deg"])
                                                                           for r in r2rows])),
                   "tdoa_err_nf_rms_us_median": float(np.median([r["tdoa_err_nf_rms_us"] for r in r2rows])),
                   "tdoa_err_nf_rms_us_max": float(max(r["tdoa_err_nf_rms_us"] for r in r2rows)),
                   "tdoa_err_ff_rms_us_median": float(np.median([r["tdoa_err_ff_rms_us"] for r in r2rows])),
                   "dist_err_cm_median": float(np.median([v for r in r2rows for v in r["dist_meas_minus_geom_cm"]])),
                   "dist_err_cm_absmax": float(max(abs(v) for r in r2rows for v in r["dist_meas_minus_geom_cm"])),
                   "first_refl_ms": {f"{d}m": [float(np.min([r["first_refl_after_direct_ms"] for r in r2rows
                                                             if r["dist_m"] == d])),
                                               float(np.median([r["first_refl_after_direct_ms"] for r in r2rows
                                                                if r["dist_m"] == d]))] for d in (1, 2)},
                   "window_ms": [float(min(r["window_ms"] for r in r2rows)), float(max(r["window_ms"] for r in r2rows))]},
            "r3": {}}
    for model in ("nf", "ff"):
        s = {}
        for key in ("kappa_gain", "kappa_phase", "kappa_pos", "kappa_eff_median_band", "kappa_eff_p90_band",
                    "kappa_eff_median_500_4000", "kappa_eff_median_4000_7500"):
            v = [r[model][key] for r in r3rows]
            s[key] = {"median": float(np.median(v)), "min": float(min(v)), "max": float(max(v))}
        if model == "nf":
            v = [r["nf"]["kappa_gain_x_mic_factors"] for r in r3rows]
            s["kappa_gain_x_mic_factors"] = {"median": float(np.median(v)), "min": float(min(v)), "max": float(max(v))}
        summ["r3"][model] = s
    # 每个物理麦克风的增益偏差（近场模型，子阵内去均值）在各条件之间的一致性
    per_mic = {}
    for r in r3rows:
        for m, k in enumerate(r["mics"]):
            per_mic.setdefault(k, []).append((r["nf"]["gain_db_band"][m], r["nf"]["phase0_deg"][m], r["nf"]["dx_cm"][m]))
    summ["r3"]["per_mic_nf"] = {str(k): {"gain_db_mean": round(float(np.mean([x[0] for x in v])), 3),
                                         "gain_db_std": round(float(np.std([x[0] for x in v])), 3),
                                         "phase0_deg_mean": round(float(np.mean([x[1] for x in v])), 2),
                                         "phase0_deg_std": round(float(np.std([x[1] for x in v])), 2),
                                         "dx_cm_mean": round(float(np.mean([x[2] for x in v])), 3),
                                         "n": len(v)} for k, v in sorted(per_mic.items())}
    summ["r3"]["windows"] = [{k: r[k] for k in ("rt60", "dist_m", "mird_deg", "sub", "window_ms",
                                                 "first_refl_after_direct_ms", "first_refl_level_db")} for r in r3rows]
    (out / "r2r3_summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps({k: v for k, v in summ.items() if k != "r3"}, indent=1))
    print(json.dumps({m: summ["r3"][m] for m in ("nf", "ff")}, indent=1))
    print(json.dumps(summ["r3"]["per_mic_nf"], indent=1))


if __name__ == "__main__":
    main()
