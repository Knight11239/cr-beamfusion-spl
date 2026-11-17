"""阻塞矩阵 B(f) = [g_1, …, g_P]（g_p = h_p − h_D）逐频点的数值秩与条件数（用户 2026-11-09）。

  python scripts/blocking_rank.py      # 写 results/g2/blocking_rank.{md,csv,png}

SVD（complex128）；数值秩 = #{σ_i > 1e-10 · σ_max}；条件数 = σ_max / σ_min（P 个奇异值中的最小值，含数值上为零的）。
f = 0 Hz 时所有 g_p = 0（B = 0），秩记为 0、条件数记为 nan。8 麦克风（δ = 1 cm）与 G2（M = 4、δ = 3 cm）使用与模型相同的字典。
另：WNG 公式 W = 1/(1/M + uᴴGu) 的前提 h_Dᴴ g_p = 0 逐频点核对（最大 |h_Dᴴ g_p| / ‖g_p‖）。
"""
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.beamformers import build_dictionary, steer, stft_freqs  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TOL = 1e-10


def analyse(M, delta, c=343.0):
    F = stft_freqs(16000, 512)
    H = build_dictionary(F, n_mics=M, spacing=delta, c=c)["H"]  # [P+1, F, M] complex128
    Bm = (H[1:] - H[:1]).transpose(1, 2, 0)  # [F, M, P]
    s = np.linalg.svd(Bm, compute_uv=False)  # [F, min(M, P)]
    P = Bm.shape[-1]
    smax = s[:, 0]
    rank = (s > TOL * smax[:, None]).sum(1) * (smax > 0)
    smin = s[:, -1]  # P 个奇异值中的最小值（M − 1 < P 时为舍入误差量级）
    cond = np.where(smax > 0, smax / np.maximum(smin, 1e-300), np.nan)
    sr = s[np.arange(len(F)), np.maximum(rank - 1, 0)]  # 第 rank 个奇异值
    cond_eff = np.where(smax > 0, smax / np.maximum(sr, 1e-300), np.nan)
    d = steer(F, 0.0, M, delta, c)
    hD = H[0]
    orth = np.abs(np.einsum("fm,fmp->fp", hD.conj(), Bm)) / np.maximum(np.linalg.norm(Bm, axis=1), 1e-300)
    dl = np.abs(np.einsum("fmp,fm->fp", Bm.conj(), d))
    return F, s, rank, cond, cond_eff, orth[1:].max(), dl[1:].max(), P


def main():
    out = ROOT / "results" / "g2"
    rows, L = [], ["# 阻塞矩阵 B(f) = [g_1, …, g_P] 的数值秩与条件数（`python scripts/blocking_rank.py`）", "",
                   f"SVD（complex128），数值秩 = #{{σ_i > {TOL:g}·σ_max}}；条件数 = σ_max/σ_min（P 个奇异值；M − 1 < P 时恒为 ∞）。"
                   "f = 0 Hz 时 B = 0，不计入。理论上 rank(B) ≤ min(P, M − 1)（g_pᴴd = 0）。", ""]
    fig, ax = plt.subplots(2, 1, figsize=(6.4, 5.0), sharex=True)
    for name, M, dl in (("8mic", 8, 0.01), ("G2", 4, 0.03)):
        F, s, rank, cond, cond_eff, orth, dres, P = analyse(M, dl)
        for i, f in enumerate(F):
            rows.append({"array": name, "f_hz": f, "rank": int(rank[i]), "cond": cond[i], "cond_eff": cond_eff[i],
                         **{f"sigma{k + 1}": s[i, k] for k in range(s.shape[1])}})
        fp = F > 0
        uniq = {int(r): int((rank[fp] == r).sum()) for r in np.unique(rank[fp])}
        first_full = F[fp][np.argmax(rank[fp] == rank[fp].max())]
        cf, ce = cond[fp], cond_eff[fp]
        fin = np.isfinite(cf)
        lo = F[fp] < 1000
        L += [f"## {name}（M = {M}，δ = {dl * 100:g} cm，P = {P}，min(P, M − 1) = {min(P, M - 1)}）", "",
              f"- 数值秩的频点数（f > 0，共 {fp.sum()} 个）：" + "，".join(f"秩 {k}：{v}" for k, v in uniq.items()),
              f"- 达到最大秩 {rank[fp].max()} 的最低频点：{first_full:.1f} Hz",
              f"- 条件数 σ_1/σ_P（全部 P 个奇异值）：中位数 {np.median(cf):.3g}，最大 {cf.max():.3g}，最小 {cf.min():.3g}"
              + ("（P > M − 1：σ_P 为舍入误差，B 的列线性相关，Gram 矩阵 G 奇异）" if P > M - 1 else ""),
              f"- 有效条件数 σ_1/σ_r（r = 数值秩）：中位数 {np.median(ce):.3g}，最大 {ce.max():.3g}（{F[fp][np.argmax(ce)]:.1f} Hz），"
              f"最小 {ce.min():.3g}；< 1 kHz 的中位数 {np.median(ce[lo]):.3g}，≥ 1 kHz 的中位数 {np.median(ce[~lo]):.3g}",
              f"- 最小奇异值 σ_min（第 {min(P, M - 1)} 个）：125 Hz {s[np.argmin(abs(F - 125)), min(P, M - 1) - 1]:.3g}，"
              f"1 kHz {s[np.argmin(abs(F - 1000)), min(P, M - 1) - 1]:.3g}，4 kHz {s[np.argmin(abs(F - 4000)), min(P, M - 1) - 1]:.3g}"
              f"（σ_max 分别为 {s[np.argmin(abs(F - 125)), 0]:.3g}、{s[np.argmin(abs(F - 1000)), 0]:.3g}、{s[np.argmin(abs(F - 4000)), 0]:.3g}）",
              f"- 前提核对：max_f,p |h_Dᴴg_p|/‖g_p‖ = {orth:.1e}；max |g_pᴴd| = {dres:.1e}", ""]
        for k in range(s.shape[1]):
            ax[0 if name == "8mic" else 1].semilogy(F[fp], s[fp, k] / s[fp, 0], label=f"σ{k + 1}/σ1")
        a = ax[0 if name == "8mic" else 1]
        a.axhline(TOL, color="k", ls=":", lw=0.8)
        a.set_title(f"{name}: M = {M}, δ = {dl * 100:g} cm (rank threshold {TOL:g})", fontsize=9)
        a.set_ylabel("σ_k / σ_1")
        a.legend(fontsize=7, ncol=4)
        a.set_ylim(1e-18, 2)
    ax[1].set_xlabel("frequency (Hz)")
    fig.tight_layout()
    fig.savefig(out / "blocking_rank.png", dpi=150)
    with open(out / "blocking_rank.csv", "w", newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader()
        w.writerows(rows)
    (out / "blocking_rank.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
