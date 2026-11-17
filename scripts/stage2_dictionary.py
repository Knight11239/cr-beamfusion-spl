"""阶段 2：设计固定波束形成器字典、计算并保存 Gram 矩阵、画波束图与 WNG 曲线、统计约束残差。

用法：python scripts/stage2_dictionary.py --config configs/data_dev.yaml
输出：results/stage2/*.png|json|csv；默认字典（重新归一化）与 Gram 矩阵保存在 {data_root}/dictionary/dict_base.npz，
未归一化的对照版本为 dict_base_raw.npz（npz，不进 git）。
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.beamformers import (beampattern, build_dictionary, gram, response, save_dictionary, steer,  # noqa: E402
                             stft_freqs, wng)
from src.config import data_root, load_config  # noqa: E402
from src.plots import GRID, INK2, SERIES, plt, polar_beampatterns, setup_style, wng_curves  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results" / "stage2"
BP_FREQS = [500.0, 1000.0, 2000.0, 4000.0, 6000.0]
WNG_TABLE_FREQS = [31.25, 62.5, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 6000.0, 8000.0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data_dev.yaml")
    ap.add_argument("--out", default=None, help="输出目录（默认 results/stage2）")
    a = ap.parse_args()
    global OUT
    if a.out:
        OUT = Path(a.out)
    cfg = load_config(a.config)
    M, delta, c = cfg["array"]["n_mics"], cfg["array"]["spacing"], cfg["signal"]["c"]
    OUT.mkdir(parents=True, exist_ok=True)
    F = stft_freqs(cfg["signal"]["fs"], 512)
    ds = steer(F, 0.0, M, delta, c)
    stats = {"M": M, "spacing_m": delta, "c": c, "n_freqs": len(F)}

    dics = {}
    for renorm in (False, True):
        tag = "renorm" if renorm else "raw"
        dic = build_dictionary(F, renormalize=renorm, n_mics=M, spacing=delta, c=c)
        G, _ = gram(dic["H"])
        dics[tag] = (dic, G)
        dd = data_root(cfg) / "dictionary"
        dd.mkdir(parents=True, exist_ok=True)
        save_dictionary(dd / ("dict_base.npz" if renorm else "dict_base_raw.npz"), dic, G)

        s = {}
        for p, name in enumerate(dic["names"]):
            res = np.abs(response(dic["H"][p], ds) - 1)[1:]
            e = {"max_constraint_residual": float(res.max()),
                 "n_bins_residual_ge_1e-6": int((res >= 1e-6).sum()),
                 "bins_residual_ge_1e-6_Hz": F[1:][res >= 1e-6].tolist()}
            if name in dic["info"]:
                inf = dic["info"][name]
                nr = inf["null_response"][1:, 0]
                e.update({"max_null_response": float(nr.max()),
                          "n_bins_null_ge_1e-4": int((nr >= 1e-4).sum()),
                          "max_cond_CHC": float(inf["cond_CHC"][1:].max()),
                          "cond_CHC_at_31Hz": float(inf["cond_CHC"][1])})
            w = wng(dic["H"][p], ds)
            e["wng_dB_at"] = {f"{f:g}": float(w[int(round(f / F[1]))]) for f in WNG_TABLE_FREQS}
            e["wng_dB_min_f>0"] = float(w[1:].min())
            s[name] = e
        eig = np.linalg.eigvalsh(G[1:])
        s["gram"] = {"shape": list(G.shape), "dtype": str(G.dtype),
                     "min_eig": float(eig.min()), "max_eig": float(eig.max()),
                     "max_eig_at_31Hz": float(eig[0].max()), "max_eig_at_1kHz": float(eig[31].max())}
        stats[tag] = s

    dic, G = dics["renorm"]  # 默认字典
    names = dic["names"]
    # 波束图（极坐标）：行 = 滤波器，列 = 频率
    idx = [int(round(f / F[1])) for f in BP_FREQS]
    bp = np.stack([beampattern(dic["H"][p][idx], F[idx], np.arange(361), M, delta, c) for p in range(len(names))])
    polar_beampatterns(bp, names, BP_FREQS, OUT / "beampatterns_polar.png",
                       title="Fixed beamformer dictionary: |hᴴd(θ)|² in dB (floor −40 dB), θ from +x (endfire), "
                             f"M={M}, δ={delta * 100:g} cm, target θs=0°")
    # WNG–频率
    wdb = np.stack([wng(h, ds) for h in dic["H"]])
    wng_curves(F, wdb, names, OUT / "wng_vs_freq.png", ref_db=10 * np.log10(M),
               title=f"White noise gain of the dictionary filters (M={M}, δ={delta * 100:g} cm)")

    # 约束残差与零点响应（两种实现）
    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    for k, name in enumerate(names[1:], start=1):
        for tag, ls in (("raw", "-"), ("renorm", "--")):
            d_, _ = dics[tag]
            res = np.abs(response(d_["H"][k], ds) - 1)
            axes[0].plot(F[1:], np.maximum(res[1:], 1e-17), color=SERIES[k], ls=ls,
                         label=f"{name} ({tag})")
            axes[1].plot(F[1:], d_["info"][name]["null_response"][1:, 0], color=SERIES[k], ls=ls,
                         label=f"{name} ({tag})")
    axes[0].axhline(1e-6, color=INK2, lw=0.8, ls=":")
    axes[0].set_title("Distortionless residual |hᴴd(θs) − 1| (test threshold 1e-6)", loc="left", fontsize=9)
    axes[1].axhline(1e-4, color=INK2, lw=0.8, ls=":")
    axes[1].set_title("Null response |hᴴd(θnull)| (test threshold 1e-4)", loc="left", fontsize=9)
    for ax in axes:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("frequency (Hz)")
        ax.set_xlim(F[1], F[-1])
    axes[0].legend(fontsize=6, ncol=2, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / "constraint_residuals.png")
    plt.close(fig)

    # WNG 表
    with open(OUT / "wng_table.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["freq_Hz"] + names)
        for fq in WNG_TABLE_FREQS:
            k = int(round(fq / F[1]))
            w.writerow([f"{fq:g}"] + [f"{wdb[p, k]:.3f}" for p in range(len(names))])
    with open(OUT / "wng_full.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["freq_Hz"] + names)
        for k in range(len(F)):
            w.writerow([f"{F[k]:g}"] + [f"{wdb[p, k]:.6f}" for p in range(len(names))])
    with open(OUT / "stage2_stats.json", "w") as f:
        json.dump(stats, f, indent=1)
    print(json.dumps({t: {n: {kk: v for kk, v in e.items() if kk not in ("bins_residual_ge_1e-6_Hz", "wng_dB_at")}
                          for n, e in stats[t].items()} for t in ("raw", "renorm")}, indent=1))
    print(open(OUT / "wng_table.csv").read())


if __name__ == "__main__":
    main()
