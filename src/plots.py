"""绘图辅助（统一的颜色与样式）。"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# 类别色（dataviz 参考调色板，固定顺序，不循环）
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def setup_style():
    plt.rcParams.update({
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
        "axes.spines.right": False, "font.size": 9, "lines.linewidth": 1.5, "figure.dpi": 110,
        "legend.frameon": False,
    })


def polar_beampatterns(bp_db, names, freqs_hz, path, floor_db=-40.0, title=None):
    """bp_db: [P, n_freq, 361]（theta = 0..360°）。行 = 滤波器，列 = 频率。"""
    setup_style()
    P, nf, _ = bp_db.shape
    th = np.deg2rad(np.arange(361))
    fig, axes = plt.subplots(P, nf, figsize=(2.6 * nf, 2.6 * P), subplot_kw={"projection": "polar"})
    for j in range(nf):
        top = max(5.0, float(np.ceil(bp_db[:, j].max() / 5) * 5))
        for i in range(P):
            ax = axes[i, j]
            ax.plot(th, np.clip(bp_db[i, j], floor_db, None), color=SERIES[i], lw=1.3)
            ax.set_ylim(floor_db, top)
            ax.set_yticks([floor_db, -20, 0] + ([top] if top > 5 else []))
            ax.tick_params(labelsize=6, colors=INK2)
            ax.set_rlabel_position(100)
            ax.set_thetagrids(range(0, 360, 45), fontsize=6)
            ax.grid(color=GRID, lw=0.6)
            if i == 0:
                ax.set_title(f"{freqs_hz[j]:g} Hz", fontsize=9, pad=12)
            if j == 0:
                ax.text(-0.45, 0.5, names[i], transform=ax.transAxes, rotation=90, va="center", ha="center",
                        fontsize=9, color=INK)
    if title:
        fig.suptitle(title, fontsize=9, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def wng_curves(freqs, wng_db, names, path, ref_db=None, title=None):
    setup_style()
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    for i, n in enumerate(names):
        ax.plot(freqs[1:], wng_db[i, 1:], color=SERIES[i], label=n)
    if ref_db is not None:
        ax.axhline(ref_db, color=INK2, lw=0.8, ls=":")
        ax.text(freqs[1], ref_db + 0.6, f"10·log10(M) = {ref_db:.2f} dB", fontsize=7, color=INK2)
    ax.set_xscale("log")
    ax.set_xlim(freqs[1], freqs[-1])
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("WNG (dB)")
    if title:
        ax.set_title(title, loc="left", fontsize=9)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
