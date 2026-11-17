"""Supplementary Figure S1: windowed oracle fits (8-microphone S2, target x_dp, kappa = 0, 100 matched utterances).

  python scripts/plot_oracle_figure.py [--scenario test_s2] [--target x_dp] [--out paper/figures/supplement/fig_oracle_s2_x_dp.pdf]

Reads results/oracle/oracle_summary.csv (written by scripts/oracle_window.py report): mean absolute SI-SDR and 95% interval
for the convex, real-affine and complex-affine fits per window length L, and the causal Affine-mm / BeamFusion-mm on the
same utterances (rows with part == "network").
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SETS = [("convex", "Convex", "#1f77b4", "o", "-"), ("real_affine", "Real affine", "#d95f02", "s", "--"),
        ("complex_affine", "Complex affine", "#1b9e77", "^", "-.")]
NETS = [("BeamFusion-mm", "BeamFusion-mm", "#555555", ":"), ("Affine-mm", "Affine-mm", "#e7298a", (0, (6, 2, 1, 2)))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="test_s2")
    ap.add_argument("--target", default="x_dp")
    ap.add_argument("--out", default=str(ROOT / "paper/figures/supplement/fig_oracle_s2_x_dp.pdf"))
    a = ap.parse_args()
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
                         "mathtext.fontset": "stix", "font.size": 8, "pdf.fonttype": 42, "ps.fonttype": 42})
    d = pd.read_csv(ROOT / "results/oracle/oracle_summary.csv")
    d = d[(d.scenario == a.scenario) & (d.target == a.target)]
    fit = d[d.part != "network"].copy()
    fit["L"] = fit["L"].astype(float).astype(int)
    net = d[d.part == "network"].set_index("set")
    tsub = {"x_dp": r"x_{\mathrm{dp}}", "x_ref": r"x_{\mathrm{ref}}"}[a.target]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.4), sharey=True)
    for ax, (part, title) in zip(axes, (("in_window", "(a) Fitted window"), ("out_of_window", "(b) Next-window transfer"))):
        for key, lab, col, mk, ls in SETS:
            s = fit[(fit.part == part) & (fit.set == key)].sort_values("L")
            ax.errorbar(s.L, s.si_sdr_mean, yerr=s.ci95, color=col, marker=mk, ls=ls, lw=1.2, ms=4, capsize=2, label=lab)
        for key, lab, col, ls in NETS:
            ax.axhline(net.loc[key, "si_sdr_mean"], color=col, ls=ls, lw=1.0, label=lab)
        ax.set_xscale("log", base=2)
        ax.set_xticks(sorted(fit.L.unique()))
        ax.set_xticklabels([str(v) for v in sorted(fit.L.unique())])
        ax.minorticks_off()
        ax.set_xlabel(r"Window length $L$ (frames)")
        ax.set_title(title)
        ax.grid(True, color="#dddddd", lw=0.5)
    axes[0].set_ylabel(rf"SI-SDR (dB), ${tsub}$ reference")
    sc = a.scenario.replace("test_", "").upper()
    fig.suptitle(rf"{sc}, $\kappa = 0$; 100 matched utterances", y=0.97)
    h, lb = axes[0].get_legend_handles_labels()
    fig.legend(h, lb, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.08, 1, 0.97), h_pad=0.2)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
