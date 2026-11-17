"""G2 三个种子的汇总（用户 2026-11-16）：写 results/g2/seeds/tables.md。

  python scripts/g2_seeds_report.py

MIRD：六条件等权的 A − B（正向条件数），以及各模型的跨种子范围；G2 仿真（κ = 0）与含失配验证集均值；−5 dB 推理时下限为事后探索。
只有三个种子，跨种子范围只作描述，不作统计结论。
"""
import glob
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from explore2_stats import mird_meta, paired  # noqa: E402

K = ROOT / "results" / "kaggle"
METRICS = [("d_si_sdr_dp", "ΔSI-SDR x_dp"), ("d_si_sdr", "ΔSI-SDR x_ref"), ("stoi_dp_out", "STOI"), ("pesq_dp_out", "PESQ"),
           ("target_direct_sdr_out", "TD SDR")]
# 种子 -> {(场景类, 下限): (路径模式, A 名, B 名)}；{s} 为 split
SRC = {
    0: {("mird", "none"): ("g2-eval-mird-a/eval/*/{s}_samples.csv*", "G2-Affine-mm", "g2-eval-bf-mird/eval/*/{s}_samples.csv*", "G2-BeamFusion-mm"),
        ("mird", "-5"): ("explore-w0-mird/eval/*/{s}_samples.csv*", "G2-Affine-mm(projW0=-5)", "explore2-e1-mird/eval/*/{s}_samples.csv*", "G2-BeamFusion-mm(postW0=-5)"),
        ("sim", "none"): ("g2-eval-sim-a/eval/*/{s}_samples.csv*", "G2-Affine-mm", "g2-eval-bf-sim/eval/*/{s}_samples.csv*", "G2-BeamFusion-mm"),
        ("val", "none"): ("explore-w0-val/eval/*/val_samples.csv*", "G2-Affine-mm", "explore2-e1-val/eval/*/val_samples.csv*", "G2-BeamFusion-mm"),
        ("val", "-5"): ("explore-w0-val/eval/*/val_samples.csv*", "G2-Affine-mm(projW0=-5)", "explore2-e1-val/eval/*/val_samples.csv*", "G2-BeamFusion-mm(postW0=-5)")},
}
for sd in (1, 2):
    j = f"g2-s{sd}-eval"
    a, b = f"G2-Affine-mm-s{sd}", f"G2-BeamFusion-mm-s{sd}"
    SRC[sd] = {("mird", "none"): (f"{j}-mird/eval/*/{{s}}_samples.csv*", a, f"{j}-mird/eval/*/{{s}}_samples.csv*", b),
               ("mird", "-5"): (f"{j}-mird/eval/*/{{s}}_samples.csv*", a + "(postW0=-5)", f"{j}-mird/eval/*/{{s}}_samples.csv*", b + "(postW0=-5)"),
               ("sim", "none"): (f"{j}-sim/eval/*/{{s}}_samples.csv*", a, f"{j}-sim/eval/*/{{s}}_samples.csv*", b),
               ("sim", "-5"): (f"{j}-sim/eval/*/{{s}}_samples.csv*", a + "(postW0=-5)", f"{j}-sim/eval/*/{{s}}_samples.csv*", b + "(postW0=-5)"),
               ("val", "none"): (f"{j}-val/eval/*/val_samples.csv*", a, f"{j}-val/eval/*/val_samples.csv*", b),
               ("val", "-5"): (f"{j}-val/eval/*/val_samples.csv*", a + "(postW0=-5)", f"{j}-val/eval/*/val_samples.csv*", b + "(postW0=-5)")}


def get(pattern, method):
    fs = sorted(glob.glob(str(K / pattern)))
    if not fs:
        return None
    d = pd.concat([pd.read_csv(f) for f in fs])
    d = d[d.method == method].set_index("id")
    assert len(d) and d.index.is_unique, (pattern, method)
    return d


def main():
    out = ROOT / "results" / "g2" / "seeds"
    out.mkdir(parents=True, exist_ok=True)
    L = ["# G2 三个种子（自动生成，`scripts/g2_seeds_report.py`）\n",
         "A = G2-Affine-mm，B = G2-BeamFusion-mm。MIRD 为六条件等权均值；−5 dB 为事后探索。只有三个种子，跨种子范围只作描述。\n",
         "## MIRD：A − B\n", "| 种子 | 下限 | 指标 | S1 A − B（正向 / 6） | S2 A − B（正向 / 6） |", "| --- | --- | --- | --- | --- |"]
    absm = {}
    for seed, src in SRC.items():
        for fl in ("none", "-5"):
            if ("mird", fl) not in src:
                continue
            pa, na, pb, nb = src[("mird", fl)]
            cells = {m: [] for m, _ in METRICS}
            ok = True
            for s in ("mird_s1", "mird_s2"):
                a, b = get(pa.format(s=s), na), get(pb.format(s=s), nb)
                if a is None or b is None:
                    ok = False
                    break
                meta = mird_meta(s)
                for m, _ in METRICS:
                    r = paired(a, b, m, meta)
                    cells[m].append(f"{r['mean']:+.3f} ({r['pos']})")
                if fl == "none":
                    for lab, x in (("A", a), ("B", b)):
                        v = x["d_si_sdr_dp"].groupby(meta.loc[x.index, "cond"]).mean().mean()
                        absm.setdefault((lab, s), {})[seed] = v
            if not ok:
                L.append(f"| {seed} | {fl} | （未收集） | | |")
                continue
            for m, lab in METRICS:
                L.append(f"| {seed} | {fl} | {lab} | {cells[m][0]} | {cells[m][1]} |")
    L += ["\n## MIRD 跨种子范围（ΔSI-SDR x_dp，无下限，六条件等权）\n", "| 模型 | 场景 | 各种子 | 范围（max − min） |", "| --- | --- | --- | --- |"]
    for (lab, s), d in sorted(absm.items()):
        vals = [d[k] for k in sorted(d)]
        L.append(f"| {lab} | {s} | " + " / ".join(f"{v:.2f}" for v in vals) + f" | {max(vals) - min(vals):.2f} |")
    L += ["\n## G2 仿真（κ = 0）与含失配验证集：ΔSI-SDR x_dp 均值\n", "| 种子 | 下限 | 数据 | A | B | A − B |", "| --- | --- | --- | --- | --- | --- |"]
    for seed, src in SRC.items():
        for (kind, fl), (pa, na, pb, nb) in src.items():
            if kind == "mird":
                continue
            for s in (("test_s1", "test_s2") if kind == "sim" else ("val",)):
                a, b = get(pa.format(s=s), na), get(pb.format(s=s), nb)
                if a is None or b is None:
                    L.append(f"| {seed} | {fl} | {s} | （未收集） | | |")
                    continue
                L.append(f"| {seed} | {fl} | {s} | {a.d_si_sdr_dp.mean():.3f} | {b.d_si_sdr_dp.mean():.3f} | "
                         f"{(a.d_si_sdr_dp - b.loc[a.index, 'd_si_sdr_dp']).mean():+.3f} |")
    (out / "tables.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
