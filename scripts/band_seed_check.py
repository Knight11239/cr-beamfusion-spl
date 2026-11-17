"""按种子核对“仿射在 MIRD 中频段输出误差更低”（用户 2026-11-16）。写 results/g2/seeds/band_consistency.md。

  python scripts/band_seed_check.py

每个种子、每个场景（mird_s1 / s2）、每个 T60–子阵条件、每个频带：A − B 的频带输出误差（dB，相对 Σ|X_dp|²，逐句后取条件内均值；
负 = 仿射误差更低）。只计条件层的符号，不给逐句区间（逐句不独立）。
种子 0：explore-band-mird（第一轮）；种子 1 / 2：band-diag-mird-s1 / -s2（种子 2 的 mird_s1 由 band-diag-mird-s2b 补跑）。
"""
import glob
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from explore2_stats import mird_meta  # noqa: E402

K = ROOT / "results" / "kaggle"
SRC = {0: ("explore-band-mird", "G2-Affine-mm", "G2-BeamFusion-mm"),
       1: ("band-diag-mird-s1", "G2-Affine-mm-s1", "G2-BeamFusion-mm-s1"),
       2: ("band-diag-mird-s2", "G2-Affine-mm-s2", "G2-BeamFusion-mm-s2")}
BANDS = ["0-500", "500-1000", "1000-2000", "2000-4000", "4000-5717", "5717-8000"]
MID = ["500-1000", "1000-2000", "2000-4000"]


def main():
    out = ROOT / "results" / "g2" / "seeds"
    out.mkdir(parents=True, exist_ok=True)
    L = ["# MIRD 分频带输出误差 A − B（按种子与条件；自动生成，`scripts/band_seed_check.py`）\n",
         "负值 = G2-Affine-mm 的频带输出误差低于 G2-BeamFusion-mm（dB，条件内均值）。“A 更低”列为 12 个（场景 × 条件）格中为负的个数。\n"]
    summary = {}
    for seed, (job, na, nb) in SRC.items():
        fs = glob.glob(str(K / job / "eval" / "*" / "band_diag.csv*")) + glob.glob(str(K / (job + "b") / "eval" / "*" / "band_diag.csv*"))
        if not fs:
            L.append(f"\n## 种子 {seed}：结果尚未收集（{job}）\n")
            continue
        d = pd.concat([pd.read_csv(f) for f in fs]).drop_duplicates(["method", "split", "id", "band"])
        if set(d.split) != {"mird_s1", "mird_s2"}:
            L.append(f"\n## 种子 {seed}：只有 {sorted(set(d.split))}，等待补跑\n")
            continue
        L.append(f"\n## 种子 {seed}（{job}）\n")
        L.append("| 场景 | 频带 | " + " | ".join(["160ms-A", "160ms-B", "360ms-A", "360ms-B", "610ms-A", "610ms-B"]) + " |")
        L.append("| --- | --- |" + " --- |" * 6)
        neg = {b: 0 for b in BANDS}
        for split in ("mird_s1", "mird_s2"):
            m = mird_meta(split)
            x = d[d.split == split]
            a = x[x.method == na].set_index(["id", "band"])
            b = x[x.method == nb].set_index(["id", "band"]).loc[a.index]
            df = (a.out_err_db - b.out_err_db).reset_index()
            df["cond"] = df.id.map(m.cond)
            pv = df.pivot_table(index="band", columns="cond", values="out_err_db", aggfunc="mean")
            for band in BANDS:
                vals = pv.loc[band]
                neg[band] += int((vals < 0).sum())
                L.append(f"| {split} | {band} | " + " | ".join(f"{vals[c]:+.2f}" for c in
                                                            ["160ms-A", "160ms-B", "360ms-A", "360ms-B", "610ms-A", "610ms-B"]) + " |")
        summary[seed] = neg
    L += ["\n## 汇总：A 的频带输出误差更低的格数（共 12 = 2 场景 × 6 条件）\n",
          "| 种子 | " + " | ".join(BANDS) + " |", "| --- |" + " --- |" * len(BANDS)]
    for seed, neg in summary.items():
        L.append(f"| {seed} | " + " | ".join(f"{neg[b]}/12" for b in BANDS) + " |")
    allmid = {b: all(summary[s][b] == 12 for s in summary) for b in MID}
    L.append(f"\n已收集种子：{sorted(summary)}。各中频段是否在全部已收集种子、全部 12 格中 A 更低：" +
             "，".join(f"{b} Hz：{'是' if v else '否'}" for b, v in allmid.items()) + "。")
    (out / "band_consistency.md").write_text("\n".join(L) + "\n")
    print("\n".join(L[-len(summary) - 6:]))


if __name__ == "__main__":
    main()
