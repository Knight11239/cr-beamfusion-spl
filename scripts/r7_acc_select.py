"""R7：ACC 的 μ 在 G2 仿真验证集上按 x_dp 选择，边界时按用户规则逐步向下扩展网格（2026-11-06）。

  python scripts/r7_acc_select.py --out results/mird/r7/acc_select.json

基础网格 {1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3}（results/kaggle/r7-g2sim-s1/eval/g2_val）；若选中最小值，依次加入 3e-4、1e-4、3e-5
（results/kaggle/r7-acc-ext-g2/eval/g2_val_ext），每加一个就重新选，直到选中的 μ 不在下边界或扩展值用完（则报告仍在边界）。
μ = 0（均匀凸权重固定不变）只作为极限参考，不参与选择。
"""
import argparse
import csv
import glob
import json
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parents[1] / "results" / "kaggle"
BASE = [1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3]
EXT = [3e-4, 1e-4, 3e-5]


def val_means():
    d = {}
    for g in ("r7-g2sim-s1/eval/g2_val", "r7-acc-ext-g2/eval/g2_val_ext"):
        for f in glob.glob(str(R / g / "*_samples.csv")):
            for r in csv.DictReader(open(f)):
                if r["method"].startswith("ACC(mu="):
                    mu = float(r["method"][7:-1])
                    d.setdefault(mu, []).append(float(r["d_si_sdr_dp"]))
    return {mu: (float(np.mean(v)), len(v)) for mu, v in d.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/mird/r7/acc_select.json")
    a = ap.parse_args()
    m = val_means()
    grid, steps = [mu for mu in BASE if mu in m], []
    for nxt in [None] + EXT:
        if nxt is not None:
            if nxt not in m:
                steps.append({"added": nxt, "error": "not evaluated"})
                break
            grid.append(nxt)
        best = max(grid, key=lambda mu: m[mu][0])
        steps.append({"grid": sorted(grid), "selected": best, "at_lower_edge": best == min(grid)})
        if best != min(grid):
            break
    res = {"criterion": "mean ΔSI-SDR (x_dp) on G2 simulated val (500)", "val_mean_db": {str(k): v[0] for k, v in sorted(m.items())},
           "n": {str(k): v[1] for k, v in sorted(m.items())}, "steps": steps, "selected": steps[-1].get("selected"),
           "at_lower_edge": steps[-1].get("at_lower_edge"), "mu0_reference_db": m.get(0.0, (None,))[0]}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
