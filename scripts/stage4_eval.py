"""阶段 4 评测（H3 的第一次检验，用户 2026-10-31 要求）：S1、S2、S3-1、S3-2（κ = 1）× 4 个模型，两套参考 + 实测 WNG 分布。

1. 掩码式 MVDR（直达声目标）的对角加载：在验证集上按直达声指标（d_si_sdr_dp）选择。
   - 复用阶段 3（crbf-eval-direct）在 {1e-4, 1e-3, 1e-2, 1e-1} 上的验证集结果（--prev-mvdr-selection）；
   - 重新评测 1e-4 作为锚点，核对与之前的结果一致（同一数据、同一代码路径），以确认复用有效；
   - 评测 1e-5、1e-6；若最优值仍在网格下边界，每次再往下加一个十倍，直到最优值不在边界或到达 --min-load
     （MaskMVDR.filters 本身有 1e-10 的绝对加载，再小没有意义）。
2. 在各测试集上评测：Affine、Affine+Proj（hard，W0 = 0 dB）、BeamFusion（直达声目标）、掩码式 MVDR（选定加载）。
   每个测试集输出逐样本 / 汇总 csv、两套参考的表、目标直达失真的逐频点统计，以及 CLAUDE.md 7.2 的实测 WNG 分布
   （所有样本、所有时频点，f > 0：总直方图、逐频点分位数、汇总）。

  python scripts/stage4_eval.py --out results/stage4/eval --models A=... P=... --bf-ckpt ... --mvdr-ckpt ... \
      --prev-mvdr-selection results/kaggle/stage3-eval-direct/eval/stage3_direct/selection.json \
      --sets test_s1:0 test_s2:0 test_s1:1 test_s2:1
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import load_config, resolve_device  # noqa: E402
from src.evaluate import (evaluate, parse_methods, summarize, write_csv, write_freq_stats, write_w0_stats,  # noqa: E402
                          write_wng_stats)
from src.models.gtcrn_fusion import BeamBank  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage3_eval import table_s1  # noqa: E402

CRIT = "d_si_sdr_dp"


def run(cfg, split, kappa, specs, bank, device, out, tag, a, pq=True, wng=False, val_mismatch=False, bss=False):
    methods = parse_methods(specs, cfg, bank, device)
    t0 = time.time()
    fstats, wstats, w0stats = {}, ({} if wng else None), ({} if wng else None)
    rows = evaluate(cfg, split, methods, bank, device, kappa=kappa, max_items=a.max_val if tag.startswith("tune") else a.max_items,
                    batch_size=a.batch_size, num_workers=a.num_workers, metric_workers=a.metric_workers, do_pq=pq,
                    freq_stats=fstats, wng_stats=wstats, val_mismatch=val_mismatch, w0_stats=w0stats,
                    do_bss=bss)
    write_csv(rows, out / f"{tag}_samples.csv")
    write_freq_stats(fstats, out / f"{tag}_target_distortion_freq.csv")
    summ = summarize(rows, ("method", "t60")) + summarize(rows, ("method",))
    write_csv(summ, out / f"{tag}_summary.csv")
    if wng:
        write_wng_stats(wstats, str(out / tag))
        if w0stats:  # 有可学习 W0 的模型时：W0 分布（逐频点分位数、处于下界 -15 dB 的比例）
            write_w0_stats(w0stats, str(out / tag))
    print(f"[{tag}] {len(rows)} rows in {(time.time() - t0) / 60:.1f} min", flush=True)
    return rows, summ


def mean_by_method(summ, key=CRIT):
    return {s["method"]: s[key] for s in summ if "t60" not in s}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-config", default="configs/data.yaml")
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", nargs="+", required=True, help="NAME=PATH（仿射模型等，无需调参）")
    ap.add_argument("--bf-ckpt", required=True)
    ap.add_argument("--mvdr-ckpt", required=True)
    ap.add_argument("--mvdr-name", default="MaskMVDR-direct")
    ap.add_argument("--prev-mvdr-selection", default=None)
    ap.add_argument("--new-loads", nargs="+", type=float, default=[1e-5, 1e-6])
    ap.add_argument("--anchor-load", type=float, default=1e-4)
    ap.add_argument("--min-load", type=float, default=1e-9)
    ap.add_argument("--sets", nargs="+", default=["test_s1:0", "test_s2:0", "test_s1:1", "test_s2:1"])
    ap.add_argument("--max-val", type=int, default=None)
    ap.add_argument("--max-items", type=int, default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--metric-workers", type=int, default=4)
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    cfg = load_config(a.data_config, a.set)
    device = resolve_device(a.device)
    bank = BeamBank.from_config(cfg)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    # 1. MVDR 加载因子（验证集，直达声指标）
    cands = {}
    sel = {"criterion": f"mean {CRIT} on val", "n_val": a.max_val or "all"}
    if a.prev_mvdr_selection:
        prev = json.loads(Path(a.prev_mvdr_selection).read_text())["mask_mvdr"]["candidates"]
        cands.update({float(k.split("load=")[1].rstrip(")")): v for k, v in prev.items()})
        sel["reused"] = {"source": a.prev_mvdr_selection, "candidates": prev}
    loads = [a.anchor_load] + [x for x in a.new_loads if x not in cands or x == a.anchor_load]
    tried = []
    while True:
        todo = [x for x in loads if x not in tried]
        if not todo:
            break
        _, s = run(cfg, "val", 0.0, [f"ckpt:{a.mvdr_name}={a.mvdr_ckpt}:load={x:g}" for x in todo], bank, device, out,
                   f"tune_mvdr_{len(tried)}", a, pq=False)
        for name, v in mean_by_method(s).items():
            x = float(name.split("load=")[1].rstrip(")"))
            if x == a.anchor_load and x in cands and a.max_val is None:
                sel["anchor_check"] = {"load": x, "previous": cands[x], "now": v, "abs_diff": abs(v - cands[x])}
                assert abs(v - cands[x]) < 1e-3, sel["anchor_check"]  # 复用的前提：同一数据与代码路径给出相同结果
            cands[x] = v
        tried += todo
        best = max(cands, key=cands.get)
        if best == min(cands) and best / 10 >= a.min_load:
            loads.append(best / 10)  # 最优值在下边界：继续往下扩展
    best = max(cands, key=cands.get)
    sel["mask_mvdr"] = {"load": best, "candidates": {f"{k:g}": cands[k] for k in sorted(cands)},
                        "at_lower_boundary": best == min(cands)}
    (out / "selection.json").write_text(json.dumps(sel, indent=1))
    print("selection:", json.dumps(sel), flush=True)

    # 2. 测试集
    specs = [f"ckpt:{m}" for m in a.models] + [f"ckpt:BeamFusion-direct={a.bf_ckpt}",
                                               f"ckpt:{a.mvdr_name}={a.mvdr_ckpt}:load={best:g}"]
    for spec in a.sets:
        split, kappa = spec.split(":")
        kappa = float(kappa)
        tag = split + (f"_kappa{kappa:g}" if kappa else "")
        _, summ = run(cfg, split, kappa, specs, bank, device, out, tag, a, wng=True)
        table_s1(summ, out / f"table_{tag}_direct.md", tag="_dp")
        table_s1(summ, out / f"table_{tag}.md")


if __name__ == "__main__":
    main()
