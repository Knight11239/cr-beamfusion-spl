"""阶段 3 评测流程：验证集调参 → 测试集评测 → 汇总表（与 BeamFusion 论文表 1 对应）。

1. ACC 的 mu 在验证集上从 {0.001, 0.003, 0.01, 0.03, 0.1, 0.3} 中选（按平均 ΔSI-SDR；CLAUDE.md 的网格向下扩展，用户已确认）；
2. 掩码式 MVDR 的对角加载因子在验证集上从 {1e-4, 1e-3, 1e-2, 1e-1} 中选；
3. 在 test_s1 / test_s2（kappa=0）上评测：各固定波束、ACC（选定 mu）、ACC(mu=0)（均匀权重、不更新，
   仅作参照行，不参与选 mu；用户确认）、BeamFusion、掩码式 MVDR（选定加载）；
4. 输出：<out>/{tune_*,test_*}_{samples,summary}.csv、<out>/selection.json、
   <out>/table_<split>.md（参考 x_ref）与 <out>/table_<split>_direct.md（直达声参考，见 src/evaluate.py）。
选择准则（ACC 的 mu、MVDR 的加载）：验证集上的平均 ΔSI-SDR，--select-ref 指定参考：
  direct（默认，CLAUDE.md 3.3 / 5.4 修改后）= 直达声参考 d_si_sdr_dp；early = x_ref 的 d_si_sdr（crbf-eval v3 使用）。
不在测试集上调参（CLAUDE.md 第 9 节）。

复用已有的测试集结果（避免重复计算）：--no-fixed 不评测固定波束；--acc-test-done 0.001 0 表示这些 mu 的 ACC
已在测试集上评测过（如 crbf-eval v3），选出的 mu 若在其中就不再评测，ACC(mu=0) 同理。
模型的方法名用 --bf-name / --mvdr-name 区分（如 BeamFusion-direct）。

用法：python scripts/stage3_eval.py --data-config configs/data.yaml --out results/stage3/eval \
          [--bf-ckpt runs/beamfusion/best.pt] [--mvdr-ckpt runs/mask_mvdr/best.pt] [--max-val N] [--max-test N] [--bss]
          [--acc-selection results/kaggle/val-acc-tuning/eval/val_acc_tuning_selection.json]
--acc-selection：复用已在完整验证集上完成的 ACC 网格搜索结果（同一网格、同一准则），避免重复计算。
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import load_config, resolve_device  # noqa: E402
from src.evaluate import evaluate, parse_methods, summarize, write_csv, write_freq_stats  # noqa: E402
from src.models.gtcrn_fusion import BeamBank  # noqa: E402

MUS = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3]  # 用户确认：向下扩展到 0.001（仍只在验证集上选）
LOADS = [1e-4, 1e-3, 1e-2, 1e-1]


def run(cfg, split, specs, bank, device, out, tag, a, pq=True, bss=False, max_items=None):
    methods = parse_methods(specs, cfg, bank, device)
    t0 = time.time()
    fstats = {}
    rows = evaluate(cfg, split, methods, bank, device, max_items=max_items, batch_size=a.batch_size,
                    num_workers=a.num_workers, metric_workers=a.metric_workers, do_bss=bss, do_pq=pq,
                    freq_stats=fstats)
    write_csv(rows, out / f"{tag}_samples.csv")
    write_freq_stats(fstats, out / f"{tag}_target_distortion_freq.csv")
    summ = summarize(rows, ("method", "t60")) + summarize(rows, ("method",))
    write_csv(summ, out / f"{tag}_summary.csv")
    print(f"[{tag}] {len(rows)} rows in {(time.time() - t0) / 60:.1f} min", flush=True)
    return rows, summ


def best_of(summ, prefix, key="d_si_sdr"):
    cands = [s for s in summ if "t60" not in s and s["method"].startswith(prefix)]
    return max(cands, key=lambda s: s[key]), {s["method"]: s[key] for s in cands}


def table_s1(summ, path, tag=""):
    """与 BeamFusion 论文表 1 相同的列：ΔSNR、STOI、ΔSI-SDR，按 T60 分组（另附 shadow SIR）。
    tag="" 用参考 x_ref，tag="_dp" 用直达声参考（shadow SIR 与参考无关）。"""
    t60s = sorted({s["t60"] for s in summ if "t60" in s})
    methods = list(dict.fromkeys(s["method"] for s in summ))
    lines = ["| Method | " + " | ".join(f"T60={t:g}s ΔSNR | STOI | ΔSI-SDR | SIR(shadow) | 目标直达失真 SDR" for t in t60s)
             + " |", "| --- |" + " --- | --- | --- | --- | --- |" * len(t60s)]
    noisy = []
    for t in t60s:
        s = next(x for x in summ if x.get("t60") == t)
        noisy.append(f"– | {s.get(f'stoi{tag}_out', float('nan')) - s.get(f'd_stoi{tag}', float('nan')):.2f} | – | "
                     f"{s['sir_shadow_out'] - s['d_sir_shadow']:.2f} | –")
    lines.append("| Noisy (center mic) | " + " | ".join(noisy) + " |")
    for m in methods:
        cells = []
        for t in t60s:
            s = next(x for x in summ if x.get("t60") == t and x["method"] == m)
            cells.append(f"{s[f'd_snr{tag}']:.2f} | {s.get(f'stoi{tag}_out', float('nan')):.2f} | "
                         f"{s[f'd_si_sdr{tag}']:.2f}±{s[f'd_si_sdr{tag}_ci95']:.2f} | {s['sir_shadow_out']:.2f} | "
                         f"{s.get('target_direct_sdr_out', float('nan')):.2f}")
        lines.append(f"| {m} | " + " | ".join(cells) + " |")
    Path(path).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-config", default="configs/data.yaml")
    ap.add_argument("--out", required=True)
    ap.add_argument("--bf-ckpt", default=None)
    ap.add_argument("--mvdr-ckpt", default=None)
    ap.add_argument("--max-val", type=int, default=None)
    ap.add_argument("--max-test", type=int, default=None)
    ap.add_argument("--splits", nargs="+", default=["test_s1", "test_s2"])
    ap.add_argument("--bss", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--metric-workers", type=int, default=4)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--acc-selection", default=None, help="已有的验证集 ACC 选择结果（json），跳过 ACC 调参")
    ap.add_argument("--select-ref", choices=["direct", "early"], default="direct",
                    help="选择 mu / 加载所用的参考：direct = 直达声（d_si_sdr_dp），early = x_ref（d_si_sdr）")
    ap.add_argument("--no-fixed", action="store_true", help="测试集上不评测固定波束（已有结果）")
    ap.add_argument("--acc-test-done", nargs="*", type=float, default=[], help="测试集上已评测过的 ACC mu")
    ap.add_argument("--bf-name", default="BeamFusion")
    ap.add_argument("--mvdr-name", default="MaskMVDR")
    a = ap.parse_args()
    crit = "d_si_sdr_dp" if a.select_ref == "direct" else "d_si_sdr"
    cfg = load_config(a.data_config, a.set)
    device = resolve_device(a.device or cfg["device"])
    bank = BeamBank.from_config(cfg).to(device)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    sel = {"criterion": f"mean {crit} on val", "select_ref": a.select_ref, "n_val": a.max_val or "all"}

    if a.acc_selection:
        prev = json.loads(Path(a.acc_selection).read_text())
        assert prev["split"] == "val" and prev["criterion"] == crit, (prev["criterion"], crit)
        assert sorted(float(k.split("=")[1].rstrip(")")) for k in prev["candidates"]) == sorted(MUS)
        best_m = prev["best_method"]
        sel["acc"] = {"best": best_m, "mu": float(best_m.split("=")[1].rstrip(")")), "candidates": prev["candidates"],
                      "source": a.acc_selection}
    else:
        _, s_acc = run(cfg, "val", [f"acc:mu={m}" for m in MUS], bank, device, out, "tune_acc", a, pq=False,
                       max_items=a.max_val)
        best, cands = best_of(s_acc, "ACC", crit)
        sel["acc"] = {"best": best["method"], "mu": float(best["method"].split("=")[1].rstrip(")")),
                      "candidates": cands}
    specs = [] if a.no_fixed else ["fixed"]
    for mu in (sel["acc"]["mu"], 0.0):  # mu=0：均匀权重参照行，不参与选择
        if mu not in a.acc_test_done:
            specs.append(f"acc:mu={mu:g}")
    sel["acc"]["test_reused"] = [m for m in (sel["acc"]["mu"], 0.0) if m in a.acc_test_done]
    if a.bf_ckpt:
        specs.append(f"ckpt:{a.bf_name}={a.bf_ckpt}")
    if a.mvdr_ckpt:
        _, s_mv = run(cfg, "val", [f"ckpt:{a.mvdr_name}={a.mvdr_ckpt}:load={x}" for x in LOADS], bank, device, out,
                      "tune_mvdr", a, pq=False, max_items=a.max_val)
        best, cands = best_of(s_mv, a.mvdr_name, crit)
        load = float(best["method"].split("load=")[1].rstrip(")"))
        sel["mask_mvdr"] = {"best": best["method"], "load": load, "candidates": cands}
        specs.append(f"ckpt:{a.mvdr_name}={a.mvdr_ckpt}:load={load}")
    for ck, key in ((a.bf_ckpt, "beamfusion_ckpt"), (a.mvdr_ckpt, "mask_mvdr_ckpt")):
        if ck:
            c = torch.load(ck, map_location="cpu", weights_only=False)
            sel[key] = {"path": ck, "epoch": c.get("epoch"), "val_loss": c.get("val_loss"),
                        "git_commit": c.get("git_commit")}
    (out / "selection.json").write_text(json.dumps(sel, indent=1))
    print("selection:", json.dumps(sel))
    for split in a.splits:
        _, summ = run(cfg, split, specs, bank, device, out, f"test_{split}", a, bss=a.bss, max_items=a.max_test)
        table_s1(summ, out / f"table_{split}_direct.md", tag="_dp")
        table_s1(summ, out / f"table_{split}.md")


if __name__ == "__main__":
    main()
