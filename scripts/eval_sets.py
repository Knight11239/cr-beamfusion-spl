"""在若干测试集（split:kappa）上评测一组方法（不调参），输出逐样本 / 汇总 csv、两套参考的表、目标直达失真的逐频点统计
与实测 WNG 分布。用于阶段 4 的 κ 扫描（用户 2026-11-01 决定）等。仿射模型的投影默认以 float64 计算。

  python scripts/eval_sets.py --methods ckpt:A=<pt> ckpt:M=<pt>:load=1e-4 --sets test_s1:0.5 test_s2:0.5 --out <dir>
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import load_config, resolve_device  # noqa: E402
from src.models.gtcrn_fusion import BeamBank  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage3_eval import table_s1  # noqa: E402
from stage4_eval import run  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-config", default="configs/data.yaml")
    ap.add_argument("--out", required=True)
    ap.add_argument("--methods", nargs="+", required=True, help="parse_methods 的方法说明（ckpt:NAME=PATH[:load=x]）")
    ap.add_argument("--sets", nargs="+", required=True, help="split:kappa")
    ap.add_argument("--max-items", type=int, default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--metric-workers", type=int, default=4)
    ap.add_argument("--no-wng", action="store_true")
    ap.add_argument("--val-mismatch", action="store_true", help="val 集加含失配训练同分布的增益 / 相位失配（每条样本固定种子）")
    ap.add_argument("--no-pq", action="store_true", help="不算 STOI / PESQ（只要 SI-SDR、SNR、shadow 指标时节省 CPU）")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--manifest", action="store_true", help="写 provenance.json（检查点 / 配置 / 元数据 / RIR 的 sha256）")
    ap.add_argument("--bss", default=None, choices=["dp", "ref"], help="另算 BSS-eval SIR（mir_eval），以 x_dp 或 x_ref 为目标")
    a = ap.parse_args()
    a.max_val = None
    cfg = load_config(a.data_config, a.set)
    device = resolve_device(a.device)
    bank = BeamBank.from_config(cfg)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.manifest:
        from src.evaluate import write_provenance
        write_provenance(out, cfg, a.data_config, a.methods, sorted({s.split(":")[0] for s in a.sets}))
    for spec in a.sets:
        split, kappa = spec.split(":")
        kappa = float(kappa)
        tag = split + (f"_kappa{kappa:g}" if kappa else "")
        _, summ = run(cfg, split, kappa, a.methods, bank, device, out, tag, a, wng=not a.no_wng, pq=not a.no_pq,
                      bss=a.bss or False, val_mismatch=a.val_mismatch and split == "val")
        table_s1(summ, out / f"table_{tag}_direct.md", tag="_dp")
        table_s1(summ, out / f"table_{tag}.md")


if __name__ == "__main__":
    main()
