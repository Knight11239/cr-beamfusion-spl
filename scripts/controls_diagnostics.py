"""F 的三项预定诊断所需的逐句冻结系数诊断（research/analysis_plan_controls.md §3；用户 2026-11-13）。

  python scripts/controls_diagnostics.py --methods ckpt:AffineMM=<pt> ckpt:free_complex_mm=<pt> \
      --sets test_s1:0 test_s1:1 test_s1:2 test_s2:0 test_s2:1 test_s2:2 --out <dir>

与 src/evaluate.py 完全相同的数据、阵列 STFT（含失配）、运行时字典与方法实现：由混合信号求一次系数 c（Ẑ = Σ_p c_p Z_p），
冻结后作用于只含目标直达分量的波束输出，得到 S_direct 与时域 y_direct（与 evaluate 的 target_direct_sdr_out 同一路径，但在 float64 中计算：
评测的 float32 路径与 float64 的相对差约 1e-6，会触发 frozen_diagnostics 逐元素恒等式检查在近零频点的容差；float32 与 float64 的 TD 差逐句记录在
td_float32_evaluator_minus_float64 列，须远小于分析代码的 1e-3 dB 一致性容差）；
再调用 research/control_analysis/diagnostics.frozen_diagnostics（标称响应偏差 D、增益、系数和残差、一般式 WNG 与退化计数、
冻结 TD，并用 Y_direct 独立重建有效滤波器 h = Σ conj(c_p) h_p 核对共轭约定）。输出 <dir>/<split>[_kappa<k>]_diagnostics.csv，
每行一个 (method, split, kappa, id)，供 research/control_analysis 的清单显式引用（不发现、不推断）。
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.control_analysis.diagnostics import frozen_diagnostics  # noqa: E402
from src.config import load_config, resolve_device  # noqa: E402
from src.dataset import BeamFusionDataset, collate_mixed  # noqa: E402
from src.evaluate import dictionary_gram, parse_methods  # noqa: E402
from src.models.gtcrn_fusion import BeamBank  # noqa: E402
from src.pipeline import array_stft  # noqa: E402
from src.stft import istft  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-config", default="configs/data.yaml")
    ap.add_argument("--methods", nargs="+", required=True)
    ap.add_argument("--sets", nargs="+", required=True)
    ap.add_argument("--max-items", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    cfg = load_config(a.data_config)
    device = resolve_device(a.device)
    bank = BeamBank.from_config(cfg).to(device)
    methods = parse_methods(a.methods, cfg, bank, device)
    H = bank.H.cpu().numpy().astype(np.complex128)  # 实际运行字典（complex64 → complex128）
    d_nom = dictionary_gram(cfg)[1].numpy()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for spec in a.sets:
        split, kappa = spec.split(":")
        kappa = float(kappa)
        tag = split + (f"_kappa{kappa:g}" if kappa else "")
        ds = BeamFusionDataset(cfg, split, kappa=kappa, mix_on="worker", return_components=True, max_items=a.max_items)
        dl = DataLoader(ds, batch_size=a.batch_size, shuffle=False, num_workers=a.num_workers, collate_fn=collate_mixed)
        by_index = {s["index"]: s for s in ds.scenes}
        rows, t0 = [], time.time()
        for b in dl:
            b = {k: v.to(device) for k, v in b.items()}
            N = b["mix"].shape[-1]
            Ym = array_stft(b["mix"], b["mm_gain_db"], b["mm_phase_deg"])
            Yd = array_stft(b["target_direct"], b["mm_gain_db"], b["mm_phase_deg"])
            Zm, Zd = bank(Ym), bank(Yd)
            rd = b["ref_direct"].cpu().numpy()
            Yd64 = Yd.to(torch.complex128)
            Zd64 = torch.einsum("pfm,bmft->bpft", torch.as_tensor(H).conj(), Yd64)  # 同一运行字典，float64
            for m in methods:
                coef = m.coef(Zm, Ym)  # 由混合信号求一次系数（与 evaluate 相同），之后冻结
                od32 = istft(m.apply(coef, Zd, Yd), N).cpu().numpy()  # evaluate 的 float32 冻结目标输出（只用于记录差异）
                c64 = coef.to(torch.complex128)
                Sd64 = (c64 * Zd64).sum(1)  # 冻结系数作用于目标直达分量，float64（避免 float32 舍入触发逐元素恒等式检查）
                od64 = istft(Sd64, N).cpu().numpy()
                for i in range(Zm.shape[0]):
                    sc = by_index[int(b["index"][i])]
                    r = frozen_diagnostics(H, c64[i].cpu().numpy(), d_nom, Zd64[i].cpu().numpy(), Sd64[i].cpu().numpy(),
                                           od64[i].astype(np.float64), rd[i].astype(np.float64), coefficient_origin="mixture",
                                           Y_direct=Yd64[i].cpu().numpy())
                    td32 = 10 * np.log10((np.dot(rd[i], rd[i]) + 1e-12) / (np.dot(od32[i] - rd[i], od32[i] - rd[i]) + 1e-12))
                    r["td_float32_evaluator_minus_float64"] = float(td32 - r["target_direct_sdr_out"])
                    rows.append({"method": m.name, "split": split, "kappa": kappa, "id": sc["id"], **r})
        with open(out / f"{tag}_diagnostics.csv", "w", newline="") as fo:
            w = csv.DictWriter(fo, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
            w.writeheader()
            w.writerows(rows)
        print(f"{tag}: {len(rows)} rows, {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
