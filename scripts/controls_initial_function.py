"""新对照资格记录：初始功能（用户 2026-11-13；research/analysis_plan_controls.md §2 第 2 项）。

  python scripts/controls_initial_function.py --data-config configs/data_dev.yaml --n 8 --out results/controls/initial_function.json

用训练时的初始化（同一 train.seed 与 seed_key 的种子设置方式不在此复现，这里固定 torch.manual_seed(1234)）构建 A、R、C、F，
在开发数据验证集前 n 条（含失配验证同分布的失配）上前向一次，统计初始有效系数 c（Ẑ = Σ_p c_p Z_p，第 0 个为 h_D）：
c_D 与 1 的偏差、其余 c_p 的幅度、Σc − 1，以及初始输出相对 h_D 输出的偏差 ‖Ẑ − Z_D‖ / ‖Z_D‖。
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import load_config  # noqa: E402
from src.dataset import BeamFusionDataset, collate_mixed  # noqa: E402
from src.models.gtcrn_fusion import BeamBank, build_model, count_params  # noqa: E402
from src.pipeline import array_stft  # noqa: E402

MODELS = {"A": "configs/model_affine.yaml", "R": "configs/model_affine_real.yaml",
          "C": "configs/model_convex_samehead.yaml", "F": "configs/model_free_complex.yaml",
          "B": "configs/model_beamfusion.yaml"}


def coef(kind, o):
    if kind == "affine":
        sw = o["s"][:, None] * o["w"]
        return torch.cat([1 - sw.sum(1, keepdim=True), sw], 1)
    return o["coef"] if "coef" in o else o["alpha"]


def q(x):
    x = np.asarray(x, float).ravel()
    return {"mean": float(x.mean()), "p50": float(np.percentile(x, 50)), "p95": float(np.percentile(x, 95)), "max": float(x.max())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-config", default="configs/data_dev.yaml")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    cfg = load_config(a.data_config, ["train_mismatch.enabled=true", "train_mismatch.val=true"])
    bank = BeamBank.from_config(cfg)
    ds = BeamFusionDataset(cfg, "val", kappa=0.0, mix_on="worker", return_components=False, max_items=a.n, val_mismatch=True)
    b = next(iter(DataLoader(ds, batch_size=a.n, collate_fn=collate_mixed)))
    Z = bank(array_stft(b["mix"], b["mm_gain_db"], b["mm_phase_deg"]))
    res = {"data_config": a.data_config, "n_utterances": a.n, "note": "validation utterances with mismatched-validation gain/phase; f > 0 bins"}
    for k, path in MODELS.items():
        torch.manual_seed(1234)
        mc = yaml.safe_load(open(ROOT / path))
        m = build_model(mc, cfg, bank).eval()
        o = m(Z)
        c = coef(mc["kind"], o)[:, :, 1:].to(torch.complex128)  # 跳过 f = 0
        S, ZD = o["S"][:, 1:], Z[:, 0, 1:]
        res[k] = {"config": path, "trainable_params": count_params(m), "n_out_channels": int(m.backbone.output_layer.conv.out_channels),
                  "last_bn": not isinstance(m.backbone.output_layer.bn, torch.nn.Identity),
                  "abs_cD_minus_1": q((c[:, 0] - 1).abs()), "abs_c_others": q(c[:, 1:].abs()),
                  "abs_sum_c_minus_1": q((c.sum(1) - 1).abs()),
                  "rel_output_dev_from_hD": float(((S - ZD).abs() ** 2).sum() / (ZD.abs() ** 2).sum()) ** 0.5}
        print(k, res[k]["trainable_params"], res[k]["n_out_channels"], {x: round(res[k][x]["p95"], 4) for x in ("abs_cD_minus_1", "abs_c_others", "abs_sum_c_minus_1")},
              round(res[k]["rel_output_dev_from_hD"], 4))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
