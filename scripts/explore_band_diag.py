"""事后探索性分析（用户 2026-11-14，不属于事前计划）：按频带的 WNG 与误差诊断。

  python scripts/explore_band_diag.py --data-config configs/data_mird.yaml --methods ckpt:A=<pt> ckpt:B=<pt> \
      --sets mird_s1:0 mird_s2:0 --out <dir>

与 src/evaluate.py 相同的数据、阵列 STFT、字典与方法实现：由混合信号求一次系数，冻结后作用于各分量（shadow filtering）。
每条语音、每个方法、每个频带输出一行：
  - WNG（标称导向向量，一般式 abs(Σc)² / vᴴKv；所有时频点）：中位数、低于 0 dB 与低于 −10 dB 的比例；
  - 只含目标直达分量时的失真：10 log10(Σ|S_d − X_dp|² / Σ|X_dp|²)（越低越好）；
  - 输出误差：10 log10(Σ|Ŝ − X_dp|² / Σ|X_dp|²)，以及其中完整目标分量（含反射）、干扰残余、噪声残余各自相对 Σ|X_dp|² 的能量（dB）。
2026-11-15 补充（事后探索第二轮，用户 2026-11-15）：
  - 推理时投影的激活率：s < 1 的时频点比例，以及按 |X_dp|² 与 |Z_D(mix)|² 加权的比例（PostProjMethod；其他方法为 0）；
  - 未投影系数的 WNG 低于各候选 W0（−15/−10/−5/0/3 dB）的比例（不加权与 |X_dp|² 加权），即在该 W0 下投影会激活的比例；
  - 各能量的线性绝对值（ref_e、out_e、td_e、tfull_e、interf_e、noise_e），使跨频带可加的误差能量账本成为可能
    （out_e = tfull_e + interf_e + noise_e + 交叉项）；
  - --val-mismatch（G2 含失配验证集），--manifest 写检查点 / 配置 / 场景元数据 / RIR 文件的 sha256。
频带（Hz）：(0, 500]、(500, 1000]、(1000, 2000]、(2000, 4000]、(4000, 5717]、(5717, 8000]（5717 Hz 为 G2 的空间混叠频率 c/(2δ)）。
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
from src.beamformers import stft_freqs  # noqa: E402
from src.config import load_config, resolve_device  # noqa: E402
from src.dataset import BeamFusionDataset, collate_mixed  # noqa: E402
from src.config import meta_path, rir_dir  # noqa: E402
from src.evaluate import _sha256, coef_wng_db, dictionary_gram, parse_methods  # noqa: E402
from src.models.gtcrn_fusion import BeamBank  # noqa: E402
from src.pipeline import array_stft  # noqa: E402
from src.stft import stft  # noqa: E402

W0_GRID = (-15, -10, -5, 0, 3)
BANDS = [(0, 500), (500, 1000), (1000, 2000), (2000, 4000), (4000, 5717), (5717, 8001)]


def db(x):
    return float(10 * np.log10(max(x, 1e-30)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-config", default="configs/data.yaml")
    ap.add_argument("--methods", nargs="+", required=True)
    ap.add_argument("--sets", nargs="+", required=True)
    ap.add_argument("--max-items", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--out", required=True)
    ap.add_argument("--val-mismatch", action="store_true")
    ap.add_argument("--manifest", action="store_true", help="写 provenance.json（各输入文件 sha256）")
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    cfg = load_config(a.data_config)
    device = resolve_device("cpu")
    bank = BeamBank.from_config(cfg)
    methods = parse_methods(a.methods, cfg, bank, device)
    K, d_nom = dictionary_gram(cfg)
    F = stft_freqs(cfg["signal"]["fs"], 512)
    bidx = [np.where((F > lo) & (F <= hi))[0] for lo, hi in BANDS]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    prov = {"data_config": {"path": a.data_config, "sha256": _sha256(a.data_config)},
            "methods": {m.name: getattr(m, "ckpt_sha256", getattr(getattr(m, "base", None), "ckpt_sha256", None)) for m in methods}}
    for spec in a.sets:
        split, kappa = spec.split(":")
        kappa = float(kappa)
        ds = BeamFusionDataset(cfg, split, kappa=kappa, mix_on="worker", return_components=True, max_items=a.max_items,
                               val_mismatch=a.val_mismatch and split == "val")
        if a.manifest:
            prov.setdefault("meta", {})[split] = {"path": str(meta_path(cfg, split)), "sha256": _sha256(meta_path(cfg, split))}
            rd = rir_dir(cfg, split, 0.0)
            prov.setdefault("rir", {})[split] = {f: _sha256(rd / f) for f in sorted({sc["rir_file"] for sc in ds.scenes})}
        dl = DataLoader(ds, batch_size=a.batch_size, shuffle=False, num_workers=a.num_workers, collate_fn=collate_mixed)
        by_index = {s["index"]: s for s in ds.scenes}
        t0 = time.time()
        for b in dl:
            comps = {"mix": b["mix"], "target": b["target"], "interf": b["interf"], "noise": b["noise"], "target_direct": b["target_direct"]}
            Ys = {c: array_stft(v, b["mm_gain_db"], b["mm_phase_deg"]) for c, v in comps.items()}
            Zs = {c: bank(Y) for c, Y in Ys.items()}
            X = stft(b["ref_direct"]).to(torch.complex128)  # [B, F, T]
            Xp = (X.abs() ** 2).numpy()  # [B, F, T]
            Xe = Xp.sum(-1)  # [B, F]
            Dp = (Zs["mix"][:, 0].abs() ** 2).double().numpy()  # |Z_D(mix)|²
            for m in methods:
                coef = m.coef(Zs["mix"], Ys["mix"])
                w = coef_wng_db(m, coef, K, d_nom).numpy()  # [B, F, T]
                ls = getattr(m, "last_s", None)
                act = np.zeros_like(w, dtype=bool) if ls is None or not torch.is_tensor(ls) else (ls < 1 - 1e-9).numpy()
                S = {c: m.apply(coef, Zs[c], Ys[c]).to(torch.complex128) for c in comps}
                e_out = ((S["mix"] - X).abs() ** 2).sum(-1).numpy()
                e_td = ((S["target_direct"] - X).abs() ** 2).sum(-1).numpy()
                e_t = ((S["target"] - X).abs() ** 2).sum(-1).numpy()
                p_i = (S["interf"].abs() ** 2).sum(-1).numpy()
                p_n = (S["noise"].abs() ** 2).sum(-1).numpy()
                for i in range(X.shape[0]):
                    sc = by_index[int(b["index"][i])]
                    for (lo, hi), ix in zip(BANDS, bidx):
                        wb = w[i, ix].ravel()
                        ref = Xe[i, ix].sum()
                        wb2, xb, db2, ab = w[i, ix], Xp[i, ix], Dp[i, ix], act[i, ix]
                        extra = {"proj_active_frac": float(ab.mean()), "proj_active_frac_wx": float((xb * ab).sum() / xb.sum()),
                                 "proj_active_frac_wmix": float((db2 * ab).sum() / db2.sum())}
                        for w0 in W0_GRID:
                            lo_ = wb2 < w0 - 1e-6  # 容差：投影后恰在边界上的点不计入
                            extra[f"frac_below_{w0}"] = float(lo_.mean())
                            extra[f"frac_below_{w0}_wx"] = float((xb * lo_).sum() / xb.sum())
                        extra.update({"ref_e": float(ref), "out_e": float(e_out[i, ix].sum()), "td_e": float(e_td[i, ix].sum()),
                                      "tfull_e": float(e_t[i, ix].sum()), "interf_e": float(p_i[i, ix].sum()),
                                      "noise_e": float(p_n[i, ix].sum())})
                        rows.append({"method": m.name, "split": split, "kappa": kappa, "id": sc["id"],
                                     "rt60": sc.get("rt60", sc.get("t60_target")), "subarray": sc.get("subarray", ""),
                                     "band": f"{lo}-{hi if hi < 8001 else 8000}", "wng_median_db": float(np.median(wb)),
                                     "wng_frac_below_0db": float((wb < 0).mean()), "wng_frac_below_m10db": float((wb < -10).mean()),
                                     "target_direct_err_db": db(e_td[i, ix].sum() / ref), "out_err_db": db(e_out[i, ix].sum() / ref),
                                     "target_full_err_db": db(e_t[i, ix].sum() / ref), "interf_resid_db": db(p_i[i, ix].sum() / ref),
                                     "noise_resid_db": db(p_n[i, ix].sum() / ref), **extra})
        print(f"{spec}: {(time.time() - t0) / 60:.1f} min", flush=True)
    with open(out / "band_diag.csv", "w", newline="") as fo:
        wr = csv.DictWriter(fo, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    if a.manifest:
        import json
        (out / "provenance.json").write_text(json.dumps(prov, indent=1))


if __name__ == "__main__":
    main()
