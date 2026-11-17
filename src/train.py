"""训练脚本（BeamFusion / 掩码式 MVDR；阶段 4 的变体复用同一流程）。

用法：
  python -m src.train --data-config configs/data_dev.yaml --model-config configs/model_beamfusion.yaml \
      --out runs/bf_smoke [--epochs 3] [--device cpu|cuda:0] [--resume-from DIR] [--max-hours 11]

- 每个 epoch 结束保存 last.pt（含模型、优化器、调度器、历史、随机数状态），验证损失最好时另存 best.pt；
- 若 --out 下已有 last.pt 则自动续跑；--resume-from DIR 会先把 DIR 下的检查点与日志复制到 --out（用于 Kaggle
  把上一次任务的输出作为输入来跨任务续跑）；
- --max-hours：开始新 epoch 前若预计会超时则正常退出（检查点已保存），下一次任务继续；
- 训练顺序由 (seed, epoch) 决定，续跑与不中断训练的数据顺序相同；每个 epoch 的训练噪声也随 epoch 变化。
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Sampler

from .config import derive_seed, dump_config, load_config, resolve_device, set_by_path
from .dataset import BeamFusionDataset, collate_mixed, collate_raw, mix_batch, train_target_key, val_mismatch_enabled
from .models.gtcrn_fusion import BeamBank, build_model, count_params
from .pipeline import prepare, stft_mse


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


class EpochSampler(Sampler):
    """每个 epoch 用 (seed, epoch) 生成确定的随机排列。"""

    def __init__(self, n, seed, shuffle=True):
        self.n, self.seed, self.shuffle, self.epoch = n, seed, shuffle, 0

    def set_epoch(self, e):
        self.epoch = e

    def __len__(self):
        return self.n

    def __iter__(self):
        if not self.shuffle:
            return iter(range(self.n))
        g = torch.Generator().manual_seed(derive_seed(self.seed, "epoch", self.epoch))
        return iter(torch.randperm(self.n, generator=g).tolist())


def load_model_config(path, overrides=()):
    with open(path) as f:
        m = yaml.safe_load(f)
    for ov in overrides:
        k, v = ov.split("=", 1)
        set_by_path(m, k, yaml.safe_load(v))
    return m


def forward(model, kind, Y, Z):
    if kind == "mask_mvdr":
        return model(Z, Y)
    return model(Z)


def git_commit():
    try:
        root = Path(__file__).resolve().parents[1]
        c = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True,
                                    stderr=subprocess.DEVNULL).strip()
        dirty = subprocess.check_output(["git", "-C", str(root), "status", "--porcelain"], text=True,
                                        stderr=subprocess.DEVNULL).strip()
        return c + ("-dirty" if dirty else "")
    except Exception:
        return os.environ.get("CRBF_GIT_COMMIT")


def make_loader(ds, bs, sampler, nw, device, drop_last):
    return DataLoader(ds, batch_size=bs, sampler=sampler, num_workers=nw, drop_last=drop_last,
                      collate_fn=collate_mixed if ds.mix_on == "worker" else collate_raw,
                      pin_memory=device.type == "cuda", persistent_workers=False)


def to_device(b, ds, cfg, device):
    if ds.mix_on == "worker":
        return {k: v.to(device, non_blocking=True) for k, v in b.items()}
    return mix_batch(b, cfg, device, return_components=False)


def save_atomic(obj, path: Path):
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def plot_history(hist, path):
    from .plots import SERIES, plt, setup_style
    setup_style()
    ep = [h["epoch"] + 1 for h in hist]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
    ax[0].plot(ep, [h["train_loss"] for h in hist], "o-", color=SERIES[0], ms=3, label="train")
    ax[0].plot(ep, [h["val_loss"] for h in hist], "o-", color=SERIES[1], ms=3, label="validation")
    ax[0].set_yscale("log")
    ax[0].set_xlabel("epoch")
    ax[0].set_ylabel("STFT MSE")
    ax[0].legend()
    ax[0].set_title("Loss", loc="left", fontsize=9)
    ax[1].plot(ep, [h["lr"] for h in hist], "o-", color=SERIES[2], ms=3)
    ax[1].set_yscale("log")
    ax[1].set_xlabel("epoch")
    ax[1].set_ylabel("learning rate")
    ax[1].set_title("Learning rate", loc="left", fontsize=9)
    from matplotlib.ticker import MaxNLocator
    for x in ax:
        x.xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-config", default="configs/data_dev.yaml")
    ap.add_argument("--model-config", default="configs/model_beamfusion.yaml")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--micro-batch", type=int, default=None, help="梯度累积的子批大小（默认 = batch size）")
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--max-train-items", type=int, default=None)
    ap.add_argument("--max-val-items", type=int, default=None)
    ap.add_argument("--resume-from", default=None)
    ap.add_argument("--max-hours", type=float, default=None)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--set", nargs="*", default=[], help="数据配置覆盖 a.b=c")
    ap.add_argument("--mset", nargs="*", default=[], help="模型配置覆盖 a.b=c")
    a = ap.parse_args(argv)
    t_start = time.time()

    cfg = load_config(a.data_config, a.set)
    mcfg = load_model_config(a.model_config, a.mset)
    tr = mcfg["train"]
    epochs = a.epochs or tr["epochs"]
    bs = a.batch_size or tr["batch_size"]
    mb = min(bs, a.micro_batch or tr.get("micro_batch") or bs)
    device = resolve_device(a.device or cfg["device"])
    amp = bool(tr.get("amp", False)) and device.type == "cuda"  # 混合精度（float16 autocast + GradScaler），仅 GPU
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    if a.threads:
        torch.set_num_threads(a.threads)
    nw = cfg["loader"]["num_workers"] if a.num_workers is None else a.num_workers
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.resume_from and not (out / "last.pt").exists():
        src = Path(a.resume_from)
        for f in ("last.pt", "best.pt", "history.json", "log.csv", "run_info.json"):
            if (src / f).exists():
                shutil.copy2(src / f, out / f)
        print(f"copied checkpoint from {src}", flush=True)

    # seed_key：派生训练种子用的名字（默认为模型名）。阶段 4 的变体设为 beamfusion，与基线的初始化 RNG 状态、
    # 数据顺序完全相同（用户要求“相同随机种子”）
    seed = derive_seed(tr["seed"], "train", mcfg.get("seed_key", mcfg["name"]))
    torch.manual_seed(seed)
    np.random.seed(seed % (2 ** 32))

    bank = BeamBank.from_config(cfg).to(device)
    model = build_model(mcfg, cfg, bank).to(device)
    kind = mcfg["kind"]
    opt = torch.optim.Adam(model.parameters(), lr=tr["lr"])
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    # “验证损失连续 patience 个 epoch 不下降时减半”：PyTorch 在 num_bad_epochs > patience 时才降，因此传 patience-1
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=tr["lr_factor"],
                                                       patience=tr["lr_patience"] - 1, min_lr=tr["min_lr"])

    tkey = train_target_key(cfg)  # 训练目标（CLAUDE.md 3.3）：ref_direct（直达声，默认）或 ref（x_ref）
    train_ds = BeamFusionDataset(cfg, "train", return_components=False, max_items=a.max_train_items)
    vmm = val_mismatch_enabled(cfg)  # 含失配训练：验证集也加失配（学习率调度与最佳检查点按含失配的验证损失）
    val_ds = BeamFusionDataset(cfg, "val", return_components=False, max_items=a.max_val_items, val_mismatch=vmm)
    sampler = EpochSampler(len(train_ds), seed)
    train_dl = make_loader(train_ds, bs, sampler, nw, device, drop_last=True)
    val_dl = make_loader(val_ds, bs, EpochSampler(len(val_ds), seed, shuffle=False), nw, device, drop_last=False)

    hist, start, best = [], 0, math.inf
    if (out / "last.pt").exists():
        ck = torch.load(out / "last.pt", map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        hist, start, best = ck["history"], ck["epoch"] + 1, ck["best_val"]
        if ck.get("scaler"):
            scaler.load_state_dict(ck["scaler"])
        torch.set_rng_state(ck["rng_cpu"].cpu())  # map_location 会把 RNG 状态也搬到 GPU，这里需取回 CPU
        if device.type == "cuda" and ck.get("rng_cuda") is not None:
            torch.cuda.set_rng_state(ck["rng_cuda"].cpu(), device)
        print(f"resumed from epoch {ck['epoch'] + 1} (best val {best:.4e})", flush=True)

    info = {
        "name": mcfg["name"], "kind": kind, "params": count_params(model), "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
        "torch": torch.__version__, "git_commit": git_commit(), "data_config": a.data_config,
        "model_config": mcfg, "n_train": len(train_ds), "n_val": len(val_ds), "batch_size": bs,
        "micro_batch": mb, "amp": amp, "train_target": tkey,
        "train_mismatch": (cfg.get("train_mismatch") or {}).get("enabled", False), "val_mismatch": vmm, "num_workers": nw, "mix_on": train_ds.mix_on,
        "argv": sys.argv,
    }
    with open(out / "run_info.json", "w") as f:
        json.dump(info, f, indent=1, default=str)
    (out / "data_config.yaml").write_text(dump_config(cfg))
    print(f"[{mcfg['name']}] params={info['params']} device={device} train={len(train_ds)} val={len(val_ds)} "
          f"bs={bs} (micro {mb}) epochs={start}->{epochs} target={tkey}", flush=True)

    for epoch in range(start, epochs):
        if a.max_hours and hist:
            per_epoch = np.mean([h["epoch_time_s"] + h["val_time_s"] for h in hist[-3:]])
            if (time.time() - t_start + per_epoch * 1.1) / 3600 > a.max_hours:
                print(f"time budget reached before epoch {epoch + 1}; stopping (resume later)", flush=True)
                break
        train_ds.set_epoch(epoch)
        sampler.set_epoch(epoch)
        if hasattr(model, "set_epoch"):  # 可学习 W0 的两阶段训练（CLAUDE.md 6.3）
            model.set_epoch(epoch)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        model.train()
        t0 = time.time()
        tot, nb = 0.0, 0
        t_data = t_comp = 0.0
        tl = time.time()
        for i, b in enumerate(train_dl):
            b = to_device(b, train_ds, cfg, device)
            sync(device)
            tc = time.time()
            t_data += tc - tl  # 等待 DataLoader + 在设备上混合
            opt.zero_grad(set_to_none=True)
            n = b["mix"].shape[0]
            loss = 0.0
            for j in range(0, n, mb):  # 梯度累积：等效 batch = bs，显存只需容纳 micro_batch 条
                sub = {k: v[j:j + mb] for k, v in b.items()}
                Y, Z, X = prepare(sub, bank, target_key=tkey)
                with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
                    S = forward(model, kind, Y, Z)["S"]
                li = stft_mse(S, X) * (sub["mix"].shape[0] / n)  # 损失始终在 float32 / complex64 中计算
                if not torch.isfinite(li):
                    raise FloatingPointError(f"non-finite loss at epoch {epoch + 1} step {i}")
                scaler.scale(li).backward()
                loss += li.detach()
            scaler.unscale_(opt)
            if tr.get("grad_clip"):
                torch.nn.utils.clip_grad_norm_(model.parameters(), tr["grad_clip"])
            scaler.step(opt)
            scaler.update()
            tot += loss.item()
            nb += 1
            tl = time.time()
            t_comp += tl - tc
            if tr.get("log_every") and (i + 1) % tr["log_every"] == 0:
                print(f"  epoch {epoch + 1} step {i + 1}/{len(train_dl)} loss {tot / nb:.4e} "
                      f"{(time.time() - t0) / (i + 1):.2f}s/step (data {t_data / (i + 1):.2f}s, "
                      f"compute {t_comp / (i + 1):.2f}s)", flush=True)
        t_train = time.time() - t0
        t1 = time.time()
        model.eval()
        vt, vn = 0.0, 0
        wst = {"min": math.inf, "margin_min": math.inf, "viol": 0, "n": 0, "sum": 0.0}  # 投影后实测 WNG（f > 0）
        with torch.no_grad():
            for b in val_dl:
                b = to_device(b, val_ds, cfg, device)
                for j in range(0, b["mix"].shape[0], mb):
                    sub = {k: v[j:j + mb] for k, v in b.items()}
                    Y, Z, X = prepare(sub, bank, target_key=tkey)
                    n = Y.shape[0]
                    with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
                        o = forward(model, kind, Y, Z)
                        S = o["S"]
                    if "wng_db" in o:
                        wd = o["wng_db"][:, 1:].float()
                        wst["min"] = min(wst["min"], wd.min().item())
                        wst["sum"] += wd.sum().item()
                        wst["n"] += wd.numel()
                        if "w0_db" in o:
                            w0 = o["w0_db"][:, 1:] if torch.is_tensor(o["w0_db"]) else o["w0_db"]
                            mg = wd - w0
                            wst["margin_min"] = min(wst["margin_min"], mg.min().item())
                            wst["viol"] += int((mg < -1e-4).sum().item())
                    vt += stft_mse(S, X).item() * n
                    vn += n
        val_loss = vt / vn
        t_val = time.time() - t1
        sched.step(val_loss)
        row = {"epoch": epoch, "train_loss": tot / nb, "val_loss": val_loss, "lr": opt.param_groups[0]["lr"],
               "epoch_time_s": t_train, "val_time_s": t_val, "train_samples_per_s": nb * bs / t_train,
               "data_time_s": t_data, "compute_time_s": t_comp,
               "peak_mem_MB": torch.cuda.max_memory_allocated(device) / 2 ** 20 if device.type == "cuda" else None,
               "wall_time": time.strftime("%Y-%m-%d %H:%M:%S")}
        if wst["n"]:
            row.update({"val_wng_min_db": wst["min"], "val_wng_mean_db": wst["sum"] / wst["n"]})
            if math.isfinite(wst["margin_min"]):
                row.update({"val_wng_margin_min_db": wst["margin_min"], "val_wng_viol_frac": wst["viol"] / wst["n"]})
        hist.append(row)
        improved = val_loss < best
        best = min(best, val_loss)
        ck = {"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "epoch": epoch,
              "history": hist, "best_val": best, "rng_cpu": torch.get_rng_state(),
              "rng_cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
              "scaler": scaler.state_dict(), "model_config": mcfg, "git_commit": info["git_commit"]}
        save_atomic(ck, out / "last.pt")
        if improved:
            save_atomic({"model": copy.deepcopy(model.state_dict()), "epoch": epoch, "val_loss": val_loss,
                         "model_config": mcfg, "git_commit": info["git_commit"]}, out / "best.pt")
        with open(out / "history.json", "w") as f:
            json.dump(hist, f, indent=1)
        with open(out / "log.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for h in hist for k in h)))  # 续跑时列可能增加
            w.writeheader()
            w.writerows(hist)
        plot_history(hist, out / "curves.png")
        print(f"epoch {epoch + 1}/{epochs} train {row['train_loss']:.4e} val {val_loss:.4e} lr {row['lr']:.1e} "
              f"time {t_train:.0f}+{t_val:.0f}s ({row['train_samples_per_s']:.1f} samples/s)"
              f"{' *' if improved else ''}"
              + (f" | val WNG min {row['val_wng_min_db']:.2f} dB" if "val_wng_min_db" in row else "")
              + (f", margin to W0 min {row['val_wng_margin_min_db']:.2e} dB, violations {row['val_wng_viol_frac']:.1e}"
                 if "val_wng_margin_min_db" in row else ""), flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
