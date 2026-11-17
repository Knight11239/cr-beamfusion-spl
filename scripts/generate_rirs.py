"""生成场景 RIR（可断点续跑、可分段）。已存在的文件会被跳过。

用法：
  python scripts/generate_rirs.py --config configs/data_dev.yaml --split train [--start 0 --end 500] [--workers 4]
  python scripts/generate_rirs.py --config configs/data_dev.yaml --split test_s1 --kappa 1.0   # 失配 RIR
"""
import argparse
import json
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import load_config, rir_dir  # noqa: E402
from src.simulate import load_meta, save_scene_rirs, simulate_scene_rirs  # noqa: E402


def _job(args):
    cfg, scene, kappa = args
    nominal_info = None
    if kappa > 0:
        import numpy as np
        with np.load(rir_dir(cfg, scene["split"], 0.0) / scene["rir_file"]) as z:
            nominal_info = json.loads(str(z["info"]))
    t = time.time()
    res = simulate_scene_rirs(cfg, scene, kappa, nominal_info)
    save_scene_rirs(cfg, scene, res, kappa)
    return scene["id"], time.time() - t, res["array"].shape


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data_dev.yaml")
    ap.add_argument("--split", required=True)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--kappa", type=float, default=0.0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-hours", type=float, default=None, help="时间预算：到时不再提交新场景（可续跑）")
    ap.add_argument("--set", nargs="*", default=[], help="配置覆盖，如 room.t60_mode=calibrated")
    a = ap.parse_args()
    cfg = load_config(a.config, a.set)
    scenes = load_meta(cfg, a.split)[a.start:a.end]
    d = rir_dir(cfg, a.split, a.kappa)
    todo = [s for s in scenes if not (d / s["rir_file"]).exists()]
    if a.kappa > 0:
        nd = rir_dir(cfg, a.split, 0.0)
        missing = [s["id"] for s in todo if not (nd / s["rir_file"]).exists()]
        assert not missing, f"nominal RIRs missing for {len(missing)} scenes, generate kappa=0 first"
    print(f"[{a.split} kappa={a.kappa}] {len(scenes)} scenes in range, {len(todo)} to generate -> {d}", flush=True)
    t0 = time.time()
    done = 0
    with ProcessPoolExecutor(a.workers) as ex:  # 有界提交：超过时间预算后不再提交新场景，已提交的跑完
        pending, it = set(), iter(todo)
        while True:
            budget_left = a.max_hours is None or (time.time() - t0) / 3600 < a.max_hours
            while budget_left and len(pending) < 2 * a.workers:
                s = next(it, None)
                if s is None:
                    break
                pending.add(ex.submit(_job, (cfg, s, a.kappa)))
            if not pending:
                break
            fin, pending = wait(pending, return_when=FIRST_COMPLETED)
            for f in fin:
                sid, dt, shp = f.result()
                done += 1
                if done % 20 == 0 or done == len(todo):
                    el = time.time() - t0
                    print(f"  {done}/{len(todo)} {sid} {dt:.1f}s shape={shp} elapsed={el / 60:.1f} min "
                          f"eta={el / done * (len(todo) - done) / 60:.1f} min", flush=True)
    if done < len(todo):
        print(f"time budget ({a.max_hours} h) reached: {len(todo) - done} scenes left; rerun to resume", flush=True)
    print(f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
