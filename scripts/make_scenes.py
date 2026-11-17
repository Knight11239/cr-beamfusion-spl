"""采样场景元数据，写入 {data_root}/meta/{split}.json。

用法：python scripts/make_scenes.py --config configs/data_dev.yaml [--splits train val test_s1 test_s2]
"""
import argparse
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import load_config, meta_path  # noqa: E402
from src.simulate import sample_scene  # noqa: E402
from src.speech import load_index  # noqa: E402


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return os.environ.get("CRBF_GIT_COMMIT")  # Kaggle 上的代码快照没有 .git，由 runner 传入


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data_dev.yaml")
    ap.add_argument("--splits", nargs="+", default=None)
    ap.add_argument("--set", nargs="*", default=[], help="配置覆盖，如 room.t60_mode=calibrated")
    a = ap.parse_args()
    cfg = load_config(a.config, a.set)
    for split in a.splits or list(cfg["splits"]):
        sp = cfg["splits"][split]
        idx = load_index(cfg, sp["speech"])
        scenes = [sample_scene(cfg, split, i, idx) for i in range(sp["n"])]
        p = meta_path(cfg, split)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            json.dump({"config": cfg, "git_commit": git_commit(), "scenes": scenes}, f, indent=1)
        sc = Counter(s["scenario"] for s in scenes)
        mv = Counter((s["scenario"], any(i["moving"] for i in s["interferers"]), len(s["interferers"]))
                     for s in scenes)
        print(f"[{split}] n={len(scenes)} scenario={dict(sc)} (scenario, has_moving, n_int)={dict(mv)}")


if __name__ == "__main__":
    main()
