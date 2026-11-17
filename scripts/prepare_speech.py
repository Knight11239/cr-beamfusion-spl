"""预处理 LibriSpeech：去静音 + 切 10 s 片段，生成 {data_root}/speech/{split}_index.json。

用法：python scripts/prepare_speech.py --config configs/data_dev.yaml [--splits train val test] [--workers 4]
"""
import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import load_config, speech_dir  # noqa: E402
from src.speech import process_speaker, select_speakers  # noqa: E402


def _job(args):
    cfg, split, subset, spk = args
    return process_speaker(cfg, split, subset, spk)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data_dev.yaml")
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--set", nargs="*", default=[], help="配置覆盖，如 room.t60_mode=calibrated")
    a = ap.parse_args()
    cfg = load_config(a.config, a.set)
    out = speech_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    all_spk = {}
    for split in a.splits:
        spks = select_speakers(cfg, split)
        all_spk[split] = {s for _, s in spks}
        with ProcessPoolExecutor(a.workers) as ex:
            res = list(ex.map(_job, [(cfg, split, sub, s) for sub, s in spks]))
        entries = [e for r, _ in res for e in r]
        stats = [st for _, st in res]
        with open(out / f"{split}_index.json", "w") as f:
            json.dump(entries, f, indent=0)
        with open(out / f"{split}_stats.json", "w") as f:
            json.dump(stats, f, indent=1)
        orig = sum(s["orig_s"] for s in stats) / 3600
        kept = sum(s["kept_s"] for s in stats) / 3600
        print(f"[{split}] speakers={len(spks)} clips={len(entries)} "
              f"orig={orig:.2f} h  after-silence-removal={kept:.2f} h ({100 * kept / orig:.1f}%)")
    names = list(all_spk)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            inter = all_spk[names[i]] & all_spk[names[j]]
            assert not inter, f"speaker overlap {names[i]}/{names[j]}: {inter}"
    print("speaker sets are disjoint:", {k: len(v) for k, v in all_spk.items()})


if __name__ == "__main__":
    main()
