"""由原数据集（configs/data.yaml）的场景元数据派生 G2 阵列（M = 4、δ = 3 cm）的元数据（用户 2026-11-03）。

  python scripts/make_g2_meta.py --src <原 meta 目录> --out <输出 meta 目录> [--config configs/data_g2.yaml]

只改阵列几何：房间、T60、阵列中心、源位置 / 轨迹、语音、SNR / SIR、S1 / S2 定义与种子规则全部沿用原元数据（逐场景复制）。
- mic_spacing 改为 G2 的 δ；
- 失配基础随机数（gain_u、phase_u、pos_z）取原 8 个中的前 4 个（同一随机流，不重新抽样）。
不直接用 make_scenes.py 重新生成的原因：make_scenes 在抽取失配基础随机数之后才抽 SNR，M 改变会改变随机流，
使 SNR 与原数据集不同；派生方式保证除阵列外的所有场景参数逐条相同。
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import load_config  # noqa: E402

SPLITS = ("train", "val", "test_s1", "test_s2")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default="configs/data_g2.yaml")
    a = ap.parse_args()
    cfg = load_config(a.config)
    M, d = cfg["array"]["n_mics"], cfg["array"]["spacing"]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rep = {}
    for sp in SPLITS:
        src = Path(a.src) / f"{sp}.json"
        meta = json.loads(src.read_text())
        for sc in meta["scenes"]:
            assert len(sc["mismatch_base"]["gain_u"]) >= M
            sc["mic_spacing"] = d
            sc["mismatch_base"] = {k: v[:M] for k, v in sc["mismatch_base"].items()}
        meta["derived_from"] = {"file": str(src), "sha256": hashlib.sha256(src.read_bytes()).hexdigest(),
                                "rule": "copy all scenes; mic_spacing -> G2 spacing; mismatch_base -> first M entries"}
        meta.setdefault("config", {})
        if isinstance(meta["config"], dict):
            meta["config"]["array"] = {**meta["config"].get("array", {}), "n_mics": M, "spacing": d}
        (out / f"{sp}.json").write_text(json.dumps(meta))
        rep[sp] = len(meta["scenes"])
    print(json.dumps({"n_mics": M, "spacing": d, "scenes": rep}))


if __name__ == "__main__":
    main()
