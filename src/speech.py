"""LibriSpeech 预处理：去除长于 0.1 s 的静音段，按说话人拼接后切成 10 s 片段（CLAUDE.md 3.2 节）。

每个说话人：按 (chapter, utterance) 顺序读取所有语句 -> 逐句做能量 VAD 去静音 -> 拼接（拼接处 5 ms 交叉淡化）
-> 切成不重叠的 10 s 片段（不足 10 s 的余量丢弃）。片段以 16-bit FLAC 存储，索引为 json。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

from .config import derive_seed, speech_dir


def list_speakers(librispeech_root: Path, subset: str) -> list[str]:
    d = Path(librispeech_root) / subset
    return sorted(p.name for p in d.iterdir() if p.is_dir())


def select_speakers(cfg, split: str) -> list[tuple[str, str]]:
    """返回 [(subset, speaker)]。同一 subset 内按固定种子打乱后按 speaker_frac 区间切分，保证说话人不交叉。"""
    sc = cfg["speech"]["splits"][split]
    lo, hi = sc.get("speaker_frac", [0.0, 1.0])
    out = []
    for subset in sc["subsets"]:
        spk = list_speakers(cfg["paths"]["librispeech_root"], subset)
        rng = np.random.default_rng(derive_seed(cfg["seed"], "speaker_partition", subset))
        spk = [spk[i] for i in rng.permutation(len(spk))]
        n = len(spk)
        out += [(subset, s) for s in spk[int(round(lo * n)):int(round(hi * n))]]
    return out


def silent_runs_mask(x: np.ndarray, fs: int, frame_ms=20, hop_ms=10, thresh_db=40.0, max_sil_s=0.1) -> np.ndarray:
    """返回逐样本的布尔掩码：True 表示属于“长于 max_sil_s 的静音段”，需要去除。"""
    hop = int(fs * hop_ms / 1000)
    frame = int(fs * frame_ms / 1000)
    n_blk = int(np.ceil(len(x) / hop))
    pad = np.pad(x, (frame // 2, frame + n_blk * hop - len(x)))
    idx = np.arange(n_blk)[:, None] * hop + np.arange(frame)[None, :]
    e = 10 * np.log10(np.mean(pad[idx] ** 2, axis=1) + 1e-12)
    silent = e < (e.max() - thresh_db)
    # 找出静音块的连续段，长度（以块计）> max_sil 的被标记
    min_blk = int(round(max_sil_s * 1000 / hop_ms))
    rm = np.zeros(n_blk, bool)
    d = np.diff(np.concatenate([[0], silent.astype(np.int8), [0]]))
    starts, ends = np.where(d == 1)[0], np.where(d == -1)[0]
    for s, e_ in zip(starts, ends):
        if e_ - s > min_blk:
            rm[s:e_] = True
    return np.repeat(rm, hop)[: len(x)]


def concat_with_fade(segs: list[np.ndarray], fade: int) -> np.ndarray:
    """相邻片段以 fade 个样本的线性交叉淡化拼接。"""
    segs = [s for s in segs if len(s) > 2 * fade]
    if not segs:
        return np.zeros(0, np.float32)
    out = segs[0].astype(np.float64)
    ramp = np.linspace(0, 1, fade, endpoint=False) + 0.5 / fade
    for s in segs[1:]:
        s = s.astype(np.float64)
        tail = out[-fade:] * (1 - ramp) + s[:fade] * ramp
        out = np.concatenate([out[:-fade], tail, s[fade:]])
    return out


def remove_silence(x: np.ndarray, fs: int, sc: dict) -> np.ndarray:
    rm = silent_runs_mask(x, fs, sc["frame_ms"], sc["hop_ms"], sc["silence_thresh_db"], sc["silence_max_s"])
    keep = ~rm
    d = np.diff(np.concatenate([[0], keep.astype(np.int8), [0]]))
    starts, ends = np.where(d == 1)[0], np.where(d == -1)[0]
    fade = int(fs * sc["join_fade_ms"] / 1000)
    return concat_with_fade([x[s:e] for s, e in zip(starts, ends)], fade)


def speaker_utterances(librispeech_root, subset, speaker) -> list[Path]:
    d = Path(librispeech_root) / subset / speaker
    return sorted(d.glob("*/*.flac"))


def process_speaker(cfg, split, subset, speaker) -> tuple[list[dict], dict]:
    fs = cfg["signal"]["fs"]
    sc = cfg["speech"]
    clip_len = int(cfg["signal"]["clip_seconds"] * fs)
    fade = int(fs * sc["join_fade_ms"] / 1000)
    out_dir = speech_dir(cfg) / split / speaker
    out_dir.mkdir(parents=True, exist_ok=True)
    pieces, orig_n = [], 0
    for f in speaker_utterances(cfg["paths"]["librispeech_root"], subset, speaker):
        x, sr = sf.read(f, dtype="float64")
        assert sr == fs, (f, sr)
        orig_n += len(x)
        y = remove_silence(x, fs, sc)
        if len(y):
            pieces.append(y)
    full = concat_with_fade(pieces, fade)
    n_clips = len(full) // clip_len
    entries = []
    for k in range(n_clips):
        seg = full[k * clip_len:(k + 1) * clip_len]
        cid = f"{speaker}-{k:04d}"
        p = out_dir / f"{cid}.flac"
        sf.write(p, seg.astype(np.float32), fs, subtype="PCM_16")
        entries.append({
            "id": cid,
            "speaker": speaker,
            "subset": subset,
            "path": str(p.relative_to(speech_dir(cfg))),
            "rms": float(np.sqrt(np.mean(seg ** 2))),
        })
    stats = {"speaker": speaker, "orig_s": orig_n / fs, "kept_s": len(full) / fs, "n_clips": n_clips}
    return entries, stats


def load_index(cfg, split) -> list[dict]:
    with open(speech_dir(cfg) / f"{split}_index.json") as f:
        return json.load(f)


def read_clip(cfg, entry) -> np.ndarray:
    x, _ = sf.read(speech_dir(cfg) / entry["path"], dtype="float32")
    return x
