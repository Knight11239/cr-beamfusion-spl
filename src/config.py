"""配置加载：yaml + 环境变量覆盖 + 命令行 key=value 覆盖。"""
from __future__ import annotations

import copy
import hashlib
import os
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]

ENV_OVERRIDES = {
    "CRBF_DATA_ROOT": ("paths", "data_root"),
    "CRBF_LIBRISPEECH_ROOT": ("paths", "librispeech_root"),
    "CRBF_DEVICE": ("device",),
}


def _parse_value(v: str):
    return yaml.safe_load(v)


def set_by_path(cfg: dict, dotted: str, value):
    keys = dotted.split(".")
    d = cfg
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def load_config(path: str | os.PathLike, overrides: list[str] | None = None) -> dict:
    """读取 yaml；之后依次应用环境变量与 `a.b.c=value` 形式的覆盖。"""
    path = Path(path)
    if not path.is_absolute() and not path.exists():
        path = REPO_ROOT / path
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for env, keys in ENV_OVERRIDES.items():
        if os.environ.get(env):
            set_by_path(cfg, ".".join(keys), os.environ[env])
    for ov in overrides or []:
        k, v = ov.split("=", 1)
        set_by_path(cfg, k, _parse_value(v))
    cfg["_config_path"] = str(path)
    return cfg


def data_root(cfg) -> Path:
    return Path(cfg["paths"]["data_root"])


def speech_dir(cfg) -> Path:
    return data_root(cfg) / "speech"


def meta_path(cfg, split: str) -> Path:
    return data_root(cfg) / "meta" / f"{split}.json"


def rir_dir(cfg, split: str, kappa: float = 0.0) -> Path:
    if kappa == 0:
        return data_root(cfg) / "rirs" / split
    return data_root(cfg) / "rirs" / f"{split}_kappa{kappa:g}"


def resolve_device(cfg_or_str):
    import torch

    dev = cfg_or_str["device"] if isinstance(cfg_or_str, dict) else cfg_or_str
    if dev == "auto":
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(dev)


def derive_seed(*parts) -> int:
    """由任意部件（全局种子、split 名、样本序号、用途……）确定性地派生 32 位种子。"""
    h = hashlib.sha256("|".join(str(p) for p in parts).encode()).digest()
    return int.from_bytes(h[:4], "little")


def dump_config(cfg) -> str:
    c = copy.deepcopy(cfg)
    return yaml.safe_dump(c, allow_unicode=True, sort_keys=False)
