"""场景采样与 RIR 生成（pyroomacoustics，CLAUDE.md 3.2–3.6 节）。

每个场景生成一个 npz（float32）：
  array        [S, M, L]  各声源位置 -> 8 个阵元的 RIR（S = 1 个目标位置 + 全部干扰位置）
  center       [S, L]     各声源位置 -> 阵列中心虚拟麦克风的 RIR（仅用于参考信号/中心混合，不进入阵列输入）
  direct       [S, Ld]    同一几何、max_order=0（仅直达声）时到中心的 RIR，用于 SNR/SIR 定义与直达峰定位
  direct_peak  [S]        直达声峰值的样本序号（array/center/direct 共用同一时间轴）
失配版本（kappa>0）只重算 array（扰动后的麦克风 x 坐标），center/direct 不变。
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import pyroomacoustics as pra

from .config import derive_seed, meta_path, rir_dir
from .geometry import mic_positions, source_position

# ----------------------------------------------------------------------------------------------
# Schroeder 积分 / T60 估计
# ----------------------------------------------------------------------------------------------


def schroeder_edc_db(h: np.ndarray) -> np.ndarray:
    e = np.cumsum((h.astype(np.float64) ** 2)[::-1])[::-1]
    e = e / e[0]
    return 10 * np.log10(np.maximum(e, 1e-300))


def estimate_t60(h: np.ndarray, fs: int, lo_db=-5.0, hi_db=-35.0) -> float:
    """Schroeder 积分 + 在 [lo_db, hi_db] 区间做最小二乘直线拟合，外推到 -60 dB。

    hi_db=-25 为 T20，hi_db=-35 为 T30（ISO 3382 惯例）。
    """
    edc = schroeder_edc_db(h)
    i0 = int(np.argmax(edc <= lo_db))
    i1 = int(np.argmax(edc <= hi_db))
    if i1 <= i0 + 2:
        return float("nan")
    t = np.arange(i0, i1) / fs
    slope, _ = np.polyfit(t, edc[i0:i1], 1)
    return float(-60.0 / slope)


def edc_truncation_index(h_all: np.ndarray, trunc_db: float) -> int:
    """所有通道能量之和的 Schroeder 曲线降到 -trunc_db 以下的第一个样本（再多保留 1 个样本）。"""
    e = (h_all.astype(np.float64) ** 2).reshape(-1, h_all.shape[-1]).sum(0)
    edc = np.cumsum(e[::-1])[::-1]
    edc /= edc[0]
    idx = int(np.argmax(edc < 10 ** (-trunc_db / 10)))
    return h_all.shape[-1] if idx == 0 else idx + 1


# ----------------------------------------------------------------------------------------------
# 场景采样
# ----------------------------------------------------------------------------------------------


def _feasible_center_box(room_dim, offsets, cfg_room):
    """给定各声源相对阵列中心的偏移 [K,3]，返回阵列中心 (x, y) 的可行区间。"""
    am, sm = cfg_room["array_wall_margin"], cfg_room["source_wall_margin"]
    lo, hi = [], []
    for ax in range(2):
        lo.append(max(am, sm - offsets[:, ax].min()))
        hi.append(min(room_dim[ax] - am, room_dim[ax] - sm - offsets[:, ax].max()))
    return np.array(lo), np.array(hi)


def _sample_room_dim(rng, cfg_room):
    if cfg_room["mode"] == "fixed":
        return [float(v) for v in cfg_room["fixed_dim"]]
    r = cfg_room["random_dim"]
    return [float(rng.uniform(*r["length"])), float(rng.uniform(*r["width"])), float(rng.uniform(*r["height"]))]


def s1_trajectory(cfg):
    s1 = cfg["sources"]["s1"]
    return [float(s1["start_deg"] + k * s1["step_deg"]) for k in range(s1["n_positions"])]


def _sample_s2_interferers(rng, cfg, moving: bool):
    s2 = cfg["sources"]["s2"]
    n_int = int(rng.integers(s2["n_interferers"][0], s2["n_interferers"][1] + 1))
    a0, a1 = s2["theta_range"]
    n0, n1 = s2["near_range"]
    n_static = n_int - 1 if moving else n_int
    while True:  # 拒绝采样：静止干扰 U[30,180]，至少一个在 [30,90]，两两间隔 >= min_sep
        ang = rng.uniform(a0, a1, n_static)
        if not np.any((ang >= n0) & (ang <= n1)):
            continue
        if n_static > 1:
            dif = np.abs(ang[:, None] - ang[None, :]) + np.eye(n_static) * 1e9
            if dif.min() < s2["min_sep_deg"]:
                continue
        break
    ints = [{"moving": False, "trajectory_deg": [float(a)]} for a in ang]
    if moving:
        n_pos = int(round(cfg["signal"]["clip_seconds"] / cfg["sources"]["segment_s"]))
        step = s2["speed_deg_per_s"] * cfg["sources"]["segment_s"]
        sweep = step * (n_pos - 1)
        direction = 1.0 if rng.random() < 0.5 else -1.0
        start = rng.uniform(a0, a1 - sweep) if direction > 0 else rng.uniform(a0 + sweep, a1)
        ints.append({"moving": True, "trajectory_deg": [float(start + direction * step * k) for k in range(n_pos)]})
    order = rng.permutation(len(ints))
    return [ints[i] for i in order]


def sample_scene(cfg, split: str, idx: int, speech_index: list[dict]) -> dict:
    """确定性地（由 seed、split、idx 派生）采样一个场景的全部元数据。"""
    rng = np.random.default_rng(derive_seed(cfg["seed"], "scene", split, idx))
    sp = cfg["splits"][split]
    src = cfg["sources"]
    room_cfg = cfg["room"]
    scenario = sp["scenario"]
    if scenario == "mixed":  # 训练/验证：S1 型与 S2 型各一半
        scenario = "S1" if idx % 2 == 0 else "S2"
        s2_rank = idx // 2
    else:
        s2_rank = idx
    is_test = split.startswith("test")

    if is_test:  # 测试集按 T60 分组（300 / 700 ms 各一半）
        vals = room_cfg["test_t60_values"]
        t60 = float(vals[idx % len(vals)])
        if scenario == "S2":  # 让“移动/静止”与 T60 分组交叉均衡
            s2_rank = idx // len(vals)
    else:
        t60 = float(rng.uniform(*room_cfg["t60"]))

    if scenario == "S1":
        interferers = [{"moving": True, "trajectory_deg": s1_trajectory(cfg)}]
        sir = float(rng.uniform(*src["s1"]["sir_db"]))
    else:
        period = int(round(1 / src["s2"]["moving_fraction"]))
        interferers = _sample_s2_interferers(rng, cfg, moving=(s2_rank % period == 1))
        sir = float(rng.uniform(*src["s2"]["sir_db"]))

    # 声源相对阵列中心的偏移
    offs = [source_position([0, 0, 0], src["target_theta_deg"], src["target_dist"])]
    for it in interferers:
        for a in it["trajectory_deg"]:
            offs.append(source_position([0, 0, 0], a, src["interferer_dist"]))
    offs = np.array(offs)

    for _ in range(1000):
        room_dim = _sample_room_dim(rng, room_cfg)
        lo, hi = _feasible_center_box(room_dim, offs, room_cfg)
        h_lo, h_hi = room_cfg["array_height"]
        h_hi = min(h_hi, room_dim[2] - room_cfg["array_wall_margin"])
        if np.all(hi >= lo) and h_hi >= h_lo:
            break
    else:
        raise RuntimeError("no feasible array position")
    center = [float(rng.uniform(lo[0], hi[0])), float(rng.uniform(lo[1], hi[1])), float(rng.uniform(h_lo, h_hi))]

    # 语音：目标与各干扰来自不同说话人
    spk = sorted({e["speaker"] for e in speech_index})
    chosen = rng.choice(len(spk), size=1 + len(interferers), replace=False)
    speech_ids = []
    for k in chosen:
        cands = [e["id"] for e in speech_index if e["speaker"] == spk[k]]
        speech_ids.append(cands[int(rng.integers(len(cands)))])

    M = cfg["array"]["n_mics"]
    mm = {  # 失配的“基础随机数”，实际值 = kappa * 范围 * 基础值（保证不同 kappa 之间可比、可复现）
        "gain_u": rng.uniform(-1, 1, M).tolist(),
        "phase_u": rng.uniform(-1, 1, M).tolist(),
        "pos_z": rng.standard_normal(M).tolist(),
    }
    return {
        "id": f"{split}_{idx:05d}",
        "split": split,
        "index": idx,
        "scenario": scenario,
        "room_dim": room_dim,
        "t60_target": t60,
        "array_center": center,
        "mic_spacing": cfg["array"]["spacing"],
        "target": {"theta_deg": src["target_theta_deg"], "dist": src["target_dist"]},
        "interferers": [dict(it, dist=src["interferer_dist"]) for it in interferers],
        "speech": {"target": speech_ids[0], "interferers": speech_ids[1:]},
        "snr_db": float(rng.uniform(*src["snr_db"])),
        "sir_db": sir,
        "mismatch_base": mm,
        "rir_file": f"{split}_{idx:05d}.npz",
    }


def scene_source_positions(scene) -> np.ndarray:
    c = scene["array_center"]
    pos = [source_position(c, scene["target"]["theta_deg"], scene["target"]["dist"])]
    for it in scene["interferers"]:
        for a in it["trajectory_deg"]:
            pos.append(source_position(c, a, it["dist"]))
    return np.array(pos)


def mismatch_params(cfg, scene, kappa: float) -> dict:
    mb, mc = scene["mismatch_base"], cfg["mismatch"]
    return {
        "gain_db": (kappa * mc["gain_db"] * np.asarray(mb["gain_u"])).tolist(),
        "phase_deg": (kappa * mc["phase_deg"] * np.asarray(mb["phase_u"])).tolist(),
        "dx_m": (kappa * mc["pos_std_m"] * np.asarray(mb["pos_z"])).tolist(),
    }


# ----------------------------------------------------------------------------------------------
# RIR 生成
# ----------------------------------------------------------------------------------------------


def absorption_for_t60(cfg, scene) -> tuple[float, int, dict]:
    """返回 (能量吸声系数, 反射阶数, 诊断信息)。

    sabine：CLAUDE.md 规定的 pra.inverse_sabine。
    calibrated：在 inverse_sabine 基础上对吸声系数做二分搜索，使中心麦克风（目标源）RIR 的
                Schroeder T30 等于设定值（用于修正 ISM 与 Sabine 公式之间的系统偏差，需用户确认后启用）。
    """
    fs, c = cfg["signal"]["fs"], cfg["signal"]["c"]
    t60, dim = scene["t60_target"], scene["room_dim"]
    e, order = pra.inverse_sabine(t60, dim, c=c)
    order = max(1, int(math.ceil(order * cfg["room"]["max_order_scale"])))
    info = {"inverse_sabine_absorption": float(e)}
    if cfg["room"]["t60_mode"] == "sabine":
        return float(e), order, info
    if cfg["room"]["t60_mode"] != "calibrated":
        raise ValueError(cfg["room"]["t60_mode"])

    tgt = scene_source_positions(scene)[0]
    ctr = np.asarray(scene["array_center"])[:, None]

    def t30(absorb):
        # T30 只依赖 -5…-35 dB 段，用 0.6 倍反射阶数即可（已验证与全阶数一致），加快搜索
        o = max(1, int(math.ceil(order * 0.6)))
        r = pra.ShoeBox(dim, fs=fs, materials=pra.Material(absorb), max_order=o, air_absorption=False)
        r.add_microphone_array(ctr)
        r.add_source(tgt)
        r.compute_rir()
        return estimate_t60(r.rir[0][0], fs, -5, -35)

    lo, hi = np.log(e * 0.2), np.log(min(0.999, e * 3.0))
    for _ in range(12):
        mid = 0.5 * (lo + hi)
        if t30(np.exp(mid)) > t60:
            lo = mid  # 混响过长 -> 增大吸声
        else:
            hi = mid
    e_cal = float(np.exp(0.5 * (lo + hi)))
    # 吸声减小时需要更高的反射阶数才能覆盖到 c*T60 的距离
    order = max(order, int(math.ceil(order * e / e_cal)))
    info["calibrated_absorption"] = e_cal
    return e_cal, order, info


def direct_array_rir(cfg, scene, kappa: float = 0.0) -> np.ndarray:
    """目标源到 8 个阵元的直达声 RIR（max_order=0，自由场），[M, L]。kappa>0 时使用扰动后的麦克风位置。

    与 center 的 direct RIR 用同一 pyroomacoustics 约定（同一全局时延），因此可直接与 x_ref 的直达声部分比较。
    用于“目标失真”指标（shadow filtering 只处理目标直达声分量）。
    """
    fs = cfg["signal"]["fs"]
    pra.constants.set("c", cfg["signal"]["c"])
    M, delta, order_conv = cfg["array"]["n_mics"], cfg["array"]["spacing"], cfg["array"]["mic_order"]
    center = np.asarray(scene["array_center"])
    dx = mismatch_params(cfg, scene, kappa)["dx_m"] if kappa > 0 else None
    room = pra.ShoeBox(scene["room_dim"], fs=fs, max_order=0, air_absorption=False)
    room.add_microphone_array(mic_positions(center, M, delta, order_conv, dx=dx))
    room.add_source(scene_source_positions(scene)[0])
    room.compute_rir()
    L = max(len(room.rir[m][0]) for m in range(M))
    out = np.zeros((M, L), np.float32)
    for m in range(M):
        out[m, :len(room.rir[m][0])] = room.rir[m][0]
    return out


def simulate_scene_rirs(cfg, scene, kappa: float = 0.0, nominal_info: dict | None = None) -> dict:
    """kappa>0 时应传入标称 RIR 的 info（复用同一吸声系数与反射阶数，只改变麦克风位置）。"""
    fs, c = cfg["signal"]["fs"], cfg["signal"]["c"]
    M, delta, order_conv = cfg["array"]["n_mics"], cfg["array"]["spacing"], cfg["array"]["mic_order"]
    pra.constants.set("c", c)
    if nominal_info is not None:
        absorb, max_order = nominal_info["absorption"], nominal_info["max_order"]
        ainfo = {}
    else:
        absorb, max_order, ainfo = absorption_for_t60(cfg, scene)
    src_pos = scene_source_positions(scene)
    center = np.asarray(scene["array_center"])
    dx = mismatch_params(cfg, scene, kappa)["dx_m"] if kappa > 0 else None
    mics = mic_positions(center, M, delta, order_conv, dx=dx)
    nominal = kappa == 0
    if nominal:
        mics = np.concatenate([mics, center[:, None]], axis=1)  # 第 M 路 = 中心虚拟麦克风

    t0 = time.time()
    # 逐个声源位置仿真（与一次性加入所有声源的结果相同），内存只需容纳单个声源的镜像源
    n_mic, n_src = mics.shape[1], len(src_pos)
    rirs = []
    for p in src_pos:
        room = pra.ShoeBox(scene["room_dim"], fs=fs, materials=pra.Material(absorb), max_order=max_order,
                           air_absorption=False)
        room.add_microphone_array(mics)
        room.add_source(p)
        room.compute_rir()
        rirs.append([np.asarray(room.rir[m][0], np.float64) for m in range(n_mic)])
        del room
    L = max(len(r) for rs in rirs for r in rs)
    H = np.zeros((n_src, n_mic, L), np.float64)
    for s_, rs in enumerate(rirs):
        for m, r in enumerate(rs):
            H[s_, m, :len(r)] = r
    del rirs
    sim_time = time.time() - t0

    out = {"absorption": absorb, "max_order": max_order, "sim_time_s": sim_time, **ainfo}
    if not nominal:  # 失配 RIR：同时保存该样本的失配参数（位置误差已用于仿真；增益/相位在 STFT 域施加），便于追溯
        out["mismatch"] = {"kappa": float(kappa), **mismatch_params(cfg, scene, kappa)}
    if nominal:
        # 直达声（max_order=0）到中心麦克风
        room0 = pra.ShoeBox(scene["room_dim"], fs=fs, materials=pra.Material(absorb), max_order=0,
                            air_absorption=False)
        room0.add_microphone_array(center[:, None])
        for p in src_pos:
            room0.add_source(p)
        room0.compute_rir()
        Ld = max(len(room0.rir[0][s]) for s in range(n_src))
        D = np.zeros((n_src, Ld))
        for s in range(n_src):
            D[s, :len(room0.rir[0][s])] = room0.rir[0][s]
        out["direct"] = D
        out["direct_peak"] = np.argmax(np.abs(D), axis=1).astype(np.int32)
    trunc_db = cfg["room"]["rir_trunc_db"]
    L_keep = edc_truncation_index(H, trunc_db) if trunc_db else L
    out["L_full"] = L
    H = H[..., :L_keep]
    out["array"] = H[:, :M]
    if nominal:
        out["center"] = H[:, M]
    return out


def save_scene_rirs(cfg, scene, res, kappa=0.0):
    d = rir_dir(cfg, scene["split"], kappa)
    d.mkdir(parents=True, exist_ok=True)
    dt = np.dtype((cfg["room"].get("rir_dtype_overrides") or {}).get(scene["split"], cfg["room"]["rir_dtype"]))
    arrays = {k: res[k].astype(dt) for k in ("array", "center", "direct") if k in res}
    if "direct_peak" in res:
        arrays["direct_peak"] = res["direct_peak"]
    info = {k: v for k, v in res.items() if k not in ("array", "center", "direct", "direct_peak")}
    tmp = d / (scene["rir_file"] + ".tmp.npz")
    np.savez(tmp, info=json.dumps(info), **arrays)
    tmp.rename(d / scene["rir_file"])


def load_scene_rirs(cfg, scene, kappa=0.0) -> dict:
    with np.load(rir_dir(cfg, scene["split"], 0.0) / scene["rir_file"]) as z:
        out = {k: z[k] for k in z.files if k != "info"}
        out["info"] = json.loads(str(z["info"]))
    if kappa > 0:
        with np.load(rir_dir(cfg, scene["split"], kappa) / scene["rir_file"]) as z:
            out["array"] = z["array"]
            out["info_mismatch"] = json.loads(str(z["info"]))
    return out


def load_meta(cfg, split) -> list[dict]:
    with open(meta_path(cfg, split)) as f:
        return json.load(f)["scenes"]
