"""评测指标。所有函数输入 1-D numpy 数组（16 kHz）。

- si_sdr(est, ref)：尺度不变 SDR（不去均值）。
- snr(est, ref)：10 log10(||ref||^2 / ||est - ref||^2)（非尺度不变；BeamFusion 论文未给出 ΔSNR 的定义，
  此为本项目设定，见阶段 3 报告）。
- stoi、pesq_wb：pystoi / pesq（宽带）。
- power_ratio_db(a, b)：10 log10(||a||^2 / ||b||^2)，用于 shadow filtering 的 SIR / SNR。
- bss_sir：mir_eval BSS-eval 的 SIR（参考源 = [x_ref, 中心处干扰]，估计 = 输出）。
"""
from __future__ import annotations

import numpy as np

EPS = 1e-12


def si_sdr(est, ref):
    est, ref = np.asarray(est, np.float64), np.asarray(ref, np.float64)
    a = np.dot(est, ref) / (np.dot(ref, ref) + EPS)
    t = a * ref
    return 10 * np.log10((np.dot(t, t) + EPS) / (np.dot(est - t, est - t) + EPS))


def snr(est, ref):
    est, ref = np.asarray(est, np.float64), np.asarray(ref, np.float64)
    return 10 * np.log10((np.dot(ref, ref) + EPS) / (np.dot(est - ref, est - ref) + EPS))


def power_ratio_db(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return 10 * np.log10((np.dot(a, a) + EPS) / (np.dot(b, b) + EPS))


def stoi(est, ref, fs=16000):
    from pystoi import stoi as _stoi
    return float(_stoi(ref, est, fs, extended=False))


def pesq_wb(est, ref, fs=16000):
    from pesq import pesq as _pesq
    try:
        return float(_pesq(fs, ref, est, "wb"))
    except Exception:  # 例如检测不到语音
        return float("nan")


def bss_sir(est, ref, interf_ref):
    from mir_eval.separation import bss_eval_sources
    refs = np.stack([ref, interf_ref])
    ests = np.stack([est, est])
    _, sir, _, _ = bss_eval_sources(refs, ests, compute_permutation=False)
    return float(sir[0])


def ci95(x):
    """均值与 95% 置信区间半宽（正态近似）。"""
    x = np.asarray(x, np.float64)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return float(np.mean(x)) if len(x) else float("nan"), float("nan")
    return float(x.mean()), float(1.96 * x.std(ddof=1) / np.sqrt(len(x)))
