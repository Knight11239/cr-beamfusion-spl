"""固定波束形成器字典、WNG、波束图与 Gram 矩阵（CLAUDE.md 第 4 节）。

约定：滤波器 h 的形状为 [F, M]（complex128），输出 Z = h^H y = sum_m conj(h_m) y_m。
所有设计在 float64 / complex128 中完成，需要时再转换为 complex64。
"""
from __future__ import annotations

import numpy as np

from .geometry import C_SOUND, steering_vector


def stft_freqs(fs=16000, n_fft=512) -> np.ndarray:
    return np.arange(n_fft // 2 + 1, dtype=np.float64) * fs / n_fft


def steer(freqs, theta_deg, n_mics=8, spacing=0.01, c=C_SOUND) -> np.ndarray:
    """单个方向的导向向量，[F, M]。"""
    return steering_vector(freqs, theta_deg, n_mics, spacing, c)[:, 0, :]


def delay_and_sum(freqs, theta_s=0.0, n_mics=8, spacing=0.01, c=C_SOUND) -> np.ndarray:
    """h_D = d(theta_s) / M（最大 WNG 波束形成器），[F, M]。"""
    return steer(freqs, theta_s, n_mics, spacing, c) / n_mics


def null_filter(freqs, null_deg, theta_s=0.0, n_mics=8, spacing=0.01, c=C_SOUND, eps_rel=1e-8,
                renormalize=True):
    """在 theta_s 无失真、在 null_deg（标量或列表）置零的最小范数滤波器。

    C = [d(theta_s), d(theta_1), ...]，g = [1, 0, ...]^T，h = C (C^H C + eps I)^{-1} g，
    eps = eps_rel * tr(C^H C) / K（K = 约束个数；单零点时即 CLAUDE.md 的 tr/2）。
    f = 0 处直接使用 h_D。
    renormalize=True（默认，CLAUDE.md 4.1 节澄清，阶段 2 已确认）：正则化解再做标量重新归一化
    h <- h / conj(h^H d_s)，使无失真约束精确成立；零点、波束图与 WNG 不变。
    renormalize=False 为未归一化的正则化解（低频约束残差可达 5.8e-5），仅用于对照。

    返回 (h [F, M], info)，info 含正则化后的约束残差、零点响应与 C^H C 条件数。
    """
    nulls = np.atleast_1d(np.asarray(null_deg, dtype=np.float64))
    ds = steer(freqs, theta_s, n_mics, spacing, c)
    Cm = np.stack([ds] + [steer(freqs, t, n_mics, spacing, c) for t in nulls], axis=-1)  # [F, M, K]
    K = Cm.shape[-1]
    A = np.einsum("fmk,fml->fkl", Cm.conj(), Cm)  # C^H C
    eps = eps_rel * np.trace(A, axis1=1, axis2=2).real / K
    g = np.zeros(K, np.complex128)
    g[0] = 1.0
    x = np.linalg.solve(A + eps[:, None, None] * np.eye(K), np.broadcast_to(g, (len(freqs), K))[..., None])[..., 0]
    h = np.einsum("fmk,fk->fm", Cm, x)
    f0 = np.asarray(freqs) == 0
    h[f0] = ds[f0] / n_mics
    resp_s = np.einsum("fm,fm->f", h.conj(), ds)
    if renormalize:
        h = h / resp_s.conj()[:, None]
        resp_s = np.einsum("fm,fm->f", h.conj(), ds)
    null_resp = np.stack([np.abs(np.einsum("fm,fm->f", h.conj(), Cm[..., k])) for k in range(1, K)], -1)
    info = {
        "constraint_residual": np.abs(resp_s - 1.0),
        "null_response": null_resp,  # [F, K-1]
        "cond_CHC": np.linalg.cond(A),
        "eps": eps,
    }
    return h, info


def build_dictionary(freqs, null_angles=(90.0, 120.0, 150.0, 180.0), second_order=(), theta_s=0.0, n_mics=8,
                     spacing=0.01, c=C_SOUND, eps_rel=1e-8, renormalize=True):
    """字典：第 0 个为 h_D，其后依次为单零点滤波器（null_angles）与二阶（两个零点）滤波器（second_order）。

    返回 dict(names, H [P_total, F, M], info{name: info})。
    """
    names = ["h_D"]
    Hs = [delay_and_sum(freqs, theta_s, n_mics, spacing, c)]
    infos = {}
    for a in null_angles:
        h, info = null_filter(freqs, a, theta_s, n_mics, spacing, c, eps_rel, renormalize)
        names.append(f"null-{a:g}")
        Hs.append(h)
        infos[names[-1]] = info
    for pair in second_order:
        h, info = null_filter(freqs, list(pair), theta_s, n_mics, spacing, c, eps_rel, renormalize)
        names.append("null-" + "-".join(f"{a:g}" for a in pair))
        Hs.append(h)
        infos[names[-1]] = info
    return {"names": names, "H": np.stack(Hs), "freqs": np.asarray(freqs), "info": infos}


def response(h, d) -> np.ndarray:
    """h^H d，逐频点。h、d: [F, M]。"""
    return np.einsum("fm,fm->f", h.conj(), d)


def wng(h, d) -> np.ndarray:
    """白噪声增益（dB），逐频点：10 log10(|h^H d|^2 / h^H h)。"""
    num = np.abs(response(h, d)) ** 2
    den = np.einsum("fm,fm->f", h.conj(), h).real
    return 10 * np.log10(num / den)


def beampattern(h, freqs, thetas=np.arange(0, 361), n_mics=8, spacing=0.01, c=C_SOUND) -> np.ndarray:
    """|h^H d(theta)|^2（dB），[F, n_theta]。h 为 [F, M]，freqs 与 h 的频点一一对应。"""
    D = steering_vector(freqs, thetas, n_mics, spacing, c)  # [F, T, M]
    r = np.einsum("fm,ftm->ft", np.asarray(h).conj(), D)
    return 10 * np.log10(np.abs(r) ** 2 + 1e-300)


def gram(H, anchor=0):
    """g_p = h_p - h_anchor（p != anchor），G_pq(f) = g_p^H g_q。返回 (G [F, P, P] complex128, Gvec [P, F, M])。"""
    idx = [p for p in range(H.shape[0]) if p != anchor]
    Gv = H[idx] - H[anchor][None]
    G = np.einsum("pfm,qfm->fpq", Gv.conj(), Gv)
    return G, Gv


def save_dictionary(path, dic, G):
    np.savez(path, names=np.array(dic["names"]), H=dic["H"], freqs=dic["freqs"], G=G)


def load_dictionary(path):
    with np.load(path) as z:
        return {"names": [str(n) for n in z["names"]], "H": z["H"], "freqs": z["freqs"], "G": z["G"]}
