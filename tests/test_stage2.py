"""阶段 2 单元测试（CLAUDE.md 4.3 节）+ Gram 矩阵与波束图的辅助检查。

字典使用默认设置（正则化解 + 标量重新归一化，CLAUDE.md 4.1 节澄清）。
test_raw_regularized_residual 记录未归一化正则化解的低频残差，仅作对照。
"""
import numpy as np
import pytest

from src.beamformers import beampattern, build_dictionary, gram, response, steer, stft_freqs, wng

F = stft_freqs()
NULLS = (90.0, 120.0, 150.0, 180.0)
M = 8


@pytest.fixture(scope="module")
def dic():
    return build_dictionary(F, NULLS)


def test_dictionary_shape(dic):
    assert dic["H"].shape == (5, 257, 8) and dic["H"].dtype == np.complex128
    assert dic["names"] == ["h_D", "null-90", "null-120", "null-150", "null-180"]


def test_distortionless(dic):
    """所有字典滤波器、所有频点（f > 0）：|h^H d(theta_s) - 1| < 1e-6。"""
    ds = steer(F, 0.0)
    res = np.stack([np.abs(response(h, ds) - 1) for h in dic["H"]])[:, 1:]
    assert res.max() < 1e-6


def test_raw_regularized_residual():
    """对照：未归一化的正则化解在低频的约束残差（阶段 2 报告：最大 5.8e-5，31–219 Hz），归一化前后零点深度相同。"""
    raw = build_dictionary(F, NULLS, renormalize=False)
    ren = build_dictionary(F, NULLS)
    ds = steer(F, 0.0)
    res = np.stack([np.abs(response(h, ds) - 1) for h in raw["H"]])[:, 1:]
    assert 1e-6 < res.max() < 1e-4
    for k, a in enumerate(NULLS, start=1):
        nr_raw = np.abs(response(raw["H"][k], steer(F, a)))[1:]
        nr_ren = np.abs(response(ren["H"][k], steer(F, a)))[1:]
        assert np.allclose(nr_raw, nr_ren, rtol=1e-3)


def test_null_response(dic):
    """零点滤波器在零点方向：|h^H d(theta_null)| < 1e-4。"""
    for name, a in zip(dic["names"][1:], NULLS):
        h = dic["H"][dic["names"].index(name)]
        r = np.abs(response(h, steer(F, a)))[1:]
        assert r.max() < 1e-4, (name, r.max())


def test_wng_delay_and_sum():
    hD = build_dictionary(F, NULLS)["H"][0]
    w = wng(hD, steer(F, 0.0))
    assert np.abs(w - 10 * np.log10(M)).max() < 1e-6


def test_convex_combination_wng_proposition1(dic):
    """命题 1：1000 组随机凸权重（逐频点），融合滤波器的 WNG >= 字典成员 WNG 的最小值。"""
    rng = np.random.default_rng(0)
    ds = steer(F, 0.0)
    H = dic["H"]
    w_members = np.stack([wng(h, ds) for h in H])  # [P, F]
    w_min = w_members.min(0)
    worst = np.inf
    for _ in range(1000):
        alpha = rng.dirichlet(np.ones(len(H)) * rng.choice([0.2, 1.0, 5.0]), size=len(F))  # [F, P]
        h = np.einsum("fp,pfm->fm", alpha, H)
        margin = wng(h, ds) - w_min
        worst = min(worst, margin[1:].min())
    assert worst >= -1e-9, worst


def test_gram_matrix(dic):
    H = dic["H"]
    G, Gv = gram(H)
    assert G.shape == (257, 4, 4) and G.dtype == np.complex128
    assert np.allclose(G, np.conj(np.swapaxes(G, 1, 2)))  # Hermitian
    assert np.linalg.eigvalsh(G).min() > -1e-12  # 半正定
    assert np.allclose(G[0], 0)  # f = 0 处所有滤波器都等于 h_D
    ds = steer(F, 0.0)
    gd = np.abs(np.einsum("pfm,fm->pf", Gv.conj(), ds))[:, 1:]  # g_p^H d = h_p^H d - 1
    assert gd.max() < 1e-6
    # 用 Gram 计算的 WNG 与显式重建的滤波器一致：W = 1 / (1/M + u^H G u)（需要精确无失真）
    rng = np.random.default_rng(1)
    u = (rng.standard_normal((257, 4)) + 1j * rng.standard_normal((257, 4))) * 0.3
    h = H[0] + np.einsum("fp,pfm->fm", u, Gv)
    w_explicit = wng(h, ds)
    w_gram = -10 * np.log10(1 / M + np.einsum("fp,fpq,fq->f", u.conj(), G, u).real)
    assert np.abs(w_explicit - w_gram).max() < 1e-9


def test_beampattern_basic():
    d = build_dictionary(F, NULLS)
    bp = beampattern(d["H"][0], F)  # [F, 361]
    assert bp.shape == (257, 361)
    assert np.abs(bp[1:, 0]).max() < 1e-9  # 无失真 -> 0 dB
    assert np.allclose(bp[:, 1:180], bp[:, 359:180:-1], atol=1e-9)  # 线阵关于阵列轴对称
    for k, a in enumerate(NULLS, start=1):
        bpk = beampattern(d["H"][k], F)
        assert bpk[16:, int(a)].max() < -70  # 500 Hz 以上零点深度 > 70 dB
