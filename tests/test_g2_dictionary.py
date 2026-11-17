"""G2 阵列（M = 4、δ = 3 cm）的字典测试（CLAUDE.md 4.3 节，用于 MIRD 准备；与 test_stage2.py 相同的检查）。"""
import numpy as np
import pytest

from src.beamformers import build_dictionary, gram, response, steer, stft_freqs, wng

F = stft_freqs()
NULLS = (90.0, 120.0, 150.0, 180.0)
M, D = 4, 0.03


@pytest.fixture(scope="module")
def dic():
    return build_dictionary(F, NULLS, n_mics=M, spacing=D)


def test_distortionless(dic):
    ds = steer(F, 0.0, M, D)
    res = np.stack([np.abs(response(h, ds) - 1) for h in dic["H"]])[:, 1:]
    assert res.max() < 1e-6


def test_wng_delay_and_sum(dic):
    w = wng(dic["H"][0], steer(F, 0.0, M, D))
    assert np.abs(w - 10 * np.log10(M)).max() < 1e-6  # 6.02 dB


def test_null_response(dic):
    """零点方向响应 < 1e-4，只有空间混叠频点例外：f = c / (δ (1 − cos θ_null)) 处 d(θ_null) 与 d(0) 重合，
    无失真与置零不能同时满足（null-180：5.72 kHz，null-150：6.13 kHz，null-120：7.62 kHz；各 1 个频点）。"""
    for k, a in enumerate(NULLS, start=1):
        r = np.abs(response(dic["H"][k], steer(F, a, M, D)))[1:]
        bad = F[1:][r >= 1e-4]
        f_alias = 343.0 / (D * (1 - np.cos(np.deg2rad(a))))
        assert all(abs(f - f_alias) < 31.25 for f in bad), (dic["names"][k], bad, f_alias)


def test_convex_combination_wng_proposition1(dic):
    rng = np.random.default_rng(0)
    ds = steer(F, 0.0, M, D)
    H = dic["H"]
    w_min = np.stack([wng(h, ds) for h in H]).min(0)
    worst = np.inf
    for _ in range(1000):
        alpha = rng.dirichlet(np.ones(len(H)) * rng.choice([0.2, 1.0, 5.0]), size=len(F))
        h = np.einsum("fp,pfm->fm", alpha, H)
        worst = min(worst, (wng(h, ds) - w_min)[1:].min())
    assert worst >= -1e-9, worst


def test_gram_matrix(dic):
    H = dic["H"]
    G, Gv = gram(H)
    assert G.shape == (257, 4, 4) and np.allclose(G, np.conj(np.swapaxes(G, 1, 2)))
    ev = np.linalg.eigvalsh(G)
    assert (ev.min(-1) > -1e-12 * np.abs(ev).max(-1).clip(1)).all()
    ds = steer(F, 0.0, M, D)
    rng = np.random.default_rng(1)
    u = (rng.standard_normal((257, 4)) + 1j * rng.standard_normal((257, 4))) * 0.3
    h = H[0] + np.einsum("fp,pfm->fm", u, Gv)
    w_gram = -10 * np.log10(1 / M + np.einsum("fp,fpq,fq->f", u.conj(), G, u).real)
    assert np.abs(wng(h, ds) - w_gram)[1:].max() < 1e-6
