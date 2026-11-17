"""阶段 1 单元测试：几何/导向向量约定、去静音、场景采样约束、移动源拼接、混合的 SNR/SIR、失配、T60 估计。"""
import numpy as np
import pyroomacoustics as pra
import pytest
import torch

from src.config import load_config
from src.dataset import conv_source, fft_conv_sum, mix_scene, reference_rir, segment_weights
from src.geometry import mic_coords, mic_positions, source_position, steering_vector
from src.simulate import (_sample_s2_interferers, estimate_t60, mismatch_params, s1_trajectory,
                          sample_scene, scene_source_positions)
from src.speech import remove_silence
from src.stft import apply_mismatch, istft, stft

FS = 16000


@pytest.fixture(scope="module")
def cfg():
    return load_config("configs/data_dev.yaml")


def _fake_index(n_spk=6, per=5):
    return [{"id": f"s{s}-{k}", "speaker": f"s{s}"} for s in range(n_spk) for k in range(per)]


# ---------------------------------------------------------------- geometry
def test_mic_coords_center_reference():
    x = mic_coords(8, 0.01)
    assert np.isclose(x.mean(), 0) and np.isclose(x[1] - x[0], 0.01) and np.isclose(x[0], -0.035)


def _anechoic_ratio(mic_order, theta, dist=2.0):
    """自由场（max_order=0）仿真：各阵元相对中心麦克风的传递函数比，[F, M]。"""
    center = np.array([5.0, 5.0, 2.0])
    mics = np.concatenate([mic_positions(center, 8, 0.01, mic_order), center[:, None]], 1)
    room = pra.ShoeBox([10, 10, 4], fs=FS, max_order=0)
    room.add_microphone_array(mics)
    room.add_source(source_position(center, theta, dist))
    room.compute_rir()
    L = max(len(r[0]) for r in room.rir)
    H = np.stack([np.fft.rfft(np.pad(r[0], (0, L - len(r[0]))), 4096) for r in room.rir])
    f = np.fft.rfftfreq(4096, 1 / FS)
    return f, (H[:8] / H[8]).T


@pytest.mark.parametrize("theta", [0.0, 45.0, 90.0, 135.0, 180.0])
def test_steering_vector_matches_simulation(theta):
    """CLAUDE.md 的 d_m = exp(-j w x_m cos(theta)/c) 与 pyroomacoustics 仿真的相位一致（mic_order=target_first）。"""
    f, ratio = _anechoic_ratio("target_first", theta)
    sel = (f > 100) & (f < 7000)
    d = steering_vector(f[sel], theta)[:, 0, :]
    err = np.abs(np.angle(ratio[sel] * np.conj(d)))
    assert err.max() < 0.1, err.max()  # 2 m 处的近场相位误差 < 0.1 rad


def test_literal_order_gives_conjugate():
    f, ratio = _anechoic_ratio("literal", 0.0)
    sel = (f > 1000) & (f < 7000)
    d = steering_vector(f[sel], 0.0)[:, 0, :]
    assert np.abs(np.angle(ratio[sel] * d)).max() < 0.1  # 字面放置时实际导向向量是 conj(d)


# ---------------------------------------------------------------- speech
def test_remove_silence_only_long_gaps(cfg):
    rng = np.random.default_rng(0)
    burst = lambda n: 0.1 * rng.standard_normal(n)  # noqa: E731
    sil = lambda s: np.zeros(int(s * FS))  # noqa: E731
    x = np.concatenate([burst(8000), sil(0.05), burst(8000), sil(0.5), burst(8000), sil(0.3), burst(8000)])
    y = remove_silence(x, FS, cfg["speech"])
    # 0.05 s 的短静音保留，0.5 s 与 0.3 s 的长静音去除（帧边界处允许 ±40 ms 误差）
    expected = 4 * 8000 + int(0.05 * FS)
    assert abs(len(y) - expected) < 0.04 * FS * 2, (len(y), expected)


# ---------------------------------------------------------------- scenes
def test_s1_trajectory(cfg):
    assert s1_trajectory(cfg) == [90.0 + 10 * k for k in range(10)]


def test_s2_interferer_constraints(cfg):
    rng = np.random.default_rng(1)
    s2 = cfg["sources"]["s2"]
    for k in range(300):
        ints = _sample_s2_interferers(rng, cfg, moving=bool(k % 2))
        assert 2 <= len(ints) <= 3
        static = [i["trajectory_deg"][0] for i in ints if not i["moving"]]
        assert any(30 <= a <= 90 for a in static)
        for i in ints:
            assert all(30 - 1e-9 <= a <= 180 + 1e-9 for a in i["trajectory_deg"])
            if i["moving"]:
                t = np.array(i["trajectory_deg"])
                assert len(t) == 10 and np.allclose(np.abs(np.diff(t)), s2["speed_deg_per_s"])
        st = np.sort(static)
        assert np.all(np.diff(st) >= s2["min_sep_deg"])
        assert sum(i["moving"] for i in ints) == (k % 2)


@pytest.mark.parametrize("split", ["train", "test_s1", "test_s2"])
def test_scene_geometry(cfg, split):
    idx = _fake_index()
    for i in range(40):
        sc = sample_scene(cfg, split, i, idx)
        dim, c = np.array(sc["room_dim"]), np.array(sc["array_center"])
        assert np.all(c[:2] >= 1.0 - 1e-9) and np.all(c[:2] <= dim[:2] - 1.0 + 1e-9)
        assert 1.0 <= c[2] <= 1.5
        pos = scene_source_positions(sc)
        assert np.all(pos[:, :2] >= 0.5 - 1e-9) and np.all(pos[:, :2] <= dim[:2] - 0.5 + 1e-9)
        assert np.allclose(np.linalg.norm(pos - c, axis=1), 2.0)
        assert np.allclose(pos[:, 2], c[2])
        spk = [s.split("-")[0] for s in [sc["speech"]["target"]] + sc["speech"]["interferers"]]
        assert len(set(spk)) == len(spk)  # 目标与干扰来自不同说话人
        assert 20 <= sc["snr_db"] <= 40 and 0 <= sc["sir_db"] <= 10
        if split.startswith("test"):
            assert sc["t60_target"] == [0.3, 0.7][i % 2]
        # 可复现
        assert sample_scene(cfg, split, i, idx) == sc


def test_mismatch_params_scale_with_kappa(cfg):
    sc = sample_scene(cfg, "test_s1", 0, _fake_index())
    m1, m2, m0 = (mismatch_params(cfg, sc, k) for k in (1.0, 2.0, 0.0))
    assert np.all(np.abs(m1["gain_db"]) <= 1.0) and np.all(np.abs(m1["phase_deg"]) <= 5.0)
    assert np.allclose(np.array(m2["gain_db"]), 2 * np.array(m1["gain_db"]))
    assert np.allclose(np.array(m2["dx_m"]), 2 * np.array(m1["dx_m"]))
    assert np.allclose(m0["gain_db"], 0) and np.allclose(m0["dx_m"], 0)


# ---------------------------------------------------------------- mixing
def test_segment_weights():
    W = segment_weights(160000, 10, 16000, 800)
    assert torch.allclose(W.sum(0), torch.ones(160000))
    assert torch.all(W >= 0)
    # 段内（远离边界）只有一个位置有效；交叉淡化区宽 50 ms
    assert W[3, 3 * 16000 + 1000] == 1 and W[3, 3 * 16000 - 1000] == 0
    assert ((W[3] > 0) & (W[3] < 1)).sum() == 2 * 799  # 两个边界，各 799 个过渡样本


def test_segmented_conv_equals_full_conv():
    torch.manual_seed(0)
    s, R = torch.randn(160000), torch.randn(10, 4, 3000) * 0.05
    a = conv_source(s, R, 16000, 800)
    b = fft_conv_sum(segment_weights(160000, 10, 16000, 800) * s[None], R, 160000)
    assert ((a - b).norm() / b.norm()) < 1e-5


def _fake_raw(n_int=2, moving=True, L=4000, seed=0):
    rng = np.random.default_rng(seed)
    N = 160000
    S = 1 + n_int - 1 + (10 if moving else 1)
    decay = np.exp(-np.arange(L) / 800.0)
    A = (rng.standard_normal((S, 8, L)) * decay).astype(np.float32) * 0.01
    Cn = (rng.standard_normal((S, L)) * decay).astype(np.float32) * 0.01
    D = np.zeros((S, 200), np.float32)
    D[:, 133] = 0.04
    Cn[:, 133] += 0.04
    slices, k = [], 1
    for i in range(n_int):
        n = 10 if (moving and i == n_int - 1) else 1
        slices.append((k, k + n))
        k += n
    return {"index": 0, "s_target": rng.standard_normal(N).astype(np.float32),
            "s_interf": [rng.standard_normal(N).astype(np.float32) for _ in range(n_int)],
            "interf_slices": slices, "array": A[:k], "center": Cn[:k], "direct": D[:k],
            "direct_peak": np.full(k, 133), "snr_db": 27.0, "sir_db": 4.0, "t60": 0.5,
            "mm_gain_db": np.zeros(8, np.float32), "mm_phase_deg": np.zeros(8, np.float32), "noise_seed": 123}


def test_mix_scene_levels(cfg):
    raw = _fake_raw()
    o = mix_scene(raw, cfg)
    p = lambda x: x.double().pow(2).mean().item()  # noqa: E731
    assert torch.allclose(o["mix"], o["target"] + o["interf"] + o["noise"], atol=1e-6)
    ptd = p(o["ref_direct"])
    assert abs(10 * np.log10(ptd / p(o["noise"])) - 27.0) < 0.1
    o2 = mix_scene(raw, cfg)  # 确定性
    assert torch.equal(o["mix"], o2["mix"])
    # 参考信号 = 截取窗口内的中心 RIR 卷积
    h = reference_rir(torch.tensor(raw["center"][0]), 133, FS, 1.0, 50.0)
    assert h[:133 - 16].abs().sum() == 0 and h[133 + 801:].abs().sum() == 0
    assert torch.equal(h[133 - 16:133 + 801], torch.tensor(raw["center"][0][133 - 16:133 + 801]))


def test_mix_scene_sir(cfg):
    raw = _fake_raw(n_int=3, moving=False)
    raw["sir_db"] = 6.0
    # 去掉噪声与混响以便解析检查：所有 RIR 只保留直达脉冲
    for k in ("array", "center"):
        raw[k] = np.zeros_like(raw[k])
        raw[k][..., 133] = 0.04
    o = mix_scene(raw, cfg)
    p = lambda x: x.double().pow(2).mean().item()  # noqa: E731
    sir = 10 * np.log10(p(o["center_target"]) / p(o["center_interf"]))
    assert abs(sir - 6.0) < 0.3  # 三个干扰相互独立时总功率 ≈ 各自之和


# ---------------------------------------------------------------- STFT / mismatch
def test_stft_shape_and_perfect_reconstruction():
    x = torch.randn(2, 8, 16000)
    X = stft(x)
    assert X.shape[-2] == 257
    assert torch.allclose(istft(X, 16000), x, atol=1e-5)


def test_apply_mismatch():
    Y = torch.randn(8, 257, 10, dtype=torch.complex64)
    g, ph = torch.linspace(-1, 1, 8), torch.linspace(-5, 5, 8)
    Z = apply_mismatch(Y, g, ph)
    ratio = Z / Y
    assert torch.allclose(ratio.abs(), (10 ** (g / 20))[:, None, None].expand_as(ratio.abs()), atol=1e-5)
    assert torch.allclose(ratio.angle(), torch.deg2rad(ph)[:, None, None].expand_as(ratio.real), atol=1e-5)


# ---------------------------------------------------------------- T60
@pytest.mark.parametrize("t60", [0.3, 0.7])
def test_estimate_t60_synthetic(t60):
    rng = np.random.default_rng(0)
    t = np.arange(int(2 * t60 * FS)) / FS
    h = rng.standard_normal(len(t)) * 10 ** (-3 * t / t60)  # 幅度每 t60 衰减 60 dB
    assert abs(estimate_t60(h, FS, -5, -35) - t60) / t60 < 0.03
