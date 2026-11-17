"""阶段 4 单元测试：仿射融合 + WNG 投影（CLAUDE.md 6.1–6.3、6.5 节）。"""
import math

import numpy as np
import pytest
import torch

from src.beamformers import build_dictionary, gram, steer, stft_freqs
from src.config import load_config, meta_path
from src.models.gtcrn_fusion import AffineFusionNet, BeamBank, build_model, causal_ema, count_params
from src.models.projection import project, radius, scale_factor

F = stft_freqs(16000, 512)
M = 8
W0S = (-10.0, -5.0, 0.0, 5.0)
CFG_PATH = "configs/data_dev.yaml"


@pytest.fixture(scope="module")
def dic():
    d = build_dictionary(F)
    G, Gv = gram(d["H"], anchor=0)
    return {"H": d["H"], "G": G, "Gv": Gv, "d": steer(F, 0.0)}  # complex128


def _model(dic, proj="hard", w0_db=0.0, init_scale=0.01, randomize=False, seed=0, **kw):
    torch.manual_seed(seed)
    m = AffineFusionNet(5, dic["G"], M, proj=proj, w0_db=w0_db, out_init_scale=init_scale, **kw)
    if randomize:  # “随机网络输出”：输出层重新随机化为较大的权重，使大量时频点触发投影
        conv = m.backbone.output_layer.conv
        with torch.no_grad():
            conv.weight.normal_(0, 1.0)
            conv.bias.normal_(0, 1.0)
    return m.eval()


def _reconstruct(dic, u, s):
    """由 u、s 在滤波器空间显式重建 h_hat = h_D + s sum_p u_p g_p，[B, F, T, M]（complex128）。"""
    Gv = torch.as_tensor(dic["Gv"])  # [P', F, M]
    hD = torch.as_tensor(dic["H"][0])  # [F, M]
    c = torch.einsum("bpft,pfm->bftm", u.to(torch.complex128), Gv)
    return hD[None, :, None] + s.to(torch.float64)[..., None] * c


def _filter_checks(dic, h):
    d = torch.as_tensor(dic["d"])  # [F, M]
    resp = torch.einsum("bftm,fm->bft", h.conj(), d)
    wng = resp.abs() ** 2 / (h.abs() ** 2).sum(-1)
    return (resp - 1).abs().max().item(), 10 * torch.log10(wng)


def _plane_wave(theta, X):
    d = torch.as_tensor(steer(F, theta), dtype=X.dtype)
    return d.T[None, :, :, None] * X[:, None]


# ------------------------------------------------------------------------------------------- 6.5 / 6.2 性质
@pytest.mark.parametrize("proj", ["hard", "smooth"])
@pytest.mark.parametrize("w0", W0S)
def test_projection_filter_space(dic, proj, w0):
    """随机网络输出、随机输入：由 u 显式重建 8 维滤波器，逐时频点 |h^H d - 1| < 1e-5，WNG >= W0 - 1e-6 dB。"""
    m = _model(dic, proj, w0, randomize=True).double()
    Z = torch.randn(2, 5, 257, 30, dtype=torch.complex128)
    with torch.no_grad():
        o = m(Z)
    h = _reconstruct(dic, o["u"], o["s"])
    err, wng = _filter_checks(dic, h)
    assert err < 1e-5
    assert wng.min().item() >= w0 - 1e-6
    assert (o["s"] < 1).float().mean() > 0.1  # 确实有相当比例的时频点被投影（W0 越低比例越小）
    assert torch.allclose(wng, o["wng_db"], atol=1e-6)  # 波束输出空间公式 = 滤波器空间实测
    # 波束输出空间的输出 = 由重建滤波器直接作用于阵列信号（Z_p = h_p^H y）
    Y = torch.randn(2, M, 257, 30, dtype=torch.complex128)
    Zy = torch.einsum("pfm,bmft->bpft", torch.as_tensor(dic["H"]).conj(), Y)
    with torch.no_grad():
        o = m(Zy)
    h = _reconstruct(dic, o["u"], o["s"])
    S_filter = torch.einsum("bftm,bmft->bft", h.conj(), Y)
    assert torch.allclose(o["S"], S_filter, atol=1e-9 * Y.abs().max())


@pytest.mark.parametrize("w0", W0S)
def test_projection_filter_space_float32(dic, w0):
    """训练精度（complex64）下：无失真误差 < 1e-5，WNG 相对 W0 的违反 < 1e-4 dB（float32 舍入）。"""
    m = _model(dic, "hard", w0, randomize=True)
    with torch.no_grad():
        o = m(torch.randn(2, 5, 257, 30, dtype=torch.complex64))
    err, wng = _filter_checks(dic, _reconstruct(dic, o["u"], o["s"]))
    assert err < 1e-5 and wng.min().item() >= w0 - 1e-4


def test_learnable_w0_filter_space(dic):
    m = _model(dic, "hard", "learn", randomize=True).double()
    with torch.no_grad():
        o = m(torch.randn(2, 5, 257, 30, dtype=torch.complex128))
    w0 = o["w0_db"]  # [B, F, T]（每帧，因果递归平均）
    assert w0.shape == o["s"].shape
    assert w0.min() >= -15 and w0.max() <= 9 < 10 * math.log10(M)
    err, wng = _filter_checks(dic, _reconstruct(dic, o["u"], o["s"]))
    assert err < 1e-5 and (wng - w0).min().item() >= -1e-6


def test_learnable_w0_two_stage(dic):
    m = _model(dic, "hard", "learn", randomize=True, w0_freeze_epochs=20, w0_init_db=0.0)
    Z = torch.randn(1, 5, 257, 20, dtype=torch.complex64)
    m.set_epoch(0)
    with torch.no_grad():
        assert torch.all(m(Z)["w0_db"] == 0.0)
    m.set_epoch(20)
    with torch.no_grad():
        assert not torch.all(m(Z)["w0_db"] == 0.0)


def test_hard_identity_when_feasible(dic):
    """原本 WNG >= W0 时 hard 投影 s = 1，输出不变。"""
    torch.manual_seed(0)
    u = 1e-3 * torch.randn(3, 4, 257, 10, dtype=torch.complex128)
    for w0 in W0S:
        s, n, wng = project(u, torch.as_tensor(dic["G"]), 10 ** (w0 / 10), M, "hard")
        feas = 1 / (1 / M + n ** 2) >= 10 ** (w0 / 10)
        assert feas.all() and torch.all(s == 1.0)
    m_none, m_hard = _model(dic, "none"), _model(dic, "hard", -10.0)  # 同一初始化
    Z = torch.randn(1, 5, 257, 20, dtype=torch.complex64)
    with torch.no_grad():
        on, oh = m_none(Z), m_hard(Z)
    assert torch.all(oh["s"] == 1.0) and torch.equal(on["S"], oh["S"])


def test_smooth_never_relaxes_constraint():
    """smooth：s <= min(1, r/n) 对所有 n 成立；n << r 时 s 接近 1。"""
    for w0 in W0S:
        r = radius(10 ** (w0 / 10), M)
        n = torch.logspace(-6, 3, 20001, dtype=torch.float64) * r
        s = scale_factor(n, r, "smooth")
        assert torch.all(s <= torch.clamp(r / n, max=1.0) * (1 + 1e-12))
        assert s[n < 0.1 * r].min() > 0.999
        sh = scale_factor(n, r, "hard")
        assert torch.all(s <= sh * (1 + 1e-12))


@pytest.mark.parametrize("mode", ["hard", "smooth"])
def test_projection_gradcheck(dic, mode):
    torch.manual_seed(1)
    G = torch.as_tensor(dic["G"][[40, 120, 200]])  # 3 个频点
    ur = torch.randn(1, 4, 3, 5, dtype=torch.float64, requires_grad=True)
    ui = torch.randn(1, 4, 3, 5, dtype=torch.float64, requires_grad=True)
    w0 = torch.tensor([[[-2.0], [0.0], [3.0]]], dtype=torch.float64, requires_grad=True)  # [B, F, 1]（可学习 W0）
    Dr = torch.randn(1, 4, 3, 5, dtype=torch.float64)

    def f(ur, ui, w0):
        u = torch.complex(ur, ui) * 30.0
        s, _, _ = project(u, G, 10 ** (w0 / 10), M, mode)
        return s * (u.conj() * Dr).sum(1).real, s * (u.conj() * Dr).sum(1).imag

    assert torch.autograd.gradcheck(f, (ur, ui, w0), eps=1e-6, atol=1e-5)


# ------------------------------------------------------------------------------------------- 6.1 初始化 / 端到端
def test_init_is_near_delay_and_sum(dic):
    m = _model(dic, "none")
    Z = torch.randn(2, 5, 257, 40, dtype=torch.complex64)
    with torch.no_grad():
        o = m(Z)
    assert o["w"].abs().max() < 0.1
    rel = (o["S"] - Z[:, 0]).abs().pow(2).sum() / Z[:, 0].abs().pow(2).sum()
    assert rel < 1e-3


@pytest.mark.parametrize("proj,w0", [("none", 0.0), ("hard", 0.0), ("smooth", -5.0), ("hard", "learn")])
def test_distortionless_plane_wave(dic, proj, w0):
    """只有来自 theta_s 的平面波目标时，任意（随机、较大的）融合系数下输出都等于目标。"""
    bank = BeamBank(dic["H"])
    X = torch.randn(2, 257, 50, dtype=torch.complex64)
    Z = bank(_plane_wave(0.0, X))
    m = _model(dic, proj, w0, randomize=True)
    with torch.no_grad():
        o = m(Z)
    assert o["w"].abs().max() > 1  # 确实是远离单纯形的仿射系数
    assert torch.allclose(o["S"], X, atol=1e-3 * X.abs().max())


def test_distortionless_end_to_end_direct_path():
    """数据集中只含目标直达声分量的样本：各变体（初始化状态）的输出与 x_dp（中心处直达声）误差很小。"""
    cfg = load_config(CFG_PATH)
    if not meta_path(cfg, "val").exists():
        pytest.skip("需要阶段 1 生成的数据")
    from src.dataset import BeamFusionDataset
    from src.metrics import snr
    from src.pipeline import array_stft
    from src.stft import istft
    bank = BeamBank.from_config(cfg)
    b = BeamFusionDataset(cfg, "val", mix_on="worker", return_components=True, max_items=1)[0]
    Z = bank(array_stft(b["target_direct"][None]))
    ref = b["ref_direct"].numpy()
    for name in ("affine", "affine_proj", "affine_proj_learn"):
        from src.train import load_model_config
        mc = load_model_config(f"configs/model_{name}.yaml")
        torch.manual_seed(0)
        m = build_model(mc, cfg, bank).eval()
        with torch.no_grad():
            S = m(Z)["S"]
        out = istft(S, ref.shape[-1])[0].numpy()
        assert snr(out, ref) > 30, (name, snr(out, ref))


def test_variant_configs_and_param_count(dic):
    cfg = load_config(CFG_PATH)
    bank = BeamBank.from_config(cfg)
    from src.train import load_model_config
    for name, n_out in (("affine", 8), ("affine_proj", 8), ("affine_proj_learn", 9)):
        mc = load_model_config(f"configs/model_{name}.yaml")
        assert mc["train"] == load_model_config("configs/model_beamfusion.yaml")["train"]  # 训练设置与基线相同
        assert mc["seed_key"] == "beamfusion"
        m = build_model(mc, cfg, bank)
        assert m.backbone.output_layer.conv.out_channels == n_out
        assert torch.all(m.backbone.output_layer.conv.bias == 0)
        assert np.allclose(m.G.numpy(), dic["G"])
        assert 80_000 < count_params(m) < 95_000


def test_fixed_w0_model_is_causal(dic):
    m = _model(dic, "hard", 0.0, randomize=True)
    Z1 = torch.randn(1, 5, 257, 60, dtype=torch.complex64)
    Z2 = Z1.clone()
    Z2[..., 40:] = torch.randn(1, 5, 257, 20, dtype=torch.complex64)
    with torch.no_grad():
        s1, s2 = m(Z1)["S"], m(Z2)["S"]
    assert torch.allclose(s1[..., :40], s2[..., :40], atol=1e-5)


# ------------------------------------------------------------------------------------------- 6.3 因果可学习 W0
def test_causal_ema_matches_recursion():
    torch.manual_seed(4)
    x = torch.randn(2, 3, 700, dtype=torch.float64)
    lam = math.exp(-1 / (0.5 * 125))
    y = causal_ema(x, lam)
    ref = torch.empty_like(x)
    ref[..., 0] = x[..., 0]
    for t in range(1, x.shape[-1]):
        ref[..., t] = lam * ref[..., t - 1] + (1 - lam) * x[..., t]
    assert torch.allclose(y, ref, atol=1e-10)
    # 时间常数：阶跃响应在 0.5 s（62.5 帧）处达到 1 - 1/e
    step = torch.cat([torch.zeros(1, 1, 1, dtype=torch.float64), torch.ones(1, 1, 200, dtype=torch.float64)], -1)
    r = causal_ema(step, lam)[0, 0]
    assert abs(r[63].item() - (1 - lam ** 63)) < 1e-12 and 0.62 < r[63].item() < 0.64
    xg = torch.randn(1, 2, 300, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda v: causal_ema(v, lam), (xg,), eps=1e-6, atol=1e-6)


def test_learnable_w0_is_causal_and_per_frame(dic):
    m = _model(dic, "hard", "learn", randomize=True)
    Z1 = torch.randn(1, 5, 257, 120, dtype=torch.complex64)
    Z2 = Z1.clone()
    Z2[..., 80:] = torch.randn(1, 5, 257, 40, dtype=torch.complex64)
    with torch.no_grad():
        o1, o2 = m(Z1), m(Z2)
    assert torch.allclose(o1["w0_db"][..., :80], o2["w0_db"][..., :80], atol=1e-5)
    assert torch.allclose(o1["S"][..., :80], o2["S"][..., :80], atol=1e-5)
    assert not torch.allclose(o1["w0_db"][..., 80:], o2["w0_db"][..., 80:], atol=1e-4)
    assert (o1["w0_db"].std(-1) > 0).all()  # 每帧不同（不是整句常数）


# ------------------------------------------------------------------------------------------- 评测（阶段 4 / 5）
def _ckpt(tmp_path, name, **mset):
    from src.train import load_model_config
    cfg = load_config(CFG_PATH)
    mc = load_model_config(f"configs/model_{name}.yaml", [f"{k}={v}" for k, v in mset.items()])
    torch.manual_seed(0)
    m = build_model(mc, cfg, BeamBank.from_config(cfg))
    with torch.no_grad():  # 较大的输出，使投影与低 WNG 都出现
        m.backbone.output_layer.conv.weight.normal_(0, 0.5)
    p = tmp_path / f"{name}.pt"
    torch.save({"model": m.state_dict(), "model_config": mc, "epoch": 0}, p)
    return cfg, p


@pytest.mark.parametrize("name", ["affine", "affine_proj", "affine_proj_learn"])
def test_eval_affine_coefficients_and_wng(tmp_path, name):
    """评测用的逐波束复系数 c 复现模型输出；滤波器空间的实测 WNG 与模型自身给出的 WNG 一致。"""
    from src.evaluate import ModelMethod, coef_wng_db, dictionary_gram
    cfg, p = _ckpt(tmp_path, name, **({"w0_freeze_epochs": 0} if name.endswith("learn") else {}))
    bank = BeamBank.from_config(cfg)
    m = ModelMethod("x", str(p), cfg, bank, torch.device("cpu"))
    Z = torch.randn(2, 5, 257, 40, dtype=torch.complex64)
    with torch.no_grad():
        o = m.model(Z)
    c = m.coef(Z, None)
    assert torch.allclose(c.sum(1), torch.ones_like(c[:, 0]), atol=1e-5)  # 无失真：sum_p c_p = 1
    assert torch.allclose(m.apply(c, Z, None), o["S"], atol=1e-5 * Z.abs().max())
    K, d = dictionary_gram(cfg)
    w = coef_wng_db(m, c, K, d)
    assert torch.allclose(w[:, 1:].double(), o["wng_db"][:, 1:].double(), atol=2e-3)
    if name != "affine":  # 评测默认以 float64 计算投影：滤波器空间的 WNG 与模型给出的一致，且不低于 W0（float64 精度）
        assert c.dtype == torch.complex128 and o["wng_db"].dtype == torch.float64
        assert torch.allclose(w[:, 1:], o["wng_db"][:, 1:], atol=1e-8)
        w0 = o["w0_db"] if torch.is_tensor(o["w0_db"]) else torch.tensor(o["w0_db"])
        assert (w[:, 1:] - w0[:, 1:] if torch.is_tensor(o["w0_db"]) else w[:, 1:] - w0).min() > -1e-9


@pytest.mark.parametrize("name", ["beamfusion", "affine", "affine_proj"])
def test_eval_apply_reproduces_model_output(tmp_path, name):
    """ModelMethod.apply(coef, Z) 与模型自身的输出 S 一致，且保持复数（回归测试：ec07666 中 Z.to(coef.dtype)
    对凸组合的实值 alpha 把 Z 转成实数、丢掉了虚部，crbf-eval-kappa-a / -b 中 BeamFusion 的 κ = 0.5、1.5、2 结果因此作废）。"""
    from src.evaluate import ModelMethod
    cfg, p = _ckpt(tmp_path, name)
    m = ModelMethod("x", str(p), cfg, BeamBank.from_config(cfg), torch.device("cpu"))
    torch.manual_seed(1)
    Z = torch.randn(2, 5, 257, 40, dtype=torch.complex64)
    with torch.no_grad():
        o = m.model(Z)
        out = m.apply(m.coef(Z, None), Z, None)
    assert out.dtype == torch.complex64 and out.shape == o["S"].shape
    assert out.imag.abs().max() > 0.1
    assert torch.allclose(out, o["S"], atol=1e-5 * Z.abs().max())


def test_eval_learnable_w0_freeze_state_follows_checkpoint(tmp_path):
    """评测时可学习 W0 的冻结状态与训练该检查点时相同：epoch >= w0_freeze_epochs 时用网络给出的 W0，
    之前为固定的 w0_init_db；缺少 epoch 时报错（回归：build_model 后默认冻结，评测曾不调用 set_epoch）。"""
    from src.evaluate import ModelMethod
    cfg, p = _ckpt(tmp_path, "affine_proj_learn")  # 配置中 w0_freeze_epochs = 20
    bank = BeamBank.from_config(cfg)
    ck = torch.load(p, weights_only=False)
    assert ck["model_config"]["w0_freeze_epochs"] == 20
    Z = torch.randn(1, 5, 257, 30, dtype=torch.complex64)
    for ep, frozen in ((19, True), (20, False), (49, False)):
        torch.save({**ck, "epoch": ep}, p)
        m = ModelMethod("x", str(p), cfg, bank, torch.device("cpu"))
        m.coef(Z, None)
        w0 = m.last_w0
        assert w0.shape == (1, 257, 30)
        if frozen:
            assert torch.all(w0 == 0.0)
        else:
            assert w0.std() > 0.1 and w0.min() >= -15.0 and w0.max() <= 9.0
    torch.save({k: v for k, v in ck.items() if k != "epoch"}, p)
    with pytest.raises(ValueError):
        ModelMethod("x", str(p), cfg, bank, torch.device("cpu"))


def test_w0_stats(tmp_path):
    """可学习 W0 分布统计：逐频点分位数、处于下界（-15 dB）的比例与精确计数一致；f = 0 不计入。"""
    import csv
    import json
    from src.evaluate import accumulate_w0, write_w0_stats
    g = torch.Generator().manual_seed(0)
    w0 = -15.0 + 24.0 * torch.rand(3, 257, 50, generator=g)
    w0[:, 1:20] = -15.0  # 低频全部在下界
    w0[:, 100, :10] = -14.95  # 下界以上 0.05 dB：计入 0.1 dB，不计入 0.01 dB
    w0[:, 0] = -15.0  # f = 0 不统计
    st = {}
    accumulate_w0(st, "M", w0[:2])
    accumulate_w0(st, "M", w0[2:])
    summ = write_w0_stats(st, str(tmp_path / "t"))["M"]
    x = w0[:, 1:].double()
    assert summ["n_tf"] == x.numel()
    assert summ["frac_at_lo_0.01db"] == pytest.approx(float((x <= -14.99).double().mean()))
    assert summ["frac_at_lo_0.1db"] == pytest.approx(float((x <= -14.9).double().mean()))
    assert summ["mean_db"] == pytest.approx(float(x.mean()))
    rows = list(csv.DictReader(open(tmp_path / "t_w0_freq.csv")))
    assert len(rows) == 256 and float(rows[0]["freq_Hz"]) == 31.25
    assert float(rows[0]["frac_at_lo_0.01db"]) == 1.0 and float(rows[0]["p99"]) == -15.0
    for e in (0.01, 0.1, 1.0):  # 逐频点比例与精确计数一致（f = 100 处有 1/5 的点在下界以上 0.05 dB）
        assert float(rows[99][f"frac_at_lo_{e:g}db"]) == pytest.approx(float((x[:, 99] <= -15 + e).double().mean()), abs=1e-6)
    assert float(rows[99]["frac_at_lo_0.1db"]) >= 0.2
    r = rows[150]
    assert -15 <= float(r["p1"]) <= float(r["p50"]) <= float(r["p99"]) <= 9
    assert abs(float(r["p50"]) - float(x[:, 150].median())) < 1.0
    json.loads((tmp_path / "t_w0_summary.json").read_text())


def test_eval_wng_fixed_beams_and_mvdr(dic):
    from src.beamformers import wng
    from src.evaluate import FixedBeam, coef_wng_db, dictionary_gram
    cfg = load_config(CFG_PATH)
    K, d = dictionary_gram(cfg)
    Z = torch.randn(1, 5, 257, 3, dtype=torch.complex64)
    for p in range(5):
        fb = FixedBeam(p, str(p))
        w = coef_wng_db(fb, fb.coef(Z, None), K, d)[0, 1:, 0].numpy()
        assert np.allclose(w, wng(dic["H"][p][1:], dic["d"][1:]), atol=1e-6)
    from src.models.gtcrn_fusion import MaskMVDR

    class _M:
        coef_kind = "h"
    mv = MaskMVDR(5, dic["d"])
    Y = torch.randn(1, M, 257, 50, dtype=torch.complex64)
    h = mv.filters(Y, torch.zeros(1, 257, 50), load=1e-3)
    w = coef_wng_db(_M(), h, K, d)[0, 1:, 0].numpy()
    ref = wng(h[0, 1:].numpy().astype(np.complex128), dic["d"][1:])
    assert np.allclose(w, ref, atol=1e-4)


def test_eval_wng_histograms(tmp_path):
    from src.evaluate import accumulate_wng, write_wng_stats
    st = {}
    w = torch.tensor(np.random.default_rng(0).normal(0, 5, (2, 257, 30)))
    accumulate_wng(st, "a", w, 30)
    accumulate_wng(st, "b", w[..., :1], 30)  # MVDR 式：每句一个滤波器，按帧数加权
    s = write_wng_stats(st, str(tmp_path / "x"))
    assert s["a"]["n_tf"] == 2 * 256 * 30 == s["b"]["n_tf"]
    assert abs(s["a"]["p50_db"]) < 0.5 and abs(s["a"]["min_db"] - float(w[:, 1:].min())) < 1e-9
    assert abs(s["a"]["frac_below_0db"] - float((w[:, 1:] < -1e-3).double().mean())) < 1e-12  # 精确计数
    st2 = {}
    accumulate_wng(st2, "p", torch.full((1, 257, 10), -1e-6, dtype=torch.float64), 10)  # 恰在投影边界上
    s2 = write_wng_stats(st2, str(tmp_path / "y"))
    assert s2["p"]["frac_below_0db"] == 0.0 and s2["p"]["p1_db"] == 0.0
    assert (tmp_path / "x_wng_freq.csv").exists() and (tmp_path / "x_wng_hist.csv").exists()


def test_evaluate_affine_end_to_end(tmp_path):
    """evaluate() 对仿射模型：shadow 分解一致，两套参考的指标与 WNG 统计都输出。"""
    cfg, p = _ckpt(tmp_path, "affine_proj")
    if not meta_path(cfg, "val").exists():
        pytest.skip("需要阶段 1 生成的数据")
    from src.evaluate import evaluate, parse_methods
    bank = BeamBank.from_config(cfg)
    methods = parse_methods([f"ckpt:AP={p}", "fixed"], cfg, bank, torch.device("cpu"))
    ws = {}
    rows = evaluate(cfg, "val", methods, bank, torch.device("cpu"), max_items=2, batch_size=2, num_workers=0,
                    metric_workers=1, do_pq=False, log=lambda *a: None, wng_stats=ws)
    ap = [r for r in rows if r["method"] == "AP"]
    assert len(ap) == 2 and all(r["shadow_consistency_err"] < 1e-4 for r in ap)
    assert all(np.isfinite(r["d_si_sdr_dp"]) and np.isfinite(r["d_si_sdr"]) for r in ap)
    assert ws["AP"]["min"] > -1e-3  # 投影后 WNG >= 0 dB
    assert abs(ws["h_D"]["min"] - 10 * math.log10(8)) < 1e-4


# ------------------------------------------------------------------------------------------- 训练时在线失配（第 3、4 批）
def _cfg_tm(enabled=True):
    cfg = load_config(CFG_PATH, [f"train_mismatch.enabled={'true' if enabled else 'false'}"])
    if not meta_path(cfg, "train").exists():
        pytest.skip("需要阶段 1 生成的数据")
    return cfg


def test_train_mismatch_params_ranges_and_determinism():
    from src.dataset import train_mismatch_params
    cfg = load_config(CFG_PATH, ["train_mismatch.enabled=true"])
    assert train_mismatch_params(load_config(CFG_PATH), 0, 0, M) is None  # 默认关闭
    a, b = train_mismatch_params(cfg, 7, 3, M), train_mismatch_params(cfg, 7, 3, M)
    assert np.array_equal(a["gain_db"], b["gain_db"]) and a["kappa"] == b["kappa"]  # 同一样本、同一 epoch 可复现
    c = train_mismatch_params(cfg, 7, 4, M)
    assert not np.array_equal(a["gain_db"], c["gain_db"])  # 每个 epoch 重新抽取
    ks, gs, ps = [], [], []
    for i in range(2000):
        t = train_mismatch_params(cfg, i, 0, M)
        ks.append(t["kappa"])
        gs.append(np.abs(t["gain_db"]).max() / max(t["kappa"], 1e-12))
        ps.append(np.abs(t["phase_deg"]).max() / max(t["kappa"], 1e-12))
    ks = np.array(ks)
    assert ks.min() >= 0 and ks.max() <= 1 and abs(ks.mean() - 0.5) < 0.03  # kappa ~ U[0, 1]
    assert max(gs) <= 1.0 + 1e-6 and max(ps) <= 5.0 + 1e-5  # |g| <= kappa * 1 dB，|phi| <= kappa * 5 度
    assert min(gs) > 0.3 and min(ps) > 1.0  # 确实按 kappa 缩放，而不是恒为 0


def test_train_mismatch_applied_to_train_only():
    from src.dataset import BeamFusionDataset
    from src.pipeline import prepare
    from src.stft import apply_mismatch, stft
    cfg_on, cfg_off = _cfg_tm(True), _cfg_tm(False)
    bank = BeamBank.from_config(cfg_on)
    on = BeamFusionDataset(cfg_on, "train", mix_on="worker", return_components=False, max_items=2)
    off = BeamFusionDataset(cfg_off, "train", mix_on="worker", return_components=False, max_items=2)
    for ds in (on, off):
        ds.set_epoch(5)
    bo, bf = on[1], off[1]
    assert torch.all(bf["mm_gain_db"] == 0) and torch.all(bf["mm_phase_deg"] == 0)
    assert bo["mm_gain_db"].abs().max() > 0 and bo["mm_gain_db"].abs().max() <= 1.0 + 1e-6
    assert bo["mm_phase_deg"].abs().max() <= 5.0 + 1e-5
    for k in ("mix", "ref", "ref_direct"):  # 时域信号与训练目标不变（失配只在 STFT 域作用于阵列观测）
        assert torch.equal(bo[k], bf[k]), k
    Yo, _, Xo = prepare({k: v[None] for k, v in bo.items()}, bank, target_key="ref_direct")
    Yf, _, Xf = prepare({k: v[None] for k, v in bf.items()}, bank, target_key="ref_direct")
    assert torch.equal(Xo, Xf)
    assert torch.allclose(Yo, apply_mismatch(stft(bf["mix"][None]), bo["mm_gain_db"][None], bo["mm_phase_deg"][None]))
    val = BeamFusionDataset(cfg_on, "val", mix_on="worker", return_components=False, max_items=1)[0]
    assert torch.all(val["mm_gain_db"] == 0) and torch.all(val["mm_phase_deg"] == 0)  # 验证集不加


# ------------------------------------------------------------------------------------------- 含失配训练的验证集（第 3、4 批）
def test_val_mismatch_switch():
    from src.dataset import val_mismatch_enabled
    base = load_config(CFG_PATH)
    assert base["train_mismatch"]["val"] is True and not val_mismatch_enabled(base)  # 未启用含失配训练：无失配验证集
    assert val_mismatch_enabled(load_config(CFG_PATH, ["train_mismatch.enabled=true"]))
    assert not val_mismatch_enabled(load_config(CFG_PATH, ["train_mismatch.enabled=true", "train_mismatch.val=false"]))


def test_val_mismatch_params_fixed_and_same_distribution():
    from src.dataset import train_mismatch_params, val_mismatch_params
    cfg = load_config(CFG_PATH, ["train_mismatch.enabled=true"])
    a, b = val_mismatch_params(cfg, 7, M), val_mismatch_params(cfg, 7, M)
    assert np.array_equal(a["gain_db"], b["gain_db"]) and np.array_equal(a["phase_deg"], b["phase_deg"])  # 固定种子
    assert not np.array_equal(a["gain_db"], val_mismatch_params(cfg, 8, M)["gain_db"])  # 按样本不同
    assert not any(np.array_equal(a["gain_db"], train_mismatch_params(cfg, 7, e, M)["gain_db"]) for e in range(5))
    ks, gs, ps = [], [], []
    for i in range(2000):
        t = val_mismatch_params(cfg, i, M)
        ks.append(t["kappa"])
        gs.append(np.abs(t["gain_db"]).max() / max(t["kappa"], 1e-12))
        ps.append(np.abs(t["phase_deg"]).max() / max(t["kappa"], 1e-12))
    ks = np.array(ks)
    assert ks.min() >= 0 and ks.max() <= 1 and abs(ks.mean() - 0.5) < 0.03  # 与训练相同：kappa ~ U[0, 1]
    assert max(gs) <= 1.0 + 1e-6 and max(ps) <= 5.0 + 1e-5 and min(gs) > 0.3 and min(ps) > 1.0
    # 与训练失配的分布一致（两样本 KS 检验，各 2000 个 kappa）
    from scipy.stats import ks_2samp
    kt = [train_mismatch_params(cfg, i, 0, M)["kappa"] for i in range(2000)]
    assert ks_2samp(ks, kt).pvalue > 0.01


def test_test_snr_override():
    """自噪声扫描：sources.test_snr_override_db 只改变测试集的白噪声 SNR；目标、干扰与噪声实现不变（噪声只是缩放）；
    验证集不受影响。"""
    from src.dataset import BeamFusionDataset
    base = load_config(CFG_PATH)
    assert base["sources"]["test_snr_override_db"] is None
    kw = dict(mix_on="worker", return_components=True, max_items=2)
    a = BeamFusionDataset(base, "test_s1", **kw)[1]
    for snr in (40.0, 10.0, 0.0):
        b = BeamFusionDataset(load_config(CFG_PATH, [f"sources.test_snr_override_db={snr}"]), "test_s1", **kw)[1]
        assert float(b["snr_db"]) == snr
        for k in ("target", "interf", "ref", "ref_direct", "mm_gain_db", "mm_phase_deg"):
            assert torch.equal(a[k], b[k]), k
        p_td = b["ref_direct"].double().pow(2).mean()
        p_n = b["noise"].double().pow(2).mean()
        assert abs(10 * torch.log10(p_td / p_n).item() - snr) < 0.2  # 实际 SNR（相对中心处直达声功率）
        ratio = b["noise"].double() / a["noise"].double()  # 同一噪声实现，只是缩放
        expect = 10 ** ((float(a["snr_db"]) - snr) / 20)
        assert torch.allclose(ratio, torch.full_like(ratio, expect), rtol=1e-4)
    cfg = load_config(CFG_PATH, ["sources.test_snr_override_db=0"])
    v0 = BeamFusionDataset(base, "val", **kw)[0]
    v1 = BeamFusionDataset(cfg, "val", **kw)[0]
    assert float(v0["snr_db"]) == float(v1["snr_db"]) and torch.equal(v0["mix"], v1["mix"])


def test_val_mismatch_dataset():
    from src.dataset import BeamFusionDataset, val_mismatch_params
    from src.pipeline import prepare
    from src.stft import apply_mismatch, stft
    cfg = _cfg_tm(True)
    bank = BeamBank.from_config(cfg)
    on = BeamFusionDataset(cfg, "val", mix_on="worker", return_components=False, max_items=2, val_mismatch=True)
    off = BeamFusionDataset(cfg, "val", mix_on="worker", return_components=False, max_items=2)
    b1 = on[1]
    on.set_epoch(9)
    b2 = on[1]
    exp = val_mismatch_params(cfg, on.scenes[1]["index"], M)
    assert np.allclose(b1["mm_gain_db"].numpy(), exp["gain_db"]) and np.allclose(b1["mm_phase_deg"].numpy(), exp["phase_deg"])
    assert torch.equal(b1["mm_gain_db"], b2["mm_gain_db"]) and torch.equal(b1["mix"], b2["mix"])  # 与 epoch 无关
    bf = off[1]
    assert torch.all(bf["mm_gain_db"] == 0)  # 默认（评测、调参）不加
    for k in ("mix", "ref_direct"):
        assert torch.equal(b1[k], bf[k])
    Yo, _, Xo = prepare({k: v[None] for k, v in b1.items()}, bank, target_key="ref_direct")
    _, _, Xf = prepare({k: v[None] for k, v in bf.items()}, bank, target_key="ref_direct")
    assert torch.equal(Xo, Xf)  # 训练目标不变
    assert torch.allclose(Yo, apply_mismatch(stft(bf["mix"][None]), b1["mm_gain_db"][None], b1["mm_phase_deg"][None]))
    with pytest.raises(ValueError):
        BeamFusionDataset(cfg, "train", max_items=1, val_mismatch=True)
    with pytest.raises(ValueError):
        BeamFusionDataset(cfg, "val", kappa=1.0, max_items=1, val_mismatch=True)


def _val_loss(cfg, ckpt, val_mismatch, n):
    from src.dataset import BeamFusionDataset, train_target_key
    from src.pipeline import prepare
    from src.train import stft_mse
    bank = BeamBank.from_config(cfg)
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    m = build_model(ck["model_config"], cfg, bank).eval()
    m.load_state_dict(ck["model"])
    ds = BeamFusionDataset(cfg, "val", mix_on="worker", return_components=False, max_items=n, val_mismatch=val_mismatch)
    with torch.no_grad():
        b = {k: torch.stack([ds[i][k] for i in range(n)]) for k in ds[0]}
        Y, Z, X = prepare(b, bank, target_key=train_target_key(cfg))
        return stft_mse(m(Z)["S"], X).item()


@pytest.mark.parametrize("mismatch", [True, False])
def test_train_selects_on_matching_val_set(tmp_path, mismatch):
    """train.py：含失配训练时验证损失（学习率调度与最佳检查点选择）在含失配的验证集上计算，否则在无失配的验证集上。"""
    import json
    from src.train import main as train_main
    cfg = _cfg_tm(mismatch)
    out = tmp_path / "run"
    sets = [f"train_mismatch.enabled={'true' if mismatch else 'false'}"]
    train_main(["--data-config", CFG_PATH, "--model-config", "configs/model_affine_proj.yaml", "--out", str(out),
                "--epochs", "1", "--batch-size", "2", "--device", "cpu", "--num-workers", "0", "--max-train-items", "2",
                "--max-val-items", "2", "--set", *sets])
    info = json.loads((out / "run_info.json").read_text())
    assert info["val_mismatch"] is mismatch and info["train_mismatch"] is mismatch
    v = json.loads((out / "history.json").read_text())[-1]["val_loss"]
    v_mm, v_clean = _val_loss(cfg, out / "best.pt", True, 2), _val_loss(cfg, out / "best.pt", False, 2)
    # 两个验证集确实不同（1 个 epoch 后模型接近延迟求和，对失配不敏感，差异约 1e-3 量级），且训练记录的验证损失
    # 与对应验证集上重算的损失一致到 float32 精度
    assert abs(v_mm - v_clean) > 2e-4 * v_clean
    assert math.isclose(v, v_mm if mismatch else v_clean, rel_tol=1e-5)


# ------------------------------------------------------------------------------------------- 推理时 float64 投影
def _big_model(dic, w0_db, scale, seed=0, **kw):
    """输出层权重很大的模型：w 的模可达上百，float32 下 u^H G u 与 c_0 = 1 - s sum w 有明显舍入。"""
    m = _model(dic, "hard", w0_db, seed=seed, **kw)
    with torch.no_grad():
        m.backbone.output_layer.conv.weight.normal_(0, scale)
        m.backbone.output_layer.conv.bias.normal_(0, scale)
    return m


@pytest.mark.parametrize("w0", W0S + ("learn",))
def test_projection_fp64_inference_strict(dic, w0):
    """proj_fp64：在滤波器空间由 u、s 显式重建 h_hat（complex128），逐时频点 |h_hat^H d - 1| < 1e-9 且 WNG >= W0 - 1e-9 dB；
    评测系数 c（complex128）的 WNG 同样不低于 W0。"""
    from src.evaluate import coef_wng_db, dictionary_gram
    kw = {"w0_freeze_epochs": 0} if w0 == "learn" else {}
    m = _big_model(dic, w0, 5.0, **kw)
    m.proj_fp64 = True
    torch.manual_seed(1)
    Z = torch.randn(3, 5, 257, 50, dtype=torch.complex64) * 10
    with torch.no_grad():
        o = m(Z)
    assert o["u"].dtype == torch.complex128 and o["s"].dtype == torch.float64 and o["S"].dtype == torch.complex64
    assert (o["s"] < 1).float().mean() > 0.5 and o["w"].abs().max() > 50  # 确实大量投影、w 很大
    h = _reconstruct(dic, o["u"], o["s"])  # [B, F, T, M]
    d = torch.as_tensor(dic["d"])
    hd = torch.einsum("bftm,fm->bft", h.conj(), d)
    wng = 10 * torch.log10(hd.abs() ** 2 / (h.abs() ** 2).sum(-1))
    w0_db = o["w0_db"] if torch.is_tensor(o["w0_db"]) else torch.tensor(o["w0_db"], dtype=torch.float64)
    assert (hd[:, 1:] - 1).abs().max() < 1e-9
    assert (wng - w0_db)[:, 1:].min() > -1e-9
    sw = o["s"][:, None] * o["w"]
    c = torch.cat([1 - sw.sum(1, keepdim=True), sw], 1)
    K, dn = dictionary_gram(load_config(CFG_PATH))
    wc = coef_wng_db(type("M", (), {"coef_kind": "alpha"})(), c, K, dn)
    assert (wc - w0_db)[:, 1:].min() > -1e-9


def test_projection_fp64_vs_fp32(dic):
    """float32 路径（训练与旧评测）在 w 很大时会有小的越界；float64 路径没有。两条路径的输出一致到 float32 精度，
    且 proj_fp64 不进入 state_dict、默认关闭（训练不变）。"""
    m = _big_model(dic, 0.0, 5.0)
    assert m.proj_fp64 is False and not any("proj_fp64" in k for k in m.state_dict())
    torch.manual_seed(2)
    Z = torch.randn(4, 5, 257, 60, dtype=torch.complex64) * 10
    with torch.no_grad():
        o32 = m(Z)
        m.proj_fp64 = True
        o64 = m(Z)
    assert o32["u"].dtype == torch.complex64 and o32["S"].dtype == torch.complex64
    rel = (o32["S"] - o64["S"]).abs().max() / o64["S"].abs().max()
    assert rel < 1e-4
    h32 = _reconstruct(dic, o32["u"].to(torch.complex128), o32["s"].double())
    h64 = _reconstruct(dic, o64["u"], o64["s"])
    d = torch.as_tensor(dic["d"])

    def wng(h):
        return 10 * torch.log10(torch.einsum("bftm,fm->bft", h.conj(), d).abs() ** 2 / (h.abs() ** 2).sum(-1))[:, 1:]
    assert wng(h64).min() > -1e-9
    assert wng(h32).min() < -1e-6  # float32 下的舍入越界确实存在（本测试的数据下），说明 float64 路径是必要的


def test_eval_proj_fp32_option(tmp_path):
    """方法说明 ckpt:N=P:proj=fp32 复现旧的 float32 投影（只用于核对）；默认 float64。"""
    from src.evaluate import parse_methods
    cfg, p = _ckpt(tmp_path, "affine_proj")
    bank = BeamBank.from_config(cfg)
    a, b = parse_methods([f"ckpt:A={p}", f"ckpt:B={p}:proj=fp32"], cfg, bank, torch.device("cpu"))
    assert a.model.proj_fp64 is True and b.model.proj_fp64 is False


# ------------------------------------------------------------------------------------------- kappa 扫描的失配 RIR 生成
def _runner(tmp_path):
    import importlib
    import sys
    sys.path.insert(0, "scripts/kaggle")  # 以可导入的模块名加载（_check_npz 要在子进程中 pickle）
    r = importlib.import_module("runner")
    r.WORK, r.INPUT, r.DATA = tmp_path / "work", tmp_path / "input", tmp_path / "tmp" / "data"
    return r


def test_runner_rir_outputs_hold_no_input_links(tmp_path):
    """rir 任务：输入的标称 RIR 只链接在临时 data 根目录下，不进入输出；要生成的目录指向输出；
    本 kernel 上一版本的同名输出不被链接（由 job_rir 复制为实体文件）。"""
    r = _runner(tmp_path)
    src = r.INPUT / "crbf-rir-eval" / "data"
    for sub in ("meta", "rirs/test_s1", "rirs/test_s1_kappa1"):
        (src / sub).mkdir(parents=True)
    (src / "rirs/test_s1/test_s1_00000.npz").write_bytes(b"x")
    prev = r.INPUT / "crbf-rir-kappa" / "data" / "rirs" / "test_s1_kappa0.5"
    prev.mkdir(parents=True)
    (prev / "test_s1_00000.npz").write_bytes(b"y")
    r.assemble_data_root(rirs_to_work=True, out_dirs=["test_s1_kappa0.5"])
    assert (r.DATA / "rirs/test_s1/test_s1_00000.npz").is_symlink()
    assert (r.DATA / "rirs/test_s1_kappa0.5").resolve() == (r.WORK / "data/rirs/test_s1_kappa0.5").resolve()
    assert not any(f.is_symlink() for f in r.WORK.rglob("*"))
    assert not any((r.WORK / "data/rirs/test_s1_kappa0.5").iterdir())  # 上一版本输出未被链接进来
    assert {p.name for p in (r.WORK / "data/rirs").iterdir()} == {"test_s1_kappa0.5"}
    assert r.rir_dir_name({"split": "test_s2", "kappa": 1.5}) == "test_s2_kappa1.5"
    assert r.rir_dir_name({"split": "test_s2"}) == "test_s2"


@pytest.mark.parametrize("ok", [True, False])
def test_runner_eval_generates_and_saves_rirs(tmp_path, monkeypatch, ok):
    """eval 任务 save_rirs：失配 RIR 生成到本 kernel 的输出（经 data 根目录的链接），核对后再评测；
    核对不通过时写 RIR_FAILED 并跳过评测。"""
    r = _runner(tmp_path)
    src = r.INPUT / "crbf-rir-eval" / "data"
    for sub in ("meta", "rirs/test_s1"):
        (src / sub).mkdir(parents=True)
    calls = []

    def fake_sh(cmd, env=None, **kw):  # 模拟 generate_rirs.py：写入 data 根目录下的 RIR 目录
        cmd = [str(c) for c in cmd]
        name = f"{cmd[cmd.index('--split') + 1]}_kappa{float(cmd[cmd.index('--kappa') + 1]):g}"
        (r.DATA / "rirs" / name / "x.npz").write_bytes(b"z")
        calls.append(name)
        return 0
    monkeypatch.setattr(r, "pip_install", lambda *a: None)
    monkeypatch.setattr(r, "sh", fake_sh)
    monkeypatch.setattr(r, "_check_rir_outputs", lambda job: {
        r.rir_dir_name(t): {"problems": [] if ok else ["missing 1"]} for t in job["tasks"]})
    ran = []
    monkeypatch.setattr(r, "run_eval_cmds", lambda job: ran.append(1))
    job = {"job_id": "x", "save_rirs": True, "generate_rirs": [{"split": "test_s1", "kappa": 3.0},
                                                               {"split": "test_s2", "kappa": 3.0}]}
    r.job_eval(job)
    assert calls == ["test_s1_kappa3", "test_s2_kappa3"]
    for n in calls:
        f = r.WORK / "data" / "rirs" / n / "x.npz"
        assert f.exists() and not f.is_symlink()
    assert not any(f.is_symlink() for f in r.WORK.rglob("*"))
    assert (r.WORK / "reports" / "rir_x.json").exists()
    assert ran == ([1] if ok else []) and (r.WORK / "RIR_FAILED").exists() == (not ok)


def test_mismatch_rir_saves_params_and_output_check(tmp_path, monkeypatch):
    """失配 RIR 的 info 保存该样本的失配参数（与 mismatch_params 一致）；生成后的输出核对能发现缺失与参数不一致。"""
    from src.simulate import load_meta, save_scene_rirs, simulate_scene_rirs
    cfg0 = load_config(CFG_PATH)
    if not meta_path(cfg0, "test_s1").exists():
        pytest.skip("需要阶段 1 生成的数据")
    r = _runner(tmp_path)
    root = r.DATA
    (root / "rirs").mkdir(parents=True)
    for sub in ("meta", "speech"):
        (root / sub).symlink_to(meta_path(cfg0, "test_s1").parents[1] / sub if sub == "meta" else
                                meta_path(cfg0, "test_s1").parents[1] / "speech")
    (root / "rirs/test_s1").symlink_to(meta_path(cfg0, "test_s1").parents[1] / "rirs/test_s1")
    out = r.WORK / "data/rirs/test_s1_kappa0.5"
    out.mkdir(parents=True)
    (root / "rirs/test_s1_kappa0.5").symlink_to(out)
    monkeypatch.setenv("CRBF_DATA_ROOT", str(root))
    cfg = load_config(CFG_PATH)
    sc = load_meta(cfg, "test_s1")[0]
    import json as _json
    with np.load(meta_path(cfg0, "test_s1").parents[1] / "rirs/test_s1" / sc["rir_file"]) as z:
        nominal_info = _json.loads(str(z["info"]))
    res = simulate_scene_rirs(cfg, sc, 0.5, nominal_info)
    save_scene_rirs(cfg, sc, res, 0.5)
    from src.simulate import mismatch_params
    with np.load(out / sc["rir_file"]) as z:
        info = _json.loads(str(z["info"]))
    exp = mismatch_params(cfg, sc, 0.5)
    assert info["mismatch"]["kappa"] == 0.5 and all(np.allclose(info["mismatch"][k], exp[k]) for k in exp)
    job = {"data_config": CFG_PATH, "tasks": [{"split": "test_s1", "kappa": 0.5, "start": 0, "end": 1}]}
    assert r._check_rir_outputs(job)["test_s1_kappa0.5"]["problems"] == []
    job["tasks"][0]["end"] = 2
    assert r._check_rir_outputs(job)["test_s1_kappa0.5"]["problems"] == ["missing 1"]


# ------------------------------------------------------------------------------------------- MVDR：失配验证集选加载（CLAUDE.md 5.3）
def test_mvdr_mmval_load_selection_grid():
    import sys
    sys.path.insert(0, "scripts")
    from stage4_mvdr_mmval import select_load
    calls = []

    def make(peak):
        def tune(loads):
            calls.append(list(loads))
            return {x: -abs(math.log10(x) - math.log10(peak)) for x in loads}
        return tune
    grid = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    best, c, r = select_load(make(1e-2), grid, 1e-9, 10.0)  # 内部最优：一轮即止
    assert math.isclose(best, 1e-2) and r == 1 and len(c) == 6
    best, c, r = select_load(make(30.0), grid, 1e-9, 10.0)  # 上边界：向上扩展到 1、10，到上限为止
    assert math.isclose(best, 10.0) and r == 3 and max(c) == pytest.approx(10.0)
    best, c, r = select_load(make(1e-8), grid, 1e-9, 10.0)  # 下边界：向下扩展，直到内部
    assert math.isclose(best, 1e-8) and min(c) == pytest.approx(1e-9) and r == 4


def test_evaluate_val_mismatch_changes_array_only():
    """evaluate(val_mismatch=True)：验证集阵列观测加失配（延迟求和输出改变），输入（中心麦克风）指标不变；默认不加。"""
    cfg = load_config(CFG_PATH)
    if not meta_path(cfg, "val").exists():
        pytest.skip("需要阶段 1 生成的数据")
    from src.evaluate import evaluate, parse_methods
    bank = BeamBank.from_config(cfg)
    ms = parse_methods(["fixed"], cfg, bank, torch.device("cpu"))
    kw = dict(max_items=2, batch_size=2, num_workers=0, metric_workers=1, do_pq=False, log=lambda *x: None)
    r0 = [r for r in evaluate(cfg, "val", ms, bank, torch.device("cpu"), **kw) if r["method"] == "h_D"]
    r1 = [r for r in evaluate(cfg, "val", ms, bank, torch.device("cpu"), val_mismatch=True, **kw) if r["method"] == "h_D"]
    for a, b in zip(r0, r1):
        assert a["si_sdr_dp_in"] == b["si_sdr_dp_in"]
        assert abs(a["d_si_sdr_dp"] - b["d_si_sdr_dp"]) > 1e-4
