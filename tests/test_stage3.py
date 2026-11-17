"""阶段 3 单元测试：GTCRN 融合模型、ACC、掩码式 MVDR、评测（shadow filtering）与训练续跑。"""
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src import metrics as mt
from src.acc import acc_fuse
from src.beamformers import steer, stft_freqs
from src.config import load_config, meta_path
from src.models.gtcrn_fusion import BeamBank, BeamFusionNet, MaskMVDR, count_params

F = stft_freqs()
CFG_PATH = "configs/data_dev.yaml"


@pytest.fixture(scope="module")
def cfg():
    return load_config(CFG_PATH)


@pytest.fixture(scope="module")
def bank(cfg):
    return BeamBank.from_config(cfg)


def _has_data(cfg, split="val"):
    return meta_path(cfg, split).exists()


def _plane_wave(theta, X):
    """单个远场平面波：Y[b, m, f, t] = d_m(f, theta) X[b, f, t]。"""
    d = torch.as_tensor(steer(F, theta), dtype=torch.complex64)  # [F, M]
    return d.T[None, :, :, None] * X[:, None]


def test_param_count_matches_paper():
    assert count_params(BeamFusionNet(5)) == 85054  # BeamFusion 论文：85 k


def test_convex_weights_on_simplex():
    torch.manual_seed(0)
    m = BeamFusionNet(5).eval()
    Z = torch.randn(2, 5, 257, 40, dtype=torch.complex64)
    a = m(Z)["alpha"]
    assert a.shape == (2, 5, 257, 40)
    assert torch.all(a >= 0) and torch.allclose(a.sum(1), torch.ones(2, 257, 40), atol=1e-5)


def test_model_is_causal():
    """改变 t0 之后的输入，不影响 t0 之前帧的输出（eval 模式）。"""
    torch.manual_seed(0)
    m = BeamFusionNet(5).eval()
    Z1 = torch.randn(1, 5, 257, 60, dtype=torch.complex64)
    Z2 = Z1.clone()
    Z2[..., 40:] = torch.randn(1, 5, 257, 20, dtype=torch.complex64)
    with torch.no_grad():
        a1, a2 = m(Z1)["alpha"], m(Z2)["alpha"]
    assert torch.allclose(a1[..., :40], a2[..., :40], atol=1e-6)
    assert not torch.allclose(a1[..., 40:], a2[..., 40:], atol=1e-4)


def test_distortionless_end_to_end(bank):
    """只有来自 theta_s 的平面波目标时，BeamFusion / ACC / 掩码 MVDR / 各固定波束的输出都等于目标。"""
    torch.manual_seed(1)
    X = torch.randn(2, 257, 50, dtype=torch.complex64)
    Y = _plane_wave(0.0, X)
    Z = bank(Y)
    assert torch.allclose(Z, X[:, None].expand_as(Z), atol=1e-4)  # 每个波束无失真
    S_bf = BeamFusionNet(5).eval()(Z)["S"]
    S_acc, _ = acc_fuse(Z, mu=0.1)
    mv = MaskMVDR(5, steer(F, 0.0)).eval()
    noise = 0.1 * torch.randn(2, 8, 257, 50, dtype=torch.complex64)  # 用于估计协方差的“噪声”
    h = mv.filters(Y + noise, torch.zeros(2, 257, 50))
    S_mv = torch.einsum("bfm,bmft->bft", h.conj(), Y)
    for S in (S_bf, S_acc, S_mv):
        assert torch.allclose(S[:, 1:], X[:, 1:], atol=1e-3 * X.abs().max())


def test_acc_tracks_interferer(bank):
    """只有 90° 平面波干扰时，ACC 应把权重集中到 null-90 上（输出功率趋近 0）。"""
    torch.manual_seed(2)
    I = torch.randn(1, 257, 400, dtype=torch.complex64)
    Z = bank(_plane_wave(90.0, I))
    S, A = acc_fuse(Z, mu=0.3)
    k = bank.names.index("null-90")
    band = (F > 500) & (F < 4000)
    assert A[0, k, band, -1].mean() > 0.9
    assert torch.allclose(A.sum(1), torch.ones_like(A[:, 0]), atol=1e-5)
    p_in = Z[0, 0, band, -50:].abs().pow(2).mean()
    p_out = S[0, band, -50:].abs().pow(2).mean()
    assert p_out < 1e-2 * p_in


def test_mvdr_nulls_interferer():
    torch.manual_seed(3)
    mv = MaskMVDR(5, steer(F, 0.0))
    I = torch.randn(1, 257, 300, dtype=torch.complex64)
    V = _plane_wave(120.0, I) + 1e-3 * torch.randn(1, 8, 257, 300, dtype=torch.complex64)
    h = mv.filters(V, torch.zeros(1, 257, 300), load=1e-6)
    d0 = torch.as_tensor(steer(F, 0.0), dtype=torch.complex64)
    d120 = torch.as_tensor(steer(F, 120.0), dtype=torch.complex64)
    band = (F > 300)
    assert torch.allclose(torch.einsum("fm,fm->f", h[0].conj(), d0)[1:], torch.ones(256, dtype=torch.complex64),
                          atol=1e-3)
    assert torch.einsum("fm,fm->f", h[0].conj(), d120)[band].abs().max() < 0.05


def test_metrics_sanity():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(16000)
    assert mt.si_sdr(x, x) > 100 and mt.si_sdr(3 * x, x) > 100  # 尺度不变
    n = rng.standard_normal(16000)
    assert abs(mt.snr(x + 0.1 * n, x) - 20) < 0.5
    assert abs(mt.power_ratio_db(x, 0.1 * x) - 20) < 1e-9


@pytest.mark.slow
def test_evaluate_shadow_consistency(cfg):
    if not _has_data(cfg):
        pytest.skip("需要阶段 1 生成的数据")
    from src.evaluate import evaluate, parse_methods
    bank = BeamBank.from_config(cfg)
    methods = parse_methods(["fixed", "acc:mu=0.1"], cfg, bank, torch.device("cpu"))
    rows = evaluate(cfg, "val", methods, bank, torch.device("cpu"), max_items=2, batch_size=2, num_workers=0,
                    metric_workers=1, do_pq=False, log=lambda *a: None)
    assert len(rows) == 2 * 6
    for r in rows:
        assert r["shadow_consistency_err"] < 1e-4  # 分量输出之和 = 混合输出（线性）
        assert np.isfinite(r["d_si_sdr"]) and np.isfinite(r["sir_shadow_out"])
    hd = [r for r in rows if r["method"] == "h_D"]
    assert all(abs(r["si_sdr_out"] - r["si_sdr_in"]) < 3 for r in hd)  # 延迟求和接近中心麦克风


@pytest.mark.slow
def test_train_resume_is_exact(cfg, tmp_path):
    """训练 2 个 epoch 与“1 个 epoch + 续跑 1 个 epoch”结果完全相同（含跨目录 --resume-from）。"""
    if not _has_data(cfg, "train"):
        pytest.skip("需要阶段 1 生成的数据")
    from src.train import main as train_main
    common = ["--data-config", CFG_PATH, "--model-config", "configs/model_beamfusion.yaml",
              "--max-train-items", "4", "--max-val-items", "2", "--batch-size", "2", "--num-workers", "0",
              "--device", "cpu", "--threads", "2"]
    train_main(common + ["--out", str(tmp_path / "a"), "--epochs", "2"])
    train_main(common + ["--out", str(tmp_path / "b"), "--epochs", "1"])
    train_main(common + ["--out", str(tmp_path / "c"), "--epochs", "2", "--resume-from", str(tmp_path / "b")])
    ka = torch.load(tmp_path / "a" / "last.pt", weights_only=False)
    kc = torch.load(tmp_path / "c" / "last.pt", weights_only=False)
    for k in ka["model"]:
        assert torch.equal(ka["model"][k], kc["model"][k]), k
    ha = json.loads((tmp_path / "a" / "history.json").read_text())
    hc = json.loads((tmp_path / "c" / "history.json").read_text())
    assert [h["train_loss"] for h in ha] == [h["train_loss"] for h in hc]
    assert Path(tmp_path / "c" / "best.pt").exists()


def test_training_target_direct_path(cfg):
    """CLAUDE.md 3.3（2026-10-30 修改）：训练目标为中心处的直达声分量；x_ref 仍可得到且与之时间对齐。"""
    if not _has_data(cfg):
        pytest.skip("需要阶段 1 生成的数据")
    from src.dataset import BeamFusionDataset, train_target_key
    from src.pipeline import prepare
    from src.stft import stft
    assert cfg["reference"]["train_target"] == "direct" and train_target_key(cfg) == "ref_direct"
    assert train_target_key({"reference": {}}) == "ref"  # 旧配置（无该字段）= 旧训练目标 x_ref
    b = BeamFusionDataset(cfg, "val", mix_on="worker", return_components=False, max_items=1)[0]
    b = {k: v[None] for k, v in b.items()}
    _, _, X = prepare(b, BeamBank.from_config(cfg), target_key=train_target_key(cfg))
    assert torch.equal(X, stft(b["ref_direct"]))
    # 直达声是 x_ref 的一部分：两者的最佳对齐时延为 0，且 x_ref 的能量更大（含早期反射）
    xd, xr = b["ref_direct"][0].double(), b["ref"][0].double()
    c = torch.fft.irfft(torch.fft.rfft(xr, 2 * len(xr)) * torch.fft.rfft(xd, 2 * len(xd)).conj())
    assert int(c.argmax()) == 0
    assert xr.pow(2).sum() > xd.pow(2).sum()
