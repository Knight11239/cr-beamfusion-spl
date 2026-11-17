"""新对照的算子资格测试（用户 2026-11-13；research/analysis_plan_controls.md §2 第 6 项）。

对 A（affine）、R（affine_real）、C（convex_samehead）、F（free_complex）：网络输出 Ẑ 与 Σ_p coef_p Z_p 一致，且与由系数重建的
有效滤波器 h = Σ_p conj(coef_p) h_p 作用于阵列信号的 hᴴy 一致（共轭约定）；实系数乘复谱保留虚部；初始功能符合设计。
"""
import pytest
import torch
import yaml

from src.config import load_config
from src.models.gtcrn_fusion import BeamBank, build_model

CFG = load_config("configs/data_dev.yaml")
BANK = BeamBank.from_config(CFG)
MODELS = {"A": "configs/model_affine.yaml", "R": "configs/model_affine_real.yaml",
          "C": "configs/model_convex_samehead.yaml", "F": "configs/model_free_complex.yaml"}


def _coef(kind, o):
    if kind == "affine":
        sw = o["s"][:, None] * o["w"]
        return torch.cat([1 - sw.sum(1, keepdim=True), sw], 1)
    return o["coef"]


def _model(name):
    torch.manual_seed(0)
    mc = yaml.safe_load(open(MODELS[name]))
    return mc, build_model(mc, CFG, BANK).eval()


@pytest.mark.parametrize("name", list(MODELS))
def test_output_equals_reconstructed_filter(name):
    mc, m = _model(name)
    g = torch.Generator().manual_seed(1)
    Y = torch.complex(torch.randn(1, CFG["array"]["n_mics"], 257, 12, generator=g),
                      torch.randn(1, CFG["array"]["n_mics"], 257, 12, generator=g))
    Z = BANK(Y)
    with torch.no_grad():
        o = m(Z)
        c = _coef(mc["kind"], o).to(torch.complex128)
        S1 = (c * Z.to(torch.complex128)).sum(1)
        h = torch.einsum("bpft,pfm->bftm", c.conj(), BANK.H.to(torch.complex128))  # h = Σ conj(c_p) h_p
        S2 = torch.einsum("bftm,bmft->bft", h.conj(), Y.to(torch.complex128))
    S = o["S"].to(torch.complex128)
    tol = 1e-4 * S.abs().max()
    assert (S - S1).abs().max() < tol and (S - S2).abs().max() < tol
    assert S.imag.abs().max() > 0  # 实系数（R、C）乘复谱后虚部保留
    if name in ("R", "C"):
        assert _coef(mc["kind"], o).imag.abs().max() == 0 if torch.is_complex(_coef(mc["kind"], o)) else True
    if name == "C":
        a = o["coef"]
        assert (a >= 0).all() and torch.allclose(a.sum(1), torch.ones_like(a.sum(1)), atol=1e-6)


def test_initial_function_bias():
    """输出层权重置零时，系数正好等于偏置初始化的设计值：C 的 α = [0.95, 0.0125 × 4]，F 的 w = e_D。
    （真实数据上的初始功能统计见 scripts/controls_initial_function.py → results/controls/initial_function.json。）"""
    Z = BANK(torch.complex(torch.randn(1, CFG["array"]["n_mics"], 257, 10), torch.randn(1, CFG["array"]["n_mics"], 257, 10)))
    for name, ref in (("C", torch.tensor([0.95, 0.0125, 0.0125, 0.0125, 0.0125])), ("F", torch.tensor([1.0, 0, 0, 0, 0]))):
        _, m = _model(name)
        with torch.no_grad():
            m.backbone.output_layer.conv.weight.zero_()
            c = m(Z)["coef"]
        assert torch.allclose(c[0, :, 5, 5].real.float(), ref, atol=1e-6) and c.imag.abs().max() == 0 if torch.is_complex(c) else torch.allclose(c[0, :, 5, 5], ref, atol=1e-6)
