"""推理时径向投影包装（PostProjMethod，事后探索 2026-11-15）：无失真、WNG 下限、频带限定、对凸系数同样适用。"""
import numpy as np
import torch

from src.config import load_config
from src.evaluate import PostProjMethod, coef_wng_db, dictionary_gram


class _Fixed:
    name, coef_kind = "X", "alpha"

    def __init__(self, c):
        self.c = c

    def coef(self, Z, Y):
        return self.c


def _cfg():
    return load_config("configs/data_g2.yaml")


def test_postproj_affine_and_convex():
    torch.manual_seed(0)
    cfg = _cfg()
    K, d = dictionary_gram(cfg)
    P1, F, T = K.shape[1], K.shape[0], 6
    w = 3 * (torch.randn(2, P1 - 1, F, T, dtype=torch.float64) + 1j * torch.randn(2, P1 - 1, F, T, dtype=torch.float64))
    aff = torch.cat([1 - w.sum(1, keepdim=True), w], 1)
    cvx = torch.softmax(torch.randn(2, P1, F, T, dtype=torch.float64) * 4, 1)
    fb = torch.as_tensor(np.arange(F) * cfg["signal"]["fs"] / 512)
    for c in (aff, cvx):
        for band in (None, (0.0, 1000.0)):
            m = PostProjMethod(_Fixed(c), cfg, -5.0, band)
            cp = m.coef(None, None)
            assert (cp.sum(1) - 1).abs().max() < 1e-12
            wdb = coef_wng_db(m, cp, K, d)[:, 1:]
            inside = torch.ones(F, dtype=torch.bool)[1:] if band is None else ((fb > band[0]) & (fb <= band[1]))[1:]
            assert wdb[:, inside].min() >= -5.0 - 1e-6
            if band is not None:  # 频带外不变
                assert torch.allclose(cp[:, :, 1:][:, :, ~inside], c.to(cp.dtype)[:, :, 1:][:, :, ~inside], atol=1e-12, rtol=0)
            wu = coef_wng_db(m, c.to(torch.complex128), K, d)  # 投影前的 WNG
            mask_f = torch.ones(F, dtype=torch.bool) if band is None else ((fb > band[0]) & (fb <= band[1]))
            exp = (wu < -5.0) & mask_f[None, :, None]
            assert torch.equal((m.last_s < 1)[:, 1:], exp[:, 1:])  # 激活 ⇔ 投影前 WNG < W0（频带内）
