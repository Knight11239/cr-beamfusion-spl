"""Minimal CPU end-to-end check with the released checkpoints (no data download).

  python scripts/cpu_smoke.py

For every checkpoint listed in results/checkpoints/MANIFEST.json, this script
  1. verifies the file's SHA-256 against the manifest,
  2. builds the model from its stored configuration and loads the weights (strict),
  3. feeds a synthetic target-only plane wave from the nominal look direction (theta = 0)
     through the fixed beamformer bank and the frozen network,
  4. reports the relative error between the fused output and the target STFT.
Fused filters that are distortionless for the nominal steering vector (convex, real and complex
affine fusion, the mask-based MVDR with nominal steering) reproduce the target up to numerical
precision; the unconstrained fusion (F) is expected not to.
It does not measure enhancement quality.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.beamformers import steer, stft_freqs  # noqa: E402
from src.config import load_config  # noqa: E402
from src.evaluate import ModelMethod  # noqa: E402
from src.models.gtcrn_fusion import BeamBank  # noqa: E402


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    torch.manual_seed(0)
    torch.set_grad_enabled(False)
    man = json.loads((ROOT / "results/checkpoints/MANIFEST.json").read_text())
    ok = True
    for rel, info in man.items():
        p = ROOT / rel
        assert sha256(p) == info["sha256"], f"hash mismatch: {rel}"
        cfg = load_config(str(ROOT / ("configs/data_g2.yaml" if "/gpu-g2-" in rel else "configs/data.yaml")))
        a = cfg["array"]
        bank = BeamBank.from_config(cfg)
        m = ModelMethod(Path(rel).stem, str(p), cfg, bank, torch.device("cpu"))
        F = stft_freqs(cfg["signal"]["fs"], 512)
        d = torch.as_tensor(steer(F, 0.0, a["n_mics"], a["spacing"], cfg["signal"]["c"])).to(torch.complex64)  # [F, M]
        X = (torch.randn(1, len(F), 60) + 1j * torch.randn(1, len(F), 60)).to(torch.complex64)
        Y = d.T[None, :, :, None] * X[:, None]  # [1, M, F, T]
        Z = bank(Y)
        out = m.apply(m.coef(Z, Y), Z, Y)
        err = float((out[:, 1:] - X[:, 1:]).abs().max() / X.abs().max())
        expect_distortionless = "free_complex" not in rel
        passed = (err < 1e-3) if expect_distortionless else True
        ok &= passed
        print(f"{'PASS' if passed else 'FAIL'}  {rel}  kind={m.kind}  rel. error (target-only, nominal) = {err:.2e}"
              + ("" if expect_distortionless else "  [unconstrained: not distortionless by design]"))
    print("all checks passed" if ok else "SOME CHECKS FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
