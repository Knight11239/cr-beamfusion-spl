"""Diagnostics API for a future evaluator, with coefficients selected on mixture.

No network is constructed or called. Current dry-run never invokes this on a
prospective model. Arrays must come from the actual runtime dictionary/path.
"""
import numpy as np
import io

VERSION = "frozen-mixture-coefficients-v1"


def frozen_diagnostics(H, c, d_nom, z_direct, S_direct_actual, y_direct_frozen, x_dp,
                       *, coefficient_origin, Y_direct=None):
    """One utterance: H[P,F,M], c[P,F,T], d[F,M], Zdirect[P,F,T].

    S_direct_actual and y_direct_frozen must be the evaluator's outputs after
    reusing these same mixture-selected c, not target-only network outputs.
    Time-domain TD deliberately uses the actual frozen output, with repository
    EPS, and no gain alignment. Caller provenance remains necessary: arrays
    alone cannot prove where a neural coefficient was estimated.
    """
    if coefficient_origin != "mixture":
        raise ValueError("Target-only re-estimated coefficients are not the planned TD")
    H, c, d = [np.asarray(x, np.complex128) for x in (H, c, d_nom)]
    z, S = [np.asarray(x, np.complex128) for x in (z_direct, S_direct_actual)]
    if np.iscomplexobj(y_direct_frozen) or np.iscomplexobj(x_dp):
        raise ValueError("Time-domain diagnostic arrays must be real")
    y, ref = [np.asarray(x, np.float64) for x in (y_direct_frozen, x_dp)]
    if H.ndim != 3 or c.ndim != 3 or H.shape[:2] != c.shape[:2] or d.shape != H.shape[1:]:
        raise ValueError("Expected actual H[P,F,M], c[P,F,T], d[F,M]")
    if H.shape[1] < 2 or z.shape != c.shape or S.shape != c.shape[1:] or y.ndim != 1 or y.shape != ref.shape:
        raise ValueError("Mismatched frozen-target arrays")
    if not all(np.isfinite(x).all() for x in (H, c, d, z, S, y, ref)):
        raise ValueError("Nonfinite diagnostic input")
    reconstructed = np.sum(c * z, axis=0)
    if not np.allclose(reconstructed, S, rtol=1e-5, atol=1e-7):
        raise ValueError("Frozen target spectrum does not match mixture-selected coefficients")
    h = np.einsum("pft,pfm->fmt", c.conj(), H)
    gain = np.einsum("fmt,fm->ft", h.conj(), d)
    error = abs(gain - 1)
    norm = np.sum(abs(h) ** 2, axis=1)
    numerator = abs(gain) ** 2
    # A scale-relative numerical degeneracy guard, not a WNG floor or claim threshold.
    tiny = np.finfo(float).tiny
    undefined = norm <= tiny
    zero_response = (~undefined) & (numerator == 0)
    defined = (~undefined) & (~zero_response)
    wng = np.full(norm.shape, np.nan)
    wng[defined] = 10 * np.log10(numerator[defined] / norm[defined])
    wng[zero_response] = -np.inf
    finite_wng = wng[1:][np.isfinite(wng[1:])]
    out = dict(diagnostic_version=VERSION, coefficient_origin="mixture", runtime_dictionary=True,
               nominal_D_p95=float(np.percentile(error[1:], 95)),
               nominal_D_rms=float(np.sqrt(np.mean(error[1:] ** 2))),
               nominal_D_max=float(error[1:].max()), nominal_D_dc_max=float(error[0].max()),
               nominal_gain_amp_mean=float(abs(gain[1:]).mean()),
               nominal_gain_phase_abs_p95=float(np.percentile(abs(np.angle(gain[1:])), 95)),
               sum_c_residual_max=float(abs(c.sum(axis=0) - 1).max()),
               wng_undefined_count=int(undefined[1:].sum()), wng_zero_response_count=int(zero_response[1:].sum()),
               near_zero_filter_count=int((norm[1:] < 1e-24).sum()),
               wng_finite_min_db=float(finite_wng.min()) if finite_wng.size else None,
               wng_finite_median_db=float(np.median(finite_wng)) if finite_wng.size else None,
               tf_non_dc_points=int(error[1:].size),
               target_direct_sdr_out=float(10 * np.log10((np.dot(ref, ref) + 1e-12) /
                                                        (np.dot(y - ref, y - ref) + 1e-12))),
               target_output_kind="frozen_mixture_coefficients",
               target_spectrum_reconstruction_relative_error=float(np.linalg.norm(reconstructed - S) /
                                                                    max(np.linalg.norm(S), 1e-30)))
    if Y_direct is not None:
        Y = np.asarray(Y_direct, np.complex128)
        if Y.shape != (H.shape[2], H.shape[1], c.shape[2]):
            raise ValueError("Expected Y_direct[M,F,T]")
        independently = np.einsum("fmt,mft->ft", h.conj(), Y)
        if not np.allclose(independently, S, rtol=1e-5, atol=1e-7):
            raise ValueError("Effective-filter conjugation/dictionary identity failed")
        out["effective_filter_reconstruction_relative_error"] = float(np.linalg.norm(independently - S) /
                                                                      max(np.linalg.norm(S), 1e-30))
    return out


def from_archive(content, expected_mics):
    """Offline derivation for a future explicitly supplied NPZ; never calls a model.

    In addition to coefficient/filter identities, validate the actual frozen
    time output against the repository's inverse STFT. Only this optional path
    imports torch. The legacy smoke manifest supplies no archives.
    """
    with np.load(io.BytesIO(content), allow_pickle=False) as archive:
        keys = ("H", "c", "d_nom", "z_direct", "S_direct_actual", "y_direct_frozen", "x_dp")
        arrays = {key: archive[key] for key in keys}
        origin = str(archive["coefficient_origin"].item())
        convention = str(archive["stft_convention"].item())
        if arrays["H"].shape != (5, 257, expected_mics):
            raise ValueError("Wrong runtime dictionary geometry/frequency layout")
        if convention != "512/sqrt-hann/128/center":
            raise ValueError("Wrong frozen-target STFT convention")
        if "Y_direct" in archive:
            arrays["Y_direct"] = archive["Y_direct"]
        out = frozen_diagnostics(**arrays, coefficient_origin=origin)
        import torch
        from src.stft import istft
        torch.set_num_threads(1)
        with torch.inference_mode():
            reproduced = istft(torch.from_numpy(arrays["S_direct_actual"]), len(arrays["x_dp"])).numpy()
        y = arrays["y_direct_frozen"]
        if not np.allclose(reproduced, y, rtol=1e-5, atol=1e-7):
            raise ValueError("Time-domain TD output is not the supplied frozen target spectrum")
        out["frozen_istft_relative_error"] = float(np.linalg.norm(reproduced - y) / max(np.linalg.norm(y), 1e-30))
        return out
