"""Math negative controls and legacy-only integration; no new model outcomes."""
import copy
import csv
import io
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import t

from research.control_analysis.diagnostics import frozen_diagnostics, from_archive
from research.control_analysis.inputs import Inputs, InputError
from research.control_analysis.pipeline import Analysis, make_slot, primary_slots, interpret
from research.control_analysis.statistics import paired, tost, holm, exact_signed_rank, stratified_bootstrap, cluster_bootstrap

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


@pytest.fixture(scope="module")
def legacy():
    manifest = json.loads((HERE / "legacy_inputs.json").read_text())
    result = Inputs(ROOT, manifest).load()
    assert not result.issues
    return result


def test_hand_computable_pair_and_no_finite_intersection():
    r = paired([1, 2, -3, 4], np.zeros(4))
    assert r["n"] == 4 and r["mean_diff"] == 1 and r["r_rb"] == .4
    assert r["ci_hi"] == pytest.approx(1 + 1.96 * np.sqrt(26 / 3) / 2)
    with pytest.raises(ValueError, match="Nonfinite"):
        paired([1, np.nan], [0, 0])
    z = paired(np.ones(4), np.ones(4))
    assert z["p_raw"] == 1 and z["r_rb"] == 0 and z["cohen_dz"] is None


def test_csv_output_uses_lf_and_preserves_missing_p(tmp_path):
    from research.control_analysis.run import write_csv
    path = tmp_path / "format.csv"
    write_csv(path, [dict(status="未执行", p_raw=None, sources={"old.csv": 600})])
    assert b"\r" not in path.read_bytes()
    row = next(csv.DictReader(path.open()))
    assert row["p_raw"] == "" and row["status"] == "未执行"
    assert json.loads(row["sources"]) == {"old.csv": 600}


@pytest.mark.parametrize("difference,expected", [(0., 0.), (.49, 0.), (.5, 1.), (-.5, 1.), (.7, 1.)])
def test_tost_degenerate_boundaries(difference, expected):
    assert tost(np.full(8, difference), np.zeros(8))["p_raw"] == expected


def test_tost_student_t_formula_and_not_significance():
    d = np.array([-.2, -.1, .1, .2])
    se = d.std(ddof=1) / 2
    r = tost(d, np.zeros(4))
    assert r["p_raw"] == pytest.approx(max(t.sf((d.mean() + .5) / se, 3), t.cdf((d.mean() - .5) / se, 3)))
    # A zero-centered but very imprecise difference is not evidence of equivalence.
    assert tost([-10, 10], [0, 0])["p_raw"] > .05


def test_fixed_holm_slots_are_not_measured_p_equals_one():
    slots = primary_slots()
    assert len(slots) == 56
    slots[0].update(eligible=True, p_raw=.001, p_for_holm=.001, correction_placeholder=False)
    holm(slots, "p_holm_global", 56)
    assert slots[0]["p_holm_global"] == .056
    assert all(r["p_raw"] is None and r["p_holm_global"] is None and r["p_for_holm"] == 1 for r in slots[1:])
    with pytest.raises(ValueError, match="56"):
        holm(slots[:-1], "p_holm_global", 56)


def test_condition_sign_enumeration_ties_zeros_and_resolution():
    assert exact_signed_rank([1, 2, 3, 4, 5, 6]) == .03125
    assert exact_signed_rank([1, -1]) == 1
    assert exact_signed_rank([0, 0, 0]) == 1
    records = [dict(eligible=True, p_for_holm=.03125) for _ in range(6)]
    holm(records, "adjusted", 6)
    assert all(r["adjusted"] == .1875 for r in records)


def scenes_with_clusters(one=False):
    scenes, effects = [], []
    for c, (rt, sub) in enumerate((rt, sub) for rt in ("0.160", "0.360", "0.610") for sub in ("A", "B")):
        for cluster in range(1 if one else 2):
            # Unequal cluster/condition sizes: a row-weighted pooled bootstrap is wrong.
            for _ in range(1 + c + cluster):
                scenes.append(dict(rt60=rt, subarray=sub, interferers=[dict(trajectory_deg=[cluster], dist=2)]))
                effects.append([100. if c == 0 else 0.])
    return np.array(effects), scenes


def test_cluster_bootstrap_keeps_fixed_condition_weights():
    d, scenes = scenes_with_clusters()
    r = cluster_bootstrap(d, scenes, repetitions=800, seed=7)
    assert r["bootstrap_ci_lo"][0] == pytest.approx(100 / 6)
    assert r["bootstrap_ci_hi"][0] == pytest.approx(100 / 6)
    assert r == cluster_bootstrap(d, scenes, repetitions=800, seed=7)
    assert set(r["clusters_by_condition"].values()) == {2}


def test_single_trajectory_is_not_a_zero_width_ci():
    d, scenes = scenes_with_clusters(one=True)
    r = cluster_bootstrap(d, scenes, repetitions=100)
    assert r["bootstrap_status"] == "not_estimable_single_trajectory_condition"
    assert r["bootstrap_ci_lo"] is None and r["bootstrap_ci_hi"] is None


def test_paired_t60_bootstrap_uses_equal_strata():
    r = stratified_bootstrap([2., 2., 8., 8.], [.3, .3, .7, .7], repetitions=500)
    assert r["bootstrap_ci_lo"] == r["bootstrap_ci_hi"] == 5
    with pytest.raises(ValueError, match="equal"):
        stratified_bootstrap([2., 2., 8.], [.3, .3, .7], repetitions=10)


def diagnostic_arrays(scale=1):
    H = np.tile(np.array([[[.5, .5]], [[1., 0.]]], complex), (1, 3, 1))
    c = np.zeros((2, 3, 4), complex)
    c[0] = scale
    Y = np.ones((2, 3, 4), complex)
    z = np.einsum("pfm,mft->pft", H.conj(), Y)
    S = np.sum(c * z, axis=0)
    ref = np.array([1., -1., .5, -.5])
    return dict(H=H, c=c, d_nom=np.ones((3, 2)), z_direct=z, S_direct_actual=S,
                y_direct_frozen=ref * scale, x_dp=ref, coefficient_origin="mixture", Y_direct=Y)


def test_response_and_td_detect_scale_which_wng_does_not():
    a = frozen_diagnostics(**diagnostic_arrays())
    b = frozen_diagnostics(**diagnostic_arrays(2))
    assert a["nominal_D_p95"] == 0 and b["nominal_D_p95"] == 1
    assert a["wng_finite_min_db"] == pytest.approx(b["wng_finite_min_db"])
    assert b["target_direct_sdr_out"] == pytest.approx(0.)
    zero = frozen_diagnostics(**diagnostic_arrays(0))
    assert zero["nominal_D_p95"] == 1 and zero["wng_undefined_count"] == 8
    assert zero["wng_finite_min_db"] is None


def test_frozen_coefficients_and_conjugation_negative_controls():
    arrays = diagnostic_arrays()
    arrays["coefficient_origin"] = "target_only"
    with pytest.raises(ValueError, match="Target-only"):
        frozen_diagnostics(**arrays)
    arrays = diagnostic_arrays(1 + 1j)
    arrays["y_direct_frozen"] = arrays["x_dp"]  # Any real frozen signal; spectral identity is checked separately.
    frozen_diagnostics(**arrays)
    arrays["S_direct_actual"] = np.sum(arrays["c"].conj() * arrays["z_direct"], axis=0)
    with pytest.raises(ValueError, match="does not match"):
        frozen_diagnostics(**arrays)


def test_optional_archive_istft_identity_without_any_network():
    import torch
    from src.stft import stft, istft
    ref = torch.sin(torch.arange(1024) * .1)
    X = stft(ref).numpy()
    H = np.ones((5, 257, 4), np.complex64) / 4
    c = np.zeros((5, *X.shape), np.complex64)
    c[0] = 1
    y = istft(torch.from_numpy(X), len(ref)).numpy()
    values = dict(H=H, c=c, d_nom=np.ones((257, 4)), z_direct=np.repeat(X[None], 5, axis=0), S_direct_actual=X,
                  y_direct_frozen=y, x_dp=ref.numpy(), coefficient_origin=np.array("mixture"),
                  stft_convention=np.array("512/sqrt-hann/128/center"))
    stream = io.BytesIO()
    np.savez(stream, **values)
    r = from_archive(stream.getvalue(), 4)
    assert r["nominal_D_p95"] == 0 and r["frozen_istft_relative_error"] == 0
    values["y_direct_frozen"] = y + 1
    stream = io.BytesIO()
    np.savez(stream, **values)
    with pytest.raises(ValueError, match="Time-domain"):
        from_archive(stream.getvalue(), 4)


def test_pending_paths_are_not_even_checked(monkeypatch):
    manifest = json.loads((HERE / "legacy_inputs.json").read_text())
    manifest["models"]["C"].update(checkpoint=dict(path="NEVER_READ", sha256="0" * 64),
        sources=[dict(path="NEVER_READ", sha256="0" * 64, dataset="8mic")],
        diagnostics=[dict(path="NEVER_READ", sha256="0" * 64, dataset="8mic")])
    seen = []
    original = Inputs.checked
    def checked(self, ref, purpose, **kwargs):
        assert ref["path"] != "NEVER_READ"
        seen.append(ref["path"])
        return original(self, ref, purpose, **kwargs)
    monkeypatch.setattr(Inputs, "checked", checked)
    data = Inputs(ROOT, manifest).load()
    assert "C" not in data.enabled and seen
    manifest["models"]["C"]["state"] = "evaluated"
    with pytest.raises(InputError, match="allow-new-results"):
        Inputs(ROOT, manifest)


def test_hash_mismatch_and_changed_legacy_manifest_are_rejected():
    manifest = json.loads((HERE / "legacy_inputs.json").read_text())
    data = Inputs(ROOT, manifest)
    ref = dict(manifest["models"]["A"]["sources"][0], sha256="0" * 64)
    with pytest.raises(InputError, match="SHA mismatch"):
        data.checked(ref, "negative test: existing file hash")
    manifest["models"]["A"]["sources"][0] = ref
    with pytest.raises(InputError, match="unchanged bundled"):
        Inputs(ROOT, manifest)


def test_conflicting_duplicate_source_blocks_model(monkeypatch):
    manifest = json.loads((HERE / "legacy_inputs.json").read_text())
    source = manifest["models"]["A"]["sources"][0]
    manifest["models"]["A"]["sources"].append(source)
    seen = 0
    original = Inputs.read_csv
    def read(self, ref, purpose):
        nonlocal seen
        result = original(self, ref, purpose)
        if ref["path"] == source["path"] and purpose.startswith("A:"):
            seen += 1
            if seen == 2:
                next(row for row in result if row["method"] == manifest["models"]["A"]["method"])["d_si_sdr_dp"] = "12345"
        return result
    monkeypatch.setattr(Inputs, "read_csv", read)
    data = Inputs(ROOT, manifest, allow_new=True).load()
    assert "Conflicting duplicate" in data.issues["A"]


def test_all_registered_slots_stay_unexecuted(legacy):
    output = Analysis(legacy).run(smoke=False)
    assert len(output["primary"]) == 56 and len(output["secondary"]) == 102
    assert len(output["t60"]) == 240 and len(output["mird_conditions"]) == 252
    assert len(output["target_auxiliary"]) == 12 and len(output["diagnostic_summary"]) == 12
    assert all(r["p_raw"] is None for r in output["target_auxiliary"])
    assert all(r["status"] == ("未执行" if r["model"] == "F" else "不可判定") for r in output["diagnostic_summary"])
    assert all(r["execution_status"] == "未执行" and r["p_raw"] is None and r["p_holm_global"] is None for r in output["primary"])
    assert all("待定" in r["interpretation"] for r in output["interpretations"])


def test_legacy_pairing_agrees_with_existing_ledger(legacy):
    prior = list(csv.DictReader((ROOT / "research/streaming_cost_provenance/extension/claim_statistics.csv").open()))
    for old in prior:
        if old["family"] != "fair8" or old["group"] != "all" or old["metric"] != "d_si_sdr_dp":
            continue
        slot = make_slot("LEGACY_A_B", "SMOKE", "8mic", old["split"], float(old["kappa"]), "A", "B", "d_si_sdr_dp")
        row, bundle = Analysis(legacy).calculate(slot, smoke=True)
        assert bundle is not None and row["n"] == 600 and not row["eligible"]
        for field, source in [("mean_diff", "mean_diff"), ("ci_lo", "ci_lo"), ("ci_hi", "ci_hi"), ("r_rb", "r_rb"), ("p_raw", "p")]:
            assert row[field] == pytest.approx(float(old[source]), rel=1e-12, abs=0 if field == "p_raw" else 1e-12)


@pytest.mark.parametrize("fault", ["missing", "nan", "nan_output", "input", "delta"])
def test_bad_pair_never_drops_ids_or_repairs_scores(legacy, fault):
    data = copy.copy(legacy)
    data.data = dict(legacy.data)
    key = ("A", "8mic", "test_s1", 0., "test_s1_00000")
    if fault == "missing":
        del data.data[key]
    else:
        data.data[key] = dict(data.data[key])
        field = {"nan": "d_si_sdr_dp", "nan_output": "si_sdr_dp_out", "input": "si_sdr_dp_in", "delta": "d_si_sdr_dp"}[fault]
        data.data[key][field] = "nan" if fault.startswith("nan") else str(float(data.data[key][field]) + 1)
    row, bundle = Analysis(data).calculate(make_slot("LEGACY", "SMOKE", "8mic", "test_s1", 0., "A", "B", "d_si_sdr_dp"), smoke=True)
    assert bundle is None and row["execution_status"] == "不可判定"
    assert row["p_raw"] is None and row["p_for_holm"] == 1


def test_unknown_qualification_preserves_descriptive_numbers_but_blocks_claim(legacy):
    row, _ = Analysis(legacy).calculate(make_slot("C1", "H8", "8mic", "test_s1", 0., "A", "B", "d_si_sdr_dp"))
    assert row["mean_diff"] is not None and row["p_raw"] is not None
    assert row["qualification_status"] == "unknown" and not row["eligible"] and row["p_for_holm"] == 1


def test_legacy_mird_conditions_and_single_cluster_status(legacy):
    analysis = Analysis(legacy)
    rows = analysis.mird("SMOKE_GL_GA", "GL", "GA", "mird_s1", smoke=True)
    primary = rows[0]
    assert primary["n"] == 6 and primary["paired_ids"] == 600
    assert all(r["n"] == 100 for r in analysis.condition_rows)
    assert all(r["bootstrap_status"] == "not_estimable_single_trajectory_condition" for r in analysis.cluster_rows)


@pytest.mark.parametrize("split,kappa", [(s, k) for s in ("test_s1", "test_s2") for k in (1., 2.)])
def test_degradation_matches_direct_four_way_id_pairing(legacy, split, kappa):
    slot = make_slot("LEGACY_DEGRADATION", "SMOKE", "8mic", split, kappa, "A", "B", "d_si_sdr_dp", kind="degradation")
    row, bundle = Analysis(legacy).calculate(slot, smoke=True)
    assert bundle is not None and row["paired_ids"] == 600
    ids, _, _, a, b = bundle
    a0, b0 = [legacy.group(m, "8mic", split, 0.) for m in ("A", "B")]
    d = np.array([(float(a[i]["d_si_sdr_dp"]) - float(a0[i]["d_si_sdr_dp"])) -
                  (float(b[i]["d_si_sdr_dp"]) - float(b0[i]["d_si_sdr_dp"])) for i in ids])
    expected = paired(d, np.zeros(600))
    for field in ("mean_diff", "ci_lo", "ci_hi", "p_raw", "r_rb"):
        assert row[field] == pytest.approx(expected[field])
    assert row["absolute_left_at_zero"] == pytest.approx(np.mean([float(a0[i]["d_si_sdr_dp"]) for i in ids]))
    assert sum(row["zero_left_sources"].values()) == 600


def test_tost_pipeline_uses_existing_pairs_without_filling_new_slots(legacy):
    row, bundle = Analysis(legacy).calculate(make_slot("LEGACY_TOST", "SMOKE", "8mic", "test_s1", 0., "A", "B", "d_si_sdr_dp", kind="tost"), smoke=True)
    assert bundle is not None
    _, a, b, _, _ = bundle
    expected = tost(a, b)
    assert row["p_raw"] == expected["p_raw"]
    assert row["tost_ci90_lo"] == expected["tost_ci90_lo"]
    assert not row["eligible"]


def test_unproven_d_and_partial_bad_a_diagnostic_do_not_pass(legacy):
    data = copy.copy(legacy)
    data.data, data.qualifications = dict(legacy.data), []
    key = ("A", "8mic", "test_s1", 0., "test_s1_00000")
    data.data[key] = dict(data.data[key], nominal_D_p95="0", nominal_D_max="0.01", nominal_D_dc_max="0")
    values, bad = Analysis.endpoint({key[-1]: data.data[key]}, [key[-1]], "nominal_D_p95")
    assert bad == [key[-1]] and np.isnan(values[0])
    data.data[key]["_diagnostic_source"] = "synthetic in-memory fault in existing A only"
    assert data.qualify("F_D", "A", "B", "8mic", "test_s1", 0.) == "fail"
    assert any(r["requirement"] == "nominal_response_audit" and r["status"] == "fail" for r in data.qualifications)
    assert data.qualify("C1", "A", "B", "8mic", "test_s1", 0.) == "fail"


def test_diagnostic_summary_keeps_global_max_and_undefined_events(legacy):
    data = copy.copy(legacy)
    data.data = dict(legacy.data)
    # Pure synthetic diagnostics attached to old A in memory, never exported as research results.
    synthetic = frozen_diagnostics(**diagnostic_arrays(0))
    for key, row in legacy.data.items():
        if key[:4] == ("A", "8mic", "test_s1", 0.):
            data.data[key] = dict(row, **synthetic, _diagnostic_source="in-memory mathematical fixture")
    summary = Analysis(data).diagnostic_summary()
    r = next(r for r in summary if (r["model"], r["split"], r["kappa"]) == ("A", "test_s1", 0.))
    assert r["global_nominal_response_max"] == 1 and r["n"] == 600
    assert r["wng_undefined_count_total"] == 4800 and r["tf_non_dc_points_total"] == 4800
    assert r["wng_finite_min_db_finite_n"] == 0 and "wng_finite_min_db_mean" not in r


@pytest.mark.parametrize("fault", [None, "origin", "residual", "reconstruction", "events", "wng", "td"])
def test_diagnostic_loader_checks_provenance_and_numerical_contract(legacy, monkeypatch, fault):
    data = copy.copy(legacy)
    data.data = dict(legacy.data)
    key = ("A", "8mic", "test_s1", 0., "test_s1_00000")
    diagnostic = dict(frozen_diagnostics(**diagnostic_arrays()), method="AffineMM", split="test_s1", kappa=0., id=key[-1])
    data.data[key] = dict(legacy.data[key], target_direct_sdr_out=diagnostic["target_direct_sdr_out"])
    if fault == "origin":
        diagnostic["coefficient_origin"] = "target_only"
    elif fault == "residual":
        diagnostic["nominal_D_p95"] = 1
    elif fault == "reconstruction":
        diagnostic["target_spectrum_reconstruction_relative_error"] = -1
    elif fault == "events":
        diagnostic["wng_undefined_count"] = 999999
    elif fault == "wng":
        diagnostic["wng_finite_min_db"] = None
    elif fault == "td":
        diagnostic["target_direct_sdr_out"] += 1
    monkeypatch.setattr(data, "diagnostic_records", lambda name, spec: [("8mic", diagnostic, "synthetic in-memory math fixture")])
    if fault:
        with pytest.raises(InputError):
            data.load_diagnostics("A", {"method": "AffineMM"})
    else:
        data.load_diagnostics("A", {"method": "AffineMM"})
        assert data.data[key]["_diagnostic_source"] == "synthetic in-memory math fixture"
