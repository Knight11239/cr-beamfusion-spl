"""Release-package test configuration.

Some tests in the development repository exercise components that are deliberately not part of this
package. They are skipped here, with the reason shown in the pytest summary (`pytest -rs`):
  * the cloud job runner and job-generation scripts (not released);
  * tests that need locally generated speech/RIR data (see README, "Full reproduction");
  * tests of the pre-registered analysis code that re-read the full, hash-pinned per-utterance CSVs,
    unless the raw-results release asset has been extracted (see README); the repository itself ships
    the frozen ledger and column-reduced copies of the per-utterance results.
"""
from pathlib import Path

import pytest

# present after extracting the release asset cr-beamfusion-spl-per-utterance-v1.tar.gz (README)
RAW_PRESENT = (Path(__file__).parent / "results/kaggle/stage4-b3eval-a/eval/b3_test_s1/test_s1_samples.csv").exists()

RUNNER = {"test_runner_rir_outputs_hold_no_input_links", "test_runner_eval_generates_and_saves_rirs", "test_mvdr_mmval_load_selection_grid"}
NEEDS_DATA = {"test_test_snr_override"}
# analysis-code tests that run on synthetic inputs (verified to pass without the raw CSVs)
OFFLINE_OK = {"test_cluster_bootstrap_keeps_fixed_condition_weights", "test_condition_sign_enumeration_ties_zeros_and_resolution",
              "test_csv_output_uses_lf_and_preserves_missing_p", "test_fixed_holm_slots_are_not_measured_p_equals_one",
              "test_frozen_coefficients_and_conjugation_negative_controls", "test_hand_computable_pair_and_no_finite_intersection",
              "test_optional_archive_istft_identity_without_any_network", "test_paired_t60_bootstrap_uses_equal_strata",
              "test_pending_paths_are_not_even_checked", "test_response_and_td_detect_scale_which_wng_does_not",
              "test_single_trajectory_is_not_a_zero_width_ci", "test_tost_degenerate_boundaries", "test_tost_student_t_formula_and_not_significance"}


def pytest_collection_modifyitems(config, items):
    for it in items:
        name = it.originalname or it.name
        if name in RUNNER:
            it.add_marker(pytest.mark.skip(reason="cloud job runner / job-generation script not included in the release"))
        elif name in NEEDS_DATA:
            it.add_marker(pytest.mark.skip(reason="needs locally generated dev data (README: Full reproduction)"))
        elif "control_analysis" in str(it.fspath) and name not in OFFLINE_OK and not RAW_PRESENT:
            it.add_marker(pytest.mark.skip(reason="re-reads full hash-pinned per-utterance CSVs (not shipped; frozen ledger included)"))


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: long-running test")
