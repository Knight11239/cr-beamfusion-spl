"""Explicit, hash-pinned inputs. No globbing or prospective result discovery."""
import csv
import hashlib
import io
import json
import math
from collections import Counter
from pathlib import Path

DATASETS = {"8mic": ("test_s1", "test_s2"), "G2sim": ("test_s1", "test_s2"),
            "MIRD": ("mird_s1", "mird_s2")}
MODEL_DATASETS = {k: ("8mic",) for k in ("A", "B", "C", "R", "F")}
MODEL_DATASETS.update({k: ("G2sim", "MIRD") for k in ("GA", "GL", "GBmm", "GB")})
PROSPECTIVE = {"C", "R", "F", "GBmm", "GB"}
INPUT_COLUMNS = ("si_sdr_dp_in", "si_sdr_in", "snr_dp_in", "snr_in")
MODEL_REQUIREMENTS = ("training_protocol", "checkpoint_selection", "input_protocol", "operator_checks", "geometry")
COMPARISON_REQUIREMENTS = {"C1": ("head_match", "initial_function"),
                           "C2": ("parameterization_recorded",), "C3": ("parameterization_recorded",),
                           "C4": ("convex_protocol",), "C5": ("no_postprocessing",),
                           "F_D": ("no_postprocessing",), "F_TD": ("no_postprocessing",),
                           "F_R": ("no_postprocessing",)}


class InputError(ValueError):
    def __init__(self, message, ids=None):
        super().__init__(message)
        self.ids = ids or []


class Inputs:
    def __init__(self, root, manifest, allow_new=False):
        self.root = Path(root).resolve()
        self.manifest = manifest
        self.allow_new = allow_new
        self.data, self.issues, self.audit, self.qualifications = {}, {}, [], []
        self.cache, self.metadata, self.enabled = {}, {}, set()
        if set(manifest["models"]) != set(MODEL_DATASETS):
            raise InputError("Manifest must contain all nine fixed model slots")
        # This happens BEFORE any prospective source, weight or evidence path is inspected.
        for name in PROSPECTIVE:
            if manifest["models"][name]["state"] != "pending" and not allow_new:
                raise InputError(f"{name}: prospective inputs require explicit --allow-new-results")
        for name, spec in manifest["models"].items():
            if spec["state"] not in ("pending", "existing", "evaluated"):
                raise InputError(f"Unknown execution state for {name}")
            if name in PROSPECTIVE and spec["state"] == "existing":
                raise InputError("Prospective models cannot be disguised as existing baselines")
        if not allow_new:
            policy = json.loads(Path(__file__).with_name("legacy_inputs.json").read_text())
            if manifest["plan"] != policy["plan"] or manifest["mird_metadata"] != policy["mird_metadata"]:
                raise InputError("Legacy-only mode requires the bundled pinned plan/metadata")
            for name, spec in manifest["models"].items():
                if spec["state"] == "pending":
                    continue
                if name not in ("A", "B", "GA", "GL") or spec != policy["models"][name]:
                    raise InputError("Legacy-only mode permits only unchanged bundled A/B/GA/GL inputs; explicit new inputs require the opt-in")

    def checked(self, ref, purpose, cache=True):
        if not ref.get("sha256") or not ref.get("path"):
            raise InputError(f"Unpinned input for {purpose}")
        path = (self.root / ref["path"]).resolve()
        if not path.is_relative_to(self.root):
            raise InputError("Inputs must be explicitly repo-relative")
        key = (str(path), ref["sha256"])
        if key not in self.cache:
            content = path.read_bytes()
            digest = hashlib.sha256(content).hexdigest()
            if digest != ref["sha256"]:
                raise InputError(f"SHA mismatch: {ref['path']}")
            if cache:
                self.cache[key] = content
        else:
            content = self.cache[key]
        self.audit.append(dict(path=ref["path"], sha256=ref["sha256"], purpose=purpose))
        return content

    def read_csv(self, ref, purpose):
        return list(csv.DictReader(io.StringIO(self.checked(ref, purpose).decode("utf-8"))))

    def load(self):
        for split, ref in self.manifest["mird_metadata"].items():
            scenes = json.loads(self.checked(ref, "fixed MIRD metadata"))["scenes"]
            indexed = {r["id"]: r for r in scenes}
            counts = Counter((r["rt60"], r["subarray"]) for r in scenes)
            if len(indexed) != len(scenes) or len(indexed) != 600 or len(counts) != 6 or set(counts.values()) != {100}:
                raise InputError("Invalid fixed MIRD condition metadata")
            self.metadata[split] = indexed
        for name, spec in self.manifest["models"].items():
            if spec["state"] == "pending":
                # Deliberately do not check existence, hash, qualify or open ANY pending path.
                continue
            self.enabled.add(name)
            try:
                expected_mics = 8 if name in ("A", "B", "C", "R", "F") else 4
                if spec.get("n_mics") != expected_mics or spec.get("seed") != 0:
                    raise InputError("Wrong geometry or training seed for fixed plan slot")
                self.checked(spec["checkpoint"], f"{name}: checkpoint hash only; no model loading")
                for source in spec["sources"]:
                    dataset = source["dataset"]
                    if dataset not in MODEL_DATASETS[name]:
                        raise InputError(f"{name} cannot be evaluated on geometry {dataset}")
                    for row in self.read_csv(source, f"{name}: existing" if name not in PROSPECTIVE else f"{name}: explicit evaluated input"):
                        if row.get("method") != spec["method"]:
                            continue
                        split, kappa = row["split"], float(row["kappa"])
                        if split not in DATASETS[dataset] or kappa not in ((0., 1., 2.) if dataset == "8mic" else (0.,)):
                            continue
                        key = (name, dataset, split, kappa, row["id"])
                        tagged = dict(row, _source=source["path"], _sha256=source["sha256"])
                        if key in self.data:
                            old = self.data[key]
                            fields = set(row) | {k for k in old if not k.startswith("_")}
                            changed = [k for k in fields if old.get(k) != row.get(k)]
                            self.audit.append(dict(purpose="duplicate; first explicit source retained", model=name,
                                                   id=row["id"], kept=old["_source"], ignored=source["path"], changed_fields=changed))
                            if changed:
                                raise InputError(f"Conflicting duplicate {key}: {changed}")
                        else:
                            self.data[key] = tagged
                self.load_diagnostics(name, spec)
            except (InputError, OSError, KeyError, ValueError, RuntimeError) as error:
                self.issues[name] = str(error)
        return self

    def diagnostic_records(self, name, spec):
        for source in spec.get("diagnostics", []):
            for row in self.read_csv(source, name + ": explicitly supplied diagnostic summary"):
                yield source["dataset"], row, source["path"]
        for index in spec.get("diagnostic_archives", []):
            from .diagnostics import from_archive
            entries = json.loads(self.checked(index, name + ": explicit diagnostic archive index"))
            if not isinstance(entries, list):
                raise InputError("Diagnostic archive index must be a list")
            for item in entries:
                dataset = item["dataset"]
                key = (name, dataset, item["split"], float(item["kappa"]), item["id"])
                if dataset not in MODEL_DATASETS[name] or key not in self.data:
                    raise InputError("Diagnostic archive index includes unmatched/out-of-plan score rows")
                content = self.checked(item, name + ": explicit coefficient archive, no inference", cache=False)
                derived = from_archive(content, spec["n_mics"])
                row = dict(derived, method=spec["method"], split=item["split"], kappa=item["kappa"], id=item["id"])
                yield dataset, row, item["path"]

    def load_diagnostics(self, name, spec):
        seen = set()
        for dataset, row, path in self.diagnostic_records(name, spec):
            if dataset not in MODEL_DATASETS[name]:
                raise InputError("Wrong diagnostic geometry")
            if row.get("method") != spec["method"]:
                continue
            key = (name, dataset, row["split"], float(row["kappa"]), row["id"])
            if key in seen:
                raise InputError(f"Duplicate diagnostic row {key}")
            seen.add(key)
            if key not in self.data:
                raise InputError(f"Diagnostic without raw score row {key}")
            if (row.get("diagnostic_version") != "frozen-mixture-coefficients-v1" or
                row.get("coefficient_origin") != "mixture" or
                str(row.get("runtime_dictionary", "")).lower() != "true" or
                row.get("target_output_kind") != "frozen_mixture_coefficients"):
                raise InputError("Missing/incorrect frozen-mixture diagnostic provenance")
            for field in ("nominal_D_p95", "nominal_D_rms", "nominal_D_max", "nominal_D_dc_max"):
                number = float(row[field])
                if not math.isfinite(number) or number < 0:
                    raise InputError("Invalid nominal response diagnostic")
                self.data[key][field] = row[field]
            if (float(row["nominal_D_p95"]) > float(row["nominal_D_max"]) + 1e-12 or
                float(row["nominal_D_rms"]) > float(row["nominal_D_max"]) + 1e-12 or
                not math.isfinite(float(row["target_spectrum_reconstruction_relative_error"])) or
                not 0 <= float(row["target_spectrum_reconstruction_relative_error"]) <= 1e-5):
                raise InputError("Diagnostic summary fails numerical self-consistency")
            for field in ("nominal_gain_amp_mean", "nominal_gain_phase_abs_p95", "sum_c_residual_max"):
                value = float(row[field])
                if not math.isfinite(value) or value < 0:
                    raise InputError("Missing/nonfinite gain or coefficient-sum diagnostic")
            event_counts = {}
            for field in ("wng_undefined_count", "wng_zero_response_count", "near_zero_filter_count", "tf_non_dc_points"):
                value = float(row[field])
                if not math.isfinite(value) or value < 0 or value != int(value):
                    raise InputError("Invalid WNG/degeneracy count")
                event_counts[field] = int(value)
            points = event_counts["tf_non_dc_points"]
            unscorable = event_counts["wng_undefined_count"] + event_counts["wng_zero_response_count"]
            if points < 1 or unscorable > points or event_counts["near_zero_filter_count"] > points:
                raise InputError("Inconsistent WNG event counts")
            wng = [row.get(field) for field in ("wng_finite_min_db", "wng_finite_median_db")]
            if unscorable < points:
                if not all(value not in (None, "") and math.isfinite(float(value)) for value in wng):
                    raise InputError("Finite WNG summary missing despite finite TF points")
                if float(wng[0]) > float(wng[1]):
                    raise InputError("WNG minimum exceeds median")
            elif any(value not in (None, "") for value in wng):
                raise InputError("Undefined/zero WNG cannot have a fabricated finite summary")
            # TD is the evaluator's actual frozen output, without scale optimization.
            if (not math.isfinite(float(row["target_direct_sdr_out"])) or
                not math.isfinite(float(self.data[key]["target_direct_sdr_out"])) or
                abs(float(self.data[key]["target_direct_sdr_out"]) - float(row["target_direct_sdr_out"])) > 1e-3):
                raise InputError("Diagnostic TD disagrees with frozen-output score")
            self.data[key]["_diagnostic_source"] = path
            for field, value in row.items():
                if field not in self.data[key]:
                    self.data[key][field] = value

    def group(self, model, dataset, split, kappa):
        return {key[4]: row for key, row in self.data.items() if key[:4] == (model, dataset, split, kappa)}

    def expected_ids(self, dataset, split):
        if dataset == "MIRD":
            return set(self.metadata[split])
        return {f"{split}_{i:05d}" for i in range(600)}

    def qualification(self, owner, requirement, item):
        state = item.get("status", "unknown")
        reason = item.get("reason", "not recorded")
        if state not in ("pass", "fail", "unknown"):
            state, reason = "fail", "Invalid qualification state"
        if state == "pass":
            evidence = item.get("evidence", [])
            if not evidence:
                state, reason = "unknown", "A claimed pass needs explicit hash-pinned evidence"
            else:
                try:
                    for ref in evidence:
                        self.checked(ref, f"qualification {owner}/{requirement}; recorded evidence, not independent fact verification")
                except (InputError, OSError) as error:
                    state, reason = "fail", str(error)
        self.qualifications.append(dict(owner=owner, requirement=requirement, status=state, reason=reason,
                                        evidence=item.get("evidence", [])))
        return state

    def qualify(self, comparison, left, right, dataset, split, kappa):
        states = []
        for name in (left, right):
            if name not in self.enabled:
                return "unknown"
            requirements = MODEL_REQUIREMENTS + (("resume_budget",) if name in ("GBmm", "GB") else ())
            spec = self.manifest["models"][name]
            for requirement in requirements:
                states.append(self.qualification(name, requirement, spec.get("qualifications", {}).get(requirement, {})))
        for requirement in COMPARISON_REQUIREMENTS.get(comparison, ()):
            states.append(self.qualification(comparison, requirement,
                          self.manifest.get("comparisons", {}).get(comparison, {}).get(requirement, {})))
        if "A" in (left, right):
            a = {key: row for key, row in self.data.items() if key[0:2] == ("A", "8mic")}
            values = [float(r[k]) for r in a.values() for k in ("nominal_D_max", "nominal_D_dc_max")
                      if k in r and r.get("_diagnostic_source")]
            if values and max(values) > 1e-3:
                states.append("fail")
                self.qualifications.append(dict(owner="A", requirement="nominal_response_audit", status="fail",
                                                reason="A maximum response residual exceeds planned 1e-3"))
            elif comparison in ("F_D", "F_TD", "F_R") and len(values) != 7200:
                states.append("unknown")
                self.qualifications.append(dict(owner="A", requirement="nominal_response_audit", status="unknown",
                                                reason="Missing complete A non-DC/DC maxima; do not assume pass"))
            elif len(values) == 7200:
                self.qualifications.append(dict(owner="A", requirement="nominal_response_audit", status="pass",
                                                reason=f"All six cells supplied; maximum residual {max(values):.17g} <= 1e-3"))
        return "fail" if "fail" in states else ("unknown" if "unknown" in states else "pass")
