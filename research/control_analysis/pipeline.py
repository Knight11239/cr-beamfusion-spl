"""Fixed slots, complete-case validation, lineage and prospective interpretation."""
import hashlib
import json
from collections import Counter

import numpy as np

from .inputs import DATASETS, INPUT_COLUMNS, InputError
from .statistics import paired, tost, holm, stratified_bootstrap, cluster_bootstrap

PRIMARY = "d_si_sdr_dp"
SECONDARY = ("d_si_sdr", "stoi_dp_out", "pesq_dp_out")
MIRD_METRICS = (PRIMARY, *SECONDARY, "target_direct_sdr_out", "target_sdr_dp_out", "target_sdr_out")
COMPARISONS = {"C1": ("A", "C"), "C2": ("R", "C"), "C3": ("A", "R"),
               "C4": ("C", "B"), "C5": ("F", "A")}
G2_COMPARISONS = {"G1": ("GA", "GBmm"), "G2": ("GBmm", "GB")}
MIRD_COMPARISONS = {"M1": ("GA", "GBmm"), "M2": ("GL", "GBmm"), "M3": ("GBmm", "GB")}
INPUT_METRIC_ATOL = 1e-5  # Numerical audit tolerance in dB; NOT an engineering claim threshold.


def make_slot(comparison, family, dataset, split, kappa, left, right, metric, kind="difference", group="all"):
    return dict(slot=f"{comparison}:{split}:k{kappa:g}:{kind}:{metric}:{group}", comparison=comparison,
                family=family, dataset=dataset, split=split, kappa=kappa, seed=0, group=group,
                left=left, right=right, metric=metric, kind=kind, execution_status="未执行",
                data_status="not_executed", qualification_status="unknown", eligible=False,
                p_raw=None, p_for_holm=1.0, correction_placeholder=True, reason="", n=None,
                mean_left=None, mean_right=None, mean_diff=None, ci_lo=None, ci_hi=None,
                median_diff=None, r_rb=None, cohen_dz=None, nonzero_n=None,
                available_left=0, available_right=0, paired_ids=None,
                paired_id_sha256=None, left_sources=None, right_sources=None,
                p_holm_global=None, p_holm_family=None, failed_ids=[])


def primary_slots():
    slots = []
    for comp, (left, right) in COMPARISONS.items():
        for split in DATASETS["8mic"]:
            for k in (0., 1., 2.):
                slots.append(make_slot(comp, "H8", "8mic", split, k, left, right, PRIMARY))
    for split in DATASETS["8mic"]:
        for k in (0., 1., 2.):
            slots.append(make_slot("C1", "H8", "8mic", split, k, "A", "C", PRIMARY, "tost"))
            slots.append(make_slot("F_D", "HF", "8mic", split, k, "F", "A", "nominal_D_p95"))
            slots.append(make_slot("F_TD", "HF", "8mic", split, k, "F", "A", "target_direct_sdr_out"))
            if k > 0:
                slots.append(make_slot("F_R", "HF", "8mic", split, k, "F", "A", PRIMARY, "degradation"))
    for comp, (left, right) in G2_COMPARISONS.items():
        for split in DATASETS["G2sim"]:
            slots.append(make_slot(comp, "HG2", "G2sim", split, 0., left, right, PRIMARY))
    assert Counter(r["family"] for r in slots) == {"H8": 36, "HF": 16, "HG2": 4}
    assert len({r["slot"] for r in slots}) == 56
    return slots


class Analysis:
    def __init__(self, inputs):
        self.inputs = inputs
        self.pair_audit = []
        self.condition_rows, self.cluster_rows = [], []
        self.bootstrap_rows, self.t60_rows = [], []
        self.smoke_bootstrap = []

    def pair(self, left, right, dataset, split, kappa):
        I = self.inputs
        a, b = (I.group(m, dataset, split, kappa) for m in (left, right))
        expected = I.expected_ids(dataset, split)
        for model, group in ((left, a), (right, b)):
            if model in I.issues:
                raise InputError(f"{model}: {I.issues[model]}")
            if set(group) != expected:
                bad = sorted(set(group) ^ expected)
                raise InputError(f"{model}: coverage != 600 complete IDs; mismatched IDs={bad}", bad)
        ids = sorted(expected)
        anchor_model = "A" if dataset == "8mic" else "GA"
        anchor = I.group(anchor_model, dataset, split, kappa)
        if set(anchor) != expected or anchor_model in I.issues:
            raise InputError("Missing trustworthy fixed A/GA reference for cohort/input validation")
        maxima = {col: 0. for col in INPUT_COLUMNS}
        for i in ids:
            reference = anchor[i]
            for group in (a, b):
                row = group[i]
                if dataset == "MIRD":
                    expected_t60 = float(I.metadata[split][i]["rt60"])
                    if abs(float(row["t60"]) - expected_t60) > 1e-9:
                        raise InputError(f"MIRD metadata/T60 mismatch {i}")
                if float(row["t60"]) != float(reference["t60"]):
                    raise InputError(f"T60 mismatch {i}")
                for col in INPUT_COLUMNS:
                    x, y = float(row[col]), float(reference[col])
                    maxima[col] = max(maxima[col], abs(x - y))
                    if not np.isfinite([x, y]).all() or abs(x - y) > INPUT_METRIC_ATOL:
                        raise InputError(f"Input metric mismatch {i}/{col}", [i])
                for col in ("scenario", "moving", "n_interf", "snr_set", "sir_set"):
                    x, y = row[col], reference[col]
                    if col in ("snr_set", "sir_set", "n_interf"):
                        if abs(float(x) - float(y)) > 1e-10:
                            raise InputError(f"Scene metadata mismatch {i}/{col}")
                    elif x != y:
                        raise InputError(f"Scene metadata mismatch {i}/{col}")
                for suffix in ("_dp", ""):
                    delta, output, inp = (float(row["d_si_sdr" + suffix]),
                                          float(row["si_sdr" + suffix + "_out"]),
                                          float(row["si_sdr" + suffix + "_in"]))
                    if suffix == "_dp" and not np.isfinite([delta, output, inp]).all():
                        raise InputError(f"Nonfinite primary delta/output/input {i}", [i])
                    if np.isfinite([delta, output, inp]).all() and abs(delta - output + inp) > 1e-8:
                        raise InputError(f"Inconsistent SI-SDR delta/output/input {i}/{suffix}", [i])
        if dataset != "MIRD":
            counts = Counter(float(a[i]["t60"]) for i in ids)
            if counts != {.3: 300, .7: 300}:
                raise InputError(f"Expected fixed 300/300 T60 strata, got {counts}")
        self.pair_audit.append(dict(dataset=dataset, split=split, kappa=kappa, left=left, right=right,
                                    input_metric_atol_db=INPUT_METRIC_ATOL, input_metric_max_abs_diff=maxima,
                                    note="Numerical proxy only; does not prove byte-identical waveforms or training qualification"))
        return ids, a, b

    @staticmethod
    def endpoint(group, ids, metric):
        values = np.array([float(group[i].get(metric, "nan"))
                           if metric != "nominal_D_p95" or group[i].get("_diagnostic_source") else np.nan
                           for i in ids])
        bad = [i for i, v in zip(ids, values) if not np.isfinite(v)]
        return values, bad

    def calculate(self, slot, *, smoke=False, descriptive=False):
        I = self.inputs
        left, right, dataset, split, k = (slot[c] for c in ("left", "right", "dataset", "split", "kappa"))
        slot["available_left"] = len(I.group(left, dataset, split, k))
        slot["available_right"] = len(I.group(right, dataset, split, k))
        pending = [m for m in (left, right) if m not in I.enabled]
        if pending:
            slot["reason"] = "未执行模型槽位: " + ", ".join(pending)
            return slot, None
        try:
            ids, a, b = self.pair(left, right, dataset, split, k)
            # A model-specific primary score failure blocks every formal comparison
            # at this cell, even if a secondary endpoint happens to remain finite.
            for group in (a, b):
                _, bad = self.endpoint(group, ids, PRIMARY)
                if bad:
                    slot["failed_ids"] = bad
                    raise InputError("Model-specific nonfinite primary output; no finite-intersection deletion")
            if slot["group"] in ("0.3", "0.7"):
                ids = [i for i in ids if float(a[i]["t60"]) == float(slot["group"])]
            av, bad_a = self.endpoint(a, ids, slot["metric"])
            bv, bad_b = self.endpoint(b, ids, slot["metric"])
            if bad_a or bad_b:
                slot["failed_ids"] = sorted(set(bad_a + bad_b))
                raise InputError("Missing/nonfinite endpoint; no imputation or selective deletion")
            if slot["kind"] == "degradation":
                zero_ids, a0, b0 = self.pair(left, right, dataset, split, 0.)
                if zero_ids != ids:
                    raise InputError("Kappa=0 IDs do not match degradation cell")
                az, az_bad = self.endpoint(a0, ids, PRIMARY)
                bz, bz_bad = self.endpoint(b0, ids, PRIMARY)
                if az_bad or bz_bad:
                    slot["failed_ids"] = sorted(set(az_bad + bz_bad))
                    raise InputError("Nonfinite kappa=0 score; no selective degradation pairs")
                for i in ids:
                    for hi, lo in ((a, a0), (b, b0)):
                        for col in INPUT_COLUMNS:
                            if abs(float(hi[i][col]) - float(lo[i][col])) > INPUT_METRIC_ATOL:
                                raise InputError("Changed input reference across kappa")
                slot.update(absolute_left_at_kappa=float(av.mean()), absolute_right_at_kappa=float(bv.mean()),
                            absolute_left_at_zero=float(az.mean()), absolute_right_at_zero=float(bz.mean()),
                            zero_left_sources=dict(Counter(a0[i]["_source"] for i in ids)),
                            zero_right_sources=dict(Counter(b0[i]["_source"] for i in ids)))
                av, bv = av - az, bv - bz
            slot.update(paired(av, bv, run_test=not descriptive))
            if slot["kind"] == "tost":
                slot.update(tost(av, bv))
            slot.update(data_status="complete", paired_ids=len(ids),
                        paired_id_sha256=hashlib.sha256("\n".join(ids).encode()).hexdigest(),
                        left_sources=dict(Counter(a[i]["_source"] for i in ids)),
                        right_sources=dict(Counter(b[i]["_source"] for i in ids)))
            if slot["metric"] in (PRIMARY, "d_si_sdr"):
                suffix = "_dp" if slot["metric"] == PRIMARY else ""
                for label, group in (("left", a), ("right", b)):
                    slot[label + "_mean_si_sdr" + suffix + "_in"] = float(np.mean([float(group[i]["si_sdr" + suffix + "_in"]) for i in ids]))
                    slot[label + "_mean_si_sdr" + suffix + "_out"] = float(np.mean([float(group[i]["si_sdr" + suffix + "_out"]) for i in ids]))
            for label, group in (("left", a), ("right", b)):
                diagnostic_sources = Counter(group[i].get("_diagnostic_source", "") for i in ids)
                if any(diagnostic_sources):
                    slot[label + "_diagnostic_sources"] = {p: n for p, n in diagnostic_sources.items() if p}
            q = "not_applicable_smoke" if smoke else I.qualify(slot["comparison"], left, right, dataset, split, k)
            slot["qualification_status"] = q
            slot["execution_status"] = "已计算（空跑，不属预定主检验）" if smoke else ("已计算" if q == "pass" else "不可判定")
            slot["eligible"] = q == "pass" and not smoke and not descriptive
            slot["p_for_holm"] = slot["p_raw"] if slot["eligible"] else 1.0
            slot["correction_placeholder"] = not slot["eligible"]
            slot["reason"] = "" if q == "pass" else ("既有模型的格式/配对检查，不作新对照结论" if smoke else "资格未确认或失败；仅保留描述性计算")
            return slot, (ids, av, bv, a, b)
        except (InputError, ValueError, KeyError) as error:
            slot["failed_ids"] = sorted(set(slot["failed_ids"] + getattr(error, "ids", [])))
            slot.update(execution_status="不可判定", data_status="invalid_or_missing", reason=str(error),
                        eligible=False, p_for_holm=1.0, correction_placeholder=True)
            return slot, None

    def diagnostic_summary(self):
        """D maxima and actual WNG events; never derive missing D from scores."""
        records = []
        fields = ("nominal_D_p95", "nominal_D_rms", "nominal_D_max", "nominal_D_dc_max",
                  "nominal_gain_amp_mean", "nominal_gain_phase_abs_p95", "sum_c_residual_max",
                  "wng_finite_min_db", "wng_finite_median_db", "target_direct_sdr_out")
        counts = ("wng_undefined_count", "wng_zero_response_count", "near_zero_filter_count", "tf_non_dc_points")
        for model in ("A", "F"):
            for split in DATASETS["8mic"]:
                for k in (0., 1., 2.):
                    record = dict(model=model, split=split, kappa=k, status="未执行", n=0,
                                  reason="Pending slot; no diagnostic paths inspected")
                    rows = self.inputs.group(model, "8mic", split, k)
                    if model in self.inputs.enabled:
                        tagged = [rows[i] for i in sorted(rows) if rows[i].get("_diagnostic_source")]
                        record.update(status="不可判定", n=len(tagged), reason="Require all 600 source-validated diagnostic rows")
                        if model not in self.inputs.issues and set(rows) == self.inputs.expected_ids("8mic", split) and len(tagged) == 600:
                            record.update(status="descriptive", reason="No additional hypothesis test; TF points are not independent utterances",
                                          sources=dict(Counter(r["_diagnostic_source"] for r in tagged)))
                            for field in fields:
                                values = [float(r[field]) for r in tagged if r.get(field) not in (None, "")]
                                finite = np.array([v for v in values if np.isfinite(v)])
                                record[field + "_finite_n"] = len(finite)
                                if len(finite):
                                    for name, value in (("mean", finite.mean()), ("min", finite.min()), ("max", finite.max()), ("p95", np.percentile(finite, 95))):
                                        record[field + "_" + name] = float(value)
                            for field in counts:
                                values = [float(r[field]) for r in tagged if r.get(field) not in (None, "")]
                                record[field + "_recorded_n"] = len(values)
                                record[field + "_total"] = sum(values) if len(values) == 600 else None
                            record["global_nominal_response_max"] = max(record["nominal_D_max_max"], record["nominal_D_dc_max_max"])
                    records.append(record)
        return records

    def mird(self, comp, left, right, split, smoke=False):
        summary = []
        for metric in MIRD_METRICS:
            family = "SMOKE_MIRD" if smoke else "HMIRD"
            base = make_slot(comp, family, "MIRD", split, 0., left, right, metric)
            whole, bundle = self.calculate(base, smoke=smoke, descriptive=True)
            metric_conditions = []
            for rt in ("0.160", "0.360", "0.610"):
                for sub in ("A", "B"):
                    row = dict(whole, slot=whole["slot"] + f":{rt}/{sub}", group=f"{rt}/{sub}",
                               p_raw=None, eligible=False, p_for_holm=1.0, correction_placeholder=True)
                    if bundle:
                        ids, av, bv, a, b = bundle
                        mask = np.array([(self.inputs.metadata[split][i]["rt60"], self.inputs.metadata[split][i]["subarray"]) == (rt, sub) for i in ids])
                        if int(mask.sum()) != 100:
                            raise InputError("Condition does not contain all 100 paired IDs")
                        row.update(paired(av[mask], bv[mask], run_test=False))
                        local_ids = [i for i, yes in zip(ids, mask) if yes]
                        row.update(paired_ids=100, paired_id_sha256=hashlib.sha256("\n".join(local_ids).encode()).hexdigest(),
                                   left_sources=dict(Counter(a[i]["_source"] for i in local_ids)),
                                   right_sources=dict(Counter(b[i]["_source"] for i in local_ids)))
                        if metric in (PRIMARY, "d_si_sdr"):
                            suffix = "_dp" if metric == PRIMARY else ""
                            for label, group in (("left", a), ("right", b)):
                                for ending in ("_in", "_out"):
                                    column = "si_sdr" + suffix + ending
                                    row[label + "_mean_" + column] = float(np.mean([float(group[i][column]) for i in local_ids]))
                    metric_conditions.append(row)
                    self.condition_rows.append(row)
            pooled = dict(whole, group="six_condition_means", kind="condition_means")
            if bundle:
                ca = np.array([r["mean_left"] for r in metric_conditions])
                cb = np.array([r["mean_right"] for r in metric_conditions])
                pooled.update(paired(ca, cb, condition_level=True, run_test=metric == PRIMARY))
                differences = ca - cb
                pooled.update(paired_ids=600, positive_conditions=int((differences > 0).sum()),
                              same_sign_as_pooled=int((np.sign(differences) == np.sign(differences.mean())).sum()),
                              condition_diff_min=float(differences.min()), condition_diff_max=float(differences.max()),
                              condition_values=differences.tolist(), ci_scope="six fixed measured conditions; not rooms")
                pooled["eligible"] = whole["qualification_status"] == "pass" and metric == PRIMARY and not smoke
                pooled["p_for_holm"] = pooled["p_raw"] if pooled["eligible"] else 1.
                pooled["correction_placeholder"] = not pooled["eligible"]
                ids, av, bv, _, _ = bundle
                boot = cluster_bootstrap((av - bv)[:, None], [self.inputs.metadata[split][i] for i in ids])
                if boot["bootstrap_status"] == "computed":
                    boot["bootstrap_ci_lo"], boot["bootstrap_ci_hi"] = boot["bootstrap_ci_lo"][0], boot["bootstrap_ci_hi"][0]
                self.cluster_rows.append(dict(slot=pooled["slot"], comparison=comp, split=split, metric=metric,
                                              n_samples=600, qualification_status=whole["qualification_status"],
                                              confirmation_scope="sensitivity only; conditional on measured RIRs", **boot))
            else:
                self.cluster_rows.append(dict(slot=pooled["slot"], comparison=comp, split=split, metric=metric,
                                              bootstrap_status="未执行", reason=whole["reason"]))
            summary.append(pooled)
        return summary

    def run(self, smoke=True):
        main = primary_slots()
        for slot in main:
            _, bundle = self.calculate(slot)
            if slot["family"] == "H8" and slot["kind"] == "difference":
                item = dict(slot=slot["slot"], status=slot["execution_status"], reason=slot["reason"])
                if bundle:
                    ids, a, b, rows, _ = bundle
                    item.update(stratified_bootstrap(a - b, [float(rows[i]["t60"]) for i in ids]))
                self.bootstrap_rows.append(item)
        holm(main, "p_holm_global", 56)
        for family, count in (("H8", 36), ("HF", 16), ("HG2", 4)):
            holm([r for r in main if r["family"] == family], "p_holm_family", count)
        secondary = []
        for metric, family in zip(SECONDARY, ("Href", "Hstoi", "Hpesq")):
            group = []
            for slot in main:
                if slot["kind"] != "difference" or slot["family"] == "HF":
                    continue
                row = make_slot(slot["comparison"], family, slot["dataset"], slot["split"], slot["kappa"],
                                slot["left"], slot["right"], metric)
                self.calculate(row)
                group.append(row)
            holm(group, "p_holm_family", 34)
            secondary.extend(group)
        for comp, (left, right) in COMPARISONS.items():
            for split in DATASETS["8mic"]:
                for k in (0., 1., 2.):
                    for t60 in ("0.3", "0.7"):
                        for metric in (PRIMARY, *SECONDARY):
                            row = make_slot(comp, "T60_descriptive", "8mic", split, k, left, right, metric, group=t60)
                            self.calculate(row, descriptive=True)
                            self.t60_rows.append(row)
        mird = []
        for comp, (left, right) in MIRD_COMPARISONS.items():
            for split in DATASETS["MIRD"]:
                mird.extend(self.mird(comp, left, right, split))
        holm([r for r in mird if r["metric"] == PRIMARY], "p_holm_family", 6)
        target_auxiliary = []
        for split in DATASETS["8mic"]:
            for k in (0., 1., 2.):
                for metric in ("target_sdr_dp_out", "target_sdr_out"):
                    row = make_slot("C5", "F_target_descriptive", "8mic", split, k, "F", "A", metric)
                    self.calculate(row, descriptive=True)
                    target_auxiliary.append(row)
        smoke_rows = []
        if smoke:
            for dataset, left, right in (("8mic", "A", "B"), ("G2sim", "GL", "GA")):
                for split in DATASETS[dataset]:
                    for k in ((0., 1., 2.) if dataset == "8mic" else (0.,)):
                        for metric in (PRIMARY, *SECONDARY, "target_direct_sdr_out"):
                            row = make_slot("SMOKE_" + left + "_" + right, "SMOKE_ONLY", dataset, split, k, left, right, metric)
                            _, bundle = self.calculate(row, smoke=True)
                            smoke_rows.append(row)
                            if dataset == "8mic" and metric == PRIMARY and bundle:
                                ids, av, bv, rows, _ = bundle
                                self.smoke_bootstrap.append(dict(slot=row["slot"], scope="legacy smoke only; not a registered hypothesis",
                                    **stratified_bootstrap(av - bv, [float(rows[i]["t60"]) for i in ids])))
            for split in DATASETS["MIRD"]:
                smoke_rows.extend(self.mird("SMOKE_GL_GA", "GL", "GA", split, smoke=True))
        # Persist model qualification statuses even when every future contrast is pending.
        for name, spec in self.inputs.manifest["models"].items():
            if name not in self.inputs.enabled:
                for requirement in ("execution", "training_protocol", "checkpoint_selection", "input_protocol", "operator_checks", "geometry") + (("resume_budget",) if name in ("GBmm", "GB") else ()):
                    self.inputs.qualifications.append(dict(owner=name, requirement=requirement, status="未执行",
                                                           reason="Pending slot; no source/checkpoint/evidence paths inspected"))
                continue
            for requirement in ("training_protocol", "checkpoint_selection", "input_protocol", "operator_checks", "geometry"):
                self.inputs.qualification(name, requirement, spec.get("qualifications", {}).get(requirement, {}))
        from .inputs import COMPARISON_REQUIREMENTS
        for comp, requirements in COMPARISON_REQUIREMENTS.items():
            for requirement in requirements:
                if not any(r["owner"] == comp and r["requirement"] == requirement for r in self.inputs.qualifications):
                    self.inputs.qualifications.append(dict(owner=comp, requirement=requirement, status="未执行",
                                                           reason="No complete executed prospective pair; evidence not accessed"))
        return dict(primary=main, secondary=secondary, t60=self.t60_rows, mird=mird,
                    mird_conditions=self.condition_rows, mird_clusters=self.cluster_rows,
                    paired_bootstrap=self.bootstrap_rows, smoke=smoke_rows, smoke_bootstrap=self.smoke_bootstrap,
                    target_auxiliary=target_auxiliary, diagnostic_summary=self.diagnostic_summary(),
                    interpretations=interpret(main, mird))


def interpret(main, mird):
    def ready(rows):
        return bool(rows) and all(r["eligible"] for r in rows)

    def significant(r, threshold):
        return r["p_holm_global"] < .05 and r["mean_diff"] >= threshold and r["ci_lo"] > 0

    out = []
    c1 = [r for r in main if r["comparison"] == "C1" and r["kind"] == "difference"]
    eq = [r for r in main if r["kind"] == "tost"]
    if not ready(c1 + eq):
        text = "待定：C1数据或同头/初始化资格尚未齐全"
    elif all(significant(r, 1.) for r in c1):
        text = "六格观测改善至少1 dB，支持匹配实现下的集合贡献；不证明唯一因果或总体下界"
    elif all(abs(r["mean_diff"]) < .5 and r["p_holm_global"] < .05 for r in eq):
        text = "六格在事前±0.5 dB界内等效；不能主要归为解除凸约束；头贡献另看C4"
    elif any(r["p_holm_global"] < .05 and r["mean_diff"] < 0 and r["ci_hi"] < 0 for r in c1):
        text = "存在A显著更差的预定格；撤回一致优越表述"
    else:
        text = "效应弱、异质或证据未定；保留全部六格，不改变阈值/参考"
    out.append(dict(rule="C1", interpretation=text))
    for comp in ("C2", "C3", "C4"):
        rows = [r for r in main if r["comparison"] == comp]
        text = "待定：数据/资格未齐" if not ready(rows) else (
            "六格均有至少0.5 dB观测改善及全局显著；只支持该比较的限定贡献"
            if all(significant(r, .5) for r in rows) else "未达到六格一致贡献门槛，完整报告异质/负面结果")
        out.append(dict(rule=comp, interpretation=text))
    for comp in ("G1", "G2"):
        rows = [r for r in main if r["comparison"] == comp]
        text = "待定：G2续跑/资格未齐" if not ready(rows) else (
            "G2两个指定场景支持左侧协议；保持训练/验证增强与结构效应的区别"
            if all(significant(r, 0.) for r in rows) else "G2效果异质或未定，保留两场景与训练/验证协议区别")
        out.append(dict(rule=comp, interpretation=text))
    free = [r for r in main if r["family"] == "HF"]
    scores = [r for r in main if r["comparison"] == "C5"]
    if not ready(free + scores):
        out.append(dict(rule="F_tradeoff", interpretation="待定：不得把未执行的D/TD/退化诊断当作未检出损害"))
    else:
        harms = []
        for r in free:
            threshold = {"F_D": .01, "F_TD": -1., "F_R": -.5}[r["comparison"]]
            positive = r["comparison"] == "F_D"
            harm = (r["mean_diff"] >= threshold and r["ci_lo"] > 0) if positive else (r["mean_diff"] <= threshold and r["ci_hi"] < 0)
            if harm and r["p_holm_global"] < .05:
                harms.append(r["slot"])
        text = ("存在预定工程代价；收益/标称响应/目标保真/退化分开逐格报告" if harms else
                ("六格评分更高，未检出预定工程代价；不证明安全、等效或约束无价值" if all(significant(r, .5) for r in scores)
                 else "无一致评分收益；不能只选失真例子证明约束普遍必要"))
        out.append(dict(rule="F_tradeoff", interpretation=text, harmful_slots=harms))
    for r in mird:
        if r["metric"] != PRIMARY:
            continue
        supported = (r["data_status"] == "complete" and r["qualification_status"] == "pass" and
                     r["mean_diff"] > 0 and r["positive_conditions"] >= 5 and r["condition_diff_min"] > -.5)
        text = ("多数固定已测条件支持；不是跨房间显著性结论" if supported else
                ("条件异质/未定，不能删除MIRD" if r["data_status"] == "complete" else "待定：未执行或缺数据"))
        out.append(dict(rule=r["comparison"] + ":" + r["split"], interpretation=text))
    return out
