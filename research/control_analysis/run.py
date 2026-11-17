"""One-command prospective analysis; default is a pinned legacy-only smoke run."""
import argparse
import csv
import hashlib
import json
import platform
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import scipy

from .inputs import Inputs
from .pipeline import Analysis
from .statistics import BOOT_REPETITIONS, BOOT_SEED

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
BASE_COLUMNS = ["slot", "comparison", "family", "dataset", "split", "kappa", "seed", "group",
                "left", "right", "metric", "kind", "execution_status", "data_status",
                "qualification_status", "eligible", "reason", "available_left", "available_right",
                "paired_ids", "paired_id_sha256", "n", "nonzero_n", "mean_left", "mean_right", "mean_diff",
                "sd_diff", "ci_lo", "ci_hi", "median_diff", "r_rb", "cohen_dz", "p_raw", "p_for_holm",
                "correction_placeholder", "p_holm_global", "p_holm_global_m", "p_holm_family", "p_holm_family_m",
                "tost_p_lower", "tost_p_upper", "tost_margin", "tost_ci90_lo", "tost_ci90_hi",
                "test", "ci_method", "left_sources", "right_sources", "failed_ids"]


def write_csv(path, records, base=()):
    columns = list(dict.fromkeys([*base, *(key for r in records for key in r)]))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for r in records:
            writer.writerow({k: json.dumps(v, ensure_ascii=False, sort_keys=True) if isinstance(v, (dict, list)) else v
                             for k, v in r.items()})


def run(manifest_path, output, *, allow_new=False, smoke=True):
    manifest_path, output = Path(manifest_path).resolve(), Path(output).resolve()
    if not output.is_relative_to(ROOT / "research"):
        raise ValueError("Outputs must remain under research/; never overwrite paper/original results")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new empty output directory, not an existing result ledger")
    config = json.loads(manifest_path.read_text())
    if config.get("schema_version") != 1:
        raise ValueError("Unsupported manifest schema")
    inputs = Inputs(ROOT, config, allow_new=allow_new)
    plan = inputs.checked(config["plan"], "frozen pre-analysis plan")
    inputs.load()
    analysis = Analysis(inputs)
    result = analysis.run(smoke=smoke)
    output.mkdir(parents=True, exist_ok=True)
    filenames = {"primary": "primary_56.csv", "secondary": "secondary_102.csv", "t60": "t60_descriptive.csv",
                 "mird": "mird_condition_summary.csv", "mird_conditions": "mird_six_conditions.csv",
                 "mird_clusters": "mird_trajectory_bootstrap.csv", "paired_bootstrap": "paired_t60_bootstrap.csv",
                 "smoke": "legacy_smoke.csv", "smoke_bootstrap": "legacy_t60_bootstrap.csv",
                 "target_auxiliary": "f_target_descriptive.csv", "diagnostic_summary": "diagnostics_summary.csv"}
    for key, filename in filenames.items():
        write_csv(output / filename, result[key], BASE_COLUMNS if key in ("primary", "secondary", "t60", "mird", "smoke") else ["slot"])
    qualifications = {json.dumps(r, sort_keys=True, ensure_ascii=False): r for r in inputs.qualifications}
    write_csv(output / "qualifications.csv", list(qualifications.values()), ["owner", "requirement", "status", "reason", "evidence"])
    audit = {json.dumps(r, sort_keys=True, ensure_ascii=False): r for r in inputs.audit}
    (output / "source_audit.json").write_text(json.dumps(list(audit.values()), indent=2, ensure_ascii=False))
    pair_audit = {json.dumps(r, sort_keys=True, ensure_ascii=False): r for r in analysis.pair_audit}
    (output / "pair_audit.json").write_text(json.dumps(list(pair_audit.values()), indent=2, ensure_ascii=False))
    diagnostic_rows = []
    for (model, dataset, split, kappa, identifier), row in inputs.data.items():
        if row.get("_diagnostic_source"):
            fields = {k: v for k, v in row.items() if k.startswith(("nominal_", "wng_", "near_zero_", "target_", "sum_c_", "tf_", "frozen_", "effective_"))}
            diagnostic_rows.append(dict(model=model, dataset=dataset, split=split, kappa=kappa, id=identifier,
                                        source=row["_diagnostic_source"], diagnostic_version=row["diagnostic_version"], **fields))
    write_csv(output / "diagnostics_by_id.csv", diagnostic_rows,
              ["model", "dataset", "split", "kappa", "id", "source", "nominal_D_p95", "nominal_D_rms", "nominal_D_max", "nominal_D_dc_max", "target_direct_sdr_out"])
    (output / "interpretation.json").write_text(json.dumps(result["interpretations"], indent=2, ensure_ascii=False))
    summary = dict(schema_version=1, allow_new_results=allow_new, mode="explicit completed inputs" if allow_new else "legacy-only smoke",
                   plan_sha256=hashlib.sha256(plan).hexdigest(), manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                   primary_slots=len(result["primary"]), primary_family_sizes=dict(Counter(r["family"] for r in result["primary"])),
                   primary_statuses=dict(Counter(r["execution_status"] for r in result["primary"])),
                   primary_eligible=sum(r["eligible"] for r in result["primary"]), secondary_slots=len(result["secondary"]),
                   enabled_models=sorted(inputs.enabled), prospective_models_not_read=[m for m in ("C", "R", "F", "GBmm", "GB") if m not in inputs.enabled],
                   model_input_errors=inputs.issues, legacy_smoke_rows=len(result["smoke"]),
                   bootstrap_repetitions=BOOT_REPETITIONS, bootstrap_seed=BOOT_SEED,
                   scipy=scipy.__version__, numpy=np.__version__, python=platform.python_version(),
                   code_sha256={name: hashlib.sha256((HERE / name).read_bytes()).hexdigest()
                                for name in ("inputs.py", "pipeline.py", "statistics.py", "diagnostics.py", "run.py")},
                   scope="No training or model inference. Main slots never filled by legacy proxies. Raw p is blank for unexecuted tests; p=1 is correction-only.")
    (output / "run_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    report = ["# Controls analysis run", "", "模式：" + summary["mode"], "",
              "主账本固定56槽，子族36/16/4；全局Holm的分母始终为56。未执行/不合格槽用1占位校正，原始和展示校正p留空；不把1伪装成实测p。", "",
              "| 状态 | 主槽数 |", "| --- | ---: |"]
    report += [f"| {status} | {n} |" for status, n in summary["primary_statuses"].items()]
    report += ["", "已启用的既有/显式输入：" + ", ".join(summary["enabled_models"]),
               "", "未读取的新模型槽：" + ", ".join(summary["prospective_models_not_read"]),
               "", "`legacy_smoke.csv`中的A−B及既有GL−GA只检查格式/配对，不进入预定主假设或新对照结论。",
               "MIRD条件内轨迹bootstrap使用固定六条件分层；S1单轨迹条件标不可估计，不生成零宽区间。六条件精确双侧p最低0.03125，六项Holm最低0.1875，不能改用600个独立房间。",
               "资格通过依赖被哈希固定的外部证据和人工声明；脚本检查记录完整性，不能仅凭CSV证明训练事实。", "", "## 事前规则输出", ""]
    report += ["- " + r["rule"] + "：" + r["interpretation"] for r in result["interpretations"]]
    if inputs.issues:
        report += ["", "## 输入错误", "", json.dumps(inputs.issues, indent=2, ensure_ascii=False)]
    (output / "REPORT.md").write_text("\n".join(report) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=HERE / "legacy_inputs.json")
    parser.add_argument("--out", type=Path, default=HERE / "runs" / "legacy-smoke")
    parser.add_argument("--allow-new-results", action="store_true", help="Future use only: explicitly enable evaluated prospective slots")
    parser.add_argument("--no-smoke", action="store_true", help="Skip legacy-only format checks; fixed registered slots unchanged")
    args = parser.parse_args()
    try:
        summary = run(args.manifest, args.out, allow_new=args.allow_new_results, smoke=not args.no_smoke)
    except (ValueError, OSError, KeyError) as error:
        parser.exit(2, str(error) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if summary["model_input_errors"]:
        sys.exit(2)


if __name__ == "__main__":
    main()
