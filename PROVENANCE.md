# Provenance of the pre-registered analysis

This repository is a release snapshot. The full development history (code, configurations, logs and review
records) is kept in a private repository; it can be made available to the editor or reviewers on request. The
commit identifiers below refer to that private repository. All times are UTC.

## Frozen analysis plan

| Item | Value |
| --- | --- |
| File | `research/analysis_plan_controls.md` (analysis plan v1.1) |
| Freezing commit (private repository) | `763419c69a3d316805500ebd89b99bebf6d0763b`, committed 2026-11-11 04:16:39 UTC |
| SHA-256 of the file at that commit | `db29ea7dadd364861a34fd4c6d03495544d82ce60708e27597daf31203d1d6c5` |
| SHA-256 of the file in this repository | `db29ea7dadd364861a34fd4c6d03495544d82ce60708e27597daf31203d1d6c5` |
| Byte-for-byte comparison | identical (`cmp` of the file at the freezing commit and the file in this repository, 2026-11-17) |

The plan was first committed as v1 on 2026-11-11 04:07:55 UTC (`c2695e0266f49062bae907161d66eb739559d209`,
different hash) and corrected to v1.1 nine minutes later at the freezing commit above (text corrections made before
any control training). The same v1.1 file was copied unchanged into the main development branch on 2026-11-13
08:51:17 UTC (`ee3f1f5a932eac0494fc0cc906d7d51413742a69`). The earlier results of the affine and original convex
models (A and B) were already known when the plan was written; the plan concerns the control models below.

## Eligibility record and control-model training

| Event | Time (UTC) | Record |
| --- | --- | --- |
| Eligibility record committed before training (`results/controls/eligibility.md`) | 2026-11-13 08:48:38 | commit `13b6a8802bb70637aec341b29f890a9743fe18cf` |
| Control training jobs submitted | 2026-11-14 00:06:59 | commit `76f90f558bf8a513ed3615b23ada9fe0ce5f2a2e` (code version used by both jobs) |
| Job `gpu-ctrl-s2` started: convex model with matched output layer (C) and real affine model (R) | 2026-11-14 00:07:37 (training start 00:08:01) | cloud job `crbf-gpu-ctrl-s2`; first epoch logged 00:27:57; epoch 50 logged 10:48:20 (C) and 10:48:38 (R) |
| Job `gpu-ctrl-s3` started: unconstrained fusion (F) | 2026-11-14 00:07:17 (training start 00:07:43) | cloud job `crbf-gpu-ctrl-s3`; first epoch logged 00:24:53; epoch 50 logged 10:40:00 |
| Results collected | 2026-11-14 11:05:33 | commit `a16a9ac0d256f972256087db9ac24a5afc4e675d` |

These were the first full training runs of C, R and F. Before them, the eligibility record documents only a CPU pipeline smoke test (C and F trained for one epoch and resumed to a second on a small development subset; outputs not used) and an initial-function check at initialization (`results/controls/initial_function.json`). The checkpoints in `results/checkpoints/gpu-ctrl-s2/` and
`results/checkpoints/gpu-ctrl-s3/` record the code commit (`git_commit` = `76f90f5…`) and are listed with their
SHA-256 in `results/checkpoints/MANIFEST.json`. The analysis ledger computed under the frozen plan is
`research/control_analysis/runs/completed-001/`; it can be reproduced from the raw per-utterance results
(see README, "Re-running the pre-registered analysis").

## Post hoc analyses

Additional training seeds and all inference-time white-noise-gain floors are outside the plan. Their plans were
committed before the corresponding results were computed (`results/explore/PLAN.md`, `results/explore/PLAN2.md`), after the
measured-data results had been seen; they are reported as exploratory.
