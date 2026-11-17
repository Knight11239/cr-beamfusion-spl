# What Relaxing Convexity Buys and Costs in Neural Fusion of Distortionless Beamformers

Code, frozen checkpoints, per-utterance evaluation results and analysis ledgers for the IEEE Signal Processing Letters submission by Lingnan Liu (manuscript under review; citation to be added). arXiv link: to be added (no preprint has been posted yet).

The letter compares convex and complex affine fusion of a fixed differential-beamformer dictionary, written as generalized sidelobe canceller (GSC) parameterizations. In simulation (8-microphone array), affine fusion improves the direct-path ΔSI-SDR by 1.3–2.2 dB over a convex model with an architecturally matched output layer (pre-registered comparison); on measured impulse responses of one 4-microphone array (MIRD), the convex model is better in all six reverberation–subarray conditions for all three training seeds. The supplementary material is `paper/spl/supplement.pdf`; the paper PDF itself is not included in this repository.

## Layout

| Path | Contents |
| --- | --- |
| `src/` | Array geometry, dictionary design, simulation, dataset, fusion models (GTCRN backbone adapted from Rong et al., MIT), WNG projection, training, evaluation |
| `configs/` | Data configurations (`data.yaml`: 8-mic simulation; `data_g2.yaml`: 4-mic simulation; `data_mird.yaml`: measured MIRD responses) and model configurations |
| `scripts/` | Data preparation, evaluation, analysis, number check (`spl_numbers.py`), supplement tables (`spl_supplement.py`), CPU smoke test |
| `research/` | Pre-registered analysis plan (`analysis_plan_controls.md`), analysis code and frozen ledger (`control_analysis/runs/completed-001/`) |
| `results/kaggle/` | Per-utterance results used in the paper (gzipped CSV, reduced to the columns the scripts read); directory names are the original job names |
| `results/checkpoints/` | 18 frozen checkpoints used in the paper (about 10 MB) with `MANIFEST.json` (SHA-256) |
| `results/` (other) | Selection records, MIRD checksums, three-seed tables, pre-registration records |
| `paper/` | Supplementary material (`spl/supplement.pdf`, its source `spl/supplement.tex` and generated tables `spl/supp/`); the letter's LaTeX source `spl/main.tex`, used by the number check (no paper PDF) |
| `EVIDENCE_INDEX.md` | Every number in the letter → inputs and script |
| `PROVENANCE.md` | Freezing commit, timestamps and hashes of the pre-registered analysis plan; control-model training records |

## Quick verification (CPU, no data download)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python -m pytest -q tests research/control_analysis      # unit tests (dictionary, projection, models, analysis code)
python scripts/spl_numbers.py                             # recompute every number in paper/spl/main.tex from the saved results
python scripts/spl_supplement.py                          # regenerate supplementary tables S1–S6 (paper/spl/supp/*.tex)
python scripts/cpu_smoke.py                               # load all checkpoints, check hashes and the nominal distortionless response
CRBF_DATA_ROOT=$PWD/local_data python scripts/stage2_dictionary.py --config configs/data_g2.yaml --out local_out/dictionary_g2
```

`spl_numbers.py` prints `54 records, 0 missing` when every number in the letter matches the saved results. To rebuild the supplement PDF: `cd paper/spl && latexmk -pdf supplement.tex` (needs a TeX installation).

## Full reproduction (requires data downloads; training requires a GPU)

| Step | Needs | Notes |
| --- | --- | --- |
| Speech preparation and simulated scenes/RIRs | LibriSpeech train-clean-100, dev-clean, test-clean ([OpenSLR SLR12](https://www.openslr.org/12/), CC BY 4.0) | `scripts/prepare_speech.py`, `scripts/make_scenes.py`, `scripts/generate_rirs.py`, `scripts/make_g2_meta.py`; set `CRBF_DATA_ROOT` and `CRBF_LIBRISPEECH_ROOT` |
| Measured test sets | MIRD 3-3-3-8-3-3-3 cm responses ([provider page](https://sharongannot.group/resources/downloads/), ODC-By v1.0) | `scripts/make_mird_data.py --mird /path/to/MIRD --config configs/data_mird.yaml --verify-sha results/mird/dataset_v2_files.json` |
| Training | GPU (about 10 h per model on one T4 for 50 epochs) | `python -m src.train --data-config ... --model-config ...`; settings per model are stored in each checkpoint (`model_config`) |
| Evaluation | CPU is sufficient (about 1–2 h per 600-mixture test set and 4–5 models) | `scripts/eval_sets.py`; band diagnostics `scripts/explore_band_diag.py`; inference-time floors via `ckpt:NAME=PATH:post_w0=-5` |

The original runs used a private cloud platform; job definitions are not included. Absolute paths and platform identifiers were removed from this package.

## Re-running the pre-registered analysis (raw per-utterance results)

The column-reduced, gzipped tables under `results/kaggle/` reproduce every number in the letter and the supplement, but
the pre-registered analysis code checks the SHA-256 of the full, original per-utterance CSVs, run records and job
definitions. These are distributed separately as a release asset (not stored in git):

| File | Size | SHA-256 |
| --- | --- | --- |
| `cr-beamfusion-spl-per-utterance-v1.tar.gz` | 12.1 MB (12,140,542 bytes; 104 entries: 48 CSV, 44 JSON, 9 YAML, `pinned_code/src/evaluate.py`, `SHA256SUMS.txt`, `README_ARCHIVE.md`) | `2af26dabce0bdac519a73ae1791fe1e8e9764a86861563ac94889258f1aeff0a` |

Download location: to be attached to a GitHub Release of this repository (not yet available). Then, from the
repository root:

```bash
echo "2af26dabce0bdac519a73ae1791fe1e8e9764a86861563ac94889258f1aeff0a  cr-beamfusion-spl-per-utterance-v1.tar.gz" | sha256sum -c
tar -xzf cr-beamfusion-spl-per-utterance-v1.tar.gz && sha256sum -c SHA256SUMS.txt
cp src/evaluate.py /tmp/evaluate.current.py && cp pinned_code/src/evaluate.py src/evaluate.py   # version pinned by the ledger
python -m research.control_analysis.run --manifest research/control_analysis/completed_inputs.json \
    --allow-new-results --out research/control_analysis/runs/rerun-001
cp /tmp/evaluate.current.py src/evaluate.py
python -m pytest -q research/control_analysis       # the hash-dependent tests now run as well
```

The files in `runs/rerun-001/` are then identical to the frozen ledger `runs/completed-001/`. The current
`src/evaluate.py` only adds the inference-time projection options and checkpoint hashing used later; without the swap,
all statistics are unchanged but four eligibility checks report a hash mismatch and the primary tests are marked
"not decidable". The archive's run records and job definitions contain the original cloud-platform account name and
container paths; they contain no audio.

## Data, licenses and attribution

- Code: MIT (see `LICENSE`, © 2026 Lingnan Liu) for the original code. `src/models/gtcrn.py` is adapted from [GTCRN](https://github.com/Xiaobin-Rong/gtcrn) (MIT, © 2024 Rong Xiaobin); the IEEEtran class files are under LPPL 1.3. See `THIRD_PARTY_NOTICES.md` and `licenses/`.
- LibriSpeech (CC BY 4.0) and MIRD (ODC-By v1.0) are **not** redistributed; download them from the providers. Attribution and the modifications we apply are described in `DATA_SOURCES.md`. Please cite E. Hadad, F. Heese, P. Vary, S. Gannot, “Multichannel audio database in various acoustic environments,” IWAENC 2014, and V. Panayotov et al., “LibriSpeech: An ASR corpus based on public domain audio books,” ICASSP 2015.
- The per-utterance result tables contain only scores and identifiers, no audio.

## Status of the analyses

Comparisons C1–C5, G1–G2 and M1–M3 follow the analysis plan in `research/analysis_plan_controls.md`, frozen before the control models were trained (`results/controls/eligibility.md`; commits, times and hashes in [`PROVENANCE.md`](PROVENANCE.md)). Additional seeds and all inference-time WNG floors are post hoc and exploratory (`results/explore/PLAN.md`, `PLAN2.md`). All measured results come from one room and one array.
