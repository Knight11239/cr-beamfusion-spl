# Data sources, attribution and modifications

No speech audio and no room impulse responses are included in this repository. Download them from the providers.

## LibriSpeech

- Source: OpenSLR SLR12, https://www.openslr.org/12/ — license **CC BY 4.0** (https://creativecommons.org/licenses/by/4.0/).
- Subsets: train-clean-100 (training), dev-clean (validation), test-clean (test), as set in `configs/data.yaml`.
- Citation: V. Panayotov, G. Chen, D. Povey, S. Khudanpur, "LibriSpeech: An ASR corpus based on public domain audio books,"
  ICASSP 2015, pp. 5206–5210, doi:10.1109/ICASSP.2015.7178964.
- Modifications (performed by `scripts/prepare_speech.py`, `src/speech.py`, `src/dataset.py`): silence removal, 10-s
  segmentation, normalization, and mixing with simulated or measured impulse responses, interferers and white noise.
- The released checkpoints were trained on mixtures built from LibriSpeech; the per-utterance result tables contain only
  scores and identifiers.

## MIRD (Multichannel Impulse Response Database)

- Source: https://sharongannot.group/resources/downloads/ — license **Open Data Commons Attribution License v1.0
  (ODC-By)**, https://opendatacommons.org/licenses/by/1-0/.
- Attribution: contains information from the Multichannel Impulse Response Database made available by E. Hadad,
  F. Heese, P. Vary and S. Gannot under ODC-By v1.0.
- Citation: E. Hadad, F. Heese, P. Vary, S. Gannot, "Multichannel audio database in various acoustic environments,"
  IWAENC 2014, pp. 313–317, doi:10.1109/IWAENC.2014.6954309.
- Used: the 3-3-3-8-3-3-3 cm array, reverberation times 0.160, 0.360 and 0.610 s, two four-microphone subarrays
  (microphones 1–4 and 5–8). These are six conditions of one measured room and array, not six independent rooms.
- Integrity: `results/mird/dataset_v2_files.json` lists the SHA-256 of the input files used; `scripts/make_mird_data.py --verify-sha results/mird/dataset_v2_files.json`
  checks them before processing (an earlier mirror of the 610-ms set was wrong and must not be used).
- Modifications (`scripts/make_mird_data.py`, `scripts/mird_r2r3.py`): removal of the 624-sample acquisition delay,
  resampling from 48 to 16 kHz, direct-path and early-reflection reference windows, fractional-delay alignment and
  averaging of the two centre microphones as the reference; the array observations use the full responses.

## Simulated responses

Scene sampling, image-method responses (pyroomacoustics), source movement, array geometry and test-only microphone
position errors are defined by `scripts/make_scenes.py`, `scripts/generate_rirs.py`, `src/simulate.py`,
`configs/data.yaml` and `configs/data_g2.yaml`, with fixed seeds.
