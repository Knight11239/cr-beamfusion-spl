# Third-party notices

The original code of this repository is released under the MIT License (`LICENSE`, © 2026 Lingnan Liu).
The following third-party material keeps its own license and attribution; it is **not** claimed as original work.

## GTCRN (adapted code)

`src/models/gtcrn.py` is adapted from GTCRN, https://github.com/Xiaobin-Rong/gtcrn (upstream commit prefix `502ebfab`),
Copyright (c) 2024 Rong Xiaobin, MIT License. The file header records the source and the changes (channel width,
activations, recurrent options, batch normalization). The upstream license text is reproduced verbatim in
`licenses/GTCRN-MIT.txt`.

## IEEEtran LaTeX class and bibliography style

`paper/spl/IEEEtran.cls` (V1.8b, 2015/08/26; Copyright (c) 1993–2000 Gerry Murray, Silvano Balemi, Jon Dixon,
Peter Nüchter and Juergen von Hagen; Copyright (c) 2001–2015 Michael Shell) and `paper/spl/IEEEtran.bst`
(Copyright (c) 2003–2015 Michael Shell) are distributed unmodified under the LaTeX Project Public License,
version 1.3; the text of LPPL 1.3c (the current revision of version 1.3) is in `licenses/LPPL-1.3c.txt`.
SHA-256: cls `da751920a317ed318b7b5cd7fa585a6cc7d28502d457856382e9be24b10a3bd7`,
bst `314f0ece704568faf827011bac498650691b2b5ee06320720830e782416d5a5f` (identical to the CTAN release).

## Python dependencies

PyTorch, NumPy, SciPy, pandas, PyYAML, Matplotlib, SoundFile, pyroomacoustics, pystoi, pesq, mir_eval and pytest are
installed from their distributions (`requirements.txt`) and are not bundled; each keeps its own license. The `pesq`
package wraps the ITU-T P.862 reference implementation, whose terms of use are set by its upstream providers.

## Data

LibriSpeech (CC BY 4.0) and MIRD (ODC-By v1.0) are not redistributed; see `DATA_SOURCES.md`.
