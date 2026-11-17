"""SPL 稿（paper/spl/main.tex）全文数字核对（用户 2026-11-17，依据 Codex PR #14 的要求）。

  python scripts/spl_numbers.py            # 写 results/paper/spl_numbers.csv，打印不一致

每个经验数字都由原始逐样本 CSV（或冻结账本 primary_56.csv 中由原始数据算出的未舍入值）**一次舍入**得到，
不经过任何三位小数的中间表。对每个数字生成稿件中应出现的 LaTeX 片段，检查该片段是否逐字出现在 main.tex 中。
另外提取正文中所有数字字面量，列出未被任何记录覆盖的数字（应只剩设计常数、引用编号、协议参数等），便于人工复核。
MIRD：每个 T60–子阵条件 100 条，六条件等权均值等于 600 条的总体均值；配对差同样。
"""
import csv
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from explore2_stats import mird_meta  # noqa: E402

K = ROOT / "results" / "kaggle"
TEX = ROOT / "paper" / "spl" / "main.tex"
REC = []


def load(job, split, method, n=None):
    fs = sorted(glob.glob(str(K / job / "eval" / "*" / f"{split}_samples.csv*")))
    assert fs, (job, split)
    d = pd.concat([pd.read_csv(f) for f in fs])
    d = d[d.method == method].drop_duplicates("id").set_index("id")
    assert len(d), (job, split, method)
    if n:
        assert len(d) == n, (job, split, method, len(d))
    return d


def sp(split, k):
    return split if k == 0 else f"{split}_kappa{k:g}"


def f(v, d, sign=False, lead=True):
    s = f"{v:+.{d}f}" if sign else f"{v:.{d}f}"
    if not lead:
        s = s.replace("0.", ".", 1)
    if s in ("-0." + "0" * d, "+0." + "0" * d) and not sign:
        s = s.lstrip("-+")
    return s


def b2(v, d):  # 斜杠后的第二个带符号值：/{+}0.75
    t = f(v, d, True)
    return "{" + t[0] + "}" + t[1:]


def m(x):  # 数学模式负号
    return f"${x}$" if x.startswith("-") else x


def add(key, raw, snippet):
    REC.append({"key": key, "raw": raw if not isinstance(raw, (list, tuple)) else json.dumps([float(r) for r in raw]),
                "snippet": snippet})


def pair(a, b, col):
    ids = a.index.intersection(b.index)
    assert len(ids) == len(a) == len(b)
    return float((a.loc[ids, col] - b.loc[ids, col]).mean())


# ------------------------------------------------------------------ Table I（8 麦克风）
T1 = [("ACC", lambda s, k: load("stage3-eval-direct", "test_" + s, "ACC(mu=0.03)", 600) if k == 0 else
        load("paper-main-" + s[-2:] if k == 1 else "b2-afdma-8mic-" + s[-2:], sp(s, k), "ACC(mu=0.03)", 600)),
      ("AF-DMA", lambda s, k: load("b2-afdma-8mic-" + s[-2:], sp(s, k), "AF-DMA(5)", 600)),
      ("DMA-GSC-MPDR", lambda s, k: load("b2-mpdr-mm-8mic-" + s[-2:], sp(s, k), "DMA-GSC-MPDR-mmval(lam=1,delta=0.0001,W0=none)", 600)),
      ("Mask-MVDR (non-causal)", lambda s, k: load("stage4-mvdr-mm", sp(s, k), "MaskMVDR-direct(load=0.01)", 600) if k < 2
       else load("stage4-mvdr-mmval", sp(s, k), "MaskMVDR-direct-mmval(load=0.01)", 600)),
      ("BeamFusion-mm (B)", lambda s, k: load("b5-eval-" + s[-2:], sp(s, k), "BeamFusion-mm", 600)),
      ("Convex, matched layer (C)", lambda s, k: load("ctrl-eval-cr-" + s[-2:], sp(s, k), "convex_samehead_mm", 600)),
      ("Real affine (R)", lambda s, k: load("ctrl-eval-cr-" + s[-2:], sp(s, k), "affine_real_mm", 600)),
      ("Complex affine (A)", lambda s, k: load("stage4-b3eval-" + ("a" if s == "test_s1" else "b"), sp(s, k), "AffineMM", 600)),
      ("Unconstrained (F)$^\\dagger$", lambda s, k: load("ctrl-eval-f-" + s[-2:], sp(s, k), "free_complex_mm", 600))]
D8 = {}
for name, fn in T1:
    cells = []
    for s in ("test_s1", "test_s2"):
        for k in (0, 1, 2):
            D8[(name, s, k)] = fn(s, k)
    for s in ("test_s1", "test_s2"):
        for k in (0, 1, 2):
            cells.append(f(D8[(name, s, k)].d_si_sdr_dp.mean(), 2))
    st = [f(D8[(name, s, 0)].stoi_dp_out.mean(), 3, lead=False) for s in ("test_s1", "test_s2")]
    if name == "Complex affine (A)":
        row = name + " & " + " & ".join(f"\\textbf{{{c}}}" for c in cells) + f" & \\textbf{{{st[0]}}}/\\textbf{{{st[1]}}} \\\\"
    else:
        row = name + " & " + " & ".join(cells) + f" & {st[0]}/{st[1]} \\\\"
    add(f"T1 {name}", [D8[(name, s, k)].d_si_sdr_dp.mean() for s in ("test_s1", "test_s2") for k in (0, 1, 2)], row)

A8, B8, C8, R8, F8 = "Complex affine (A)", "BeamFusion-mm (B)", "Convex, matched layer (C)", "Real affine (R)", "Unconstrained (F)$^\\dagger$"
cells = [(s, k) for s in ("test_s1", "test_s2") for k in (0, 1, 2)]


def diffs(l, r, col="d_si_sdr_dp"):
    return [pair(D8[(l, s, k)], D8[(r, s, k)], col) for s, k in cells]


c1 = diffs(A8, C8)
add("C1 S1", c1[:3], "$" + ",".join(f(v, 2, True) for v in c1[:3]) + "$\\,dB on S1")
add("C1 S2", c1[3:], "$" + ",".join(f(v, 2, True) for v in c1[3:]) + "$\\,dB on S2")
lo = []
for s, k in cells:
    a, c = D8[(A8, s, k)], D8[(C8, s, k)]
    d = a.d_si_sdr_dp - c.loc[a.index, "d_si_sdr_dp"]
    lo.append(d.mean() - 1.96 * d.std(ddof=1) / np.sqrt(len(d)))
add("C1 min lower bound (exceeds)", min(lo), "exceeds 1.15\\,dB" if 1.15 < min(lo) < 1.16 else f"exceeds {f(min(lo), 2)}")
add("C1 range abstract/conclusion", (min(c1), max(c1)), f"by {f(min(c1), 1)}--{f(max(c1), 1)}\\,dB")
c4 = diffs(C8, B8)
add("C4 range", (min(c4), max(c4)), f"is ${f(max(c4), 2)}$ to ${f(min(c4), 2)}$\\,dB")
ab0 = diffs(A8, B8)
add("A-B seed0 S1/S2", ab0, "$" + ",".join(f(v, 2, True) for v in ab0[:3]) + "$ and $" + ",".join(f(v, 2, True) for v in ab0[3:]) + "$\\,dB for seed 0")
ab1 = []
for s, k in cells:
    a = load("b6-seed1-oracle-rho", sp(s, k), "AffineMM-seed1", 600)
    b = load("ctrl-eval-bfs1-" + s[-2:], sp(s, k), "BeamFusion-mm-seed1", 600)
    ab1.append(pair(a, b, "d_si_sdr_dp"))
add("A-B seed1 S1", ab1[:3], "$" + ",".join(f(v, 2, True) for v in ab1[:3]) + "$\\,dB (S1)")
add("A-B seed1 S2", ab1[3:], "$" + ",".join(f(v, 2, True) for v in ab1[3:]) + "$\\,dB (S2)")
rc = diffs(R8, C8)
add("R-C k0", (rc[0], rc[3]), f"(${f(rc[0], 2, True)}/{b2(rc[3], 2)}$\\,dB)")
add("R-C k2", (rc[2], rc[5]), f"(${f(rc[2], 2, True)}/{b2(rc[5], 2)}$\\,dB)")
acx = diffs(A8, C8, "d_si_sdr")
add("A-C x_ref range", (min(acx), max(acx)), f"A is below C by {f(-max(acx), 1)}--{f(-min(acx), 1)}\\,dB")
xr = {s: (load("b5-eval-" + s[-2:], s, "Affine-xref", 600), load("b5-eval-" + s[-2:], s, "BeamFusion-xref", 600)) for s in ("test_s1", "test_s2")}
xref = [pair(*xr[s], "d_si_sdr") for s in ("test_s1", "test_s2")]
xdp = [pair(*xr[s], "d_si_sdr_dp") for s in ("test_s1", "test_s2")]
add("xref-trained on x_ref", xref, f"${f(xref[0], 2, True)}/{b2(xref[1], 2)}$\\,dB")
add("xref-trained on x_dp", xdp, f"by ${f(-xdp[0], 2)}/{f(-xdp[1], 2)}$\\,dB")
c5 = diffs(F8, A8)
add("C5 range", (min(c5), max(c5)), f"(C5: ${f(min(c5), 2, True)}$ to ${f(max(c5), 2, True)}$\\,dB")
# F / A 诊断（六格）
dg = {}
for s in ("test_s1", "test_s2"):
    for k in (0, 1, 2):
        fn = glob.glob(str(K / f"ctrl-diag-{s[-2:]}" / "eval" / "*" / f"{sp(s, k)}_diagnostics.csv*"))[0]
        dg[(s, k)] = pd.read_csv(fn)
Dp = {mm: [dg[c][dg[c].method == mm].nominal_D_p95.mean() for c in cells] for mm in ("free_complex_mm", "AffineMM")}
add("F D_p95 six-cell mean", np.mean(Dp["free_complex_mm"]), f"averages {f(np.mean(Dp['free_complex_mm']), 2)} over the six cells")
ea = np.mean(Dp["AffineMM"])
e = int(np.floor(np.log10(ea)))
add("A D_p95 six-cell mean", ea, f"(A: ${ea / 10 ** e:.1f}\\times10^{{{e}}}$)")
g = [dg[c][dg[c].method == "free_complex_mm"].nominal_gain_amp_mean.mean() for c in cells]
add("F gain range", (min(g), max(g)), f"is {f(min(g), 2)}--{f(max(g), 2)}")
td = [dg[c][dg[c].method == "free_complex_mm"].target_direct_sdr_out.mean() - dg[c][dg[c].method == "AffineMM"].target_direct_sdr_out.mean() for c in cells]
add("F-A TD range", (min(td), max(td)), f"with {f(-max(td), 1)}--{f(-min(td), 1)}\\,dB lower SDR")

# ------------------------------------------------------------------ Table II（G2）
G = {}


def g2(job_sim, job_mird, meth, sim_split=("test_s1", "test_s2")):
    out = {}
    for s in sim_split:
        out[s] = load(job_sim.format(s=s[-2:]), s, meth, 600)
    for s in ("mird_s1", "mird_s2"):
        out[s] = load(job_mird.format(s=s[-2:]), s, meth, 600)
    return out


SEEDJ = {0: ("g2-eval-sim-a", "g2-eval-mird-a", "G2-Affine-mm", "g2-eval-bf-sim", "g2-eval-bf-mird", "G2-BeamFusion-mm"),
         1: ("g2-s1-eval-sim", "g2-s1-eval-mird", "G2-Affine-mm-s1", "g2-s1-eval-sim", "g2-s1-eval-mird", "G2-BeamFusion-mm-s1"),
         2: ("g2-s2-eval-sim", "g2-s2-eval-mird", "G2-Affine-mm-s2", "g2-s2-eval-sim", "g2-s2-eval-mird", "G2-BeamFusion-mm-s2")}
rows2 = [("ACC", "r7-g2sim-{s}", "r7-mird-{s}", "ACC(mu=0.001)"),
         ("AF-DMA", "b2-afdma-g2sim", "b2-afdma-mird", "AF-DMA(4)"),
         ("DMA-GSC-MPDR", "b2-mpdr-mm-g2sim", "b2-mpdr-mm-mird", "DMA-GSC-MPDR-mmval(lam=1,delta=0.001,W0=none)")]
for sd, (sa, ma, na, sb, mb, nb) in SEEDJ.items():
    G[("A", sd)] = g2(sa, ma, na)
    G[("B", sd)] = g2(sb, mb, nb)
for name, js, jm, meth in rows2:
    G[(name, None)] = g2(js, jm, meth)


def row2(label, d):
    v = lambda s, c, dd=2, lead=True: f(d[s][c].mean(), dd, lead=lead)  # noqa: E731
    return (f"{label} & {v('test_s1', 'd_si_sdr_dp')}/{v('test_s2', 'd_si_sdr_dp')} & {v('mird_s1', 'd_si_sdr_dp')}/{v('mird_s2', 'd_si_sdr_dp')} & "
            f"{v('mird_s1', 'stoi_dp_out', 3, False)}/{v('mird_s2', 'stoi_dp_out', 3, False)} & "
            f"{v('mird_s1', 'pesq_dp_out')}/{v('mird_s2', 'pesq_dp_out')} \\\\")


for name, *_ in rows2:
    add(f"T2 {name}", G[(name, None)]["mird_s1"].d_si_sdr_dp.mean(), row2(name, G[(name, None)]))
for who in ("B", "A"):
    for sd in (0, 1, 2):
        add(f"T2 {who} seed {sd}", G[(who, sd)]["mird_s1"].d_si_sdr_dp.mean(), row2(f"{who}, seed {sd}", G[(who, sd)]))

# ------------------------------------------------------------------ III-C 正文与 Table III
META = {s: mird_meta(s) for s in ("mird_s1", "mird_s2")}


def mird_pair(a, b, s, col="d_si_sdr_dp"):
    ids = a.index.intersection(b.index)
    assert len(ids) == 600
    d = (a.loc[ids, col] - b.loc[ids, col]).to_frame("d").join(META[s])
    cm = d.groupby("cond")["d"].mean()
    return float(cm.mean()), int((cm > 0).sum()), cm


g1 = {sd: [pair(G[("A", sd)][s], G[("B", sd)][s], "d_si_sdr_dp") for s in ("test_s1", "test_s2")] for sd in (0, 1, 2)}
add("G1 seed0", g1[0], f"is ${f(g1[0][0], 2, True)}/{b2(g1[0][1], 2)}$\\,dB (S1/S2)")
add("G1 seeds 1,2", g1[1] + g1[2], f"and ${f(g1[1][0], 2, True)}/{b2(g1[1][1], 2)}$ and ${f(g1[2][0], 2, True)}/{b2(g1[2][1], 2)}$\\,dB for seeds 1 and 2")
mab, allc = {}, []
for sd in (0, 1, 2):
    mab[sd] = []
    for s in ("mird_s1", "mird_s2"):
        mean, pos, cm = mird_pair(G[("A", sd)][s], G[("B", sd)][s], s)
        mab[sd].append((mean, pos))
        allc += list(cm.values)
add("M1 averages (B-A)", [x[0] for sd in mab for x in mab[sd]],
    "by " + ", ".join(f"${f(-mab[sd][0][0], 2)}/{f(-mab[sd][1][0], 2)}$" for sd in (0, 1)) + f", and ${f(-mab[2][0][0], 2)}/{f(-mab[2][1][0], 2)}$\\,dB on average")
add("M1 per-condition range", (min(allc), max(allc)), f"(per condition {f(-max(allc), 2)}--{f(-min(allc), 2)}\\,dB)")
rng = {}
for who in ("A", "B"):
    rng[who] = []
    for s in ("mird_s1", "mird_s2"):
        v = [G[(who, sd)][s].d_si_sdr_dp.mean() for sd in (0, 1, 2)]
        rng[who].append(max(v) - min(v))
add("seed ranges", rng["A"] + rng["B"], f"ranges over ${f(rng['A'][0], 2)}/{f(rng['A'][1], 2)}$\\,dB and B's over ${f(rng['B'][0], 2)}/{f(rng['B'][1], 2)}$\\,dB")

# Table III 与下限段落
val = {}
for sd, (an, bn, aj, bj, a5, b5) in {
        0: ("G2-Affine-mm", "G2-BeamFusion-mm", "explore-w0-val", "explore2-e1-val", "G2-Affine-mm(projW0=-5)", "G2-BeamFusion-mm(postW0=-5)"),
        1: ("G2-Affine-mm-s1", "G2-BeamFusion-mm-s1", "g2-s1-eval-val", "g2-s1-eval-val", "G2-Affine-mm-s1(postW0=-5)", "G2-BeamFusion-mm-s1(postW0=-5)"),
        2: ("G2-Affine-mm-s2", "G2-BeamFusion-mm-s2", "g2-s2-eval-val", "g2-s2-eval-val", "G2-Affine-mm-s2(postW0=-5)", "G2-BeamFusion-mm-s2(postW0=-5)")}.items():
    val[(sd, "none")] = (load(aj, "val", an, 500).d_si_sdr_dp.mean(), load(bj, "val", bn, 500).d_si_sdr_dp.mean())
    val[(sd, "-5")] = (load(aj, "val", a5, 500).d_si_sdr_dp.mean(), load(bj, "val", b5, 500).d_si_sdr_dp.mean())
M5 = {0: ("explore-w0-mird", "G2-Affine-mm(projW0=-5)", "explore2-e1-mird", "G2-BeamFusion-mm(postW0=-5)"),
      1: ("g2-s1-eval-mird", "G2-Affine-mm-s1(postW0=-5)", "g2-s1-eval-mird", "G2-BeamFusion-mm-s1(postW0=-5)"),
      2: ("g2-s2-eval-mird", "G2-Affine-mm-s2(postW0=-5)", "g2-s2-eval-mird", "G2-BeamFusion-mm-s2(postW0=-5)")}
P5 = {}
for sd, (ja, na, jb, nb) in M5.items():
    P5[sd] = {s: (load(ja, s, na, 600), load(jb, s, nb, 600)) for s in ("mird_s1", "mird_s2")}


def m2(x):
    return f"${x}$"


for sd in (0, 1, 2):
    for fl, lab in (("none", "none"), ("-5", "$-5$\\,dB")):
        if fl == "none":
            r = [mird_pair(G[("A", sd)][s], G[("B", sd)][s], s) for s in ("mird_s1", "mird_s2")]
        else:
            r = [mird_pair(*P5[sd][s], s) for s in ("mird_s1", "mird_s2")]
        va, vb = val[(sd, fl)]
        row = (f"{sd} & {lab} & {f(va, 2)} / {f(vb, 2)} & {m2(f(r[0][0], 2, True))} / {m2(f(r[1][0], 2, True))} & "
               f"{r[0][1]} / {r[1][1]} \\\\")
        add(f"T3 seed {sd} {fl}", [va, vb, r[0][0], r[1][0]], row)
# −15 dB 对 MIRD 的影响（种子 0，A）
d15 = []
for s in ("mird_s1", "mird_s2"):
    a15 = load("explore-w0-mird", s, "G2-Affine-mm(projW0=-15)", 600)
    d15.append(pair(a15, G[("A", 0)][s], "d_si_sdr_dp"))
add("-15 dB effect < 0.01", max(abs(x) for x in d15), "by less than 0.01\\,dB" if max(abs(x) for x in d15) < 0.01 else "CHECK -15")
imp = {}
for who, idx in (("A", 0), ("B", 1)):
    imp[who] = [pair(P5[0][s][idx], G[(who, 0)][s], "d_si_sdr_dp") for s in ("mird_s1", "mird_s2")]
add("seed0 floor improvements", imp["A"] + imp["B"],
    f"A ${f(imp['A'][0], 2, True)}/{b2(imp['A'][1], 2)}$\\,dB, B ${f(imp['B'][0], 2, True)}/{b2(imp['B'][1], 2)}$\\,dB")
sec = {}
for col, dd in (("d_si_sdr", 2), ("stoi_dp_out", 3), ("pesq_dp_out", 3)):
    sec[col] = [mird_pair(*P5[0][s], s, col)[0] for s in ("mird_s1", "mird_s2")]
add("seed0 -5 secondary", sum(sec.values(), []),
    f"in $x_{{\\mathrm{{ref}}}}$ (${f(sec['d_si_sdr'][0], 2)}/{b2(sec['d_si_sdr'][1], 2)}$\\,dB), STOI (${f(sec['stoi_dp_out'][0], 3)}/{b2(sec['stoi_dp_out'][1], 3)}$), "
    f"and PESQ (${f(sec['pesq_dp_out'][0], 3)}/{b2(sec['pesq_dp_out'][1], 3)}$)")
e2 = {}
for tag, name in (("full", "G2-Affine-mm(postW0=-5)"), ("low", "G2-Affine-mm(postW0=-5@0-1000Hz)"), ("high", "G2-Affine-mm(postW0=-5@5717-8001Hz)")):
    e2[tag] = [pair(load("explore2-e2-mird", s, name, 600), G[("A", 0)][s], "d_si_sdr_dp") for s in ("mird_s1", "mird_s2")]
add("band intervention low", e2["low"], f"reproduces its improvement (${f(e2['low'][0], 2, True)}/{b2(e2['low'][1], 2)}$\\,dB)")
add("band intervention high", e2["high"], f"yields ${f(max(e2['high']), 3, True)}$\\,dB" if abs(e2["high"][0] - e2["high"][1]) < 5e-4 else "CHECK high")
# 频带一致性（三个种子）
bc = (ROOT / "results" / "g2" / "seeds" / "band_consistency.md").read_text()
add("band 2-4 kHz 12/12 all seeds", bc.count("| 12/12 | 8/12 | 0/12 |") + bc.count("| 12/12 | 2/12 | 0/12 |"),
    "at 2--4\\,kHz in all twelve scenario--condition cells for each of the three seeds")

# R3 等效 kappa（近场模型，160 / 360 ms，results/mird/r2r3_summary.json）
r3 = json.loads((ROOT / "results" / "mird" / "r2r3_summary.json").read_text())["r3"]["nf"]
k1, k2 = r3["kappa_eff_median_500_4000"]["median"], r3["kappa_eff_median_4000_7500"]["median"]
add("R3 kappa", (k1, k2), f"about {f(k1, 1)} at 0.5--4\\,kHz and {f(k2, 1)} at 4--7.5\\,kHz")

# ------------------------------------------------------------------ 核对
tex = TEX.read_text()
body = tex[tex.index("\\begin{document}"):tex.index("\\bibliography")]
bad = 0
for r in REC:
    r["found"] = r["snippet"] in tex
    bad += not r["found"]
out = ROOT / "results" / "paper" / "spl_numbers.csv"
with open(out, "w", newline="") as fo:
    w = csv.DictWriter(fo, fieldnames=["key", "found", "snippet", "raw"])
    w.writeheader()
    w.writerows(REC)
for r in REC:
    if not r["found"]:
        print("MISSING:", r["key"], "->", r["snippet"])
# 未覆盖的数字字面量（去掉注释行）
lines = [ln for ln in body.splitlines() if not ln.lstrip().startswith("%")]
txt = "\n".join(re.sub(r"(?<!\\)%.*", "", ln) for ln in lines)
nums = re.findall(r"[-+]?\d*\.\d+|\d+", txt)
covered = {x.lstrip("+-") for r in REC for x in re.findall(r"[-+]?\d*\.\d+|\d+", r["snippet"])}
DESIGN = {"0.95": "C 的初始 alpha_0（设计常数）", "0.01": "输出层权重缩放 / -15 dB 影响阈值（已作为记录）",
          "0.05": "显著性水平", "-7.5": "频带 4--7.5 kHz 的上限（正则把 -- 读成负号）",
          "0.1": "位置误差标准差 0.1 kappa cm（协议）", "5.7": "空间混叠频率 c/(2 delta)", "0.5": "频带边界 / TOST 界",
          "2.4": "MIRD 房间高度", "0.2": "T60 范围", "0.8": "T60 范围", "0.3": "测试 T60", "0.7": "测试 T60", "7.5": "R3 频带上限"}
unc = sorted({n for n in nums if "." in n and n.lstrip("+-") not in covered})
print("design/protocol constants among uncovered:", {u: DESIGN.get(u.lstrip("+-"), "??") for u in unc})
print(f"{len(REC)} records, {bad} missing. Uncovered decimal literals (manual check):", unc)
