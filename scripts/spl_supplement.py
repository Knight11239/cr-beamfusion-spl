"""SPL 在线补充材料的表格（用户 2026-11-17）：由原始逐样本 CSV / 冻结账本生成 paper/spl/supp/*.tex（一次舍入）。

  python scripts/spl_supplement.py && (cd paper/spl && latexmk -pdf supplement.tex)
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from explore2_stats import mird_meta  # noqa: E402

K = ROOT / "results" / "kaggle"
OUT = ROOT / "paper" / "spl" / "supp"
OUT.mkdir(parents=True, exist_ok=True)
LEDGER = ROOT / "research" / "control_analysis" / "runs" / "completed-001"


def f(v, d, sign=False):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "--"
    s = f"{v:+.{d}f}" if sign else f"{v:.{d}f}"
    return s.replace("-", "$-$") if s.startswith("-") else s


def pv(p):
    if not np.isfinite(p):
        return "--"
    if p >= 0.001:
        return f"{p:.3f}"
    m, e = f"{p:.1e}".split("e")
    return f"${m}\\times10^{{{int(e)}}}$"


def load(job, split, method, n=None):
    fs = sorted(glob.glob(str(K / job / "eval" / "*" / f"{split}_samples.csv*")))
    assert fs, (job, split)
    d = pd.concat([pd.read_csv(x) for x in fs])
    d = d[d.method == method].drop_duplicates("id").set_index("id")
    assert len(d) and (n is None or len(d) == n), (job, split, method, len(d))
    return d


def write(name, lines):
    (OUT / f"{name}.tex").write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------- S1：事前主族 56 项与 MIRD 族
p = pd.read_csv(LEDGER / "primary_56.csv")
L = ["\\begin{longtable}{@{}llllrrrr@{}}", "\\caption{All 56 primary tests (frozen plan; global Holm correction).}\\label{tab:primary}\\\\", "\\toprule",
     "Comp. & Family & Scen. & $\\kappa$ & Kind & Mean diff. & 95\\% CI & Holm-56 $p$ \\\\", "\\midrule\\endhead"]
for _, r in p.iterrows():
    kind = {"difference": "diff.", "tost": "TOST", "degradation": "degr."}.get(r["kind"], r["kind"])
    met = r["metric"].replace("_", "\\_")
    ci = f"[{f(r.ci_lo, 2)}, {f(r.ci_hi, 2)}]" if r["kind"] != "tost" else f"[{f(r.tost_ci90_lo, 2)}, {f(r.tost_ci90_hi, 2)}]$^{{90}}$"
    L.append(f"{r.comparison.replace('_', '\\_')} & {r.family} & {r.split.replace('test_', '').upper()} & {r.kappa:g} & {kind}, \\texttt{{{met}}} & "
             f"{f(r.mean_diff, 3, True) if abs(r.mean_diff) >= 1e-3 else f'{r.mean_diff:.1e}'} & {ci} & {pv(r.p_holm_global)} \\\\")
L += ["\\bottomrule", "\\end{longtable}"]
write("s1_primary", L)
mc = pd.read_csv(LEDGER / "mird_condition_summary.csv")
mc = mc[mc.metric == "d_si_sdr_dp"] if "metric" in mc else mc
L = ["\\begin{tabular}{@{}llrrrr@{}}", "\\toprule", "Comp. & Scen. & Mean (6 cond.) & $t_5$ 95\\% CI & Cond.\\ $>0$ & Holm-6 $p$ \\\\", "\\midrule"]
for _, r in mc.iterrows():
    npos = int(r.positive_conditions)
    L.append(f"{r.comparison} & {r.split.replace('mird_', '').upper()} & {f(r.mean_diff, 2, True)} & [{f(r.ci_lo, 2)}, {f(r.ci_hi, 2)}] & {npos} & {pv(r.p_holm_family)} \\\\")
L += ["\\bottomrule", "\\end{tabular}"]
write("s1_mird", L)

# ---------------------------------------------------------------- S2：分窗 oracle
o = pd.read_csv(ROOT / "results" / "oracle" / "oracle_summary.csv")
L = ["\\begin{tabular}{@{}llllrrr@{}}", "\\toprule", "Scen. & Target & Part & $L$ & Convex & Real affine & Complex affine \\\\", "\\midrule"]
for (sc, tg, pt, Lw), g in o.groupby(["scenario", "target", "part", "L"], sort=False):
    v = g.set_index("set").si_sdr_mean
    L.append(f"{sc.replace('test_', '').upper()} & {tg.replace('_', '\\_')} & {pt.replace('_', '-')} & {Lw} & {f(v['convex'], 2)} & "
             f"{f(v['real_affine'], 2)} & {f(v['complex_affine'], 2)} \\\\")
L += ["\\bottomrule", "\\end{tabular}"]
write("s2_oracle", L)

# ---------------------------------------------------------------- S3：基线调参
L = ["\\begin{tabular}{@{}p{3.3cm}p{10.2cm}p{2.6cm}@{}}", "\\toprule", "Baseline (array; validation set) & Candidates (mean $\\Delta$SI-SDR, dB) & Selected \\\\", "\\midrule"]
s3 = json.loads((K / "stage3-eval-direct" / "eval" / "stage3_direct" / "selection.json").read_text())
acc = s3["acc"]["candidates"]
L.append("ACC $\\mu$ ($M{=}8$; nominal) & " + ", ".join(f"{k[7:-1]}: {f(v, 2)}" for k, v in acc.items()) + f" & {s3['acc']['mu']:g} \\\\")
mv = json.loads((K / "stage4-mvdr-mmval" / "eval" / "mvdr_mmval" / "selection.json").read_text())
L.append("Mask-MVDR loading ($M{=}8$; mismatched) & " + ", ".join(f"{k}: {f(v, 2)}" for k, v in mv["candidates"].items()) + f" & {mv['load']:g} \\\\")
ag = json.loads((ROOT / "results" / "mird" / "r7" / "acc_select.json").read_text())
L.append("ACC $\\mu$ (G2; nominal) & " + ", ".join(f"{k}: {f(v, 2)}" for k, v in ag["val_mean_db"].items()) + " & 0.001 \\\\")
for key, lab in (("8mic", "$M{=}8$"), ("g2", "G2")):
    for fn, vs in (("mpdr_select.json", "nominal"), ("mpdr_select_mmval.json", "mismatched")):
        d = json.loads((ROOT / "results" / "baselines2" / fn).read_text())[key]
        top = d["ranking"][:3]
        L.append(f"DMA-GSC-MPDR ({lab}; {vs}) & top 3: " + "; ".join(
            f"{t['method'].split('(')[1].rstrip(')').replace(',W0=none', '').replace('delta', '$\\delta_{\\mathrm{load}}$').replace('lam', '$\\lambda$')}: {f(t['d_si_sdr_dp'], 2)}"
            for t in top) + f" & {d['selected'].split('(')[1].rstrip(')').replace(',W0=none', '').replace('delta', '$\\delta_{\\mathrm{load}}$').replace('lam', '$\\lambda$')} \\\\")
L += ["\\bottomrule", "\\end{tabular}"]
write("s3_tuning", L)

# ---------------------------------------------------------------- S4：x_ref 训练的对比（计划外）
L = ["\\begin{tabular}{@{}llrrrr@{}}", "\\toprule", "Scen. & $\\kappa$ & $x_{\\mathrm{ref}}$: diff. & 95\\% CI & $x_{\\mathrm{dp}}$: diff. & 95\\% CI \\\\", "\\midrule"]
for s in ("test_s1", "test_s2"):
    for k in (0, 1, 2):
        sp_ = s if k == 0 else f"{s}_kappa{k}"
        a, b = load("b5-eval-" + s[-2:], sp_, "Affine-xref", 600), load("b5-eval-" + s[-2:], sp_, "BeamFusion-xref", 600)
        cells = []
        for col in ("d_si_sdr", "d_si_sdr_dp"):
            d = a[col] - b.loc[a.index, col]
            h = 1.96 * d.std(ddof=1) / np.sqrt(len(d))
            cells += [f(d.mean(), 2, True), f"[{f(d.mean() - h, 2)}, {f(d.mean() + h, 2)}]"]
        L.append(f"{s[-2:].upper()} & {k} & " + " & ".join(cells) + " \\\\")
L += ["\\bottomrule", "\\end{tabular}"]
write("s4_xref", L)

# ---------------------------------------------------------------- S5：相同 WNG 下限（探索）
META = {s: mird_meta(s) for s in ("mird_s1", "mird_s2")}


def cond_pair(a, b, s, col="d_si_sdr_dp"):
    d = (a[col] - b.loc[a.index, col]).to_frame("d").join(META[s])
    cm = d.groupby("cond")["d"].mean()
    return cm.mean(), int((cm > 0).sum())


SEED = {0: dict(Am=("g2-eval-mird-a", "G2-Affine-mm"), Bm=("g2-eval-bf-mird", "G2-BeamFusion-mm"),
                Av=("explore-w0-val", "G2-Affine-mm"), Bv=("explore2-e1-val", "G2-BeamFusion-mm"),
                A5=lambda w: ("explore-w0-mird", f"G2-Affine-mm(projW0={w})"), B5=lambda w: ("explore2-e1-mird", f"G2-BeamFusion-mm(postW0={w})"),
                A5v=lambda w: ("explore-w0-val", f"G2-Affine-mm(projW0={w})"), B5v=lambda w: ("explore2-e1-val", f"G2-BeamFusion-mm(postW0={w})"),
                floors=(-15, -10, -5, 0, 3))}
for sd in (1, 2):
    j = f"g2-s{sd}-eval"
    SEED[sd] = dict(Am=(f"{j}-mird", f"G2-Affine-mm-s{sd}"), Bm=(f"{j}-mird", f"G2-BeamFusion-mm-s{sd}"),
                    Av=(f"{j}-val", f"G2-Affine-mm-s{sd}"), Bv=(f"{j}-val", f"G2-BeamFusion-mm-s{sd}"),
                    A5=lambda w, j=j, sd=sd: (f"{j}-mird", f"G2-Affine-mm-s{sd}(postW0={w})"),
                    B5=lambda w, j=j, sd=sd: (f"{j}-mird", f"G2-BeamFusion-mm-s{sd}(postW0={w})"),
                    A5v=lambda w, j=j, sd=sd: (f"{j}-val", f"G2-Affine-mm-s{sd}(postW0={w})"),
                    B5v=lambda w, j=j, sd=sd: (f"{j}-val", f"G2-BeamFusion-mm-s{sd}(postW0={w})"), floors=(-5,))
L = ["\\begin{tabular}{@{}llrrrrrr@{}}", "\\toprule",
     "Seed & Floor & Val.\\ A & Val.\\ B & MIRD A (S1/S2) & MIRD B (S1/S2) & A$-$B S1 & A$-$B S2 \\\\", "\\midrule"]
L2 = ["\\begin{tabular}{@{}llrrrr@{}}", "\\toprule", "Seed & Floor & $x_{\\mathrm{ref}}$ (S1/S2) & STOI (S1/S2) & PESQ (S1/S2) & TD SDR (S1/S2) \\\\", "\\midrule"]
for sd, c in SEED.items():
    for w in (None,) + c["floors"]:
        am = c["Am"] if w is None else c["A5"](w)
        bm = c["Bm"] if w is None else c["B5"](w)
        av = c["Av"] if w is None else c["A5v"](w)
        bv = c["Bv"] if w is None else c["B5v"](w)
        A = {s: load(am[0], s, am[1], 600) for s in META}
        B = {s: load(bm[0], s, bm[1], 600) for s in META}
        va, vb = load(av[0], "val", av[1], 500).d_si_sdr_dp.mean(), load(bv[0], "val", bv[1], 500).d_si_sdr_dp.mean()
        pr = {s: cond_pair(A[s], B[s], s) for s in META}
        lab = "none" if w is None else f"${w:g}$\\,dB".replace("$-", "$-$")
        lab = "none" if w is None else f"{w:g}\\,dB".replace("-", "$-$")
        L.append(f"{sd} & {lab} & {f(va, 2)} & {f(vb, 2)} & {f(A['mird_s1'].d_si_sdr_dp.mean(), 2)}/{f(A['mird_s2'].d_si_sdr_dp.mean(), 2)} & "
                 f"{f(B['mird_s1'].d_si_sdr_dp.mean(), 2)}/{f(B['mird_s2'].d_si_sdr_dp.mean(), 2)} & "
                 f"{f(pr['mird_s1'][0], 2, True)} ({pr['mird_s1'][1]}) & {f(pr['mird_s2'][0], 2, True)} ({pr['mird_s2'][1]}) \\\\")
        if w in (None, -5):
            cells = []
            for col, dd in (("d_si_sdr", 2), ("stoi_dp_out", 3), ("pesq_dp_out", 3), ("target_direct_sdr_out", 2)):
                cells.append("/".join(f(cond_pair(A[s], B[s], s, col)[0], dd, True) for s in META))
            L2.append(f"{sd} & {lab} & " + " & ".join(cells) + " \\\\")
L += ["\\bottomrule", "\\end{tabular}"]
L2 += ["\\bottomrule", "\\end{tabular}"]
write("s5_floors", L)
write("s5_secondary", L2)
# 频带限定干预（种子 0，A）
A0 = {s: load("g2-eval-mird-a", s, "G2-Affine-mm", 600) for s in META}
L = ["\\begin{tabular}{@{}lrr@{}}", "\\toprule", "$-5$\\,dB projection applied to & S1 & S2 \\\\", "\\midrule"]
for lab, name in (("all bins", "G2-Affine-mm(postW0=-5)"), ("0--1\\,kHz only", "G2-Affine-mm(postW0=-5@0-1000Hz)"),
                  ("1--5.717\\,kHz only", "G2-Affine-mm(postW0=-5@1000-5717Hz)"), ("5.717--8\\,kHz only", "G2-Affine-mm(postW0=-5@5717-8001Hz)")):
    v = [(load("explore2-e2-mird", s, name, 600).d_si_sdr_dp - A0[s].d_si_sdr_dp).mean() for s in META]
    L.append(f"{lab} & {f(v[0], 3, True)} & {f(v[1], 3, True)} \\\\")
L += ["\\bottomrule", "\\end{tabular}"]
write("s5_band_intervention", L)
# 激活率（种子 0，−5 dB）
L = ["\\begin{tabular}{@{}llrrrr@{}}", "\\toprule", "Data & Band (Hz) & A: active & A: $|X|^2$-wt. & B: active & B: $|X|^2$-wt. \\\\", "\\midrule"]
for tag, job in (("MIRD", "explore2-e3-mird"), ("Val.", "explore2-e3-val")):
    d = pd.read_csv(glob.glob(str(K / job / "eval" / "*" / "band_diag.csv*"))[0])
    for band in ("0-500", "500-1000", "1000-2000", "2000-4000", "4000-5717", "5717-8000"):
        x = d[d.band == band]
        a = x[x.method == "G2-Affine-mm(postW0=-5)"]
        b = x[x.method == "G2-BeamFusion-mm(postW0=-5)"]
        L.append(f"{tag} & {band.replace('-', '--')} & {f(a.proj_active_frac.mean(), 4)} & {f(a.proj_active_frac_wx.mean(), 3)} & "
                 f"{f(b.proj_active_frac.mean(), 4)} & {f(b.proj_active_frac_wx.mean(), 3)} \\\\")
L += ["\\bottomrule", "\\end{tabular}"]
write("s5_activation", L)

# ---------------------------------------------------------------- S6：三个种子的分频带诊断
SRC = {0: ("explore-band-mird", "G2-Affine-mm", "G2-BeamFusion-mm"), 1: ("band-diag-mird-s1", "G2-Affine-mm-s1", "G2-BeamFusion-mm-s1"),
       2: ("band-diag-mird-s2", "G2-Affine-mm-s2", "G2-BeamFusion-mm-s2")}
CONDS = ["160ms-A", "160ms-B", "360ms-A", "360ms-B", "610ms-A", "610ms-B"]
BANDS = ["0-500", "500-1000", "1000-2000", "2000-4000", "4000-5717", "5717-8000"]
L = ["\\begin{longtable}{@{}lll" + "r" * 6 + "r@{}}", "\\caption{MIRD band-wise output error, A$-$B (dB), per seed, scenario, band, and condition.}\\label{tab:bands}\\\\", "\\toprule",
     "Seed & Scen. & Band (Hz) & " + " & ".join(c.replace("ms-", "\\,ms/") for c in CONDS) + " & A lower \\\\", "\\midrule\\endhead"]
summ = []
for sd, (job, na, nb) in SRC.items():
    fs = glob.glob(str(K / job / "eval" / "*" / "band_diag.csv*")) + glob.glob(str(K / (job + "b") / "eval" / "*" / "band_diag.csv*"))
    d = pd.concat([pd.read_csv(x) for x in fs]).drop_duplicates(["method", "split", "id", "band"])
    cnt = {b: 0 for b in BANDS}
    for s in ("mird_s1", "mird_s2"):
        x = d[d.split == s]
        a = x[x.method == na].set_index(["id", "band"])
        b = x[x.method == nb].set_index(["id", "band"]).loc[a.index]
        df = (a.out_err_db - b.out_err_db).reset_index()
        df["cond"] = df.id.map(META[s].cond)
        pvt = df.pivot_table(index="band", columns="cond", values="out_err_db", aggfunc="mean")
        for band in BANDS:
            vals = pvt.loc[band, CONDS]
            cnt[band] += int((vals < 0).sum())
            L.append(f"{sd} & {s[-2:].upper()} & {band.replace('-', '--')} & " + " & ".join(f(v, 2, True) for v in vals) + f" & {int((vals < 0).sum())}/6 \\\\")
    summ.append(f"{sd} & " + " & ".join(f"{cnt[b]}/12" for b in BANDS) + " \\\\")
L += ["\\bottomrule", "\\end{longtable}"]
write("s6_bands", L)
write("s6_summary", ["\\begin{tabular}{@{}l" + "c" * 6 + "@{}}", "\\toprule", "Seed & " + " & ".join(b.replace("-", "--") for b in BANDS) + " \\\\",
                     "\\midrule", *summ, "\\bottomrule", "\\end{tabular}"])
print("written", sorted(x.name for x in OUT.iterdir()))
