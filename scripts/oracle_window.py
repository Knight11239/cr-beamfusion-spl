"""分窗 oracle 可行集合分析（用户 2026-11-08；只作为表达能力的上界分析，不作为因果机制的证明）。

  python scripts/oracle_window.py run --sets test_s1:0 test_s2:0 --out DIR      # Kaggle CPU：逐样本结果
  python scripts/oracle_window.py report --in DIR --out results/oracle          # 汇总（含与网络实际结果的描述性对照）

数据：8 麦克风 S1 / S2，κ = 0，各 100 条（固定规则：样本序号 i 满足 i mod 12 ∈ {0, 1}，T60 0.3 / 0.7 各 50）。
对每个频点、每个长度为 L 帧的不重叠时间窗（L ∈ {8, 32, 125}），用混合信号的 5 个波束输出 Z_p（h_D、null-90/120/150/180）
在窗内固定一组权重，最小化 Σ_t |Σ_p c_p Z_p(t) − X(t)|²（X 为目标的 STFT：x_dp，另做 x_ref）。三种可行集合：
  凸（c 实数、非负、和为 1；单纯形约束的二次规划，逐面枚举精确求解）；
  实仿射（Ẑ = Z_D + Σ u_p e_p，e_p = Z_p − Z_D，u 实数；最小二乘）；
  复仿射（同上，u 复数；最小二乘）。
三者可行集合依次包含（凸 ⊂ 实仿射 ⊂ 复仿射），故窗内最优误差依次不增。
输出三部分（不混用）：(a) 窗内最优拟合（表达能力上界）；(b) 窗外泛化（窗口 k 拟合的权重用于窗口 k + 1；第一个窗口与末尾不足 L 帧的部分
用 h_D）；(c) 实际因果网络（BeamFusion-mm、Affine-mm）在同一批样本上的结果（由已有评测读出）。
指标：时域 SI-SDR（输出 iSTFT 后与参考比较；h_D 与网络同一定义）与 STFT 域 SDR（只统计被窗覆盖的帧；窗外泛化不含第一个窗口）。
数值：正则化为相对对角加载 ρ = 1e-9·tr(H)/n（所有集合相同）；记录 Gram 矩阵条件数（窗口中位数 / 99% 分位 / 最大）与失败窗口数（解非有限，回退为 h_D）。
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from src import metrics as mt  # noqa: E402
from src.config import load_config  # noqa: E402
from src.models.gtcrn_fusion import BeamBank  # noqa: E402
from src.pipeline import array_stft  # noqa: E402
from src.stft import istft, stft  # noqa: E402

LS = (8, 32, 125)
SETS_ = ("convex", "real_affine", "complex_affine")
RHO = 1e-9


def _solve(A, b):
    """批量求解 A x = b（A [..., n, n]，b [..., n]），返回 (x, ok)。"""
    try:
        x = np.linalg.solve(A, b[..., None])[..., 0]
    except np.linalg.LinAlgError:
        x = np.full(b.shape, np.nan, dtype=np.result_type(A, b))
        for i in np.ndindex(A.shape[:-2]):
            try:
                x[i] = np.linalg.solve(A[i], b[i])
            except np.linalg.LinAlgError:
                pass
    return x, np.isfinite(x).all(-1)


def load_reg(H):
    n = H.shape[-1]
    tr = np.real(np.trace(H, axis1=-2, axis2=-1))
    return H + (RHO * tr / n + 1e-30)[..., None, None] * np.eye(n)


def fit_affine(Zw, Xw, real):
    """Zw [P, W, L]（W = F·K 个窗），Xw [W, L] → u [W, P−1]（复数或实数），cond [W]，ok [W]。"""
    E = Zw[1:] - Zw[:1]  # [P', W, L]
    y = Xw - Zw[0]
    A = np.einsum("pwl,qwl->wpq", E.conj(), E)  # E^H E
    b = np.einsum("pwl,wl->wp", E.conj(), y)
    if real:
        A, b = A.real, b.real
    cond = np.linalg.cond(A)
    u, ok = _solve(load_reg(A), b)
    return u, cond, ok


def fit_convex(Zw, Xw):
    """单纯形约束二次规划：J(c) = cᵀHc − 2cᵀg（H = Re(ZᴴZ)，g = Re(ZᴴX)），逐面枚举精确求解。返回 c [W, P]、cond、ok。"""
    P = Zw.shape[0]
    H = np.einsum("pwl,qwl->wpq", Zw.conj(), Zw).real
    g = np.einsum("pwl,wl->wp", Zw.conj(), Xw).real
    cond = np.linalg.cond(H)
    Hr = load_reg(H)
    W = H.shape[0]
    best = np.full(W, np.inf)
    cbest = np.zeros((W, P))
    for k in range(1, P + 1):
        for S in itertools.combinations(range(P), k):
            S = list(S)
            n = len(S)
            K = np.zeros((W, n + 1, n + 1))
            K[:, :n, :n] = 2 * Hr[:, S][:, :, S]
            K[:, :n, n] = 1.0
            K[:, n, :n] = 1.0
            rhs = np.zeros((W, n + 1))
            rhs[:, :n] = 2 * g[:, S]
            rhs[:, n] = 1.0
            x, okS = _solve(K, rhs)
            c = np.zeros((W, P))
            c[:, S] = x[:, :n]
            feas = okS & (c[:, S] >= -1e-12).all(-1)
            J = np.einsum("wp,wpq,wq->w", c, H, c) - 2 * np.einsum("wp,wp->w", c, g)
            upd = feas & (J < best)
            best[upd] = J[upd]
            cbest[upd] = c[upd]
    ok = np.isfinite(best)
    return np.clip(cbest, 0, None), cond, ok


def apply_w(Zw, w, kind):
    """窗内权重作用于 Zw [P, W, L]，返回 [W, L]。"""
    if kind == "convex":
        return np.einsum("wp,pwl->wl", w.astype(Zw.dtype), Zw)
    E = Zw[1:] - Zw[:1]
    return Zw[0] + np.einsum("wp,pwl->wl", w.astype(E.dtype), E)


def run_sample(Z, X, L):
    """Z [P, F, T] complex128，X [F, T] → {set: (S_in [F, T], S_out [F, T], cond, n_fail, Tcov, K)}。"""
    P, F, T = Z.shape
    K = T // L
    Tc = K * L
    Zw = Z[:, :, :Tc].reshape(P, F, K, L).reshape(P, F * K, L)
    Xw = X[:, :Tc].reshape(F, K, L).reshape(F * K, L)
    res = {}
    for kind in SETS_:
        if kind == "convex":
            w, cond, ok = fit_convex(Zw, Xw)
            w0 = np.zeros_like(w)
            w0[:, 0] = 1.0  # 回退：h_D
        else:
            w, cond, ok = fit_affine(Zw, Xw, real=(kind == "real_affine"))
            w0 = np.zeros_like(w)
        w = np.where(ok[:, None], w, w0)
        S_in = Z[0].copy()
        S_in[:, :Tc] = apply_w(Zw, w, kind).reshape(F, K, L).reshape(F, Tc)
        # 窗外：窗口 k 的权重用于窗口 k + 1；第一个窗口用 h_D
        wk = w.reshape(F, K, -1)
        wshift = np.concatenate([w0.reshape(F, K, -1)[:, :1], wk[:, :-1]], 1).reshape(F * K, -1)
        S_out = Z[0].copy()
        S_out[:, :Tc] = apply_w(Zw, wshift, kind).reshape(F, K, L).reshape(F, Tc)
        cf = cond[np.isfinite(cond)]
        res[kind] = (S_in, S_out, {"cond_median": float(np.median(cf)) if len(cf) else np.nan,
                                   "cond_p99": float(np.percentile(cf, 99)) if len(cf) else np.nan,
                                   "cond_max": float(cf.max()) if len(cf) else np.nan,
                                   "n_fail": int((~ok).sum()), "n_windows": int(ok.size)}, Tc, K)
    return res


def sdr_stft(S, X, t0, t1):
    e = np.sum(np.abs(S[1:, t0:t1] - X[1:, t0:t1]) ** 2)
    return float(10 * np.log10(np.sum(np.abs(X[1:, t0:t1]) ** 2) / max(e, 1e-30)))


def cmd_run(a):
    from stage4_diag import loader, parse_sets, stack
    cfg = load_config(a.data_config)
    bank = BeamBank.from_config(cfg)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for split, kappa in parse_sets(a.sets):
        t0 = time.time()
        ds, dl = loader(cfg, split, kappa, components=False, pick=lambda i: (i % 12) in a.pick_mod,
                        batch_size=2, workers=a.num_workers)
        n = 0
        for batch in dl:
            g = torch.stack([m["mm_gain_db"] for _, m in batch])
            p = torch.stack([m["mm_phase_deg"] for _, m in batch])
            mix = stack(batch, "mix")
            N = mix.shape[-1]
            Zb = bank(array_stft(mix, g, p)).to(torch.complex128)  # [B, P, F, T]
            refs = {"x_dp": stack(batch, "ref_direct"), "x_ref": stack(batch, "ref")}
            for i in range(len(batch)):
                sc = ds.scenes[n + i]
                Z = Zb[i].numpy()
                hd_t = istft(torch.as_tensor(Z[0])[None].to(torch.complex64), N)[0].numpy()
                for rname, rsig in refs.items():
                    xr = rsig[i].numpy()
                    X = stft(rsig[i:i + 1])[0].numpy().astype(np.complex128)
                    base = {"id": sc["id"], "scenario": split, "t60": sc["t60_target"], "target": rname}
                    rows.append({**base, "L": 0, "set": "h_D", "part": "-", "si_sdr": float(mt.si_sdr(hd_t, xr)),
                                 "sdr_stft": sdr_stft(Z[0], X, 0, Z.shape[-1])})
                    for rho in a.rho:
                                globals()["RHO"] = rho  # load_reg 读取模块级 RHO
                                for L in LS:
                                    res = run_sample(Z, X, L)
                                    for kind, (S_in, S_out, cst, Tc, K) in res.items():
                                        for part, S, ta in (("in_window", S_in, 0), ("out_of_window", S_out, L)):
                                            y = istft(torch.as_tensor(S)[None].to(torch.complex64), N)[0].numpy()
                                            rows.append({**base, "rho": rho, "L": L, "set": kind, "part": part,
                                                         "si_sdr": float(mt.si_sdr(y, xr)), "sdr_stft": sdr_stft(S, X, ta, Tc), **cst})
            n += len(batch)
            print(f"{time.strftime('%H:%M:%S')} [{split}] {n}/{len(ds)} {(time.time() - t0) / 60:.1f} min", flush=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(out / "oracle_window_samples.csv", "w", newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def cmd_report(a):
    rows = list(csv.DictReader(open(Path(a.inp) / "oracle_window_samples.csv")))
    K = ROOT / "results" / "kaggle"
    net = {}
    for m, globs in (("BeamFusion-mm", ["b5-eval-s1", "b5-eval-s2"]), ("Affine-mm", ["stage4-b3eval-a", "stage4-b3eval-b"])):
        mm = {"BeamFusion-mm": "BeamFusion-mm", "Affine-mm": "AffineMM"}[m]
        for gl in globs:
            for f in K.glob(f"{gl}/eval/**/*_samples.csv"):
                for r in csv.DictReader(open(f)):
                    if r["method"] == mm and float(r["kappa"]) == 0:
                        net[(m, r["split"], r["id"])] = r
    ids = {(r["scenario"], r["id"]) for r in rows}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    def ci(v):
        v = np.asarray(v, float)
        v = v[np.isfinite(v)]
        return v.mean(), 1.96 * v.std(ddof=1) / np.sqrt(len(v)), len(v)

    L_ = ["# 分窗 oracle 可行集合分析（`python scripts/oracle_window.py`；只作表达能力的上界分析，不作为因果机制的证明）", "",
          "8 麦克风 S1 / S2，κ = 0，各 100 条（i mod 12 ∈ {0, 1}）。时域 SI-SDR（dB，均值 ± 95% CI）；数值见文末。", ""]
    summ = []
    for sc in ("test_s1", "test_s2"):
        for tgt in ("x_dp", "x_ref"):
            sel = [r for r in rows if r["scenario"] == sc and r["target"] == tgt]
            hd = ci([float(r["si_sdr"]) for r in sel if r["set"] == "h_D"])
            L_ += [f"## {sc.replace('test_', '').upper()}，目标 {tgt}", "",
                   f"h_D（参照）：{hd[0]:.2f} ± {hd[1]:.2f} dB", ""]
            for part, title in (("in_window", "(a) 窗内最优拟合（表达能力上界）"), ("out_of_window", "(b) 窗外泛化（窗口 k 的权重用于窗口 k + 1）")):
                L_ += [f"### {title}", "", "| L（帧） | 凸 | 实仿射 | 复仿射 | 复仿射 − 凸 |", "| --- | --- | --- | --- | --- |"]
                for L in LS:
                    v = {k: {r["id"]: float(r["si_sdr"]) for r in sel if r["set"] == k and r["part"] == part and int(r["L"]) == L} for k in SETS_}
                    c = {k: ci(list(v[k].values())) for k in SETS_}
                    common = sorted(set(v["convex"]) & set(v["complex_affine"]))
                    d = ci([v["complex_affine"][i] - v["convex"][i] for i in common])
                    L_.append(f"| {L} | " + " | ".join(f"{c[k][0]:.2f} ± {c[k][1]:.2f}" for k in SETS_) + f" | {d[0]:+.2f} ± {d[1]:.2f} |")
                    for k in SETS_:
                        summ.append({"scenario": sc, "target": tgt, "part": part, "L": L, "set": k, "si_sdr_mean": c[k][0], "ci95": c[k][1], "n": c[k][2]})
                L_.append("")
            col = "si_sdr_dp_out" if tgt == "x_dp" else "si_sdr_out"
            L_ += ["### (c) 实际因果网络（同一批样本，已有评测）", "", "| 网络 | SI-SDR（dB） | n |", "| --- | --- | --- |"]
            for m in ("BeamFusion-mm", "Affine-mm"):
                vv = [float(net[(m, sc, i)][col]) for (s, i) in ids if s == sc and (m, sc, i) in net]
                c = ci(vv)
                L_.append(f"| {m} | {c[0]:.2f} ± {c[1]:.2f} | {c[2]} |")
                summ.append({"scenario": sc, "target": tgt, "part": "network", "L": "", "set": m, "si_sdr_mean": c[0], "ci95": c[1], "n": c[2]})
            L_.append("")
    L_ += ["## 数值条件（Gram 矩阵条件数：窗口中位数的样本中位数 / 99% 分位的最大值 / 最大值；失败窗口总数）", "",
           "| L | 集合 | cond 中位数 | cond p99（最大） | cond 最大 | 失败窗口 / 总窗口 |", "| --- | --- | --- | --- | --- | --- |"]
    for L in LS:
        for k in SETS_:
            s = [r for r in rows if r["set"] == k and r["part"] == "in_window" and int(r["L"]) == L]
            L_.append(f"| {L} | {k} | {np.median([float(r['cond_median']) for r in s]):.2e} | {max(float(r['cond_p99']) for r in s):.2e} | "
                      f"{max(float(r['cond_max']) for r in s):.2e} | {sum(int(r['n_fail']) for r in s)} / {sum(int(r['n_windows']) for r in s)} |")
    L_ += ["", f"正则化：相对对角加载 ρ = {RHO:g}·tr(H)/n（三种集合相同）。失败窗口（解非有限）回退为 h_D。"]
    with open(out / "oracle_summary.csv", "w", newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=list(summ[0]))
        w.writeheader()
        w.writerows(summ)
    (out / "README.md").write_text("\n".join(L_) + "\n")
    print("\n".join(L_[:40]))


def cmd_report_rho(a):
    """正则化敏感性（用户 2026-11-09）：同一批样本上 ρ 放大 / 缩小 10 倍，比较 (a) 窗内、(b) 窗外的结论是否改变。"""
    rows = list(csv.DictReader(open(Path(a.inp) / "oracle_window_samples.csv")))
    rhos = sorted({float(r["rho"]) for r in rows if r.get("rho")})
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    def ci(v):
        v = np.asarray(v, float)
        v = v[np.isfinite(v)]
        return v.mean(), 1.96 * v.std(ddof=1) / np.sqrt(len(v)), len(v)

    n_ids = len({(r["scenario"], r["id"]) for r in rows})
    L_ = ["# 分窗 oracle：正则化敏感性（`python scripts/oracle_window.py report-rho`）", "",
          f"8 麦克风 S1 / S2，κ = 0，共 {n_ids} 条（i mod 12 = 0，为主分析样本的子集）。相对对角加载 ρ ∈ {{{', '.join(f'{x:g}' for x in rhos)}}}"
          "（主分析 ρ = 1e-9）。时域 SI-SDR（dB，均值 ± 95% CI）。", ""]
    summ = []
    for tgt in ("x_dp", "x_ref"):
        for part, title in (("in_window", "(a) 窗内最优拟合"), ("out_of_window", "(b) 窗外泛化")):
            L_ += [f"## 目标 {tgt}，{title}", "", "| 场景 | L | ρ | 凸 | 实仿射 | 复仿射 | 复仿射 − 凸 | 复仿射 − 实仿射 | 失败窗口 |",
                   "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
            for sc in ("test_s1", "test_s2"):
                for L in LS:
                    for rho in rhos:
                        sel = [r for r in rows if r["scenario"] == sc and r["target"] == tgt and r["part"] == part
                               and int(r["L"]) == L and float(r["rho"]) == rho]
                        v = {k: {r["id"]: float(r["si_sdr"]) for r in sel if r["set"] == k} for k in SETS_}
                        c = {k: ci(list(v[k].values())) for k in SETS_}
                        ids = sorted(set(v["convex"]) & set(v["complex_affine"]) & set(v["real_affine"]))
                        d1 = ci([v["complex_affine"][i] - v["convex"][i] for i in ids])
                        d2 = ci([v["complex_affine"][i] - v["real_affine"][i] for i in ids])
                        nf = sum(int(r["n_fail"]) for r in sel if r["part"] == part)
                        L_.append(f"| {sc.replace('test_', '').upper()} | {L} | {rho:g} | " + " | ".join(f"{c[k][0]:.2f} ± {c[k][1]:.2f}" for k in SETS_)
                                  + f" | {d1[0]:+.2f} ± {d1[1]:.2f} | {d2[0]:+.2f} ± {d2[1]:.2f} | {nf} |")
                        summ.append({"scenario": sc, "target": tgt, "part": part, "L": L, "rho": rho,
                                     **{f"{k}_mean": c[k][0] for k in SETS_}, "complex_minus_convex": d1[0], "complex_minus_real": d2[0], "n": len(ids)})
            L_.append("")
    with open(out / "oracle_rho_summary.csv", "w", newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=list(summ[0]))
        w.writeheader()
        w.writerows(summ)
    (out / "README_rho.md").write_text("\n".join(L_) + "\n")
    print("\n".join(L_))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "report", "report-rho"])
    ap.add_argument("--data-config", default="configs/data.yaml")
    ap.add_argument("--sets", nargs="*", default=["test_s1:0", "test_s2:0"])
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--in", dest="inp", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rho", type=float, nargs="*", default=[RHO], help="相对对角加载（正则化敏感性检查时给多个值）")
    ap.add_argument("--pick-mod", type=int, nargs="*", default=[0, 1], help="选样：i mod 12 属于这些值")
    a = ap.parse_args()
    a.pick_mod = tuple(a.pick_mod)
    torch.set_grad_enabled(False)
    {"run": cmd_run, "report": cmd_report, "report-rho": cmd_report_rho}[a.cmd](a)


if __name__ == "__main__":
    main()
