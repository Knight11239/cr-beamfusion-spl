"""Fixed methods from analysis_plan_controls.md; no silent finite-value deletion."""
import itertools

import numpy as np
from scipy.stats import rankdata, t, wilcoxon

BOOT_REPETITIONS = 20_000
BOOT_SEED = 20261110


def vectors(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.ndim != 1 or a.shape != b.shape or len(a) < 2:
        raise ValueError("Expected equally sized paired vectors with n >= 2")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Nonfinite endpoint: do not silently delete model-specific failures")
    return a, b


def exact_signed_rank(d):
    """Exhaustive two-sided sign flips of nonzero average ranks (MIRD, n<=6)."""
    d = np.asarray(d, float)
    if not np.isfinite(d).all() or len(d) > 6:
        raise ValueError("Exact condition test requires finite n <= 6")
    nz = d[d != 0]
    if not len(nz):
        return 1.0
    ranks = rankdata(abs(nz), method="average")
    observed = abs(np.dot(np.sign(nz), ranks))
    statistics = [abs(np.dot(signs, ranks))
                  for signs in itertools.product((-1, 1), repeat=len(nz))]
    return float(np.mean(np.asarray(statistics) >= observed - 1e-12))


def paired(a, b, condition_level=False, run_test=True):
    a, b = vectors(a, b)
    d = a - b
    n = len(d)
    sd = float(d.std(ddof=1))
    se = sd / np.sqrt(n)
    critical = float(t.ppf(.975, n - 1)) if condition_level else 1.96
    nz = d[d != 0]
    ranks = rankdata(abs(nz), method="average")
    if not run_test:
        p = None
    elif condition_level:
        p = exact_signed_rank(d)
    else:
        p = float(wilcoxon(d, zero_method="wilcox", alternative="two-sided",
                           correction=False, method="asymptotic").pvalue) if len(nz) else 1.0
    return dict(n=n, nonzero_n=len(nz), mean_left=float(a.mean()), mean_right=float(b.mean()),
                mean_diff=float(d.mean()), sd_diff=sd, ci_lo=float(d.mean() - critical * se),
                ci_hi=float(d.mean() + critical * se), median_diff=float(np.median(d)),
                p_raw=p, r_rb=float(np.dot(np.sign(nz), ranks) / ranks.sum()) if len(nz) else 0.0,
                cohen_dz=float(d.mean() / sd) if sd > 0 else None,
                ci_method="Student-t df=5; fixed conditions" if condition_level else "paired normal 1.96 SE",
                test=("none; descriptive only" if not run_test else
                      ("exact condition signed-rank" if condition_level else "Wilcoxon asymptotic, wilcox zeros, two-sided")))


def tost(a, b, margin=.5):
    a, b = vectors(a, b)
    d = a - b
    mean = float(d.mean())
    se = float(d.std(ddof=1) / np.sqrt(len(d)))
    if se == 0:
        # Conservative boundary handling explicitly specified in the plan.
        lower = 0.0 if mean > -margin else 1.0
        upper = 0.0 if mean < margin else 1.0
    else:
        lower = float(t.sf((mean + margin) / se, len(d) - 1))
        upper = float(t.cdf((mean - margin) / se, len(d) - 1))
    half90 = float(t.ppf(.95, len(d) - 1) * se)
    return dict(p_raw=max(lower, upper), tost_p_lower=lower, tost_p_upper=upper,
                tost_margin=margin, tost_ci90_lo=mean - half90, tost_ci90_hi=mean + half90,
                test="paired-mean Student-t TOST; max(one-sided p)")


def holm(records, column, expected_m):
    """Ineligible slots occupy p=1, but never display it as a measured p-value."""
    if len(records) != expected_m:
        raise ValueError(f"Fixed family requires {expected_m} slots, got {len(records)}")
    order = sorted(range(len(records)), key=lambda i: records[i]["p_for_holm"])
    previous = 0.0
    for rank, i in enumerate(order):
        previous = max(previous, min(1., (expected_m - rank) * records[i]["p_for_holm"]))
        records[i][column] = previous if records[i]["eligible"] else None
        records[i][column + "_m"] = expected_m


def stratified_bootstrap(d, strata, repetitions=BOOT_REPETITIONS, seed=BOOT_SEED):
    d = np.asarray(d, float)
    strata = np.asarray(strata)
    if not np.isfinite(d).all() or set(strata) != {.3, .7} or len(d) != len(strata):
        raise ValueError("Need finite paired differences and both fixed T60 strata")
    groups = [d[strata == key] for key in (.3, .7)]
    if len(groups[0]) != len(groups[1]):
        raise ValueError("Plan requires equal T60 weights and equal sample counts")
    rng = np.random.default_rng(seed)
    samples = np.zeros(repetitions)
    for group in groups:
        for start in range(0, repetitions, 512):
            stop = min(repetitions, start + 512)
            indices = rng.integers(0, len(group), size=(stop - start, len(group)))
            samples[start:stop] += .5 * group[indices].mean(axis=1)
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return dict(bootstrap_status="computed", bootstrap_ci_lo=float(lo), bootstrap_ci_hi=float(hi),
                bootstrap_repetitions=repetitions, bootstrap_seed=seed,
                bootstrap_method="paired ID resampling within each T60; fixed 1/2 weights")


def cluster_bootstrap(differences, scenes, repetitions=BOOT_REPETITIONS, seed=BOOT_SEED):
    """Condition-stratified trajectory resampling; all speech rows stay in a cluster.

    differences is n x endpoints. No condition is resampled or reweighted by
    its bootstrap row count. Resetting the seed per contrast makes output
    independent of whether other prospective slots are still pending.
    """
    differences = np.asarray(differences, float)
    if differences.ndim != 2 or len(differences) != len(scenes) or not np.isfinite(differences).all():
        raise ValueError("Invalid cluster inputs")
    conditions = {}
    for i, scene in enumerate(scenes):
        condition = (scene["rt60"], scene["subarray"])
        trajectory = tuple(sorted((tuple(x["trajectory_deg"]), x["dist"]) for x in scene["interferers"]))
        conditions.setdefault(condition, {}).setdefault(trajectory, []).append(i)
    if len(conditions) != 6:
        raise ValueError("MIRD requires all six conditions")
    counts = {f"{rt}/{sub}": len(clusters) for (rt, sub), clusters in sorted(conditions.items())}
    base = dict(bootstrap_repetitions=repetitions, bootstrap_seed=seed, clusters_by_condition=counts,
                bootstrap_method="trajectory clusters within fixed measured conditions; equal six-condition weights")
    if min(counts.values()) < 2:
        return dict(base, bootstrap_status="not_estimable_single_trajectory_condition",
                    reason="No within-condition trajectory replication; no zero-width CI or IID fallback",
                    bootstrap_ci_lo=None, bootstrap_ci_hi=None)
    rng = np.random.default_rng(seed)
    draws = np.zeros((repetitions, differences.shape[1]))
    for _, clusters in sorted(conditions.items()):
        indices = list(clusters.values())
        sums = np.array([differences[ix].sum(axis=0) for ix in indices])
        ns = np.array([len(ix) for ix in indices])
        for start in range(0, repetitions, 512):
            stop = min(repetitions, start + 512)
            sampled = rng.integers(0, len(indices), size=(stop - start, len(indices)))
            means = sums[sampled].sum(axis=1) / ns[sampled].sum(axis=1)[:, None]
            draws[start:stop] += means / 6
    lo, hi = np.percentile(draws, [2.5, 97.5], axis=0)
    return dict(base, bootstrap_status="computed", bootstrap_ci_lo=lo.tolist(), bootstrap_ci_hi=hi.tolist())
