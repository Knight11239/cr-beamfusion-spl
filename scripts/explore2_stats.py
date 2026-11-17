"""事后探索第二轮的统计工具（用户 2026-11-15）：MIRD 配对差按 T60 × 子阵条件等权，考虑共享实测 RIR 的相关性。

- 条件层：六个条件（3 个 RT60 × 2 个子阵）各自的配对差均值，等权平均，正向条件个数；t_5 区间只作描述。
- 聚类层：聚类 = 条件 × 干扰配置（mird_cluster_pairs.cluster_key）；在条件内对聚类分层有放回重采样（每个条件抽与原来
  同样多的聚类），统计量为六条件等权平均。以这些实测 RIR 为条件，不代表跨房间推断。S1 每个条件只有 1 个聚类，
  其分层 bootstrap 退化为 0 宽度，因此 S1 只报告条件层（列 boot_ok = False）。
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
META = ROOT / "results" / "kaggle" / "mird-prep" / "reports"


def cluster_key(sc):
    return (sc["split"], sc["rt60"], sc["subarray"],
            tuple(sorted((tuple(i["trajectory_deg"]), float(i["dist"])) for i in sc["interferers"])))


def mird_meta(split):
    sc = json.loads((META / f"meta_{split}.json").read_text())["scenes"]
    return pd.DataFrame([{"id": s["id"], "cond": f"{float(s['rt60']) * 1000:.0f}ms-{s['subarray']}",
                          "cluster": str(cluster_key(s))} for s in sc]).set_index("id")


def paired(df_a, df_b, metric, meta, nb=10000, seed=0):
    """df_a / df_b：同一 split 的逐样本表（index = id）。返回 dict。"""
    ids = df_a.index.intersection(df_b.index)
    assert len(ids) == len(df_a) == len(df_b), "配对不完整"
    d = (df_a.loc[ids, metric] - df_b.loc[ids, metric]).to_frame("d").join(meta)
    cm = d.groupby("cond")["d"].mean()
    out = {"mean": float(cm.mean()), "pos": int((cm > 0).sum()), "n_cond": int(len(cm)),
           "per_cond": cm.round(4).to_dict(), "n_clusters": int(d["cluster"].nunique())}
    t = stats.t.ppf(0.975, len(cm) - 1) * cm.std(ddof=1) / np.sqrt(len(cm))
    out["t_lo"], out["t_hi"] = float(cm.mean() - t), float(cm.mean() + t)
    per = {c: g.groupby("cluster")["d"].agg(["sum", "count"]).to_numpy() for c, g in d.groupby("cond")}
    out["boot_ok"] = all(len(v) > 1 for v in per.values())
    if out["boot_ok"]:
        rng = np.random.default_rng(seed)
        bs = np.zeros(nb)
        for v in per.values():
            k = rng.integers(0, len(v), (nb, len(v)))
            bs += v[k, 0].sum(1) / v[k, 1].sum(1)
        bs /= len(per)
        out["b_lo"], out["b_hi"] = (float(x) for x in np.percentile(bs, [2.5, 97.5]))
    return out
