#!/usr/bin/env python3
import csv, numpy as np, itertools
from pathlib import Path
from collections import defaultdict
from math import sqrt
from scipy.stats import wilcoxon, rankdata

EXP_ROOT = Path("/workspace/experiment_root").resolve()
SUMMARY  = EXP_ROOT / "grid_results.csv"
STATS    = EXP_ROOT / "statistical_tests.csv"

def read_grid(path: str):
    """
    Try reading grid_results.csv first with ',' delimiter, then with ';'.
    Returns list[dict].
    """
    for delim in (",", ";"):
        with open(path, newline="") as f:
            reader = csv.DictReader(f, delimiter=delim)
            if "val_dice" in reader.fieldnames:
                return list(reader)
    raise RuntimeError(
        f"'test_dice' column not found with either ',' or ';' in {path}"
    )

rows = read_grid(SUMMARY)

# ---- load summary ------------------------------------------------------
rows.sort(key=lambda r: float(r["val_dice"]), reverse=True)

with open(SUMMARY, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(rows)
print("Re-sorted summary CSV (best Dice on top)")

# ---- statistical tests -------------------------------------------------
GRID_IDX = [int(r["comb_idx"]) for r in rows]
metrics  = {idx: np.load(EXP_ROOT / f"comb_{idx:02d}/metrics/val.npy")[:, 0, :2]
            for idx in GRID_IDX}

meta = [{k: r[k] for k in ("loss", "optim", "sched", "lr")} | {"idx": int(r["comb_idx"])}
        for r in rows]

# factor -> level -> vector
agg = defaultdict(lambda: defaultdict(dict))
for m_idx, (metric_name, col) in enumerate([("dice", 0), ("hd95", 1)]):
    for fac in ("loss", "optim", "sched", "lr"):
        for r in meta:
            lvl = r[fac]
            vec = metrics[r["idx"]][:, col]
            agg[(metric_name, fac)].setdefault(lvl, []).append(vec)
        # average replicas per patient
        for lvl in agg[(metric_name, fac)]:
            agg[(metric_name, fac)][lvl] = np.mean(agg[(metric_name, fac)][lvl], axis=0)

out_rows = []
for (metric, fac), levels in agg.items():
    lvls = sorted(levels)
    for a, b in itertools.combinations(lvls, 2):
        da, db = levels[a], levels[b]
        diff = db - da
        nz_mask = diff != 0
        if np.count_nonzero(nz_mask) == 0:
            p = 1.0
            z = 0.0
            r_rb = 0.0
            r_rosenthal = 0.0
            dmed = float(np.median(diff))
        else:
            diff_nz = diff[nz_mask]
            res = wilcoxon(diff_nz, alternative="two-sided", zero_method="wilcox", correction=True, mode="auto")
            p = float(res.pvalue)
            n = diff_nz.size
            # ranks for RBC
            ranks = rankdata(np.abs(diff_nz), method="average")
            R_tot = n * (n + 1) / 2.0
            Wplus = float(np.sum(ranks[diff_nz > 0]))
            Wminus = float(np.sum(ranks[diff_nz < 0]))
            r_rb = (Wplus - Wminus) / R_tot  # rank-biserial correlation in [-1, 1]
            # z from Wplus (not from p)
            mean_W = R_tot / 2.0
            var_W  = n * (n + 1) * (2 * n + 1) / 24.0
            cc = 0.5 * np.sign(Wplus - mean_W)  # continuity correction
            z = ((Wplus - mean_W) - cc) / np.sqrt(var_W)
            r_rosenthal = z / sqrt(n)
            dmed = float(np.median(diff))

        if metric == "hd95":
            delta_med_str = f"{dmed:+.2f} mm"
        else:
            delta_med_str = f"{dmed:+.4f}"

        out_rows.append({
            "metric": metric,
            "comparison": f"{fac}: {b} vs {a}",
            "delta_med": delta_med_str,
            "wilcoxon_Z": f"{z:.3f}",
            "r_effect": f"{r_rb:.3f}",           # rank-biserial correlation
            "r_rosenthal": f"{r_rosenthal:.3f}", 
            "p_raw": p
        })

# Holm-Bonferroni (step-down, monotone)
for metric in ("dice", "hd95"):
    idxs = [i for i, r in enumerate(out_rows) if r["metric"] == metric]
    if not idxs:
        continue
    ps = np.array([float(out_rows[i]["p_raw"]) for i in idxs])
    order = np.argsort(ps)
    m = len(ps)
    adj = np.empty_like(ps, dtype=float)
    running_max = 0.0
    for rank, oidx in enumerate(order, 1):
        val = min(ps[oidx] * (m - rank + 1), 1.0)
        running_max = max(running_max, val)
        adj[oidx] = running_max
    for i, padj in zip(idxs, adj):
        out_rows[i]["p_holm"] = f"{padj:.6g}"

fields = ["metric", "comparison", "delta_med", "wilcoxon_Z", "r_effect", "r_rosenthal", "p_raw", "p_holm"]
with open(STATS, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for r in out_rows:
        # ensure all keys present
        for k in fields:
            r.setdefault(k, "")
        w.writerow(r)
print("Statistical tests written to", STATS)
