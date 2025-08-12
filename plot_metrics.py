#!/usr/bin/env python3

import sys, pathlib, numpy as np, pandas as pd, matplotlib.pyplot as plt
plt.rcParams.update({"figure.dpi": 300, "font.size": 13,
                     "axes.spines.top": False, "axes.spines.right": False})

# ------------------- CLI parsing -------------------
import argparse, pathlib, numpy as np, pandas as pd, matplotlib.pyplot as plt
plt.rcParams.update({"figure.dpi": 300, "font.size": 13})

parser = argparse.ArgumentParser(description="Plot training / test figures")
parser.add_argument("--log_csv",     required=True,  help="training_log.csv")
parser.add_argument("--metrics_npy", default=None,   help="test metrics .npy")
parser.add_argument("--out_dir",     default="figures")
args = parser.parse_args()

log_csv     = args.log_csv
metrics_npy = args.metrics_npy
out_dir     = pathlib.Path(args.out_dir)
out_dir.mkdir(exist_ok=True, parents=True)

# ------------------- colours -------------------
PURPLE_HEAVY = "#5a2ca0"  # smoothed train
PURPLE_LIGHT = "#fbbdf2"  # raw      train
BLUE_HEAVY   = "#138de5"  # smoothed val
BLUE_LIGHT   = "#c0e1fb"  # raw      val
VIOLET       = "#5a2ca0"  # scatter / hist :contentReference[oaicite:0]{index=0}

# ------------------- helpers -------------------
def ema(series, alpha=0.9):
    out, prev = np.empty_like(series, dtype=float), series.iloc[0]
    for i, v in enumerate(series):
        prev = alpha*prev + (1-alpha)*v
        out[i] = prev
    return pd.Series(out, index=series.index)

def plot_metric(tr, va, title, ylabel, fname, spe):
    tr_ep = tr.groupby("epoch")["value"].mean()
    va_ep = va.groupby("epoch")["value"].mean()
    plt.figure(figsize=(6,4))
    plt.plot(tr_ep.index, tr_ep,           color=PURPLE_LIGHT, lw=1,  label="train (raw)")
    plt.plot(tr_ep.index, ema(tr_ep),      color=PURPLE_HEAVY, lw=2.2,label="train (EMA)")
    if not va_ep.empty:
        plt.plot(va_ep.index, va_ep,       color=BLUE_LIGHT,   lw=1,  label="val (raw)")
        plt.plot(va_ep.index, ema(va_ep),  color=BLUE_HEAVY,   lw=2.2,label="val (EMA)")
    plt.xlabel("Epoch"); plt.ylabel(ylabel); plt.title(title); plt.legend()
    plt.tight_layout(); plt.savefig(out_dir/f"{fname}.png"); plt.close()

# load training log & derive epoch column
raw = pd.read_csv(log_csv)
loss_tr = raw.query('key=="training_loss"').copy()
dice_tr = raw.query('key=="train_dice"') .copy()
loss_va = raw.query('key=="val_loss"')   .copy()
dice_va = raw.query('key=="val_dice"')   .copy()

SPE = int(round(len(loss_tr) / len(dice_tr)))  # robust inference
loss_tr["epoch"] = loss_tr["step"] // SPE
for df in (dice_tr, loss_va, dice_va):
    df["epoch"] = df["step"]

# curves
plot_metric(loss_tr, loss_va, "Train vs Val Loss",  "Loss",  "loss_epoch",  SPE)
plot_metric(dice_tr, dice_va, "Train vs Val Dice",  "Dice",  "dice_epoch",  SPE)

# scatter + histogram from test metrics
if metrics_npy:
    a = np.load(metrics_npy)
    if a.shape[-1] < 5:
        raise ValueError(f"{metrics_npy} lacks TP/FN counts (shape {a.shape})")
    dice         = a[:,0,0]
    tumour_size  = a[:,0,2] + a[:,0,4]          # TP + FN

    # scatter (log-x)
    fig, ax = plt.subplots(figsize=(6,5))
    ax.scatter(tumour_size, dice, color=VIOLET, s=35, alpha=.85, edgecolors='none')
    ax.set_xscale('log')
    coef = np.polyfit(tumour_size, dice, 1)     # trend line
    order = np.argsort(tumour_size)
    ax.plot(tumour_size[order], np.polyval(coef, tumour_size[order]),
            '--', lw=2, color='black', alpha=.7)
    ax.set_xlabel("Tumour size"); ax.set_ylabel("Dice coefficient")
    ax.set_title("Test‑set performance vs tumour size")
    ax.grid(True, linestyle=':', alpha=.3)
    fig.tight_layout(); fig.savefig(out_dir/"dice_vs_tumoursize.png"); plt.close()

    # histogram
    bins = np.linspace(0.0, 1.0, 21)            # 20 buckets :contentReference[oaicite:2]{index=2}
    fig_h, ax_h = plt.subplots(figsize=(6,4.5))
    ax_h.hist(dice, bins=bins, color=VIOLET, edgecolor='white', alpha=.95)
    ax_h.set_xlabel("Dice coefficient"); ax_h.set_ylabel("Number of cases")
    ax_h.set_title("Distribution of Dice across test cases")
    ax_h.set_xlim(0,1); ax_h.grid(axis='y', linestyle=':', alpha=.6)
    fig_h.tight_layout(); fig_h.savefig(out_dir/"dice_histogram.png", dpi=300); plt.close()

print("✔  All plots saved to", out_dir)
