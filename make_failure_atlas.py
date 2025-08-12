#!/usr/bin/env python3
"""
make_failure_atlas.py
=====================

Generate a 12-panel failure-atlas (3 worst, 3 median, 3 best Dice) for Centre 1.

Usage
-----
python make_failure_atlas.py \
    --pred_dir   /path/to/predictions            \
    --gt_dir     /path/to/gt_root                \
    --results_np /path/to/test.npy               \
    --out_dir    /path/to/atlas_out
"""

import argparse, glob, os
from pathlib import Path

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

plt.rcParams["figure.dpi"] = 300        # high-res

# ------------------- CLI -------------------─
ap = argparse.ArgumentParser()
ap.add_argument("--pred_dir",   required=True,
                help="Folder with caseID.nii.gz predictions")
ap.add_argument("--gt_dir",     required=True,
                help="Root folder with caseID/seg.nii.gz and t2.nii.gz")
ap.add_argument("--results_np", required=True,
                help="*.npy from 5_bladder_compute_metrics.py")
ap.add_argument("--out_dir",    required=True,
                help="Destination for PNG panels")
args = ap.parse_args()

pred_dir   = Path(args.pred_dir).resolve()
gt_root    = Path(args.gt_dir).resolve()
out_dir    = Path(args.out_dir).resolve()
out_dir.mkdir(parents=True, exist_ok=True)

# ------------------- helper: IO + orientation -------------------
def load_lps(path):
    """Load NIfTI as LPS-oriented ndarray (z,y,x)."""
    img = nib.load(str(path))
    img = nib.as_closest_canonical(img)           # -> LPS, preserves spacing
    return img.get_fdata(), img.header.get_zooms()

def best_slice(mask):
    """Return index of axial slice with max tumour area."""
    areas = mask.sum(axis=(0, 1))
    return int(areas.argmax())

def to_rgba(mask, rgb, alpha=0.35):
    h, w = mask.shape
    layer = np.zeros((h, w, 4), dtype=np.float32)
    layer[..., :3] = rgb
    layer[..., 3]  = mask * alpha
    return layer

# ------------------- load ranking data -------------------
case_ids = sorted(Path(p).name.replace(".nii.gz", "")          # 1113.nii.gz → 1113
                  for p in glob.glob(str(pred_dir / "*.nii.gz")))


results  = np.load(args.results_np)          # (N, variants, 5)
dice_raw = results[:, 0, 0]                  # variant 0 = raw Dice



# --------- fixed list of Centre 1 cases to compare across models -------------------
fixed_cases = ["1113", "1076", "1026", "1112", "1097", "1091", "1108", "1043", "1018"]

# build a lookup from case-ID -> Dice (variant 0)
dice_lookup = {cid: dice_raw[i] for i, cid in enumerate(case_ids)}

# keep only those IDs that actually exist in the current prediction folder
sel = [cid for cid in fixed_cases if cid in dice_lookup]

if len(sel) < len(fixed_cases):
    missing = set(fixed_cases) - set(sel)
    print("Warning: missing predictions for", ", ".join(missing))



# ------------------- main loop -------------------

for cid in sel:                                 # sel already holds the IDs
    dice = float(dice_lookup[cid])              # look-up Dice by ID

    # paths
    pr_path  = pred_dir / f"{cid}.nii.gz"
    gt_path  = gt_root / cid / "seg.nii.gz"
    img_path = gt_root / cid / "t2.nii.gz"

    # load & orient
    pred, _      = load_lps(pr_path)
    gt, spacing  = load_lps(gt_path)
    img, _       = load_lps(img_path)

    pred = (pred > 0.5).astype(np.uint8)
    gt   = (gt   > 0.5).astype(np.uint8)

    # resample prediction if shape mismatch after LPS orient
    if pred.shape != gt.shape:
        zoom_f = np.asarray(gt.shape) / np.asarray(pred.shape)
        pred   = zoom(pred.astype(float), zoom_f, order=0).astype(np.uint8)

    # resample T2 to seg grid if needed (keeps header logic simple)
    if img.shape != gt.shape:
        zoom_f = np.asarray(gt.shape) / np.asarray(img.shape)
        img    = zoom(img, zoom_f, order=1)

    # choose slice
    z = best_slice(gt)
    sl_img  = img[..., z]
    sl_gt   = gt [..., z]
    sl_pred = pred[..., z]

    # normalise to [0,1]
    sl_img = (sl_img - sl_img.min()) / (np.ptp(sl_img) + 1e-6)

    # ------------------- plotting -------------------
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(sl_img, cmap="gray")
    ax.imshow(to_rgba(sl_gt,   (0.00, 1.00, 0.00), alpha=0.9))   # green
    ax.imshow(to_rgba(sl_pred, (0.55, 0.00, 0.55), alpha=0.45))   # purple
    ax.axis("off")
    ax.set_title(f"{cid}  •  Dice = {dice:.3f}", fontsize=8, pad=2)
    fig.tight_layout(pad=0)

    fname = f"{int(round(dice * 10000)):04d}.png"      # 0.4343 -> 4343.png
    fig.savefig(out_dir / fname, bbox_inches="tight")
    plt.close(fig)

print(f"Saved {len(sel)} panels → {out_dir}")
