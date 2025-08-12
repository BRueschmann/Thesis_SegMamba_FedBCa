#!/usr/bin/env python3
"""
Compute Dice and HD95 for binary bladder-tumour masks.
"""

import argparse, glob, os, csv
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from medpy import metric
from tqdm import tqdm
from light_training.debug_utils import dbg

from skimage.morphology import remove_small_objects, binary_closing, ball
from scipy.ndimage         import binary_fill_holes

# ------------------- CLI -------------------─
parser = argparse.ArgumentParser()
parser.add_argument("--pred_dir", required=True,
                    help="Folder with <case>.nii.gz predictions")
parser.add_argument("--gt_dir",   required=True,
                    help="Root folder with <case>/seg.nii.gz ground-truth")
parser.add_argument("--out_dir",  default=None,
                    help="Where to save the .npy metrics "
                         "(default: ../predictions/result_metrics)")
parser.add_argument("--csv",      default=None,
                    help="Optional path to write per-case CSV")
parser.add_argument("--inspect",  type=int, default=0,
                    metavar="N", help="Print full headers for first N cases")
# legacy flags still work
parser.add_argument("--show-header-diffs", action="store_true",
                    help="(kept for backward-compat) identical to default behaviour")
parser.add_argument("--diff-limit", type=int, default=None,
                    help="Cap number of resample lines (default: unlimited)")
args = parser.parse_args()

# ------------------- helper fns -------------------
def to_bin(a): return (a > 0).astype(np.uint8)

def dice_hd95(gt, pred, spacing):
    if gt.sum() == 0 and pred.sum() == 0:
        return 1.0, 0.0
    d = metric.binary.dc(pred, gt)
    try:
        h = metric.binary.hd95(pred, gt, voxelspacing=spacing)
    except RuntimeError:
        h = np.inf
    return d, h

# ------------------- voxel confusion counts -------------------
def confusion(gt, pred):
    """Return TP, FP, FN voxel counts (TN ignored)."""
    tp = np.logical_and(pred == 1, gt == 1).sum()
    fp = np.logical_and(pred == 1, gt == 0).sum()
    fn = np.logical_and(pred == 0, gt == 1).sum()
    return tp, fp, fn

# ------------------- post-processing fns ──────────────────
def _min_vox_for_volume(img_spacing, ml=0.1):
    """Return #voxels that correspond to <ml> millilitres."""
    vox_vol_mm3 = np.prod(img_spacing)              # mm^3
    return max(1, int(round(ml * 1000 / vox_vol_mm3)))

def rso(mask, min_vox):
    """Remove small objects (<min_vox) – keeps ≥ min_vox components."""
    return remove_small_objects(mask.astype(bool), min_size=min_vox).astype(np.uint8)

def closing(mask, radius=1):
    """3‑D binary closing + hole filling."""
    closed = binary_closing(mask.astype(bool), ball(radius))
    filled = binary_fill_holes(closed)
    return filled.astype(np.uint8)


def ensure_3d(img):
    if img.GetDimension() == 2:
        img = sitk.JoinSeries(img)
        img.SetSpacing(img.GetSpacing() + (1.0,))
        img.SetDirection((1,0,0, 0,1,0, 0,0,1))
    return img

# re-orient every image to LPS so directions always match
def to_lps(img):
    orienter = sitk.DICOMOrientImageFilter()
    orienter.SetDesiredCoordinateOrientation("LPS")
    return orienter.Execute(img)

def resample_to_ref(mov, ref):
    mov = ensure_3d(mov)
    ref = ensure_3d(ref)
    return sitk.Resample(mov, ref, sitk.Transform(),
                         sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)

def hdr(img):
    return (img.GetSize(), img.GetSpacing(),
            img.GetOrigin(), img.GetDirection())

def hdr_mismatch(pred_img, gt_img):
    diffs = []
    if pred_img.GetSize()      != gt_img.GetSize():
        diffs.append("SIZE")
    if tuple(np.round(pred_img.GetSpacing(),5)) != tuple(np.round(gt_img.GetSpacing(),5)):
        diffs.append("SPACING")
    if any(abs(np.array(pred_img.GetOrigin()) - np.array(gt_img.GetOrigin())) > 1e-3):
        diffs.append("ORIGIN")
    if any(abs(np.array(pred_img.GetDirection()) - np.array(gt_img.GetDirection())) > 1e-5):
        diffs.append("DIRECTION")
    return "/".join(diffs) if diffs else "NONE"

# ------------------- path handling -------------------
pred_root = Path(args.pred_dir).resolve()
gt_root   = Path(args.gt_dir).resolve()
if not pred_root.is_dir(): raise FileNotFoundError(pred_root)
if not gt_root.is_dir():   raise FileNotFoundError(gt_root)

out_dir = (Path(args.out_dir) if args.out_dir
           else pred_root.parent / "result_metrics")
out_dir.mkdir(parents=True, exist_ok=True)

case_ids = sorted(Path(p).name.removesuffix(".nii.gz")
                  for p in glob.glob(str(pred_root / "*.nii.gz")))
if not case_ids:
    raise RuntimeError(f"No predictions found in {pred_root}")

VARIANTS = ("raw", "rso", "close", "close+rso")
results  = np.zeros((len(case_ids), len(VARIANTS), 5), dtype=np.float32)

per_case_table  = []        # [(case,dice,hd95,resample_reason), ...]
resample_cases  = []

# ------------------- loop -------------------
for i, case in enumerate(tqdm(case_ids, desc="Cases")):
    gt_path   = gt_root / case / "seg.nii.gz"
    pred_path = pred_root / f"{case}.nii.gz"
    if not gt_path.exists():   raise FileNotFoundError(gt_path)
    if not pred_path.exists(): raise FileNotFoundError(pred_path)

    # read -> ensure 3D -> orient to LPS
    gt_img   = to_lps(ensure_3d(sitk.ReadImage(str(gt_path))))
    pred_img = to_lps(ensure_3d(sitk.ReadImage(str(pred_path))))

    # handle 2-channel vector predictions
    if pred_img.GetNumberOfComponentsPerPixel() > 1:
        pred_img = sitk.VectorIndexSelectionCast(pred_img, 1)
        pred_img = sitk.Cast(pred_img, sitk.sitkUInt8)

    if args.inspect and i < args.inspect:
        print(f"\n[{case}] GT hdr : {hdr(gt_img)}")
        print(f"[{case}] PR hdr : {hdr(pred_img)}")

    resample_reason = "no resample"
    if hdr(gt_img) != hdr(pred_img):
        resample_reason = hdr_mismatch(pred_img, gt_img)
        pred_img        = resample_to_ref(pred_img, gt_img)
        assert hdr(gt_img) == hdr(pred_img)
        resample_cases.append((case, resample_reason))


    gt_arr   = to_bin(sitk.GetArrayFromImage(gt_img))
    pred_raw = to_bin(sitk.GetArrayFromImage(pred_img))

    min_vox = _min_vox_for_volume(gt_img.GetSpacing(), ml=0.015)

    # ------- build the four masks -------
    pred_rso        = rso(pred_raw, min_vox)
    pred_close      = closing(pred_raw, radius=1)
    pred_close_rso  = rso(pred_close, min_vox)

    preds = [pred_raw, pred_rso, pred_close, pred_close_rso]

    # ------- metric loop -------
    case_scores = {}
    for v_idx, (variant, pred_arr) in enumerate(zip(VARIANTS, preds)):
        d, h = dice_hd95(gt_arr, pred_arr, gt_img.GetSpacing()[::-1])
        h = np.nan if np.isinf(h) else h
        tp, fp, fn = confusion(gt_arr, pred_arr)

        results[i, v_idx, :] = (d, h, tp, fp, fn)
        case_scores[variant] = (d, h, tp, fp, fn)

    per_case_table.append((case, case_scores, resample_reason))



# ------------------- save + report -------------------
np_path = out_dir / f"{pred_root.name}.npy"
np.save(np_path, results)

print("\n--------------- Per-case results ---------------")
hdr_fmt = f"{'case':>15} |  variant |  Dice   HD95(mm) |   TP        FP        FN"
print(hdr_fmt)
for case, var_dict, reason in per_case_table:
    tag = f"[{reason}]" if reason != "no resample" else ""
    for v, (d, h, tp, fp, fn) in var_dict.items():
        print(f"{case:>15}: {v:>10}  {d:6.3f}  {h:6.2f}   | "
              f"{tp:7d}  {fp:7d}  {fn:7d}  {tag}")
if resample_cases:
    limit = args.diff_limit or len(resample_cases)
    print("\nCases resampled (reason):")
    for case, reason in resample_cases[:limit]:
        print(f"  {case}  ->  {reason}")
    if len(resample_cases) > limit:
        print(f"... {len(resample_cases) - limit} more truncated")

print(f"\nSaved per-case array -> {np_path}")
print(f"Total cases          : {len(case_ids)}")
print(f"Cases resampled      : {len(resample_cases)}")
print(f"Mean Dice            : {results[:,0,0].mean():.4f}")
print(f"Mean HD95 (mm)       : {np.nanmean(results[:,0,1]):.2f}")


print("\n--------------- Mean over all cases ---------------")
for v_idx, v in enumerate(VARIANTS):
    d_mean  = results[:, v_idx, 0].mean()
    h_mean  = np.nanmean(results[:, v_idx, 1])
    tp_mean, fp_mean, fn_mean = results[:, v_idx, 2:].mean(axis=0)
    print(f"{v:>10}:  Dice={d_mean:.4f}  HD95={h_mean:.2f} mm  "
          f"TP¯={tp_mean:7.0f}  FP¯={fp_mean:7.0f}  FN¯={fn_mean:7.0f}")


# ---------- per-centre means ----------
centre_tbl = {}
for case, var_dict, _ in per_case_table:      # new tuple layout
    cid = case[0]                             # first char = centre id
    d, h, tp, fp, fn = var_dict["raw"]        # centre stats use RAW variant
    centre_tbl.setdefault(cid, []).append((d, h, tp, fp, fn))

print("\n-------------- Per‑centre mean ---------")
d_means, h_means = [], []
tp_means, fp_means, fn_means = [], [], []
for cid in sorted(centre_tbl):
    arr     = np.array(centre_tbl[cid])
    d_mean = arr[:,0].mean()
    h_mean = np.nanmean(arr[:,1])
    tp_mean, fp_mean, fn_mean = arr[:,2:].mean(axis=0)
    d_means.append(d_mean)
    h_means.append(h_mean)
    tp_means.append(tp_mean); fp_means.append(fp_mean); fn_means.append(fn_mean)

    print(f"Centre {cid}: Dice={d_mean:.4f}  HD95={h_mean:.2f} mm  "
          f"TP¯={tp_mean:7.0f}  FP¯={fp_mean:7.0f}  FN¯={fn_mean:7.0f}  "
          f"(n={len(arr)})")
          
print(f"Macro‑avg ︱ Dice={np.mean(d_means):.4f}  HD95={np.nanmean(h_means):.2f} mm  "
      f"TP¯={np.mean(tp_means):.0f}  FP¯={np.mean(fp_means):.0f}  "
      f"FN¯={np.mean(fn_means):.0f}  (centres={len(d_means)})")


# optional CSV
if args.csv:
    with open(args.csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case_id", "variant", "dice", "hd95_mm",
                    "tp_vox", "fp_vox", "fn_vox", "resample"])
        for case, var_dict, reason in per_case_table:
            for v, (d, h, tp, fp, fn) in var_dict.items():
                w.writerow([case, v, d, h, tp, fp, fn, reason])
    print(f"Per‑case CSV written → {args.csv}")