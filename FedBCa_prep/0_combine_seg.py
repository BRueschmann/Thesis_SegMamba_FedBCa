#!/usr/bin/env python3
# combine_seg.py
"""
Union-combine multiple tumour masks per patient into one seg.nii.gz.

Assumes current layout:
    root/
        Center1/Annotation/*.nii.gz
        Center2/Annotation/*.nii.gz
        ...

File naming inside Annotation:
    075_1.nii.gz, 075_2.nii.gz 
    001.nii.gz

Result:
    For each patient, writes <PatientID>.nii.gz
    (075.nii.gz in the example) to the SAME Annotation folder,
    overwriting if --override is set.

Usage:
    python combine_seg.py /path/to/dataset/root [--override] [--keep-originals]
"""

import argparse, glob, os, re, sys, shutil
from collections import defaultdict

import nibabel as nib
import numpy as np


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="Dataset root that contains Center1, Center2, ...")
    ap.add_argument("--override", action="store_true",
                    help="Overwrite an existing combined mask if present")
    ap.add_argument("--keep-originals", action="store_true",
                    help="Keep the individual *_1.nii.gz files instead of deleting them")
    return ap.parse_args()


def group_masks(ann_dir):
    """
    Returns a dict {patient_id: [file1, file2, ...]}
    patient_id is the part before the first underscore,
    or the full name without extension if no underscore.
    """
    groups = defaultdict(list)
    for f in glob.glob(os.path.join(ann_dir, "*.nii*")):
        base = os.path.basename(f)
        pid = base.split("_")[0].split(".")[0]  # "075_1.nii.gz" -> "075"
        groups[pid].append(f)
    return groups


def combine_one(out_path, files, override):
    if os.path.exists(out_path) and not override:
        return "skip (exists)"

    first_img = nib.load(files[0])
    combined = np.zeros(first_img.shape, dtype=np.uint8)

    for f in files:
        data = nib.load(f).get_fdata()
        combined |= (data > 0).astype(np.uint8)

    nib.save(nib.Nifti1Image(combined, first_img.affine, first_img.header), out_path)
    return "written"


def main():
    args = parse_args()
    root = args.root

    centers = sorted(glob.glob(os.path.join(root, "Center*")))
    if not centers:
        sys.exit("No Center* directories found under {}".format(root))

    for c in centers:
        ann_dir = os.path.join(c, "Annotation")
        if not os.path.isdir(ann_dir):
            print(f"[warn] Missing Annotation folder in {c}")
            continue

        groups = group_masks(ann_dir)
        for pid, files in groups.items():
            if len(files) == 1:
                single = files[0]
                target = os.path.join(ann_dir, f"{pid}.nii.gz")
                if os.path.basename(single) != os.path.basename(target):
                    shutil.copy2(single, target)
                    status = "copied"
                else:
                    status = "ok"
            else:
                target = os.path.join(ann_dir, f"{pid}.nii.gz")
                status = combine_one(target, files, args.override)

            print(f"{os.path.relpath(target, root):<40}  {status}")

        if not args.keep_originals:
            # Remove any *_1.nii.gz etc.
            for f in glob.glob(os.path.join(ann_dir, "*_*.*")):
                os.remove(f)

    print("Done")


if __name__ == "__main__":
    main()