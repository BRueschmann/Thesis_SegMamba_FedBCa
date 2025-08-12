#!/usr/bin/env python3
# reorganise_fedbca.py  (v3 – fixed .nii/.nii.gz base extraction)

"""
Re-organise FedBCa data into the structure expected by the training pipeline.
After running combine_seg.py 

Resulting layout (inside --out):
    Center1/1001/{t2.nii.gz, seg.nii.gz}
    Center1/1002/{...}
    Center2/2001/{...}
    ...
    all_centers/1001/{...}  (duplicates, for a single pooled dataset)

Usage:
    python reorganise_fedbca.py /path/to/FedBCa_raw /path/to/FedBCa_clean [--verify] [--seg-suffix _seg]
"""

import argparse
import glob
import os
import shutil
import sys

import nibabel as nib


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="Existing dataset root (Center*/ subdirs)")
    ap.add_argument("out", help="Output root where new structure is created")
    ap.add_argument("--verify", action="store_true",
                    help="Check that image and seg shapes match before copying")
    ap.add_argument("--seg-suffix", default="",
                    help='Suffix between patient id and ".nii.gz" in Annotation '
                         '(e.g. "_seg" if files are "075_seg.nii.gz")')
    return ap.parse_args()


def centre_number(name):
    """Extract integer from 'CenterX' → X."""
    return int("".join(filter(str.isdigit, name)))


def copy_pair(t2_path, seg_path, dest_dir, verify):
    """Copy t2 and seg into dest_dir as t2.nii.gz / seg.nii.gz."""
    os.makedirs(dest_dir, exist_ok=True)
    if verify:
        img = nib.load(t2_path)
        seg = nib.load(seg_path)
        if img.shape != seg.shape:
            print(f"[ERROR] Shape mismatch; skipping {os.path.basename(t2_path)}")
            return False
    shutil.copy2(t2_path, os.path.join(dest_dir, "t2.nii.gz"))
    shutil.copy2(seg_path, os.path.join(dest_dir, "seg.nii.gz"))
    return True


def main():
    args = parse_args()
    root = os.path.abspath(args.root)
    out_root = os.path.abspath(args.out)

    centers = sorted(glob.glob(os.path.join(root, "Center*")))
    if not centers:
        sys.exit(f"No Center* directories found under {root}")

    all_dir = os.path.join(out_root, "all_centers")
    os.makedirs(all_dir, exist_ok=True)

    total_copied = 0
    for c_path in centers:
        cname = os.path.basename(c_path)        # e.g. "Center1"
        cnum = centre_number(cname)             # e.g. 1
        t2_dir = os.path.join(c_path, "T2WI")
        ann_dir = os.path.join(c_path, "Annotation")

        if not os.path.isdir(t2_dir) or not os.path.isdir(ann_dir):
            print(f"[WARN] {cname}: missing T2WI or Annotation – skipped")
            continue

        center_out = os.path.join(out_root, cname)
        os.makedirs(center_out, exist_ok=True)

        copied_this_center = 0
        for t2_path in sorted(glob.glob(os.path.join(t2_dir, "*.nii*"))):
            fname = os.path.basename(t2_path)
            # robust base extraction:
            if fname.lower().endswith(".nii.gz"):
                base = fname[:-7]
            elif fname.lower().endswith(".nii"):
                base = fname[:-4]
            else:
                base, _ = os.path.splitext(fname)

            seg_name = f"{base}{args.seg_suffix}.nii.gz"
            seg_path = os.path.join(ann_dir, seg_name)
            if not os.path.exists(seg_path):
                print(f"[WARN] {cname}: no seg for {fname} → looked for {seg_name}")
                continue

            # build new patient ID: CenterX + zero-padded base
            pid = f"{cnum}{int(base):03d}"

            dest_center = os.path.join(center_out, pid)
            if copy_pair(t2_path, seg_path, dest_center, args.verify):
                # also copy to pooled all_centers
                dest_all = os.path.join(all_dir, pid)
                copy_pair(t2_path, seg_path, dest_all, False)

                copied_this_center += 1
                total_copied += 1

        print(f"{cname}: {copied_this_center} cases copied.")

    print(f"\nRe-organisation complete. Total cases copied: {total_copied}")
    print(f"New root directory: {out_root}")
    print("Now point your preprocessing script to:")
    print(f"  --base_dir \"{os.path.join(out_root, 'all_centers')}\"")
    print("with data_filenames=[\"t2.nii.gz\"], seg_filename=\"seg.nii.gz\".")


if __name__ == "__main__":
    main()