from light_training.preprocessing.preprocessors.preprocessor_mri import (
    MultiModalityPreprocessor,
)

import numpy as np, pickle, json, time, csv           # time & csv added
from pathlib import Path

# ------------------------------------------------------------------
# 1) immutable config
# ------------------------------------------------------------------
data_filename = ["t2.nii.gz"]
seg_filename  = "seg.nii.gz"

import argparse
# ---------------------------------------------------------------------
# Command-line arguments
# ---------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Pre-process FedBCa bladder dataset for SegMamba")
parser.add_argument(
    "--input_dir", default="all_centers",
    help="Folder *inside* /workspace/data/FedBCa_clean/ that contains the raw "
         "cases to convert (e.g. 'centre_1').")
parser.add_argument(
    "--output_dir",
    default="/workspace/data/FedBCa_clean/full_logs_all_centers_preprocessed",
    help="Destination folder for the .npz cases and properties.pkl files.")
parser.add_argument(
    "--out_spacing", nargs=3, type=float, metavar=("Z", "Y", "X"),
    default=[3.5, 0.75, 0.75],
    help="Target voxel spacing in mm, order Z Y X.")
args = parser.parse_args()

base_dir  = "/workspace/data/FedBCa_clean"   # root that holds all centres

# Use the parsed values throughout the script
image_dir   = args.input_dir          # raw images live here
output_dir  = args.output_dir         # pre-processed dataset goes here
out_spacing = args.out_spacing        # list of three floats

# ------------------------------------------------------------------
# 2) helpers
# ------------------------------------------------------------------
def make_preprocessor() -> MultiModalityPreprocessor:
    return MultiModalityPreprocessor(
        base_dir=base_dir,
        image_dir=image_dir,
        data_filenames=["t2.nii.gz"],
        seg_filename="seg.nii.gz",
    )

# Make every slice width/height even
def pad_even_xy_dataset(directory: str) -> None:
    """
    Pads each .npz in `directory` so Y and X are even.
    Adds/updates properties["even_pad_xy"] = (pad_y, pad_x).
    """
    for npz_path in Path(directory).glob("*.npz"):
        with np.load(npz_path, allow_pickle=True) as npz:
            data = npz["data"]          # (C, Z, Y, X)
            seg  = npz["seg"]           # (Z, Y, X)  or  (1, Z, Y, X)
            props = npz["properties"].item() if "properties" in npz.files else {}

        _, z, y, x = data.shape
        pad_y = y % 2
        pad_x = x % 2
        if pad_y == 0 and pad_x == 0:
            continue

        data = np.pad(data,
                      ((0, 0), (0, 0), (0, pad_y), (0, pad_x)),
                      mode="constant")
        if seg.ndim == 3:
            seg = np.pad(seg,
                         ((0, 0), (0, pad_y), (0, pad_x)),
                         mode="constant")
        else:
            seg = np.pad(seg,
                         ((0, 0), (0, 0), (0, pad_y), (0, pad_x)),
                         mode="constant")

        props["even_pad_xy"] = (pad_y, pad_x)

        np.savez_compressed(npz_path, data=data, seg=seg, properties=props)
        print(f"[pad_even_xy] padded {npz_path.name}: +({pad_y},{pad_x})")

# ------------------------------------------------------------------
# 3) entry points
# ------------------------------------------------------------------
def process_train() -> None:
    preprocessor = make_preprocessor()
    preprocessor.run(
        output_spacing = out_spacing,
        output_dir     = output_dir,
        all_labels     = [1],
    )
    pad_even_xy_dataset(output_dir)

def plan() -> None:
    preprocessor = make_preprocessor()
    preprocessor.run_plan()

# ------------------------------------------------------------------
# 4) timed main
# ------------------------------------------------------------------
if __name__ == "__main__":
    timings = []

    t0 = time.perf_counter()
    plan()
    timings.append(("plan", time.perf_counter() - t0))

    t0 = time.perf_counter()
    process_train()
    timings.append(("process_train", time.perf_counter() - t0))

    total = sum(t for _, t in timings)
    print("\n-------------- Pre-processing timing --------------")
    for stage, sec in timings:
        print(f"{stage:<15} : {sec:7.2f} s")
    print(f"{'TOTAL':<15} : {total:7.2f} s")

    # save CSV next to processed data
    csv_path = Path(output_dir) / "preprocessing_time.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["stage", "seconds"])
        w.writerows(timings)
    print(f"\nTimings written to: {csv_path}")
