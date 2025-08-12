#!/bin/bash
#SBATCH --job-name=atlas_C2_to_C1
#SBATCH --partition=small-long
#SBATCH --time=01:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --chdir=/sharedscratch/br61/thesis/slurm_scripts
#SBATCH --output=/sharedscratch/br61/thesis/MVA_SegMamba/logs/atlas_C2_to_C1_%j.out
#SBATCH --error=/sharedscratch/br61/thesis/MVA_SegMamba/logs/atlas_C2_to_C1_%j.err

APPTAINER=/software/apptainer/bin/apptainer
REPO=/sharedscratch/br61/thesis/MVA_SegMamba
IMG=/sharedscratch/br61/containers/segmamba_wheels.sif

# ── experiment‑specific paths ────────────────────────────────────────────────
GT_DIR=/sharedscratch/br61/thesis/data/FedBCa_clean/center1_raw
EXP_ROOT=/sharedscratch/br61/thesis/Experiments/Experiment_1556730_C2_external
PRED_DIR=${EXP_ROOT}/predictions/all           # contains all centres
RESULTS_NP=${EXP_ROOT}/metrics/all.npy         # contains all centres
OUT_DIR=${EXP_ROOT}/metrics/figures
mkdir -p "${OUT_DIR}"

"$APPTAINER" exec --nv --cleanenv \
    --bind ${REPO}:/workspace \
    --bind ${GT_DIR}:/workspace/data/gt \
    --bind ${EXP_ROOT}:/workspace/experiment \
    "$IMG" \
    bash -lc "set -euo pipefail
              python /workspace/make_failure_atlas.py \
                     --pred_dir   /workspace/experiment/predictions/all \
                     --gt_dir     /workspace/data/gt \
                     --results_np /workspace/experiment/metrics/all.npy \
                     --out_dir    /workspace/experiment/metrics/figures"
