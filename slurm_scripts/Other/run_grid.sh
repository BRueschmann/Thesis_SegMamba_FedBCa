#!/bin/bash
#SBATCH --job-name=bladder_grid_p${PART}
#SBATCH --partition=gpu.L40S
#SBATCH --gres=gpu:2
#SBATCH --time=48:00:00
#SBATCH --mem=40G
#SBATCH --chdir=/sharedscratch/br61/thesis/slurm_scripts
#SBATCH --output=/sharedscratch/br61/thesis/MVA_SegMamba/logs/grid_p${PART}_%j.out
#SBATCH --error=/sharedscratch/br61/thesis/MVA_SegMamba/logs/grid_p${PART}_%j.err

# ----- user parameters -------------------------------------------------
PART=${PART:-0}      # export when submitting: 0 / 1 / 2
NPARTS=3
EXP_ROOT=/sharedscratch/br61/thesis/Experiments/bladder_grid_new
EPOCHS=200
# -----------------------------------------------------------------------

mkdir -p "${EXP_ROOT}"

APPTAINER=/software/apptainer/bin/apptainer
IMG=/sharedscratch/br61/containers/segmamba_wheels.sif
REPO=/sharedscratch/br61/thesis/MVA_SegMamba
TRAIN_DIR=/sharedscratch/br61/thesis/data/FedBCa_clean/center1_processed_063
GT_DIR=/sharedscratch/br61/thesis/data/FedBCa_clean/center1_raw

${APPTAINER} exec --nv --cleanenv \
    --bind ${REPO}:/workspace \
    --bind ${TRAIN_DIR}:/workspace/data/train \
    --bind ${GT_DIR}:/workspace/data/gt \
    --bind ${EXP_ROOT}:/workspace/experiment_root \
    ${IMG} bash -lc "
        pip install --quiet --no-cache-dir 'acvl-utils>=0.2.1' nibabel SimpleITK scipy==1.13.0
        python /workspace/grid_runner.py --part ${PART} --nparts ${NPARTS} --epoch ${EPOCHS}
    "
