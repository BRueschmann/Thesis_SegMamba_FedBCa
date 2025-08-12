#!/bin/bash
#SBATCH --job-name=bladder_grid_eval
#SBATCH --partition=small-long
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --chdir=/sharedscratch/br61/thesis/slurm_scripts
#SBATCH --output=/sharedscratch/br61/thesis/MVA_SegMamba/logs/eval_%j.out
#SBATCH --error=/sharedscratch/br61/thesis/MVA_SegMamba/logs/eval_%j.err

# ---------- user parameters ----------------------------------------------
EXP_ROOT=/sharedscratch/br61/thesis/Experiments/bladder_grid_new
APPTAINER=/software/apptainer/bin/apptainer
IMG=/sharedscratch/br61/containers/segmamba_wheels.sif
REPO=/sharedscratch/br61/thesis/MVA_SegMamba
# -------------------------------------------------------------------------

${APPTAINER} exec --cleanenv \
    --bind ${REPO}:/workspace \
    --bind ${EXP_ROOT}:/workspace/experiment_root \
    ${IMG} bash -lc "
        pip install --quiet --no-cache-dir scipy
        python /workspace/evaluate_grid.py
    "
