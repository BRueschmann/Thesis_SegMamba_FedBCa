#!/bin/bash
#SBATCH --job-name=bladder_figures
#SBATCH --partition=small-long
#SBATCH --time=00:20:00             # plenty for plotting ~40 kB of CSV
#SBATCH --mem=8G
#SBATCH --chdir=/sharedscratch/br61/thesis/slurm_scripts
#SBATCH --output=/sharedscratch/br61/thesis/MVA_SegMamba/logs/fig_%j.out
#SBATCH --error=/sharedscratch/br61/thesis/MVA_SegMamba/logs/fig_%j.err

APPTAINER=/software/apptainer/bin/apptainer

########## host‑side paths (unchanged) ########################################
REPO=/sharedscratch/br61/thesis/MVA_SegMamba
IMG=/sharedscratch/br61/containers/segmamba_wheels.sif
EXP_ROOT=/sharedscratch/br61/thesis/Experiments/Experiment_1555556
################################################################################

"$APPTAINER" exec --cleanenv \
        --bind ${REPO}:/workspace \
        --bind ${EXP_ROOT}:/workspace/experiment \
        "$IMG" \
        bash -lc 'set -euxo pipefail

    # --------------- PLOT ONLY -------------------------------------------------
    pip install --quiet --no-cache-dir pandas matplotlib

    mkdir -p /workspace/experiment/metrics/figures

    python /workspace/plot_metrics.py \
        --log_csv     /workspace/experiment/model/training_log.csv \
        --metrics_npy /workspace/experiment/metrics/test.npy \
        --out_dir     /workspace/experiment/metrics/figures
'
