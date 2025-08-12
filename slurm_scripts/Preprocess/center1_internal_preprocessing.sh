#!/bin/bash
#SBATCH --job-name=segmamba_preproc
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --mem=32G
#SBATCH --chdir=/sharedscratch/br61/thesis/slurm_scripts
#SBATCH --output=/sharedscratch/br61/thesis/MVA_SegMamba/logs/center1_process_%j.out
#SBATCH --error=/sharedscratch/br61/thesis/MVA_SegMamba/logs/center1_process_%j.err

# ---------- modules ----------
source /etc/profile.d/modules.sh
module use /gpfs01/software/modules/nvidia
module use /gpfs01/software/modules/apps

export PROJ=/sharedscratch/br61/thesis/MVA_SegMamba
export PYTHONPATH=$PROJ:$PYTHONPATH
export PYTHONDONTWRITEBYTECODE=1

# ---------- host paths ----------
APPTAINER=/software/apptainer/bin/apptainer
REPO=/sharedscratch/br61/thesis/MVA_SegMamba
DATA=/sharedscratch/br61/thesis/data/FedBCa_clean
IMG=/sharedscratch/br61/containers/segmamba_wheels.sif

# ---------- centre-1 settings ----------
CENTER_IN=/workspace/data/FedBCa_clean/all_centers
CENTER_OUT=/workspace/data/FedBCa_clean/center1_for_center1_processed
Z=4.0
Y=0.62
X=0.62



# ---------- run container ----------
"$APPTAINER" exec --nv --cleanenv \
    --bind ${REPO}:/workspace \
    --bind ${DATA}:/workspace/data/FedBCa_clean \
    "$IMG" \
    bash -lc "set -euo pipefail
              pip install --quiet --no-cache-dir acvl-utils>=0.2.1
              mkdir -p ${CENTER_OUT}
              python /workspace/2_bladder_preprocessing_mri.py \
                     --input_dir  ${CENTER_IN} \
                     --output_dir ${CENTER_OUT} \
                     --out_spacing ${Z} ${Y} ${X}"