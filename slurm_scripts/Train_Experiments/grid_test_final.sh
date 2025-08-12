#!/bin/bash
#SBATCH --job-name=bladder_experiment
#SBATCH --partition=gpu.L40S
#SBATCH --gres=gpu:2  # For 1 GPU recommended to switch env to "pytorch" in 3_bladder_train.py
#SBATCH --time=08:00:00
#SBATCH --mem=40G
#SBATCH --chdir=/sharedscratch/br61/thesis/slurm_scripts
#SBATCH --output=/sharedscratch/br61/thesis/MVA_SegMamba/logs/test_grid_%j.out
#SBATCH --error=/sharedscratch/br61/thesis/MVA_SegMamba/logs/test_grid_%j.err

APPTAINER=/software/apptainer/bin/apptainer

########## host-side paths you MAY tweak ######################
REPO=/sharedscratch/br61/thesis/MVA_SegMamba
TRAIN_DIR=/sharedscratch/br61/thesis/data/FedBCa_clean/center1_for_center1_processed
TEST_DIR=/sharedscratch/br61/thesis/data/FedBCa_clean/center1_for_center1_processed
GT_DIR=/sharedscratch/br61/thesis/data/FedBCa_clean/center1_raw
IMG=/sharedscratch/br61/containers/segmamba_wheels.sif
################################################################

# ---------------- random‑seed handling -----------------
#  ➜ 1st cmd‑line arg wins, else $SEED env, else 42
: "${SEED:=1234}"
if [ $# -ge 1 ]; then SEED=$1; fi
echo "Using SEED=$SEED"
# sbatch --export=ALL,SEED=1 center1_internal_experiment.sh


# Create experiment scaffold (one folder per SLURM job id)
EXP_ROOT=/sharedscratch/br61/thesis/Experiments/Experiment_${SLURM_JOB_ID}
MODEL_DIR=${EXP_ROOT}/model
PRED_DIR=${EXP_ROOT}/predictions
METRICS_DIR=${EXP_ROOT}/metrics
mkdir -p "${MODEL_DIR}" "${PRED_DIR}" "${METRICS_DIR}"


"$APPTAINER" exec --nv --cleanenv \
    --env SEED=${SEED} \
    --bind ${REPO}:/workspace \
    --bind ${TRAIN_DIR}:/workspace/data/train \
    --bind ${TEST_DIR}:/workspace/data/test \
    --bind ${GT_DIR}:/workspace/data/gt \
    --bind ${EXP_ROOT}:/workspace/experiment \
    "$IMG" \
    bash -lc 'set -euxo pipefail


    ############## 1) TRAIN ####################################
    pip install --quiet --no-cache-dir "acvl-utils>=0.2.1" nibabel SimpleITK

    python /workspace/3_bladder_train.py \
          --data_dir  /workspace/data/train \
          --logdir    /workspace/experiment/model \
          --max_epoch 200 \
          --batch_size 2 \
          --val_every 2 \
          --roi       128 128 128 \
          --sched poly_with_warmup \
          --optim adamw \
          --warmup 0.05 \
          --lr 0.0001 \
          --oversample_p 0.5 \
          --val_oversample_p 0.5 \
          --loss ce_dice \
          --test_rate 0.2 \
          --seed ${SEED}

    # grab the best checkpoint (created by the train script)
    CKPT=$(ls -1 /workspace/experiment/model/model/best_model_*.pt | sort | tail -n1)

    ############## 2) PREDICT ##################################
    python /workspace/4_bladder_predict.py \
          --model_path "${CKPT}" \
          --save_dir   /workspace/experiment/predictions \
          --data_dir   /workspace/data/test

    ############## 3) METRICS ##################################
    pip install --quiet --no-cache-dir medpy SimpleITK scikit-image scikit-learn tqdm nibabel

    python /workspace/5_bladder_compute_metrics.py \
          --pred_dir /workspace/experiment/predictions/test \
          --gt_dir   /workspace/data/gt \
          --out_dir  /workspace/experiment/metrics


    ############## 4) PLOTS ###################################
    # deps: pandas & matplotlib are enough; seaborn optional
    pip install --quiet --no-cache-dir pandas matplotlib

    mkdir -p /workspace/experiment/metrics/figures

    python /workspace/plot_metrics.py \
          --log_csv     /workspace/experiment/model/training_log.csv \
          --metrics_npy /workspace/experiment/metrics/test.npy \
          --out_dir     /workspace/experiment/metrics/figures
    '
