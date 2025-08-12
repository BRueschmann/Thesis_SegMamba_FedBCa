#!/usr/bin/env python3
"""
Grid runner for the bladder-MRI SegMamba study.
"""

import itertools, json, os, subprocess, argparse, csv, datetime, time
from pathlib import Path
import numpy as np

# ---------------- CLI ---------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--part",   type=int, default=0, help="0‑based partition index")
parser.add_argument("--nparts", type=int, default=3, help="total number of partitions")
parser.add_argument("--epoch",  type=int, default=100, help="epochs for 3_bladder_train.py")
args = parser.parse_args()

# ---------------- grid definition ---------------------------
LOSS  = ["dice", "ce_dice", "focal_tversky", "ce"]
P     = ["0.50"]
OPTIM = ["adamw", "sgd"]
SCHED = ["cosine_with_warmup", "poly_with_warmup"]

BASE_LR = {("adamw","dice"):["1e-4","5e-5"], ("adamw","ce_dice"):["1e-4","5e-5"],
           ("adamw","focal_tversky"):["5e-5","3e-5"],
           ("sgd","dice"):["3e-3","1e-3"], ("sgd","ce_dice"):["3e-3","1e-3"],
           ("adamw","ce"):["1e-4","5e-5"],
           ("sgd","ce"):["3e-3","1e-3"],
           ("sgd","focal_tversky"):["1e-3","5e-4"]}

GRID = [{ "loss":l,"p":p,"optim":o,"sched":s,"lr":lr }
        for l,p,o,s in itertools.product(LOSS,P,OPTIM,SCHED)
        for lr in BASE_LR[(o,l)]]

# ---------------- paths --------------------------------------------------
EXP_ROOT  = Path("/workspace/experiment_root").resolve()
EXP_ROOT.mkdir(parents=True, exist_ok=True)

PROGRESS  = EXP_ROOT / "progress.json"
SUMMARY   = EXP_ROOT / "grid_results.csv"

def load_done():
    if PROGRESS.exists():
        return set(json.loads(PROGRESS.read_text()))
    return set()

def save_done(done):
    PROGRESS.write_text(json.dumps(sorted(done)))

# ---------------- helpers ------------------------------------------------
def launch(idx, cfg, max_epoch):
    exp_dir     = EXP_ROOT / f"comb_{idx:02d}"
    pred_dir    = exp_dir / "predictions"
    metrics_dir = exp_dir / "metrics"
    for d in (exp_dir, pred_dir, metrics_dir): d.mkdir(parents=True, exist_ok=True)

    warmup = "0.20" if cfg["sched"] == "constant_with_warmup" else "0.10"
    cmd_train = f"""
        python /workspace/3_bladder_train.py \
            --data_dir /workspace/data/train \
            --logdir {exp_dir} \
            --max_epoch {max_epoch} \
            --batch_size 2 \
            --val_every 5 \
            --roi 128 128 128 \
            --sched {cfg['sched']} \
            --optim {cfg['optim']} \
            --warmup {warmup} \
            --lr {cfg['lr']} \
            --oversample_p {cfg['p']} \
            --val_oversample_p {cfg['p']} \
            --loss {cfg['loss']} \
            --test_rate 0.2
    """
    subprocess.run(cmd_train, shell=True, check=True)

    ckpt = subprocess.check_output(
        f"ls -1 {exp_dir}/model/best_model_*.pt | sort | tail -n1", shell=True
    ).decode().strip()

    subprocess.run(
        f"python /workspace/4_bladder_predict.py "
        f"--model_path {ckpt} --save_dir {pred_dir} --data_dir /workspace/data/train --split val",
        shell=True, check=True)

    subprocess.run(
        f"python /workspace/5_bladder_compute_metrics.py "
        f"--pred_dir {pred_dir}/val --gt_dir /workspace/data/gt --out_dir {metrics_dir}",
        shell=True, check=True)

    # -------- one-row summary ------------------------------------------
    val_arr  = np.load(metrics_dir / "val.npy")          # (N,1,5)
    dice     = float(np.mean(val_arr[:,0,0]))
    hd95     = float(np.nanmean(val_arr[:,0,1]))

    log_csv  = exp_dir / "training_log.csv"
    best_val = max((float(r["value"]) for r in csv.DictReader(open(log_csv))
                    if r["key"]=="val_dice"), default=float("-inf"))
    epoch_ts = [float(r["value"]) for r in csv.DictReader(open(log_csv))
                if r["key"]=="epoch_time"]
    mean_ep  = sum(epoch_ts)/len(epoch_ts) if epoch_ts else float("nan")

    write_hdr = not SUMMARY.exists()
    with open(SUMMARY, "a", newline="") as f:
        fields = ["comb_idx","loss","optim","sched","lr","p",
                  "val_dice","val_hd95","best_val_dice","mean_epoch_time"]        
        w = csv.DictWriter(f, fieldnames=fields)
        if write_hdr: w.writeheader()
        w.writerow({**cfg,"comb_idx":idx,
                    "val_dice":f"{dice:.5f}",
                    "val_hd95":f"{hd95:.2f}",
                    "best_val_dice":f"{best_val:.5f}",
                    "mean_epoch_time":f"{mean_ep:.3f}"})


# ---------------- main ---------------------------------------------------
done = load_done()
for idx, cfg in enumerate(GRID):
    if idx % args.nparts != args.part:   continue  # not my slice
    if idx in done:                      continue  # already finished
    print(f"[{datetime.datetime.now()}] PART {args.part}: launching {idx}/{len(GRID)-1}")
    launch(idx, cfg, args.epoch)
    done.add(idx)
    save_done(done)

print("### complete - now run evaluation ###")
