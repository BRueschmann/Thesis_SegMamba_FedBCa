import argparse, os, math
import numpy as np
import torch
import torch.nn as nn
from monai.utils import set_determinism
from monai.inferers import SlidingWindowInferer
from monai.losses import DiceLoss
import torch.nn.functional as F

from light_training.dataloading.dataset import get_train_val_test_loader_from_train
from light_training.trainer import Trainer
from light_training.utils.files_helper import save_new_model_and_delete_last
from light_training.debug_utils import dbg
from light_training.dataloading.base_data_loader import DataLoaderMultiProcess

import random

# ----------------------------------------------------------------------------- 
# CLI arguments                                                                 
# ----------------------------------------------------------------------------- 
parser = argparse.ArgumentParser(description="Train SegMamba on FedBCa bladder MRI")
parser.add_argument("--data_dir", type=str, default="./data/fullres/train")
parser.add_argument("--logdir",   type=str, default="./logs/segmamba")
parser.add_argument("--max_epoch",   type=int, default=1000)
parser.add_argument("--batch_size",  type=int, default=2)
parser.add_argument("--val_every",   type=int, default=2)
parser.add_argument("--device",      type=str, default="cuda:0")
parser.add_argument("--roi", nargs=3, type=int, default=[128,128,128],
                    metavar=("D","H","W"))
parser.add_argument("--sched",  choices=["poly", "poly_with_warmup",
                                         "cosine_with_warmup",
                                         "constant_with_warmup"],
                    default="poly_with_warmup")
parser.add_argument("--optim", choices=["sgd", "adamw"], default="sgd",
                    help="Which optimiser to use (case‑insensitive).")
parser.add_argument("--warmup", type=float, default=0.1)
parser.add_argument("--lr",     type=float, default=1e-2)
# Foreground-patch oversampling probability  (train time only)
parser.add_argument("--oversample_p", type=float, default=0.50,
                    help="Probability that each sampled patch contains foreground (tumour).")
parser.add_argument("--val_oversample_p", type=float, default=0.0,
                    help="Foreground‑patch oversampling prob. for VALIDATION. "
                         "Leave unset for unbiased 0.0; set to the same value "
                         "as --oversample_p if you want matched sampling.")
parser.add_argument("--loss", choices=["ce", "ce_dyn", "dice", "ce_dice", "ce_dice_dyn", "focal_tversky"], default="ce_dice",
                    help="Training loss: standard Dice+CE or focal‑Tversky.")
parser.add_argument("--val_rate",  type=float, default=0.10,
                    help="fraction of each centre held out for validation")
parser.add_argument("--test_rate", type=float, default=0.20,
                    help="fraction of each centre held out for test")
parser.add_argument("--seed", type=int, default=123,
                    help="global random seed for reproducibility")


args = parser.parse_args()


# ---------------- reproducibility -----------------
def _set_global_seed(s: int):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

_set_global_seed(args.seed)
set_determinism(args.seed)


# ------------- Focal-Tversky loss -------------
class FocalTverskyLoss(nn.Module):
    """
    Focal‑Tversky loss for binary (foreground/background) segmentation.
    TI = TP / (TP + α·FP + β·FN);  L = (1 – TI)^γ
    Defaults follow the original paper (α=0.7, β=0.3, γ=0.75).
    """
    def __init__(self, alpha=0.7, beta=0.3, gamma=0.75, smooth=1e-2):
        super().__init__()
        self.alpha, self.beta, self.gamma, self.smooth = alpha, beta, gamma, smooth

    def forward(self, logits, target):
        # logits : (N,2,*) ; target : (N,*), foreground label == 1
        p     = torch.softmax(logits, dim=1)[:, 1]     # foreground prob
        g     = (target == 1).float()
        dims  = tuple(range(1, g.dim()))               # sum over spatial dims
        tp    = (p * g).sum(dims)
        fp    = (p * (1 - g)).sum(dims)
        fn    = ((1 - p) * g).sum(dims)
        ti    = (tp + self.smooth) / (tp + self.alpha*fp + self.beta*fn + self.smooth)
        ti = torch.clamp(ti, 0.0001, 0.9999)        # ← Clamping to avoid NA
        loss  = torch.pow((1.0 - ti), self.gamma)
        return loss.mean()




# ----------------------------------------------------------------------------- #
# Trainer                                                                       #
# ----------------------------------------------------------------------------- #
class BladderTrainer(Trainer):
    def __init__(self):
        gpus_available = torch.cuda.device_count() or 1
        env_mode = "DDP" if gpus_available > 1 else "pytorch"
        super().__init__(env_type=env_mode,
                         max_epochs=args.max_epoch,
                         batch_size=args.batch_size,
                         device="cuda",
                         val_every=args.val_every,
                         num_gpus=gpus_available,
                         logdir=args.logdir,
                         master_port=17759,
                         training_script=__file__)

        # model & losses -------------------------------------------------------
        self.oversample_p = args.oversample_p
        self.val_oversample_p = args.val_oversample_p

        self.patch_size = tuple(args.roi)
        self.window_infer = SlidingWindowInferer(roi_size=self.patch_size,
                                                 sw_batch_size=1,
                                                 overlap=0.5)
        from model_segmamba.segmamba import SegMamba
        self.model = SegMamba(in_chans=1, out_chans=2,
                              depths=[2,2,2,2], feat_size=[48,96,192,384])

        w_bg, w_fg = 0.05, 0.95 # Background / Foreground weighting
        class_w    = torch.tensor([w_bg, w_fg], device=args.device)


        self.loss_type = args.loss     
        self._dyn = self.loss_type in ("ce_dyn", "ce_dice_dyn")

        self.ce   = nn.CrossEntropyLoss(weight=class_w, ignore_index=-1)
        self.dice = DiceLoss(to_onehot_y=True, softmax=True,
                             include_background=False, reduction="mean")

        # focal-Tversky is created only if requested
        self.ftv  = FocalTverskyLoss() if self.loss_type == "focal_tversky" else None


        # ---------------- choose optimiser ----------------
        if args.optim.lower() == "adamw":
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=args.lr,
                weight_decay=1e-4,
                betas=(0.9, 0.999),
            )
            print("[init] Using AdamW ︱ lr =", args.lr,
                  "︱ weight_decay = 1e‑4")
        else:          # SGD (default)
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=args.lr,
                weight_decay=0.0, 
                momentum=0.99,
                nesterov=True,
            )
            print("[init] Using SGD ︱ lr =", args.lr,
                  "︱ weight_decay = 0.0")


        self.scheduler_type = args.sched
        self.warmup         = args.warmup
        self.best_val_dice  = 0.0

        # running sums for epoch-wise train Dice -------------------------------
        self._epoch_inter = 0.0   # stores 2*TP  (numerator)
        self._epoch_union = 0.0   # stores |P|+|G|+eps  (denominator)

    # ------------------------------------------------------------------------- #
    # data helper                                                               #
    # ------------------------------------------------------------------------- #

    def _make_temp_val_loader(self, p: float = 0.0):
        # unwrap LimitedLenWrapper if present
        dl = getattr(self.val_loader, "data_loader", self.val_loader)
        ds = dl.dataset
        return DataLoaderMultiProcess(
            ds, batch_size=1, patch_size=self.patch_size,
            oversample_foreground_percent=p,
        )

    def get_input(self, batch):
        img   = batch["data"]
        raw_lbl = batch["seg"][:, 0].long()
        label   = (raw_lbl == 1).long()         # tumour label
        if self.global_step < 3:
            dbg("train_img",   img.cpu())
            dbg("train_label", label.cpu())
        return img, label

    # ------------------------------------------------------------------------- #
    # training step                                                             #
    # ------------------------------------------------------------------------- #
    def training_step(self, batch):
        img, seg  = self.get_input(batch)
        logits    = self.model(img)

        if self._dyn:
            # ---------- batch-adaptive ENet weights ----------------------------------
            with torch.no_grad():
                p_fg = (batch["seg"][:, 0] == 1).float().mean().item()
                p_bg = 1.0 - p_fg
                w_bg = 1 / math.log(1.02 + p_bg)
                w_fg = 1 / math.log(1.02 + p_fg)
                w_fg = min(w_fg, 50.0)              # safety clip
                self.ce.weight[:] = torch.tensor([w_bg, w_fg],
                                                device=self.ce.weight.device)       

        # ---- loss ---------------------------------------------------------------
        # already inside outer autocast -> just compute the loss
        if self.loss_type in ("ce", "ce_dyn"):
            loss = self.ce(logits, seg)
        elif self.loss_type == "focal_tversky":
            loss = self.ftv(logits, seg)
        elif self.loss_type == "dice":
            seg_1ch = seg.unsqueeze(1)          # (N,1, ...) one-hot target
            loss = self.dice(logits, seg_1ch)            
        elif self.loss_type in ("ce_dice", "ce_dice_dyn"):
            seg_1ch = seg.unsqueeze(1)
            loss = 0.3 * self.ce(logits, seg) + 0.7 * self.dice(logits, seg_1ch) # 30/70 better than 50/50


        # ---- NaN/Inf guard -------------------------------------------------------
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"[step {self.global_step}] NaN/Inf loss – zero surrogate")
            # keeps graph so backward() is valid
            loss = logits.sum() * 0.0


        # ---- accumulate Dice (no-grad) --------------------------------------
        with torch.no_grad():
            pred   = logits.argmax(1)
            inter  = ((pred == 1) & (seg == 1)).sum().float()
            union  = (pred == 1).sum() + (seg == 1).sum() + 1e-6
            self._epoch_inter += (2.0 * inter).item()
            self._epoch_union += union.item()

            # end-ofepoch?  ->  log & reset
            if self.global_step % self.num_step_per_epoch == 0:
                epoch_dice = self._epoch_inter / self._epoch_union
                self.log("train_dice", epoch_dice, step=self.epoch)
                self._epoch_inter = 0.0
                self._epoch_union = 0.0

        self.log("training_loss", loss.detach().item(), step=self.global_step)

        return loss

    # ------------------------------------------------------------------------- #
    # validation step                                                           #
    # ------------------------------------------------------------------------- #
    def validation_step(self, batch):
        img, seg  = self.get_input(batch)
        logits    = self.model(img)
        pred      = logits.softmax(1).argmax(1)

        if self.epoch < 2:
            dbg("val_pred", pred)
            dbg("val_gt",   seg)

        inter = ((pred == 1) & (seg == 1)).sum()
        denom = (pred == 1).sum() + (seg == 1).sum() + 1e-6
        dice_val = (2.0 * inter.float() / denom).item()

        if self.loss_type in ("ce", "ce_dyn"):
            loss_val = self.ce(logits, seg).item()
        elif self.loss_type == "focal_tversky":
            loss_val = self.ftv(logits, seg).item()
        elif self.loss_type == "dice":
            seg_1ch  = seg.unsqueeze(1)
            loss_val = self.dice(logits, seg_1ch).item()
        elif self.loss_type in ("ce_dice", "ce_dice_dyn"):
            seg_1ch  = seg.unsqueeze(1)
            loss_val = (0.3 * self.ce(logits, seg) + # i changed this from 0.5
                        0.7 * self.dice(logits, seg_1ch)).item()


        if not math.isfinite(loss_val):          # catches +-Inf as well as NaN
            loss_val = float("nan")   

        return [dice_val, loss_val]      # list so Trainer.validate handles it

    # ------------------------------------------------------------------------- #
    # validation end                                                            #
    # ------------------------------------------------------------------------- #
    def validation_end(self, val_outputs):
        if isinstance(val_outputs, list) and len(val_outputs) >= 2:
            dice_vals = val_outputs[0].float()
            loss_vals = val_outputs[1].float()
        else:                           # fallback (shouldn't happen)
            dice_vals = torch.as_tensor(val_outputs).float()
            loss_vals = None

        mean_dice = float(dice_vals.mean().item())
        self.log("val_dice", mean_dice, step=self.epoch)

        if loss_vals is not None:
            mean_loss = float(loss_vals.mean().item())
            self.log("val_loss", mean_loss, step=self.epoch)
            print(f"[epoch {self.epoch}]  val Dice = {mean_dice:.4f}   val Loss = {mean_loss:.4f}")
        else:
            print(f"[epoch {self.epoch}]  val Dice = {mean_dice:.4f}")

        # ---------------- checkpointing --------------------------------------
        if mean_dice > self.best_val_dice:
            self.best_val_dice = mean_dice
            save_new_model_and_delete_last(
                self.model,
                os.path.join(args.logdir, "model",
                             f"best_model_{mean_dice:.4f}.pt"),
                delete_symbol="best_model")

        if (self.epoch + 1) % 40 == 0:
            torch.save(self.model.state_dict(),
                       os.path.join(args.logdir, "model",
                                    f"epoch_{self.epoch:04d}_{mean_dice:.3f}.pt"))

# ----------------------------------------------------------------------------- #
# main                                                                          #
# ----------------------------------------------------------------------------- #
if __name__ == "__main__":
    trainer = BladderTrainer()
    train_ds, val_ds, _ = get_train_val_test_loader_from_train(
        args.data_dir,
        val_rate=args.val_rate,
        test_rate=args.test_rate)
    trainer.train(train_dataset=train_ds, val_dataset=val_ds)