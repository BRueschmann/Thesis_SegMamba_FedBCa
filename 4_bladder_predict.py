#!/usr/bin/env python3
# 4_bladder_predict.py  -  SegMamba inference (header-safe, timed)
# ---------------------------------------------------------------
import argparse, os, time, csv
import torch, numpy as np, SimpleITK as sitk
from monai.inferers import SlidingWindowInferer
from monai.utils import set_determinism
from monai.data import DataLoader
from light_training.dataloading.dataset import get_train_val_test_loader_from_train, get_test_loader_from_test
from light_training.prediction import Predictor as LT_Predictor
from light_training.trainer import Trainer
from light_training.debug_utils import dbg

set_determinism(123)

# ----------------- CLI -----------------
parser = argparse.ArgumentParser("SegMamba bladder prediction")
parser.add_argument("--model_path", required=True)
parser.add_argument("--data_dir",  required=True)
parser.add_argument("--save_dir",  default="./prediction_results/segmamba")
parser.add_argument("--device",    default="cuda:0")
parser.add_argument("--patch",     nargs=3, type=int, default=[128,128,128])
parser.add_argument("--split",
                    choices=["train", "val", "test", "all"],
                    default="test",
                    help="'all' = infer on EVERY .npz in --data_dir")
args = parser.parse_args()

# ----------------- Predictor class -----------------
class BladderPredictor(Trainer):
    def __init__(self):
        super().__init__(env_type="pytorch",
                         max_epochs=1, batch_size=1, num_gpus=1,
                         device=args.device, logdir="", val_every=1,
                         master_port=17759, training_script=__file__)

        # model ---------------------------------------------------------------
        from model_segmamba.segmamba import SegMamba
        self.model = SegMamba(in_chans=1, out_chans=2,
                              depths=[2,2,2,2], feat_size=[48,96,192,384]).to(args.device)
        ckpt = torch.load(args.model_path, map_location=args.device)
        self.model.load_state_dict(self._strip_ddp(ckpt))
        self.model.eval()

        # inferer -------------------------------------------------------------
        sw = SlidingWindowInferer(roi_size=tuple(args.patch),
                                  sw_batch_size=2, overlap=0.5, mode="gaussian")
        self.pred = LT_Predictor(window_infer=sw, mirror_axes=[0,1,2])

        # io ------------------------------------------------------------------
        self.split_dir = os.path.join(args.save_dir, args.split)
        os.makedirs(self.split_dir, exist_ok=True)
        self._dbg = True

    # dataset -> tensors -------------------------------------------------------
    @staticmethod
    def _bin_gt(lbl): return (lbl == 1).float()

    def get_input(self, batch):
        return batch["data"], self._bin_gt(batch["seg"]), batch["properties"]

    # single-GPU inference ----------------------------------------------------
    @torch.no_grad()
    def validation_step(self, batch):
        img, _, prop_l = self.get_input(batch)
        prop = prop_l[0] if isinstance(prop_l, (list, tuple)) else prop_l
        img  = img.to(args.device)

        raw = self.pred.maybe_mirror_and_predict(img, self.model, device=args.device)
        if self._dbg:
            dbg("raw_prob", raw.cpu());  self._dbg = False

        raw  = self.pred.predict_raw_probability(raw, prop)
        segc = raw.argmax(dim=0, keepdim=True)
        segf = self.pred.predict_noncrop_probability(segc, prop)

        self._save_nifti(segf, prop)
        return 0.0

    # write mask with correct header -----------------------------------------
    def _save_nifti(self, seg_t, prop):
        if torch.is_tensor(seg_t):
            seg_np = seg_t.squeeze(0).cpu().numpy().astype(np.uint8)
        else:
            seg_np = np.squeeze(seg_t, axis=0).astype(np.uint8)

        case = prop["name"] if isinstance(prop["name"], str) else prop["name"][0]
        gt_path = os.path.join("/workspace/data/gt", case, "seg.nii.gz")

        try:
            ref_img = sitk.ReadImage(gt_path)
        except RuntimeError:
            ref_img = None

        if ref_img is not None:
            out_img = sitk.GetImageFromArray(seg_np)
            out_img.CopyInformation(ref_img)
        else:
            out_img = sitk.GetImageFromArray(seg_np)
            spc = prop.get("target_spacing_trans", prop.get("spacing", (1,1,1)))
            out_img.SetSpacing(tuple(float(s) for s in spc[::-1]))

        sitk.WriteImage(out_img,
                        os.path.join(self.split_dir, f"{case}.nii.gz"))

    @staticmethod
    def _strip_ddp(sd):
        if isinstance(sd, dict) and "module" in sd:
            sd = sd["module"]
        return {k.replace("module.", "", 1): v for k, v in sd.items()}

    # ------- Timed inference over a dataset ----------------------------
    def run_inference_timed(self, val_dataset):
        """Run inference and print/save timing statistics."""
        val_loader = DataLoader(val_dataset,
                                batch_size=1, shuffle=False, pin_memory=True)

        if self.model is not None:
            self.model.to(self.device)
            self.model.eval()

        per_case_times = []
        total_start = time.perf_counter()

        for batch in val_loader:
            # Extract case-ID before tensors hit GPU
            prop = batch["properties"]
            case = prop[0]["name"] if isinstance(prop, list) else prop["name"]

            # normal device transfer path
            batch = self.before_data_to_device(batch)
            batch = self.to_device(batch)

            t0 = time.perf_counter()
            self.validation_step(batch)            # prediction & saving
            t1 = time.perf_counter()

            per_case_times.append((case, t1 - t0))

        total_time = time.perf_counter() - total_start
        mean_time  = total_time / max(1, len(per_case_times))

        print(f"\n--------------- Inference timing ({args.split}) --------------- ")
        print(f"Total wall-clock time : {total_time:.3f} s "
              f"for {len(per_case_times)} cases")
        print(f"Mean   wall-clock time : {mean_time:.3f} s per case\n")

        # save CSV next to masks
        csv_path = os.path.join(self.split_dir,
                                f"inference_time_{args.split}.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["case_id", "seconds"])
            w.writerows(per_case_times)
        print(f"Per-case timings written → {csv_path}")


if __name__ == "__main__":
    predictor = BladderPredictor()
    if args.split == "all": # For external experiments
        test_ds = get_test_loader_from_test(args.data_dir)
        predictor.run_inference_timed(test_ds)
    else:
        tr, val, test = get_train_val_test_loader_from_train(args.data_dir)
        split2ds = {"train": tr, "val": val, "test": test}
        predictor.run_inference_timed(split2ds[args.split])
