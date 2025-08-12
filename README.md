### **Bladder Tumour Segmentation with 3D State-Space Models**
University of St Andrews, MSc Health Data Science, Master's Thesis.

This thesis applies a 3D state-space segmentation model (SegMamba) to the multi-centre FedBCa bladder MRI dataset to study generalisation across hospitals. 
Code here reproduces preprocessing, training, inference, metrics, grid search utilities, and plotting used for the experiments.


**1. Directory layout (top level, short explanations)**

* FedBCa_prep/ — helper assets for dataset organisation and preparation.
* mamba/ — selective state-space backbone components (vendored for reproducibility).
* model_segmamba/ — SegMamba 3D segmentation model implementation.
* monai/ — minimal MONAI utilities used by the pipeline (vendored).
* causal-conv1d/ — 1- causal convolution kernel/ops required by Mamba stack.
* light_training/ — lightweight training utilities used by scripts.
* slurm_scripts/ — Slurm launcher helpers for all conducted experiments.
* SegMamba_original_README.md — upstream SegMamba README kept for reference.
* 0_inference.py — simple end-to-end inference runner for a single volume/folder.
* 2_bladder_preprocessing_mri.py — preprocessing (crop/ROI, resample, normalise) for FedBCa.
* 3_bladder_train.py — training entry point (DDP/AMP supported when GPUs are visible).
* 4_bladder_predict.py — sliding-window inference with overlap and mirroring.
* 5_bladder_compute_metrics.py — compute Dice and HD95 per case/centre and write CSVs.
* grid_runner.py — launch hyperparameter/grid experiments.
* evaluate_grid.py — aggregate/compare runs (e.g., paired tests) and summarise results.
* plot_metrics.py — plot learning curves and performance distributions.
* make_failure_atlas.py — collect and render failure examples.


**2. Dataset**

* Paper: https://www.nature.com/articles/s41597-024-03971-0 
* Zenodo record: https://zenodo.org/records/10409145


**3. Environment (minimal)**

Library Version

* numpy 2.2.4
* pandas 2.3.0
* SimpleITK 2.5.2
* scikit-image 0.25.2
* scikit-learn 1.6.1
* matplotlib 3.10.1
* medpy 0.5.2
* torch 2.7.1+cu126


**4. Releases**

Pre-release “Experiments” contains trained weights, logs, and result artifacts for a subset of runs (e.g., Centre-1/2 internal+external, pooled, grid/test). Other  runs weren’t uploaded due to file-size limits; ask if you need a specific artifact. 


**5. Contact**

br61@st-andrews.ac.uk


**6. Acknowledgements**

* Thank you to Dr David Harris-Birtill for supervision.
* Thank you to Xing et al. for the SegMamba Repo.
* Thank you to Cao et al. for the FedBCa dataset.
