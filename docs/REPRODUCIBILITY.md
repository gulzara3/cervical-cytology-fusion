# Reproducibility Guide

Every measure taken to make the reported experiments independently
reproducible, addressing the journal's Data Availability and Reproducibility
requirements and the reviewers' requests.

## 1. Seeding
A single global seed (`SEED = 42`, `src/config.py::set_seed`) is applied to
Python `random`, NumPy, PyTorch (CPU and CUDA), with `cudnn.deterministic = True`.
Residual GPU non-determinism across hardware/driver versions can shift metrics
by small amounts; the shipped `results/` correspond to the paper's final run
(NVIDIA Tesla T4).

## 2. Data split
Stratified 70/15/15 train/validation/test split (`train_test_split`,
`random_state = 42`), performed at the individual-cell level BEFORE any
augmentation. The exact held-out test indices are shipped in
`results/test_idx.npy` (n = 1,028: 735 SIPaKMeD, 293 Herlev).

## 3. Leakage control
- Augmentation only in the training transform; validation/test use resize +
  ImageNet normalisation.
- SIPaKMeD and Herlev are distinct public single-cell releases (no cross-split
  patient/cell overlap).
- Synthetic genomic features are capped at |Pearson corr| < 0.3 with the label,
  verified at generation (`src/genomic_simulation.py`).
- Clinical vectors are assigned by seeded sampling-with-replacement from the
  unpaired UCI cohort, independent of the image label (clinical-only AUC ≈ 0.51).

## 4. Statistical analysis
- 95% confidence intervals: PERCENTILE bootstrap, 1,000 resamples of the
  held-out test set; they characterise evaluation variability, not variability
  across retraining runs.
- Configuration comparisons: EXACT McNemar tests on shared test samples
  (`src/metrics.py::mcnemar_exact`); the paper reports discordant counts (b, c)
  and exact p-values.
- Decision threshold: 0.5. Calibration: Brier score and 15-bin ECE.

## 5. Analyses shipped in results/revision2_results.json
Headline CIs; confusion matrix (490/20/10/508); 7-configuration ablation +
static equal-weight fusion baseline (AUC 0.982); exact McNemar outcomes
(identical predictions among image-containing configs, p = 1.0); per-dataset
metrics; bidirectional cross-dataset transfer (AUC 0.636 / 0.747);
T-sensitivity (T ∈ {5,10,20,30,50}) with per-sample latency and throughput.

## 6. Training configuration
See `configs/default.yaml`: AdamW (wd 1e-4), OneCycleLR (max lr 3e-4),
30 epochs, FP16, gradient clip 1.0, focal loss (α 0.25, γ 2.0), dropout 0.4,
MC-Dropout T = 20, early stopping (patience 10) on validation AUC.

## 7. Environment
Reference run: Google Colab, NVIDIA Tesla T4, PyTorch 2.x, timm 0.9.12,
albumentations 1.3.1. Exact package list in `requirements.txt`.

## 8. Cross-dataset transfer protocol
Train exclusively on one dataset (85/15 stratified train/val within the source
for early stopping) and evaluate on the ENTIRE other dataset, with identical
pipeline, hyperparameters, and seed. Class priors differ markedly between the
benchmarks (73.6% vs 40.3% abnormal), which depresses fixed-threshold accuracy
under transfer; AUC is the threshold-free summary.

## 9. Model checkpoint
`models/best_model.pth` (~150 MB) is distributed via the repository Releases
page or Git LFS rather than tracked in-tree.
