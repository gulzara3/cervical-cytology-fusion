# Uncertainty-Aware Adaptive Fusion for Trustworthy Cervical Cytology

Reference implementation and full reproducibility package for the paper
**"Uncertainty-Aware Adaptive Fusion with Graceful Degradation for Trustworthy
Cervical Cytology."** The framework couples an ensemble image encoder
(ResNet50 + DenseNet121 + EfficientNet-B0) with clinical and genomic encoders
through an **uncertainty-aware adaptive fusion operator** (a convex combination
over the class-probability simplex, weighted per sample by Monte Carlo Dropout
predictive entropy), producing **calibrated** predictions with selective,
human-in-the-loop deferral.

> **Scope note.** This is a methodological proof-of-concept. The clinical
> features come from an *unpaired* public cohort and the genomic features are
> *synthetically simulated* (`src/genomic_simulation.py`); the study makes no
> claim of validated multimodal clinical decision support. See the paper's
> Limitations.

---

## Key results (final run, held-out test set, n = 1,028)

| Metric | Value | 95% CI |
|---|---|---|
| Accuracy | 97.08% | 96.01 – 98.05% |
| Sensitivity | 98.07% | 96.91 – 99.07% |
| Specificity | 96.08% | 94.29 – 97.73% |
| F1-score | 97.13% | 96.05 – 98.09% |
| AUC-ROC | 0.993 | 0.988 – 0.997 |
| Brier score | 0.025 | 0.017 – 0.033 |
| ECE (15 bins) | 0.014 | 0.010 – 0.027 |

Confidence intervals are percentile-bootstrap (1,000 resamples). Confusion
matrix: TN 490 / FP 20 / FN 10 / TP 508.

**Graceful degradation (headline finding).** All image-containing adaptive
configurations produce *identical* test-set predictions (zero discordant pairs;
exact McNemar p = 1.0): the operator suppresses uninformative auxiliary
modalities so completely that they never flip a label. A **static equal-weight
baseline** degrades AUC from 0.993 to 0.982, isolating the value of the
adaptive weighting. **Cross-dataset transfer** (train on one benchmark, test on
the other) yields AUCs of 0.636 (SIPaKMeD→Herlev) and 0.747 (Herlev→SIPaKMeD),
quantifying the generalisation gap that motivates external validation.
Throughput at T = 20: ≈433 samples/min on an NVIDIA Tesla T4.

All numbers ship in [`results/revision2_results.json`](results/revision2_results.json).

---

## Repository layout

```
cervical-cytology-fusion/
├── README.md
├── requirements.txt
├── LICENSE                       # MIT
├── CITATION.cff
├── configs/default.yaml          # all hyperparameters
├── train.py                      # training contract (see notebooks/ for the run)
├── src/
│   ├── config.py                 # Config + set_seed() (global seed 42)
│   ├── genomic_simulation.py     # leakage-limited synthetic genomics
│   ├── data.py                   # dataset, transforms, stratified split
│   ├── model.py                  # encoders, MC-Dropout, adaptive fusion
│   ├── train_utils.py            # focal loss, early stopping, train/eval
│   └── metrics.py                # ECE, Brier, percentile bootstrap, exact McNemar
├── scripts/
│   ├── download_data.py          # fetch SIPaKMeD / Herlev / UCI
│   ├── run_revision_analyses.py  # CIs + calibration from saved predictions
│   └── make_figures.py           # 300-DPI figures from saved predictions
├── notebooks/
│   ├── Reproducible_Pipeline.ipynb               # clean, fully executable
│   └── Reproducible_Pipeline_with_outputs.ipynb  # executed copy (inspect w/o running)
├── results/                      # final-run metrics, split indices, JSON summaries
├── figures/                      # final-run 300-DPI figures
├── models/                       # place best_model.pth here (Git LFS / Release)
└── docs/REPRODUCIBILITY.md
```

## Installation

```bash
git clone https://github.com/<username>/cervical-cytology-fusion.git
cd cervical-cytology-fusion
python -m venv .venv && source .venv/bin/activate   # optional
pip install -r requirements.txt
```

Python ≥ 3.9; a single CUDA GPU (Tesla T4 / L4 class) is sufficient.

## Reproducing the paper

The executable end-to-end path is the notebook (Colab-ready):

1. Open `notebooks/Reproducible_Pipeline.ipynb` on a GPU runtime.
2. Run cells **1–15**: data download → feature building → training → evaluation
   (~30 min on a T4).
3. Run cells **R1–R9**: bootstrap CIs, per-dataset analysis, T-sensitivity +
   throughput, exact McNemar tests, static equal-weight baseline, bidirectional
   cross-dataset transfer, 300-DPI figures, and a consolidated JSON identical in
   structure to `results/revision2_results.json`.

Reviewers who prefer not to re-run can inspect every output in
`notebooks/Reproducible_Pipeline_with_outputs.ipynb`.

## Reproducibility guarantees

- **Single global seed (42)** across Python, NumPy, PyTorch, CUDA
  (`cudnn.deterministic = True`).
- **Stratified 70/15/15 split at the individual-cell level before augmentation**;
  the exact test indices are shipped in `results/test_idx.npy`.
- **Augmentation on the training transform only**; validation/test are
  resize + normalise.
- **Genomic leakage cap**: every synthetic feature verified to have
  |Pearson corr| < 0.3 with the label.
- **Label-independent clinical assignment** from the unpaired UCI cohort
  (clinical-only AUC ≈ 0.51, at chance by design).

Full details: [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).

## Datasets

| Dataset | Source | Use |
|---|---|---|
| SIPaKMeD | https://www.cs.uoi.gr/~marina/sipakmed.html | cytology images (5,015 after preprocessing) |
| Herlev | http://mde-lab.aegean.gr/ | cytology images (1,834 after preprocessing) |
| UCI Cervical Cancer (Risk Factors) | https://archive.ics.uci.edu/dataset/383 | clinical features (unpaired) |
| Genomic | synthetic (this repo) | `src/genomic_simulation.py` |

## Model checkpoint

The trained checkpoint (~150 MB) exceeds GitHub's file limit and is not tracked
here. Obtain it via the repository's **Releases** page (attach on publication) or
track with [Git LFS](https://git-lfs.com) (`git lfs track "*.pth"`), placing it at
`models/best_model.pth`.

## Citation

```bibtex
@article{benchaabane2026cervical,
  title   = {Uncertainty-Aware Adaptive Fusion with Graceful Degradation
             for Trustworthy Cervical Cytology},
  author  = {Ben Chaabane, Slim and Bushnag, Anas and Oyouni, Atif and
             Massoudi, Wassim and Abuzneid, Shakour and Aluneizi, Shatha},
  journal = {Scientific Reports},
  year    = {2026}
}
```

## License

MIT : see [LICENSE](LICENSE). The public datasets retain their own licenses.
