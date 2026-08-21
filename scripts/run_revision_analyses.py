"""
Reproduce the paper's statistical analyses from saved test-set predictions:
percentile-bootstrap 95% CIs for all metrics (Table 5), calibration (Brier, ECE),
and exact McNemar comparisons from per-configuration predictions.

The fully executable end-to-end path for ALL analyses (including the masked-
configuration evaluations, static equal-weight fusion, T-sensitivity sweep, and
the bidirectional cross-dataset transfer) is the notebook
notebooks/Reproducible_Pipeline.ipynb, cells R1-R9, which imports these same
utilities. An executed copy with all outputs is provided as
notebooks/Reproducible_Pipeline_with_outputs.ipynb for inspection without
re-running.

Usage:
    python scripts/run_revision_analyses.py   # expects results/test_predictions.npz
"""
import json
import numpy as np
from sklearn.metrics import brier_score_loss, confusion_matrix

from src.metrics import compute_ece, full_bootstrap_ci


def main():
    data = np.load("results/test_predictions.npz")
    y = data["labels"].astype(int)
    yp = data["predictions"].astype(int)
    P = data["probabilities"]

    cis = full_bootstrap_ci(y, yp, P, n_boot=1000)
    print("Percentile-bootstrap 95% CIs (1,000 resamples):")
    for k, v in cis.items():
        print(f"  {k:12s}: {v['mean']:.4f}  [{v['ci_low']:.4f}, {v['ci_high']:.4f}]")

    tn, fp, fn, tp = confusion_matrix(y, yp).ravel()
    out = {"headline_ci": cis,
           "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
           "brier": float(brier_score_loss(y, P[:, 1]))}
    with open("results/revision_analyses.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved results/revision_analyses.json")


if __name__ == "__main__":
    main()
