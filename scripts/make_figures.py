"""
Regenerate publication figures at 300 DPI from saved predictions: ROC, confusion
matrix, ablation bars (including the static equal-weight baseline), and the
reliability diagram. The figures shipped in figures/ were produced by the
notebook's R7 cell from the paper's final run.

Usage:
    python scripts/make_figures.py   # expects results/test_predictions.npz
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, confusion_matrix, roc_auc_score

from src.metrics import compute_ece


def reliability_diagram(y, probs, out="figures/fig_reliability.png", n_bins=15):
    conf = probs.max(1); correct = (probs.argmax(1) == y).astype(float)
    edges = np.linspace(0, 1, n_bins + 1); ctr = (edges[:-1] + edges[1:]) / 2
    acc, dens = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        dens.append(m.mean()); acc.append(correct[m].mean() if m.sum() else np.nan)
    ece = compute_ece(y, conf, correct, n_bins)
    fig, ax = plt.subplots(figsize=(7, 5.5)); ax2 = ax.twinx()
    ax2.bar(ctr, dens, width=(edges[1] - edges[0]) * 0.9, color="#BBD3E8", alpha=0.55,
            label="Prediction density")
    ax2.set_ylabel("Prediction density")
    ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Perfect calibration")
    ax.plot(ctr, acc, "o-", color="#C0392B", lw=2, label=f"Model (ECE = {ece:.3f})")
    ax.set_xlabel("Confidence"); ax.set_ylabel("Empirical accuracy")
    ax.set_title(f"Reliability Diagram (ECE = {ece:.3f})")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left")
    plt.tight_layout(); plt.savefig(out, dpi=300, bbox_inches="tight")
    print(f"saved {out} (ECE={ece:.4f})")


if __name__ == "__main__":
    d = np.load("results/test_predictions.npz")
    reliability_diagram(d["labels"].astype(int), d["probabilities"])
