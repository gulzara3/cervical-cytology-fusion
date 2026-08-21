"""Calibration (ECE, Brier), percentile-bootstrap CIs, and exact McNemar test."""
import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, brier_score_loss)


def compute_ece(y_true, confidence, correct, n_bins: int = 15) -> float:
    """Expected Calibration Error over equal-width confidence bins."""
    edges = np.linspace(0, 1, n_bins + 1); ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (confidence > lo) & (confidence <= hi)
        if m.sum() == 0:
            continue
        ece += (m.sum() / len(confidence)) * abs(correct[m].mean() - confidence[m].mean())
    return float(ece)


def full_bootstrap_ci(y, y_pred, probs, n_boot: int = 1000, ci: float = 0.95, seed: int = 42):
    """Percentile-bootstrap CIs for all headline metrics, including Brier and ECE.

    Note: intervals characterise test-set resampling variability, not variability
    across independent retraining runs (as stated in the paper's Methods).
    """
    rng = np.random.RandomState(seed); n = len(y)
    acc, sens, spec, prec, f1s, aucs, briers, eces = ([] for _ in range(8))
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        if len(np.unique(y[idx])) < 2:
            continue
        yt, ypp, pp = y[idx], y_pred[idx], probs[idx]
        acc.append(accuracy_score(yt, ypp)); f1s.append(f1_score(yt, ypp, zero_division=0))
        prec.append(precision_score(yt, ypp, zero_division=0)); sens.append(recall_score(yt, ypp, zero_division=0))
        tn = ((yt == 0) & (ypp == 0)).sum(); fp = ((yt == 0) & (ypp == 1)).sum()
        spec.append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)
        aucs.append(roc_auc_score(yt, pp[:, 1])); briers.append(brier_score_loss(yt, pp[:, 1]))
        ct = pp.max(1); cr = (ypp == yt).astype(float); eces.append(compute_ece(yt, ct, cr))

    def summ(v):
        v = np.array(v); a = (1 - ci) / 2
        return {"mean": float(v.mean()), "ci_low": float(np.percentile(v, a * 100)),
                "ci_high": float(np.percentile(v, (1 - a) * 100))}
    return {"Accuracy": summ(acc), "Sensitivity": summ(sens), "Specificity": summ(spec),
            "Precision": summ(prec), "F1": summ(f1s), "AUC": summ(aucs),
            "Brier": summ(briers), "ECE": summ(eces)}


def mcnemar_exact(b: int, c: int) -> float:
    """Exact McNemar p-value from discordant counts b and c."""
    try:
        from statsmodels.stats.contingency_tables import mcnemar
        return float(mcnemar([[0, b], [c, 0]], exact=True).pvalue)
    except Exception:
        from scipy.stats import binomtest
        n = b + c
        return 1.0 if n == 0 else float(binomtest(min(b, c), n, 0.5).pvalue)
