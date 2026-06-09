"""
Evaluation metrics: ROC curves, AUC, Pd @ fixed Pfa.
"""

import numpy as np
from sklearn.metrics import roc_curve, auc


def compute_roc(scores: np.ndarray, labels: np.ndarray):
    """
    scores: (N,) higher = H1
    labels: (N,) binary 0/1
    Returns: fpr, tpr, thresholds, auc_score
    """
    fpr, tpr, thresholds = roc_curve(labels, scores)
    auc_score = auc(fpr, tpr)
    return fpr, tpr, thresholds, auc_score


def pd_at_pfa(scores: np.ndarray, labels: np.ndarray,
              pfa: float = 1e-2) -> float:
    """Probability of detection at a fixed false-alarm rate."""
    fpr, tpr, _, _ = compute_roc(scores, labels)
    # interpolate
    return float(np.interp(pfa, fpr, tpr))


def snr_sweep(detector_fn, dataset_factory, snr_db_list: list,
              pfa: float = 1e-2, **factory_kwargs) -> dict:
    """
    Run detector_fn over a range of SNRs and return Pd at fixed Pfa.

    detector_fn: callable(y, labels, **extra) -> scores
    dataset_factory: callable(snr_db=x, **kw) -> (y, labels, noise_samples)
    """
    results = {}
    for snr in snr_db_list:
        y, labels, noise = dataset_factory(snr_db=snr, **factory_kwargs)
        scores = detector_fn(y, noise)
        results[snr] = pd_at_pfa(scores, labels, pfa=pfa)
    return results
