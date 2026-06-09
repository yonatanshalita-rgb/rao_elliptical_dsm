"""
Classical AMF detectors using various covariance estimators.

All detectors implement the interface:
    detector(y, s, W) -> scores   (higher = more likely H1)

where:
    y : (N, n)  test observations
    s : (n,)    target steering vector
    W : (M, n)  training (noise-only) samples

The AMF test statistic is:
    T(y) = s' Q^{-1} y / sqrt(s' Q^{-1} s)

(the denominator normalises to a CFAR detector)

Covariance estimators
---------------------
1. Oracle       : uses true Q  (upper bound)
2. Sample SCM   : hat_Q = (1/M) sum w_i w_i^T
3. Tyler        : fixed-point M-estimator, scale-invariant
4. Huber        : hybrid SCM/Tyler with tuning parameter beta
"""

import numpy as np
from typing import Callable


# ---------------------------------------------------------------------------
# Covariance estimators
# ---------------------------------------------------------------------------

def sample_scm(W: np.ndarray) -> np.ndarray:
    """Standard sample covariance matrix.  W: (M, n)"""
    return (W.T @ W) / W.shape[0]


def tyler_estimator(W: np.ndarray, max_iter: int = 200,
                    tol: float = 1e-6) -> np.ndarray:
    """
    Tyler's M-estimator of scatter (Tyler 1987).

    Fixed-point iteration:
        C_{t+1} = (n/M) sum_i  (w_i w_i^T) / (w_i^T C_t^{-1} w_i)

    Normalised so that trace(C) = n at each step (scale ambiguity).
    Converges to the ML estimator under any elliptical distribution
    with unknown density generator.

    W: (M, n)  training samples
    Returns: (n, n) scatter matrix estimate (proportional to Q)
    """
    M, n = W.shape
    C = np.eye(n)

    for _ in range(max_iter):
        C_inv = np.linalg.inv(C)
        # quadratic forms: (M,)
        qf = np.einsum("mi,ij,mj->m", W, C_inv, W)   # w_i^T C^{-1} w_i
        # weighted outer products
        weights = n / qf                               # (M,)
        C_new = (W.T * weights[None, :]) @ W / M      # (n, n)
        # normalise trace
        C_new *= n / np.trace(C_new)
        # convergence check
        if np.linalg.norm(C_new - C, "fro") < tol:
            C = C_new
            break
        C = C_new

    return C


def huber_estimator(W: np.ndarray, beta: float = 0.9,
                    max_iter: int = 200, tol: float = 1e-6) -> np.ndarray:
    """
    Huber M-estimator of scatter (Kent & Tyler 1991).

    Interpolates between SCM (beta=1) and Tyler (beta->0).
    The weight function is:
        u(t) = beta / max(beta, t/n)
    where t = w^T C^{-1} w is the Mahalanobis distance squared.

    Fixed-point iteration:
        C_{t+1} = (1/M) sum_i u(w_i^T C_t^{-1} w_i) * w_i w_i^T

    beta: fraction of inlier mass (0 < beta <= 1).
    W: (M, n)  training samples
    Returns: (n, n) scatter matrix estimate
    """
    M, n = W.shape
    C = sample_scm(W)   # warm start

    for _ in range(max_iter):
        C_inv = np.linalg.inv(C)
        qf = np.einsum("mi,ij,mj->m", W, C_inv, W)    # (M,)  t_i
        # Huber weight function
        threshold = beta * n
        weights = np.where(qf <= threshold,
                           np.ones_like(qf),
                           threshold / qf)             # (M,)
        C_new = (W.T * weights[None, :]) @ W / M
        if np.linalg.norm(C_new - C, "fro") < tol:
            C = C_new
            break
        C = C_new

    return C


def t_mle_scatter(W: np.ndarray, nu: float, max_iter: int = 200,
                  tol: float = 1e-6) -> np.ndarray:
    """
    MLE scatter matrix for t_nu(0, Sigma) via EM algorithm (known nu).

    EM updates:
        E-step:  w_i = (nu + d) / (nu + y_i' Sigma^{-1} y_i)
        M-step:  Sigma = (1/M) sum_i w_i * y_i y_i'

    Unlike Tyler's M-estimator, this produces a properly scaled Sigma
    (not just the shape), and explicitly exploits the known degrees of
    freedom nu.  The fixed point is the MLE under the t_nu model.

    W:  (M, d)  noise-only training samples
    nu: degrees of freedom (assumed known)
    Returns: (d, d) scatter matrix Sigma_hat.
    """
    M, d = W.shape
    Sigma = np.cov(W.T)   # warm start: sample covariance

    for _ in range(max_iter):
        Q = np.linalg.inv(Sigma)
        A = np.einsum("mi,ij,mj->m", W, Q, W)   # (M,)  Mahalanobis^2
        weights = (nu + d) / (nu + A)            # (M,)  t-EM weights
        Sigma_new = (W.T * weights) @ W / M      # (d, d)
        diff = np.linalg.norm(Sigma_new - Sigma, "fro") / max(np.linalg.norm(Sigma, "fro"), 1e-10)
        Sigma = Sigma_new
        if diff < tol:
            break

    return Sigma


# ---------------------------------------------------------------------------
# AMF test statistic
# ---------------------------------------------------------------------------

def amf_statistic(y: np.ndarray, s: np.ndarray,
                  Q_inv: np.ndarray) -> np.ndarray:
    """
    T(y) = (s^T Q^{-1} y) / sqrt(s^T Q^{-1} s)

    y:     (N, n)
    s:     (n,)
    Q_inv: (n, n)

    Returns: (N,) test statistics
    """
    Qinv_s = Q_inv @ s                              # (n,)
    num = y @ Qinv_s                                # (N,)
    denom = np.sqrt(s @ Qinv_s)                     # scalar
    return num / denom


# ---------------------------------------------------------------------------
# Detector wrappers
# ---------------------------------------------------------------------------

def oracle_detector(y: np.ndarray, s: np.ndarray,
                    W: np.ndarray, Q_true: np.ndarray) -> np.ndarray:
    """AMF with true covariance. Optimal for Gaussian noise; sub-optimal for spike mixture."""
    Q_inv = np.linalg.inv(Q_true)
    return amf_statistic(y, s, Q_inv)


def _log_spike_mixture_density(
    y: np.ndarray,
    mu: np.ndarray,
    Q_inv: np.ndarray,
    log_det_Q: float,
    n: int,
    p_outlier: float,
    outlier_scale: float,
) -> np.ndarray:
    """
    log [(1-p)·N(y; mu, Q) + p·N(y; mu, c²Q)]

    y: (N, n), returns (N,) log-densities.
    Computed via logaddexp for numerical stability.
    """
    dy  = y - mu[None, :]                                           # (N, n)
    qf  = np.einsum("ni,ij,nj->n", dy, Q_inv, dy)                  # (N,)
    c2  = float(outlier_scale) ** 2
    const = -0.5 * n * np.log(2.0 * np.pi)

    log_N_inlier  = const - 0.5 * log_det_Q - 0.5 * qf
    log_N_outlier = const - 0.5 * (log_det_Q + n * np.log(c2)) - 0.5 / c2 * qf

    return np.logaddexp(
        np.log(1.0 - p_outlier) + log_N_inlier,
        np.log(p_outlier)       + log_N_outlier,
    )


def oracle_lrt_detector(
    y: np.ndarray,
    s: np.ndarray,
    W: np.ndarray,
    Q_true: np.ndarray,
    a: float,
    p_outlier: float = 0.01,
    outlier_scale: float = 1000.0,
) -> np.ndarray:
    """
    Oracle LRT for the spike mixture noise model.

    Computes log Λ(y) = log p(y|H1) - log p(y|H0), where
        p(y | H_k) = (1-p)·N(y; k·a·s, Q)  +  p·N(y; k·a·s, c²Q)

    This is the Neyman-Pearson optimal test: it knows Q, the signal
    amplitude a, and the exact texture distribution.  Unlike the Oracle
    AMF (which assumes pure Gaussian noise), this test accounts for the
    spike mixture and is the true performance upper bound.

    a: signal amplitude at this SNR — sqrt(snr_linear / s'Q^{-1}s)
    """
    n = y.shape[1]
    Q_inv = np.linalg.inv(Q_true)
    _, log_det_Q = np.linalg.slogdet(Q_true)

    kw = dict(Q_inv=Q_inv, log_det_Q=log_det_Q, n=n,
              p_outlier=p_outlier, outlier_scale=outlier_scale)

    log_H0 = _log_spike_mixture_density(y, np.zeros(n), **kw)
    log_H1 = _log_spike_mixture_density(y, a * s,        **kw)
    return log_H1 - log_H0


def scm_detector(y: np.ndarray, s: np.ndarray,
                 W: np.ndarray, **kwargs) -> np.ndarray:
    """AMF with sample covariance. Baseline (breaks under outliers)."""
    Q_hat = sample_scm(W)
    Q_inv = np.linalg.inv(Q_hat)
    return amf_statistic(y, s, Q_inv)


def tyler_detector(y: np.ndarray, s: np.ndarray,
                   W: np.ndarray, **kwargs) -> np.ndarray:
    """AMF with Tyler's robust estimator."""
    C_hat = tyler_estimator(W)
    C_inv = np.linalg.inv(C_hat)
    return amf_statistic(y, s, C_inv)


def huber_detector(y: np.ndarray, s: np.ndarray,
                   W: np.ndarray, beta: float = 0.9, **kwargs) -> np.ndarray:
    """AMF with Huber M-estimator."""
    C_hat = huber_estimator(W, beta=beta)
    C_inv = np.linalg.inv(C_hat)
    return amf_statistic(y, s, C_inv)


CLASSICAL_DETECTORS = {
    "oracle":      oracle_detector,
    "oracle_lrt":  oracle_lrt_detector,
    "scm":         scm_detector,
    "tyler":       tyler_detector,
    "huber":       huber_detector,
}
