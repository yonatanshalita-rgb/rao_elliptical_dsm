"""
Elliptical noise data generation.

Model:
    y = a * s + w,   w_i = mu_i * v_i
    v_i ~ N(0, Q)      (correlated Gaussian)
    mu_i ~ p_mu        (heavy-tailed positive scalar)
    s                  (known target vector, unit norm)
    a ~ Bernoulli(0.5) or fixed SNR  (presence/absence)
"""

import numpy as np
import torch
from torch import Tensor
from typing import Literal
from scipy.special import gammaln


# ---------------------------------------------------------------------------
# Covariance constructors
# ---------------------------------------------------------------------------

def ar1_covariance(n: int, rho: float = 0.9) -> np.ndarray:
    """AR(1) covariance: Q[i,j] = rho^|i-j|.  Typical in array processing."""
    idx = np.arange(n)
    return rho ** np.abs(idx[:, None] - idx[None, :])


def exponential_covariance(n: int, rho: float = 0.9) -> np.ndarray:
    """Same as AR(1) — alias for clarity."""
    return ar1_covariance(n, rho)


def random_covariance(n: int, condition_number: float = 100.0, seed: int = 0) -> np.ndarray:
    """Random covariance with controlled condition number."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n))
    Q = A @ A.T / n + np.eye(n) * 0.01
    # rescale eigenvalues to achieve desired condition number
    eigvals, eigvecs = np.linalg.eigh(Q)
    eigvals = np.linspace(1.0, condition_number, n)
    return eigvecs @ np.diag(eigvals) @ eigvecs.T


# ---------------------------------------------------------------------------
# mu distributions  (heavy-tailed positive scalars)
# ---------------------------------------------------------------------------

def sample_mu_spike(n_samples: int, p_outlier: float = 0.01,
                    outlier_scale: float = 1000.0, rng=None) -> np.ndarray:
    """Spike mixture: mu = 1 w.p. (1-p) and mu = outlier_scale w.p. p."""
    if rng is None:
        rng = np.random.default_rng()
    mu = np.ones(n_samples)
    mask = rng.random(n_samples) < p_outlier
    mu[mask] = outlier_scale
    return mu


def sample_mu_compound_gaussian(n_samples: int, shape: float = 0.5,
                                 scale: float = 1.0, rng=None) -> np.ndarray:
    """mu ~ sqrt(Gamma(shape, scale)), giving a K-distribution for w."""
    if rng is None:
        rng = np.random.default_rng()
    tau = rng.gamma(shape, scale, size=n_samples)   # texture variable
    return np.sqrt(tau)


def sample_mu_t(n_samples: int, df: float = 3.0, rng=None) -> np.ndarray:
    """mu = sqrt(df / chi2(df)), consistent with multivariate-t noise."""
    if rng is None:
        rng = np.random.default_rng()
    chi2 = rng.chisquare(df, size=n_samples)
    return np.sqrt(df / chi2)


MU_DISTRIBUTIONS = {
    "spike": sample_mu_spike,
    "compound_gaussian": sample_mu_compound_gaussian,
    "t": sample_mu_t,
}


# ---------------------------------------------------------------------------
# MGGD scale normalization
# ---------------------------------------------------------------------------

def compute_m_norm(beta: float, p: int) -> float:
    """
    Compute m such that E[tau^2] = p for MGGD(p, M, m, beta).

    Derivation: tau^{2*beta} ~ Gamma(p/(2*beta), 2*m^beta), so
        E[tau^2] = 2^{1/beta} * m * Gamma((p+2)/(2*beta)) / Gamma(p/(2*beta)).
    Setting E[tau^2] = p and solving for m (in log-space to avoid overflow):
        log m = log(p) + gammaln(p/(2*beta)) - log(2)/beta - gammaln((p+2)/(2*beta))
    """
    log_m = (np.log(p)
             + gammaln(p / (2.0 * beta))
             - np.log(2.0) / beta
             - gammaln((p + 2.0) / (2.0 * beta)))
    return float(np.exp(log_m))


# ---------------------------------------------------------------------------
# MGGD sampler and true score
# ---------------------------------------------------------------------------

def sample_mggd(n: int, p: int, M: np.ndarray, m: float, beta: float,
                seed: int = 0) -> np.ndarray:
    """
    Sample from MGGD(M, m, beta) via the stochastic representation
        x = tau * L * u
    where u is uniform on the unit sphere, L is the Cholesky factor of M,
    and tau^{2*beta} ~ Gamma(p/(2*beta), 2*m^beta).

    Returns: (n, p) float32 array.
    """
    rng = np.random.default_rng(seed)
    L = np.linalg.cholesky(M)          # M = L L^T

    z = rng.standard_normal((n, p))
    norms = np.linalg.norm(z, axis=1, keepdims=True)
    u = z / norms                      # uniform on unit sphere, (n, p)

    shape = p / (2.0 * beta)
    scale = 2.0 * (m ** beta)
    g = rng.gamma(shape, scale, size=n)   # g = tau^{2*beta}
    tau = g ** (1.0 / (2.0 * beta))       # (n,)

    x = tau[:, None] * (u @ L.T)          # (n, p)
    return x.astype(np.float32)


def mggd_true_score(x: np.ndarray, M_inv: np.ndarray,
                    m: float, beta: float) -> np.ndarray:
    """
    True score of MGGD(M, m, beta):
        ∇_x log p = -(beta / m^beta) * (x^T M^{-1} x)^{beta-1} * M^{-1} x

    x:     (n, p)
    Returns: (n, p) float32
    """
    Minv_x = x @ M_inv.T                               # (n, p)
    y = (x * Minv_x).sum(axis=1)                       # (n,)  x^T M^{-1} x
    weight = -(beta / (m ** beta)) * (y ** (beta - 1)) # (n,)
    return (weight[:, None] * Minv_x).astype(np.float32)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class EllipticalDataset(torch.utils.data.Dataset):
    """
    Returns noise samples w_i and (optionally) labelled observations y_i.

    Each item: dict with keys
        'w'     : noise vector  (n,)
        'y'     : observation   (n,)   -- a*s + w
        'label' : 0 or 1        scalar -- target absence/presence
    """

    def __init__(
        self,
        n_samples: int,
        n: int = 16,
        snr_db: float = 10.0,
        cov_type: Literal["ar1", "random"] = "ar1",
        cov_kwargs: dict | None = None,
        mu_dist: Literal["spike", "compound_gaussian", "t"] = "spike",
        mu_kwargs: dict | None = None,
        seed: int = 42,
    ):
        self.n = n
        rng = np.random.default_rng(seed)

        # --- covariance ---
        cov_kwargs = cov_kwargs or {}
        if cov_type == "ar1":
            Q = ar1_covariance(n, **cov_kwargs)
        elif cov_type == "random":
            Q = random_covariance(n, **cov_kwargs)
        else:
            raise ValueError(f"Unknown cov_type: {cov_type}")
        self.Q = Q
        self.Q_inv = np.linalg.inv(Q)
        L = np.linalg.cholesky(Q)   # Q = L L^T

        # --- target vector s (unit norm) ---
        s = np.ones(n) / np.sqrt(n)
        self.s = s

        # --- SNR: amplitude a = sqrt(SNR * s' Q^{-1} s)^{-1} ... ---
        # we define SNR = a^2 * s' Q^{-1} s
        snr_linear = 10 ** (snr_db / 10.0)
        a = np.sqrt(snr_linear / (s @ self.Q_inv @ s))
        self.a = a

        # --- generate noise ---
        mu_kwargs = mu_kwargs or {}
        mu_fn = MU_DISTRIBUTIONS[mu_dist]
        mu = mu_fn(n_samples, rng=rng, **mu_kwargs)           # (N,)
        v = rng.standard_normal((n_samples, n)) @ L.T          # (N, n)
        w = mu[:, None] * v                                    # (N, n)  elliptical

        # --- generate observations ---
        labels = rng.integers(0, 2, size=n_samples)            # 0/1
        y = labels[:, None] * a * s[None, :] + w

        self.w = torch.tensor(w, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.s_tensor = torch.tensor(s, dtype=torch.float32)

    def __len__(self):
        return len(self.w)

    def __getitem__(self, idx):
        return {
            "w": self.w[idx],
            "y": self.y[idx],
            "label": self.labels[idx],
        }


def make_dataloaders(
    n_train: int = 50_000,
    n_val: int = 10_000,
    n_test: int = 10_000,
    batch_size: int = 512,
    **dataset_kwargs,
):
    train_ds = EllipticalDataset(n_train, seed=0, **dataset_kwargs)
    val_ds   = EllipticalDataset(n_val,   seed=1, **dataset_kwargs)
    test_ds  = EllipticalDataset(n_test,  seed=2, **dataset_kwargs)

    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl   = torch.utils.data.DataLoader(val_ds,   batch_size=batch_size)
    test_dl  = torch.utils.data.DataLoader(test_ds,  batch_size=batch_size)

    return train_dl, val_dl, test_dl, train_ds
