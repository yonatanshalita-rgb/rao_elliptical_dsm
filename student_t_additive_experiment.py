"""
Experiment: additive signal detection under multivariate-t background noise.

Observation model:
    y_i = w_i + theta_i * s
    w_i ~ t_nu(0, Sigma)     (multivariate Student-t via scale-mixture)
    H0: theta_i = 0
    H1: theta_i ~ Uniform(0, theta_max)

theta_max is defined by the SNR grid:  SNR_linear = theta_max^2 * s'Q^{-1}s
(i.e., SNR corresponds to the maximum possible signal power under H1)

Oracle benchmarks (know Q^{-1} and nu):
    - Oracle t-Rao  : optimal score test near theta=0
                      T = w(A)*B / sqrt([(nu+d)/(nu+d+2)]*C)
    - Oracle t-GLRT : generalised LRT via MLE substitution
                      T = (nu+d)/2 * log((nu+A)/(nu+A-B^2/C))  when B>0, else 0
    - Oracle Gaussian AMF : standard AMF with true Q^{-1} (ignores t-tails)
                      T = B / sqrt(C)

Classical baselines (estimate Q from noise samples):
    - SCM AMF, Tyler AMF, Huber AMF

DSM neural models (trained on pure noise w_i ~ t_nu(0, Sigma)):
    - DSM Linear (MSE), DSM Linear (Huber), DSM Two-Branch
"""

import csv
import datetime
import numpy as np
import torch
import torch.utils.data
import matplotlib.pyplot as plt
from pathlib import Path

from data.generate import ar1_covariance, sample_mu_t
from models.score_models import build_model
from models.train import train
from baselines.classical import sample_scm, tyler_estimator, huber_estimator, t_mle_scatter, amf_statistic
from eval.detector import NeuralDetector
from eval.metrics import compute_roc, pd_at_pfa


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

N_DIM         = 16
NU            = 3.0          # Student-t degrees of freedom
COV_TYPE      = "ar1"
COV_KWARGS    = {"rho": 0.9}

N_TRAIN       = 50_000
N_VAL         = 10_000
N_TEST        = 50_000
N_CALIB       = 5_000
BATCH_SIZE    = 512
N_EPOCHS      = 60
LR            = 1e-3

SNR_DB_LIST      = [-10, -5, 0, 5, 10, 15]
SNR_TRAIN_DB     = 10
SNR_SMALL_EVAL_DB = 0     # defines theta_small for the "fixed_small" evaluation
PFA              = 1e-2
N_SEEDS          = 3
WARMUP_EPOCHS    = 12

BEST_PARAMS = {
    "two_branch":   {"sigma": 0.5, "delta": None, "warmup_epochs": WARMUP_EPOCHS, "grad_clip": None, "per_sample_clip": None},
    "mlp_score":    {"sigma": 0.5, "delta": None, "warmup_epochs": 0,             "grad_clip": None, "per_sample_clip": None},
}

DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_DIR    = Path("checkpoints_t")
FIG_DIR     = Path("figures_t")
RESULTS_DIR = Path("results")
FIG_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

MODEL_TYPES = ["two_branch", "mlp_score"]
MODEL_LABELS = {
    "two_branch":   "DSM Two-Branch (sigma=0.5)",
    "mlp_score":    "DSM Unconstrained MLP (sigma=0.5)",
}


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class StudentTAdditiveDataset(torch.utils.data.Dataset):
    """
    y_i = w_i + theta_i * s
    w_i ~ t_nu(0, Sigma)  via scale-mixture: w = sqrt(nu/chi2) * N(0, Sigma)
    H0: theta_i = 0
    H1: theta_i ~ Uniform(0, theta_max)  if theta_mode="uniform"
        theta_i  = theta_max             if theta_mode="fixed"
    theta_max = sqrt(SNR_linear / C),  C = s' Q^{-1} s.

    Labels are Bernoulli(0.5) — roughly equal H0 / H1 split.
    """

    def __init__(
        self,
        n_samples: int,
        n: int = 16,
        nu: float = 3.0,
        snr_db: float = 10.0,
        cov_type: str = "ar1",
        cov_kwargs: dict | None = None,
        theta_mode: str = "uniform",
        seed: int = 42,
    ):
        rng = np.random.default_rng(seed)
        cov_kwargs = cov_kwargs or {}

        if cov_type == "ar1":
            Sigma = ar1_covariance(n, **cov_kwargs)
        else:
            raise ValueError(f"Unknown cov_type: {cov_type}")

        Q_inv = np.linalg.inv(Sigma)
        L     = np.linalg.cholesky(Sigma)   # Sigma = L L^T

        s = np.ones(n) / np.sqrt(n)
        C = float(s @ Q_inv @ s)            # signal energy in Q^{-1}-metric

        snr_linear = 10 ** (snr_db / 10.0)
        theta_max  = np.sqrt(snr_linear / C)

        # Student-t noise via scale mixture
        mu_scale = sample_mu_t(n_samples, df=nu, rng=rng)    # (N,)
        v        = rng.standard_normal((n_samples, n)) @ L.T  # (N, n)
        w        = mu_scale[:, None] * v                      # (N, n)

        # labels: H0 = 0, H1 = 1  (50/50 split)
        labels = rng.integers(0, 2, size=n_samples)
        if theta_mode == "uniform":
            theta_h1 = rng.uniform(0.0, theta_max, size=n_samples)
        elif theta_mode == "fixed":
            theta_h1 = np.full(n_samples, theta_max)
        else:
            raise ValueError(f"Unknown theta_mode: {theta_mode}")
        theta = np.where(labels == 1, theta_h1, 0.0)
        y = w + theta[:, None] * s[None, :]

        self.Sigma     = Sigma
        self.Q_inv     = Q_inv
        self.s         = s
        self.C         = C
        self.theta_max = theta_max
        self.nu        = nu
        self.n         = n

        self.w      = torch.tensor(w,      dtype=torch.float32)
        self.y      = torch.tensor(y,      dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.theta  = torch.tensor(theta,  dtype=torch.float32)

    def __len__(self):
        return len(self.w)

    def __getitem__(self, idx):
        return {"w": self.w[idx], "y": self.y[idx], "label": self.labels[idx]}


def make_t_dataloaders(n_train, n_val, n_test, batch_size, **dataset_kwargs):
    # training/validation always use noise samples only (theta_mode irrelevant for w)
    train_ds = StudentTAdditiveDataset(n_train, seed=0, theta_mode="uniform", **dataset_kwargs)
    val_ds   = StudentTAdditiveDataset(n_val,   seed=1, theta_mode="uniform", **dataset_kwargs)
    test_ds  = StudentTAdditiveDataset(n_test,  seed=2, theta_mode="uniform", **dataset_kwargs)
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl   = torch.utils.data.DataLoader(val_ds,   batch_size=batch_size)
    test_dl  = torch.utils.data.DataLoader(test_ds,  batch_size=batch_size)
    return train_dl, val_dl, test_dl, train_ds


# ---------------------------------------------------------------------------
# Oracle detectors  (know Q_inv and nu)
# ---------------------------------------------------------------------------

def oracle_gaussian_amf(y: np.ndarray, s: np.ndarray, W: np.ndarray,
                        Q_inv: np.ndarray, **kwargs) -> np.ndarray:
    """
    Standard AMF with true Q^{-1}.  T = B / sqrt(C).
    Optimal for Gaussian noise; ignores t-distributed tails.
    """
    B = y @ (Q_inv @ s)                 # (N,)
    C = float(s @ Q_inv @ s)
    return B / np.sqrt(C)


def oracle_t_rao(y: np.ndarray, s: np.ndarray, W: np.ndarray,
                 Q_inv: np.ndarray, nu: float, **kwargs) -> np.ndarray:
    """
    Optimal score test (Rao test) for H0: theta=0 under t_nu background.

        w(A) = (nu + d) / (nu + A)          adaptive down-weighting
        T    = w(A) * B / sqrt([(nu+d)/(nu+d+2)] * C)

    Reduces to the Gaussian AMF as nu -> inf.
    """
    d   = y.shape[1]
    A   = np.einsum("ni,ij,nj->n", y, Q_inv, y)    # (N,)  y' Q^{-1} y
    B   = y @ (Q_inv @ s)                            # (N,)  y' Q^{-1} s
    C   = float(s @ Q_inv @ s)
    w_A = (nu + d) / (nu + A)
    fisher_scale = np.sqrt((nu + d) / (nu + d + 2) * C)
    return w_A * B / fisher_scale


def oracle_t_glrt(y: np.ndarray, s: np.ndarray, W: np.ndarray,
                  Q_inv: np.ndarray, nu: float, **kwargs) -> np.ndarray:
    """
    Generalised LRT via MLE substitution (one-sided in theta).

        T_GLRT = (nu+d)/2 * log((nu+A) / (nu+A-B^2/C))   if B > 0
               = 0                                          if B <= 0

    By Cauchy-Schwarz:  nu + A - B^2/C >= nu > 0,
    so the argument of log is always >= 1 and T_GLRT >= 0.
    """
    d     = y.shape[1]
    A     = np.einsum("ni,ij,nj->n", y, Q_inv, y)   # (N,)
    B     = y @ (Q_inv @ s)                           # (N,)
    C     = float(s @ Q_inv @ s)
    denom = nu + A - B ** 2 / C                       # (N,)  >= nu > 0
    T = np.where(
        B > 0,
        0.5 * (nu + d) * np.log((nu + A) / denom),
        0.0,
    )
    return T


def oracle_np(y: np.ndarray, s: np.ndarray, W: np.ndarray,
              Q_inv: np.ndarray, nu: float, theta: float, **kwargs) -> np.ndarray:
    """
    Neyman-Pearson optimal test for the simple vs simple problem
    H0: y ~ t_nu(0, Sigma)  vs  H1: y ~ t_nu(theta*s, Sigma)  with known theta.

        T_NP = (nu+d)/2 * log( (nu+A) / (nu + ||y - theta*s||^2_Q) )
             = (nu+d)/2 * log( (nu+A) / (nu + A - 2*theta*B + theta^2*C) )

    The denominator nu + (y-theta*s)'Q(y-theta*s) >= nu > 0 always.
    By the NP lemma this is UMP — highest Pd at every Pfa, hence highest AUC.
    Requires knowing the true theta (oracle).
    """
    d     = y.shape[1]
    A     = np.einsum("ni,ij,nj->n", y, Q_inv, y)   # (N,)
    B     = y @ (Q_inv @ s)                           # (N,)
    C     = float(s @ Q_inv @ s)
    denom = nu + A - 2.0 * theta * B + theta ** 2 * C  # = nu + ||y-theta*s||^2_Q
    return 0.5 * (nu + d) * np.log((nu + A) / denom)


def oracle_q_sigma_rao(y: np.ndarray, s: np.ndarray, W: np.ndarray,
                       Sigma: np.ndarray, nu: float, sigma_dsm: float,
                       n_tau: int = 500, **kwargs) -> np.ndarray:
    """
    Oracle Rao test for the convolved distribution q_sigma = t_nu * N(0, sigma_dsm^2 I).

    Uses Tweedie's formula:
        score = nabla log q_sigma(y) = -E[(tau*Sigma + sigma^2*I)^{-1} | y] @ y

    where tau | y has density proportional to:
        N(y; 0, tau*Sigma + sigma^2*I) * InvGamma(tau; nu/2, nu/2)

    Computed via quadrature on a log-uniform tau grid.
    All N samples are processed simultaneously via two matrix multiplies:
        (N,d)@(d,K) and (N,K)@(K,d).

    This is the exact theoretical target that a perfectly trained two-branch
    at DSM noise level sigma_dsm converges to.
    """
    N, d = y.shape

    # Eigendecomposition of Sigma: Sigma = U diag(eigvals) U^T
    eigvals, U = np.linalg.eigh(Sigma)          # eigvals (d,), U (d,d)
    y_tilde = y @ U                              # (N, d)  — project to eigenbasis
    s_tilde = U.T @ s                            # (d,)

    # Log-uniform quadrature grid for tau in (tau_min, tau_max)
    taus = np.exp(np.linspace(np.log(1e-3), np.log(1e3), n_tau))   # (K,)
    log_dtau = np.log(taus[-1] / taus[0]) / (n_tau - 1)            # uniform spacing in log

    # scale[k, i] = tau_k * eigval_i + sigma_dsm^2  — eigenvalues of (tau*Sigma + sigma^2*I)
    scale = taus[:, None] * eigvals[None, :] + sigma_dsm ** 2      # (K, d)

    # Log unnormalized posterior weights for each (sample, tau) pair.
    # log p(tau | y) ∝ log N(y; 0, tau*Sigma + sigma^2*I) + log IG(tau; nu/2, nu/2)
    #                 + log(tau)  [Jacobian for log-tau integration]
    #
    # log N = -0.5 * sum_i log(scale_i) - 0.5 * sum_i y_tilde_i^2 / scale_i
    log_det_term = -0.5 * np.log(scale).sum(axis=1)                # (K,)

    # quad_form[n, k] = sum_i y_tilde[n,i]^2 / scale[k,i]
    # = (N,d) @ (d,K)  via  y_tilde^2 @ (1/scale)^T
    quad_form = (y_tilde ** 2) @ (1.0 / scale).T                   # (N, K)

    alpha = nu / 2.0
    log_prior = -(alpha + 1.0) * np.log(taus) - alpha / taus       # (K,)  InvGamma log-density
    log_jac   = np.log(taus)                                        # (K,)  Jacobian for d(log tau)

    log_w = (log_det_term + log_prior + log_jac)[None, :] - 0.5 * quad_form  # (N, K)
    log_w -= log_w.max(axis=1, keepdims=True)   # numerical stability
    w = np.exp(log_w)
    w *= log_dtau                                # trapezoidal weights (uniform on log scale)
    w /= w.sum(axis=1, keepdims=True)           # normalize to sum-1  (N, K)

    # E[(tau*Sigma + sigma^2*I)^{-1} | y] in eigenbasis = diag(E[1/scale_i | y])
    # e_inv_scale[n, i] = sum_k w[n,k] / scale[k,i]  = (N,K) @ (K,d)
    e_inv_scale = w @ (1.0 / scale)             # (N, d)

    # Detection statistic: -s' nabla log q_sigma(y)
    #   = -s' E[-(tau*Sigma+sigma^2*I)^{-1} | y] y
    #   = sum_i s_tilde_i * y_tilde_i * E[1/(tau*d_i+sigma^2) | y]
    # Positive for H1 (y aligned with s), consistent with t-Rao = w(A)*B > 0.
    return (y_tilde * s_tilde[None, :] * e_inv_scale).sum(axis=1)  # (N,)


# ---------------------------------------------------------------------------
# Practical detectors  (estimated covariance, known nu)
# ---------------------------------------------------------------------------

def practical_t_rao(y: np.ndarray, s: np.ndarray, Q_est: np.ndarray,
                    nu: float) -> np.ndarray:
    """t-Rao using estimated precision Q_est = Sigma_hat^{-1} (from t-MLE)."""
    d   = y.shape[1]
    A   = np.einsum("ni,ij,nj->n", y, Q_est, y)
    B   = y @ (Q_est @ s)
    C   = float(s @ Q_est @ s)
    w_A = (nu + d) / (nu + A)
    fisher_scale = np.sqrt((nu + d) / (nu + d + 2) * C)
    return w_A * B / fisher_scale


def practical_t_glrt(y: np.ndarray, s: np.ndarray, Q_est: np.ndarray,
                     nu: float) -> np.ndarray:
    """t-GLRT using estimated precision Q_est = Sigma_hat^{-1} (from t-MLE)."""
    d     = y.shape[1]
    A     = np.einsum("ni,ij,nj->n", y, Q_est, y)
    B     = y @ (Q_est @ s)
    C     = float(s @ Q_est @ s)
    denom = nu + A - B ** 2 / C
    return np.where(B > 0, 0.5 * (nu + d) * np.log((nu + A) / denom), 0.0)


def compute_covariance_estimates(W: np.ndarray, nu: float) -> dict:
    """
    Compute all scatter/precision estimates from noise samples W once.
    Returns dict: name -> Q_inv  (precision matrix ready for detector formulas).
    Reusing a single call avoids redundant estimation across SNR sweep.
    """
    t_mle_cov = t_mle_scatter(W, nu)
    tyler_cov = tyler_estimator(W)
    huber_cov = huber_estimator(W)
    scm_cov   = sample_scm(W)
    return {
        "t_mle": np.linalg.inv(t_mle_cov),
        "tyler": np.linalg.inv(tyler_cov),
        "huber": np.linalg.inv(huber_cov),
        "scm":   np.linalg.inv(scm_cov),
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_all_models(train_dl, val_dl, best_params: dict, seed: int = 0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    models = {}
    for mtype in MODEL_TYPES:
        params          = best_params.get(mtype, {})
        sigma           = params.get("sigma", 0.1)
        delta           = params.get("delta") or 1.0
        warmup          = params.get("warmup_epochs", 0)
        grad_clip       = params.get("grad_clip", None)
        per_sample_clip = params.get("per_sample_clip", None)

        print(f"\n{'='*50}")
        print(f"Training {MODEL_LABELS[mtype]}  (sigma={sigma}"
              + (f", delta={delta}" if mtype == "linear_huber" else "")
              + (f", warmup={warmup}" if warmup else "")
              + (f", psc={per_sample_clip}" if per_sample_clip else "")
              + ")")
        model = build_model(mtype, n=N_DIM)
        train(
            model, mtype, train_dl, val_dl,
            n_epochs=N_EPOCHS, lr=LR,
            sigma=sigma, huber_delta=delta,
            device=DEVICE, save_dir=SAVE_DIR, verbose=True,
            warmup_epochs=warmup, grad_clip=grad_clip,
            per_sample_clip=per_sample_clip,
        )
        ckpt = SAVE_DIR / f"{mtype}_best.pt"
        if ckpt.exists():
            model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
        models[mtype] = model
    return models


# ---------------------------------------------------------------------------
# Evaluate at one SNR
# ---------------------------------------------------------------------------

def evaluate_all(models, snr_db: float, s: np.ndarray,
                 Q_inv: np.ndarray, Sigma: np.ndarray, nu: float,
                 calib_noise: np.ndarray, cov_ests: dict,
                 theta_mode: str = "uniform"):
    """
    Runs all detectors at the given SNR.

    cov_ests: precomputed precision matrices from compute_covariance_estimates().
              Passed in so covariance estimation runs only once per seed, not
              once per SNR point.

    Returns dict: method_name -> (fpr, tpr, auc, pd)
    """
    test_ds = StudentTAdditiveDataset(
        N_TEST, seed=99, n=N_DIM, nu=nu, snr_db=snr_db,
        cov_type=COV_TYPE, cov_kwargs=COV_KWARGS, theta_mode=theta_mode,
    )
    y_np      = test_ds.y.numpy()
    labels_np = test_ds.labels.numpy()

    def _record(name, scores):
        fpr, tpr, _, auc_val = compute_roc(scores, labels_np)
        results[name] = (fpr, tpr, auc_val, pd_at_pfa(scores, labels_np, pfa=PFA))

    results = {}

    # oracle detectors — know true Q_inv, Sigma, nu
    sigma_dsm = BEST_PARAMS["two_branch"]["sigma"]
    theta_max = test_ds.theta_max
    _record("oracle_np",          oracle_np(y_np, s, None, Q_inv=Q_inv, nu=nu, theta=theta_max))
    _record("oracle_gaussian",    oracle_gaussian_amf(y_np, s, None, Q_inv=Q_inv))
    _record("oracle_t_rao",       oracle_t_rao(y_np, s, None, Q_inv=Q_inv, nu=nu))
    _record("oracle_t_glrt",      oracle_t_glrt(y_np, s, None, Q_inv=Q_inv, nu=nu))
    _record("oracle_q_sigma_rao", oracle_q_sigma_rao(y_np, s, None, Sigma=Sigma, nu=nu, sigma_dsm=sigma_dsm))

    # practical detectors — t-MLE estimated covariance, known nu
    Q_tmle = cov_ests["t_mle"]
    _record("practical_t_rao",  practical_t_rao(y_np, s, Q_tmle, nu))
    _record("practical_t_glrt", practical_t_glrt(y_np, s, Q_tmle, nu))

    # AMF baselines — estimated covariance, ignore nu
    for name, key in [("scm", "scm"), ("tyler", "tyler"), ("huber", "huber")]:
        _record(name, amf_statistic(y_np, s, cov_ests[key]))

    # neural DSM detectors
    for mtype, model in models.items():
        detector = NeuralDetector(model, s, device=DEVICE)
        detector.calibrate(torch.tensor(calib_noise, dtype=torch.float32))
        _record(mtype, detector.score(y_np))

    return results, labels_np


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

COLORS = {
    "oracle_np":          "crimson",
    "oracle_gaussian":    "black",
    "oracle_t_rao":       "darkgreen",
    "oracle_t_glrt":      "purple",
    "oracle_q_sigma_rao": "magenta",
    "practical_t_rao":    "olive",
    "practical_t_glrt":   "darkorange",
    "scm":                "gray",
    "tyler":              "steelblue",
    "huber":              "teal",
    "linear_mse":         "red",
    "linear_huber":       "salmon",
    "two_branch":         "green",
    "mlp_score":          "darkcyan",
}

ALL_LABELS = {
    "oracle_np":          "Oracle NP (true Q, nu, theta — UMP)",
    "oracle_gaussian":    "Oracle Gaussian AMF (true Q; no nu)",
    "oracle_t_rao":       "Oracle t-Rao (true Q, true nu)",
    "oracle_t_glrt":      "Oracle t-GLRT (true Q, true nu)",
    "oracle_q_sigma_rao": f"Oracle q_sigma-Rao (numerical, true Q+nu, sigma={BEST_PARAMS['two_branch']['sigma']})",
    "practical_t_rao":    "Practical t-Rao (t-MLE Qhat, known nu) [+nu]",
    "practical_t_glrt":   "Practical t-GLRT (t-MLE Qhat, known nu) [+nu]",
    "scm":                "SCM AMF (no nu)",
    "tyler":              "Tyler AMF (no nu)",
    "huber":              "Huber AMF (no nu)",
    **MODEL_LABELS,
}


def plot_roc(results: dict, snr_db: float):
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, (fpr, tpr, auc_val, _) in results.items():
        label = f"{ALL_LABELS.get(name, name)}  (AUC={auc_val:.3f})"
        ax.plot(fpr, tpr, color=COLORS.get(name, "purple"), label=label)
    ax.set_xlabel("False Alarm Rate (Pfa)")
    ax.set_ylabel("Detection Rate (Pd)")
    ax.set_title(f"ROC — t_nu={NU} background, SNR = {snr_db} dB\n"
                 f"(SNR = theta_max² × s'Q⁻¹s)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"roc_t_snr{snr_db:+.0f}dB.png", dpi=150)
    plt.close(fig)
    print(f"Saved ROC plot for SNR={snr_db} dB")


def plot_pd_vs_snr(pd_curves: dict):
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, pd_list in pd_curves.items():
        ax.plot(SNR_DB_LIST, pd_list, marker="o",
                color=COLORS.get(name, "purple"),
                label=ALL_LABELS.get(name, name))
    ax.set_xlabel("SNR (dB)  [= 10 log10(theta_max² × s'Q⁻¹s)]")
    ax.set_ylabel(f"Pd @ Pfa={PFA}")
    ax.set_title(f"Detection vs SNR — t_nu={NU} background, additive signal")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "pd_vs_snr_t.png", dpi=150)
    plt.close(fig)
    print("Saved Pd-vs-SNR plot")


def plot_weight_function(model, label: str):
    from models.score_models import TwoBranchScore
    if not isinstance(model, TwoBranchScore):
        return
    # overlay the theoretical t-weight: u(d) = (nu + n) / (nu + d)
    d_vals = torch.linspace(0, 200, 500)
    with torch.no_grad():
        u_learned = model.get_weight_function(d_vals).numpy()
    d_np = d_vals.numpy()
    u_theory = (NU + N_DIM) / (NU + d_np)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(d_np, u_learned, label="Learned u(d)")
    ax.plot(d_np, u_theory,  label=f"Theoretical t-weight (nu={NU})", linestyle="--")
    ax.set_xlabel("d = ||Ay||²  (Mahalanobis proxy)")
    ax.set_ylabel("u(d)  [scalar weight]")
    ax.set_title(f"Learned vs theoretical weight — {label}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "weight_fn_t.png", dpi=150)
    plt.close(fig)
    print("Saved weight function plot")


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

def save_results_csv(
    all_results: dict,
    pd_curves: dict,
    theta_mode: str,
    run_id: str,
):
    """
    Writes two CSV files to RESULTS_DIR:

    student_t_summary_{run_id}.csv
        detector, theta_mode, snr_train_db, auc, pd_at_pfa

    student_t_pdcurve_{run_id}.csv
        detector, theta_mode, snr_db, pd
    """
    meta = dict(
        nu=NU, n_dim=N_DIM, cov_rho=COV_KWARGS["rho"],
        n_train=N_TRAIN, n_test=N_TEST, pfa=PFA,
        theta_mode=theta_mode, run_id=run_id,
    )

    # --- summary ---
    summary_path = RESULTS_DIR / f"student_t_summary_{run_id}.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "run_id", "theta_mode", "nu", "n_dim", "cov_rho",
            "n_train", "n_test", "pfa",
            "detector", "snr_train_db", "auc", "pd_at_pfa",
        ])
        for name, (_, _, auc_val, pd) in all_results.items():
            writer.writerow([
                run_id, theta_mode, NU, N_DIM, COV_KWARGS["rho"],
                N_TRAIN, N_TEST, PFA,
                name, SNR_TRAIN_DB, f"{auc_val:.6f}", f"{pd:.6f}",
            ])
    print(f"Saved summary  -> {summary_path}")

    # --- Pd vs SNR curve ---
    curve_path = RESULTS_DIR / f"student_t_pdcurve_{run_id}.csv"
    with open(curve_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "run_id", "theta_mode", "nu", "n_dim", "cov_rho",
            "n_train", "n_test", "pfa",
            "detector", "snr_db", "pd",
        ])
        for name, pd_list in pd_curves.items():
            for snr, pd in zip(SNR_DB_LIST, pd_list):
                writer.writerow([
                    run_id, theta_mode, NU, N_DIM, COV_KWARGS["rho"],
                    N_TRAIN, N_TEST, PFA,
                    name, snr, f"{pd:.6f}",
                ])
    print(f"Saved Pd curve -> {curve_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    print(f"Student-t noise: nu={NU},  AR(1) covariance rho={COV_KWARGS['rho']}")
    print(f"Signal model: y = w + theta*s,  H1: theta = theta_max (fixed)")

    # covariance and steering vector are fixed (deterministic from cov_type/kwargs)
    _ref_ds = StudentTAdditiveDataset(
        10, seed=0, n=N_DIM, nu=NU, snr_db=SNR_TRAIN_DB,
        cov_type=COV_TYPE, cov_kwargs=COV_KWARGS,
    )
    s     = _ref_ds.s
    Q_inv = _ref_ds.Q_inv
    Sigma = _ref_ds.Sigma

    # per-seed accumulators: {eval_label: {detector: [auc, ...], ...}}
    # "fixed_small": theta = theta_max(SNR_SMALL_EVAL_DB)  — local-optimality regime
    # "fixed_large": theta = theta_max(SNR_TRAIN_DB)       — suppression regime
    EVAL_MODES = {
        "fixed_small": SNR_SMALL_EVAL_DB,
        "fixed_large": SNR_TRAIN_DB,
    }
    seed_auc = {mode: {} for mode in EVAL_MODES}
    seed_pd  = {mode: {} for mode in EVAL_MODES}
    last_models   = None
    last_cov_ests = None

    for seed in range(N_SEEDS):
        print(f"\n{'#'*60}\n# Seed {seed+1}/{N_SEEDS}\n{'#'*60}")

        # fresh data and calibration for this seed
        train_dl, val_dl, _, _ = make_t_dataloaders(
            n_train=N_TRAIN, n_val=N_VAL, n_test=N_TEST,
            batch_size=BATCH_SIZE,
            n=N_DIM, nu=NU, snr_db=SNR_TRAIN_DB,
            cov_type=COV_TYPE, cov_kwargs=COV_KWARGS,
        )
        calib_ds = StudentTAdditiveDataset(
            N_CALIB, seed=100 + seed, n=N_DIM, nu=NU, snr_db=SNR_TRAIN_DB,
            cov_type=COV_TYPE, cov_kwargs=COV_KWARGS,
        )
        calib_noise = calib_ds.w.numpy()

        models = train_all_models(train_dl, val_dl, best_params=BEST_PARAMS, seed=seed)
        last_models = models

        # compute all covariance estimates once from this seed's calibration noise
        print("  Computing covariance estimates...")
        cov_ests = compute_covariance_estimates(calib_noise, NU)
        last_cov_ests = cov_ests

        for mode, snr_eval in EVAL_MODES.items():
            all_res, _ = evaluate_all(
                models, snr_eval, s, Q_inv, Sigma, NU,
                calib_noise, cov_ests, theta_mode="fixed",
            )
            for name, (_, _, auc_val, pd_val) in all_res.items():
                seed_auc[mode].setdefault(name, []).append(auc_val)
                seed_pd[mode].setdefault(name, []).append(pd_val)

    # -------------------------------------------------------------------------
    # Aggregate and report
    # -------------------------------------------------------------------------
    calib_noise_last = StudentTAdditiveDataset(
        N_CALIB, seed=100 + N_SEEDS - 1, n=N_DIM, nu=NU, snr_db=SNR_TRAIN_DB,
        cov_type=COV_TYPE, cov_kwargs=COV_KWARGS,
    ).w.numpy()

    for mode, snr_eval in EVAL_MODES.items():
        out_dir = FIG_DIR / mode
        out_dir.mkdir(exist_ok=True)

        print(f"\n{'='*60}")
        print(f"Evaluation mode: '{mode}'  theta=theta_max(SNR={snr_eval} dB)  -> {out_dir}")
        print(f"{'='*60}")
        print(f"\n--- Results at SNR={snr_eval} dB  (mean±std over {N_SEEDS} seeds) ---")

        mean_results = {}
        for name in seed_auc[mode]:
            aucs = np.array(seed_auc[mode][name])
            pds  = np.array(seed_pd[mode][name])
            mean_results[name] = (np.mean(aucs), np.std(aucs), np.mean(pds), np.std(pds))
            label = ALL_LABELS.get(name, name)
            print(f"  {label:46s}  AUC={np.mean(aucs):.4f}±{np.std(aucs):.4f}  "
                  f"Pd={np.mean(pds):.4f}±{np.std(pds):.4f}")

        # Pd vs SNR curves — last seed's models and covariance estimates
        pd_curves = {name: [] for name in mean_results}
        for snr in SNR_DB_LIST:
            res, _ = evaluate_all(
                last_models, snr, s, Q_inv, Sigma, NU,
                calib_noise_last, last_cov_ests, theta_mode="fixed",
            )
            for name in res:
                pd_curves[name].append(res[name][3])

        mode_str = rf"$\theta = \theta_{{max}}$ (SNR = {snr_eval} dB, fixed)"

        # ROC at eval SNR — last seed
        roc_results, _ = evaluate_all(
            last_models, snr_eval, s, Q_inv, Sigma, NU,
            calib_noise_last, last_cov_ests, theta_mode="fixed",
        )

        fig, ax = plt.subplots(figsize=(7, 6))
        for name, (fpr, tpr, auc_val, _) in roc_results.items():
            mean_auc = mean_results[name][0]
            label = f"{ALL_LABELS.get(name, name)}  (AUC={mean_auc:.3f})"
            ax.plot(fpr, tpr, color=COLORS.get(name, "purple"), label=label)
        ax.set_xlabel("False Alarm Rate (Pfa)")
        ax.set_ylabel("Detection Rate (Pd)")
        ax.set_title(f"ROC — t_nu={NU}, SNR={snr_eval} dB\n{mode_str}")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / f"roc_snr{snr_eval:+.0f}dB.png", dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 5))
        for name, pd_list in pd_curves.items():
            ax.plot(SNR_DB_LIST, pd_list, marker="o",
                    color=COLORS.get(name, "purple"),
                    label=ALL_LABELS.get(name, name))
        ax.set_xlabel("SNR (dB)")
        ax.set_ylabel(f"Pd @ Pfa={PFA}")
        ax.set_title(f"Detection vs SNR — t_nu={NU}, fixed theta\n"
                     f"(summary point: SNR={snr_eval} dB)")
        ax.axvline(snr_eval, color="gray", linestyle=":", alpha=0.6, label=f"Summary SNR={snr_eval} dB")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "pd_vs_snr.png", dpi=150)
        plt.close(fig)
        print(f"Figures saved to {out_dir}")

        # CSV — summarise mean±std
        run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{mode}"
        summary_path = RESULTS_DIR / f"student_t_summary_{run_id}.csv"
        with open(summary_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "run_id", "theta_mode", "nu", "n_dim", "cov_rho",
                "n_train", "n_test", "n_seeds", "pfa",
                "detector", "snr_eval_db",
                "auc_mean", "auc_std", "pd_mean", "pd_std",
            ])
            for name, (auc_m, auc_s, pd_m, pd_s) in mean_results.items():
                writer.writerow([
                    run_id, mode, NU, N_DIM, COV_KWARGS["rho"],
                    N_TRAIN, N_TEST, N_SEEDS, PFA,
                    name, snr_eval,
                    f"{auc_m:.6f}", f"{auc_s:.6f}",
                    f"{pd_m:.6f}",  f"{pd_s:.6f}",
                ])
        print(f"Saved summary -> {summary_path}")

    # weight function: learned u(d) vs theoretical t-weight (last seed)
    from models.score_models import TwoBranchScore
    tb_model = last_models.get("two_branch")
    if isinstance(tb_model, TwoBranchScore):
        d_vals = torch.linspace(0, 200, 500)
        u_theory = (NU + N_DIM) / (NU + d_vals.numpy())
        with torch.no_grad():
            u_learned = tb_model.get_weight_function(d_vals).numpy()
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(d_vals.numpy(), u_theory, "k--", label=f"Theoretical t-weight (nu={NU})")
        ax.plot(d_vals.numpy(), u_learned, color=COLORS["two_branch"],
                label=MODEL_LABELS["two_branch"])
        ax.set_xlabel("d = ||Ay||²  (Mahalanobis proxy)")
        ax.set_ylabel("u(d)  [scalar weight]")
        ax.set_title("Learned vs theoretical weight function")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(FIG_DIR / "weight_fn.png", dpi=150)
        plt.close(fig)
        print("Saved weight function plot")

    print("\nDone. Figures saved to", FIG_DIR)
