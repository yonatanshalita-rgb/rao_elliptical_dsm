"""
dsm_vs_amf_exp.py
-----------------
Pd vs SNR curve at Pfa=1%, comparing oracle baselines against DSM models.

Evaluation: for each SNR in EVAL_SNR_LIST, generate a test set with
FIXED theta = theta_max(SNR) and report Pd @ Pfa=1%.  No AUC averaging.

Settings: d=64, nu=3, AR(1) rho=0.9, 5 seeds.

Detectors
---------
oracle_amf      T = B/sqrt(C),  true Q
oracle_t_rao    T = w(A)*B/sqrt(I_0),  true Q + nu
oracle_t_glrt   T = (nu+d)/2 * log((nu+A)/(nu+A-B^2/C)),  true Q + nu
tyler_amf       T = B_hat/sqrt(C_hat),  Tyler scatter
linear_dsm      psi = -AA^T y,  sigma=0.5  (should converge to AMF)
fixed_weight    psi = w_t(d)*(-AA^T y),  fixed t-weight,  sigma=0.5
two_branch      psi = u(d)*(-AA^T y),  learned scalar,  sigma=0.5

Usage
-----
python dsm_vs_amf_exp.py
python dsm_vs_amf_exp.py --n_train 2000
"""

import argparse
import csv
import datetime
import numpy as np
import torch
import torch.utils.data
from pathlib import Path

from data.generate import ar1_covariance, sample_mu_t
from models.score_models import build_model
from models.train import train
from baselines.classical import tyler_estimator, amf_statistic
from eval.detector import NeuralDetector
from eval.metrics import compute_roc, pd_at_pfa

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

N_DIM         = 64
NU            = 3.0
COV_KWARGS    = {"rho": 0.9}

N_VAL         = 10_000
N_TEST        = 50_000   # per fixed-theta evaluation point
N_CALIB       = 5_000
BATCH_SIZE    = 512
N_EPOCHS      = 60
LR            = 1e-3
WARMUP_EPOCHS = 12

PFA           = 0.01
N_SEEDS       = 5

# SNR values at which Pd is evaluated (fixed theta per point)
EVAL_SNR_LIST = [1, 3, 5, 10, 15, 20]

DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_DIR   = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

DSM_MODELS = [
    ("linear_mse",   "DSM Linear",         {"sigma": 0.5, "warmup_epochs": 0}),
    ("fixed_weight", "DSM Fixed t-weight", {"sigma": 0.5, "warmup_epochs": 0}),
    ("two_branch",   "DSM Two-Branch",     {"sigma": 0.5, "warmup_epochs": WARMUP_EPOCHS}),
]

DETECTOR_NAMES = ["oracle_amf", "oracle_t_rao", "oracle_t_glrt", "tyler_amf"] \
                 + [m[0] for m in DSM_MODELS]

LABELS = {
    "oracle_amf":    "Oracle Gaussian AMF (true Q)",
    "oracle_t_rao":  "Oracle t-Rao       (true Q, nu)",
    "oracle_t_glrt": "Oracle t-GLRT      (true Q, nu)",
    "tyler_amf":     "Tyler AMF          (estimated Q)",
    "linear_mse":    "DSM Linear,        sigma=0.5",
    "fixed_weight":  "DSM Fixed t-weight,sigma=0.5",
    "two_branch":    "DSM Two-Branch,    sigma=0.5",
}

# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def _geometry():
    Sigma = ar1_covariance(N_DIM, **COV_KWARGS)
    Q     = np.linalg.inv(Sigma)
    s     = np.ones(N_DIM) / np.sqrt(N_DIM)
    C     = float(s @ Q @ s)
    return Sigma, Q, s, C


def make_noise_dataset(n_samples: int, seed: int):
    """Pure noise dataset for DSM training / covariance estimation."""
    rng   = np.random.default_rng(seed)
    Sigma, _, _, _ = _geometry()
    L     = np.linalg.cholesky(Sigma)
    mu    = sample_mu_t(n_samples, df=NU, rng=rng)
    w     = mu[:, None] * (rng.standard_normal((n_samples, N_DIM)) @ L.T)
    return w


def make_test_fixed_theta(snr_db: float, seed: int):
    """Test set: 50% H0, 50% H1 with theta = theta_max(snr_db) fixed."""
    rng   = np.random.default_rng(seed)
    Sigma, Q, s, C = _geometry()
    L     = np.linalg.cholesky(Sigma)

    theta_max = np.sqrt(10 ** (snr_db / 10.0) / C)

    mu    = sample_mu_t(N_TEST, df=NU, rng=rng)
    w     = mu[:, None] * (rng.standard_normal((N_TEST, N_DIM)) @ L.T)

    labels = rng.integers(0, 2, size=N_TEST)
    theta  = np.where(labels == 1, theta_max, 0.0)
    y      = w + theta[:, None] * s[None, :]
    return y, labels.astype(float)


class NoiseDataset(torch.utils.data.Dataset):
    def __init__(self, w: np.ndarray):
        self.w = torch.tensor(w, dtype=torch.float32)
    def __len__(self):          return len(self.w)
    def __getitem__(self, idx): return {"w": self.w[idx]}


# ---------------------------------------------------------------------------
# Oracle detectors
# ---------------------------------------------------------------------------

def _ABc(y, Q, s):
    d = y.shape[1]
    A = np.einsum("ni,ij,nj->n", y, Q, y)
    B = y @ (Q @ s)
    C = float(s @ Q @ s)
    return A, B, C, d


def oracle_amf_scores(y, Q, s):
    _, B, C, _ = _ABc(y, Q, s)
    return B / np.sqrt(C)


def oracle_t_rao_scores(y, Q, s, nu):
    A, B, C, d = _ABc(y, Q, s)
    return (nu + d) / (nu + A) * B / np.sqrt((nu + d) / (nu + d + 2) * C)


def oracle_t_glrt_scores(y, Q, s, nu):
    A, B, C, d = _ABc(y, Q, s)
    return np.where(B > 0, 0.5*(nu+d)*np.log((nu+A)/(nu+A-B**2/C)), 0.0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(n_train: int, eval_only: bool = False):
    _, Q, s, C = _geometry()

    print(f"\nDevice: {DEVICE}")
    print(f"d={N_DIM}  nu={NU}  rho={COV_KWARGS['rho']}  N_train={n_train}")
    print(f"seeds={N_SEEDS}  Pfa={PFA}")
    print(f"Eval SNRs (fixed theta): {EVAL_SNR_LIST} dB")
    theta_vals = [np.sqrt(10**(snr/10)/C) for snr in EVAL_SNR_LIST]
    print(f"  -> theta_max: {[f'{t:.2f}' for t in theta_vals]}")

    # seed_pds / seed_pfas[detector][seed] = list across EVAL_SNR_LIST
    seed_pds  = {name: [] for name in DETECTOR_NAMES}
    seed_pfas = {name: [] for name in DETECTOR_NAMES}

    ckpt_dir = Path("checkpoints_amf_exp")

    for seed in range(N_SEEDS):
        print(f"\n{'='*56}\nSeed {seed+1}/{N_SEEDS}\n{'='*56}")

        # --- Noise data for training and covariance estimation ---
        w_train = make_noise_dataset(n_train,  seed=seed*10+1)
        w_val   = make_noise_dataset(N_VAL,    seed=seed*10+2)
        w_calib = make_noise_dataset(N_CALIB,  seed=seed*10+4)

        train_dl = torch.utils.data.DataLoader(
            NoiseDataset(w_train), batch_size=BATCH_SIZE, shuffle=True)
        val_dl   = torch.utils.data.DataLoader(
            NoiseDataset(w_val),   batch_size=BATCH_SIZE)

        # Tyler covariance from calibration noise
        tyler_Q = np.linalg.inv(tyler_estimator(w_calib))

        # --- Train or load DSM models ---
        trained = {}
        for mtype, label, tkwargs in DSM_MODELS:
            extra = {"nu": NU} if mtype == "fixed_weight" else {}
            model = build_model(mtype, n=N_DIM, **extra)
            ckpt  = ckpt_dir / f"{mtype}_seed{seed}_best.pt"
            if eval_only and ckpt.exists():
                print(f"\n  Loading {label} from {ckpt.name} ...")
                model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
            else:
                torch.manual_seed(seed)
                np.random.seed(seed)
                print(f"\n  Training {label} ...")
                train(model, mtype, train_dl, val_dl,
                      n_epochs=N_EPOCHS, lr=LR, sigma=tkwargs["sigma"],
                      device=DEVICE, save_dir=ckpt_dir, verbose=True,
                      warmup_epochs=tkwargs["warmup_epochs"])
                generic = ckpt_dir / f"{mtype}_best.pt"
                if generic.exists():
                    generic.rename(ckpt)
                if ckpt.exists():
                    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
            detector = NeuralDetector(model, s, device=DEVICE)
            detector.calibrate(torch.tensor(w_calib, dtype=torch.float32))
            trained[mtype] = detector

        # --- Calibration-derived thresholds (set from w_calib, not test data) ---
        # For each detector: score all calibration noise samples, take (1-PFA) quantile.
        # Empirical Pfa on fresh test H0 samples then tests CFAR.
        def _threshold(calib_scores):
            return float(np.quantile(calib_scores, 1.0 - PFA))

        thresholds = {
            "oracle_amf":   _threshold(oracle_amf_scores(w_calib, Q, s)),
            "oracle_t_rao": _threshold(oracle_t_rao_scores(w_calib, Q, s, NU)),
            "oracle_t_glrt":_threshold(oracle_t_glrt_scores(w_calib, Q, s, NU)),
            "tyler_amf":    _threshold(amf_statistic(w_calib, s, tyler_Q)),
        }
        for mtype, _, _ in DSM_MODELS:
            thresholds[mtype] = _threshold(trained[mtype].score(w_calib))

        # --- Evaluate at each fixed-theta SNR point ---
        pds_this_seed  = {name: [] for name in DETECTOR_NAMES}
        pfas_this_seed = {name: [] for name in DETECTOR_NAMES}

        for snr_db in EVAL_SNR_LIST:
            y_np, labels_np = make_test_fixed_theta(snr_db, seed=seed*10+3)
            h0 = labels_np == 0
            h1 = labels_np == 1

            def _record(name, scores):
                thr = thresholds[name]
                pfas_this_seed[name].append(float((scores[h0] > thr).mean()))
                pds_this_seed[name].append( float((scores[h1] > thr).mean()))

            _record("oracle_amf",    oracle_amf_scores(y_np, Q, s))
            _record("oracle_t_rao",  oracle_t_rao_scores(y_np, Q, s, NU))
            _record("oracle_t_glrt", oracle_t_glrt_scores(y_np, Q, s, NU))
            _record("tyler_amf",     amf_statistic(y_np, s, tyler_Q))
            for mtype, _, _ in DSM_MODELS:
                _record(mtype, trained[mtype].score(y_np))

        for name in DETECTOR_NAMES:
            seed_pds[name].append(pds_this_seed[name])
            seed_pfas[name].append(pfas_this_seed[name])

    # ---------------------------------------------------------------------------
    # Report
    # ---------------------------------------------------------------------------
    run_id   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_rows = []

    print(f"\n{'='*90}")
    print(f"d={N_DIM}, nu={NU}, rho={COV_KWARGS['rho']}, N_train={n_train}, {N_SEEDS} seeds")
    print(f"Threshold set from calibration noise  |  Pfa target={PFA}")
    print(f"{'='*90}")

    for metric, store, label in [("Pd  ", seed_pds, "Pd  (mean±std)"),
                                  ("Pfa ", seed_pfas, "Pfa (mean±std)  [should be ~0.010]")]:
        snr_header = "  ".join(f"SNR={snr:>2}dB" for snr in EVAL_SNR_LIST)
        print(f"\n{label}")
        print(f"{'Detector':<38}  {snr_header}")
        print("-" * 88)
        for name in DETECTOR_NAMES:
            arr   = np.array(store[name])
            means = arr.mean(axis=0)
            stds  = arr.std(axis=0)
            vals  = "  ".join(f"{m:.3f}±{s:.3f}" for m, s in zip(means, stds))
            print(f"{LABELS.get(name, name):<38}  {vals}")
            if metric == "Pd  ":
                csv_rows.append([name] + [f"{m:.6f}" for m in means]
                                       + [f"{s:.6f}" for s in stds])

    csv_path = RESULTS_DIR / f"dsm_vs_amf_pdcurve_{run_id}_d{N_DIM}_ntrain{n_train}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        mean_hdrs = [f"pd_mean_snr{s}" for s in EVAL_SNR_LIST]
        std_hdrs  = [f"pd_std_snr{s}"  for s in EVAL_SNR_LIST]
        writer.writerow(["detector", "n_dim", "nu", "cov_rho", "n_train",
                         "n_seeds", "pfa"] + mean_hdrs + std_hdrs)
        for row in csv_rows:
            writer.writerow([row[0], N_DIM, NU, COV_KWARGS["rho"],
                             n_train, N_SEEDS, PFA] + row[1:])
    print(f"\nSaved -> {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_train",   type=int,  default=50_000)
    parser.add_argument("--eval_only", action="store_true",
                        help="Skip training, load saved per-seed checkpoints")
    args = parser.parse_args()
    run(args.n_train, eval_only=args.eval_only)
