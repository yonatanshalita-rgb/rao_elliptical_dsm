"""
sigma_sweep_exp.py
------------------
Sweep DSM noise level sigma across [0.05, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0]
for both the Two-Branch and Unconstrained MLP models.

Theory predicts:
  sigma -> 0   : both converge to t-Rao  (score of t_nu)
  sigma = 0.5  : two-branch ~ t-GLRT, MLP ~ oracle q_sigma-Rao (suppressed)
  sigma -> inf : both converge to Gaussian AMF  (score of N(0, sigma^2 I))

Oracle baselines (AMF, t-Rao, t-GLRT) are included as fixed reference lines.

Settings: d=64, nu=3, AR(1) rho=0.9, N_train=50K, 3 seeds.
Evaluation: Pd @ Pfa=1%, fixed theta at SNR in [1, 3, 5, 10, 15, 20] dB.
"""

import csv
import datetime
import numpy as np
import torch
import torch.utils.data
from pathlib import Path

from data.generate import ar1_covariance, sample_mu_t
from models.score_models import build_model
from models.train import train
from eval.detector import NeuralDetector
from eval.metrics import pd_at_pfa, compute_roc

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

N_DIM         = 64
NU            = 3.0
COV_KWARGS    = {"rho": 0.9}

N_TRAIN       = 50_000
N_VAL         = 10_000
N_TEST        = 50_000
N_CALIB       = 5_000
BATCH_SIZE    = 512
N_EPOCHS      = 60
LR            = 1e-3
WARMUP_EPOCHS = 12

PFA           = 0.01
N_SEEDS       = 3

EVAL_SNR_LIST = [1, 3, 5, 10, 15, 20]

SIGMA_LIST    = [0.05, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0]

DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_DIR   = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Geometry helpers (identical to dsm_vs_amf_exp.py)
# ---------------------------------------------------------------------------

def _geometry():
    Sigma = ar1_covariance(N_DIM, **COV_KWARGS)
    Q     = np.linalg.inv(Sigma)
    s     = np.ones(N_DIM) / np.sqrt(N_DIM)
    C     = float(s @ Q @ s)
    return Sigma, Q, s, C

def make_noise_dataset(n_samples, seed):
    rng   = np.random.default_rng(seed)
    Sigma, _, _, _ = _geometry()
    L     = np.linalg.cholesky(Sigma)
    mu    = sample_mu_t(n_samples, df=NU, rng=rng)
    return mu[:, None] * (rng.standard_normal((n_samples, N_DIM)) @ L.T)

def make_test_fixed_theta(snr_db, seed):
    rng   = np.random.default_rng(seed)
    Sigma, Q, s, C = _geometry()
    L     = np.linalg.cholesky(Sigma)
    theta_max = np.sqrt(10 ** (snr_db / 10.0) / C)
    mu    = sample_mu_t(N_TEST, df=NU, rng=rng)
    w     = mu[:, None] * (rng.standard_normal((N_TEST, N_DIM)) @ L.T)
    labels = rng.integers(0, 2, size=N_TEST)
    theta  = np.where(labels == 1, theta_max, 0.0)
    return w + theta[:, None] * s[None, :], labels.astype(float)

class NoiseDataset(torch.utils.data.Dataset):
    def __init__(self, w):
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

def oracle_amf(y, Q, s):
    _, B, C, _ = _ABc(y, Q, s)
    return B / np.sqrt(C)

def oracle_t_rao(y, Q, s, nu):
    A, B, C, d = _ABc(y, Q, s)
    return (nu+d)/(nu+A) * B / np.sqrt((nu+d)/(nu+d+2)*C)

def oracle_t_glrt(y, Q, s, nu):
    A, B, C, d = _ABc(y, Q, s)
    return np.where(B > 0, 0.5*(nu+d)*np.log((nu+A)/(nu+A-B**2/C)), 0.0)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _, Q, s, C = _geometry()

    print(f"Device: {DEVICE}")
    print(f"d={N_DIM}  nu={NU}  rho={COV_KWARGS['rho']}  N_train={N_TRAIN}")
    print(f"seeds={N_SEEDS}  Pfa={PFA}  sigma sweep: {SIGMA_LIST}")

    ckpt_dir = Path("checkpoints_sigma_sweep")
    ckpt_dir.mkdir(exist_ok=True)

    # accumulators: (model_key) -> seed -> [pd at each snr]
    # model_key = "oracle_amf" | "oracle_t_rao" | "oracle_t_glrt"
    #           | "two_branch_s{sigma}" | "mlp_score_s{sigma}"
    oracle_keys = ["oracle_amf", "oracle_t_rao", "oracle_t_glrt"]
    dsm_keys    = [f"{mtype}_s{sigma}"
                   for sigma in SIGMA_LIST
                   for mtype in ["two_branch", "mlp_score"]]
    all_keys    = oracle_keys + dsm_keys

    seed_pds = {k: [] for k in all_keys}

    for seed in range(N_SEEDS):
        print(f"\n{'='*60}\nSeed {seed+1}/{N_SEEDS}\n{'='*60}")

        w_train = make_noise_dataset(N_TRAIN, seed=seed*10+1)
        w_val   = make_noise_dataset(N_VAL,   seed=seed*10+2)
        w_calib = make_noise_dataset(N_CALIB, seed=seed*10+4)

        train_dl = torch.utils.data.DataLoader(
            NoiseDataset(w_train), batch_size=BATCH_SIZE, shuffle=True)
        val_dl   = torch.utils.data.DataLoader(
            NoiseDataset(w_val), batch_size=BATCH_SIZE)

        # --- Oracle Pd values (same for every seed since Q is fixed) ---
        # Recompute per seed to use same test data seed
        oracle_pds = {k: [] for k in oracle_keys}
        for snr_db in EVAL_SNR_LIST:
            y_np, lbl = make_test_fixed_theta(snr_db, seed=seed*10+3)
            oracle_pds["oracle_amf"].append(
                pd_at_pfa(oracle_amf(y_np, Q, s),        lbl, PFA))
            oracle_pds["oracle_t_rao"].append(
                pd_at_pfa(oracle_t_rao(y_np, Q, s, NU),  lbl, PFA))
            oracle_pds["oracle_t_glrt"].append(
                pd_at_pfa(oracle_t_glrt(y_np, Q, s, NU), lbl, PFA))
        for k in oracle_keys:
            seed_pds[k].append(oracle_pds[k])

        # --- DSM models: train one (two_branch, mlp_score) per sigma ---
        for sigma in SIGMA_LIST:
            for mtype in ["two_branch", "mlp_score"]:
                key = f"{mtype}_s{sigma}"
                warmup = WARMUP_EPOCHS if mtype == "two_branch" else 0
                print(f"\n  sigma={sigma}  model={mtype}")

                torch.manual_seed(seed + int(sigma * 100))
                np.random.seed(seed + int(sigma * 100))

                model = build_model(mtype, n=N_DIM)
                train(model, mtype, train_dl, val_dl,
                      n_epochs=N_EPOCHS, lr=LR, sigma=sigma,
                      device=DEVICE,
                      save_dir=ckpt_dir,
                      verbose=False,
                      warmup_epochs=warmup)

                # load best checkpoint
                ckpt = ckpt_dir / f"{mtype}_best.pt"
                saved = ckpt_dir / f"{mtype}_s{sigma}_seed{seed}_best.pt"
                if ckpt.exists():
                    ckpt.rename(saved)
                if saved.exists():
                    model.load_state_dict(torch.load(saved, map_location=DEVICE))

                detector = NeuralDetector(model, s, device=DEVICE)
                detector.calibrate(torch.tensor(w_calib, dtype=torch.float32))

                pds_this = []
                for snr_db in EVAL_SNR_LIST:
                    y_np, lbl = make_test_fixed_theta(snr_db, seed=seed*10+3)
                    pds_this.append(pd_at_pfa(detector.score(y_np), lbl, PFA))
                seed_pds[key].append(pds_this)

    # ---------------------------------------------------------------------------
    # Report
    # ---------------------------------------------------------------------------
    print(f"\n{'='*90}")
    print(f"Pd @ Pfa={PFA}  |  d={N_DIM}, nu={NU}, N_train={N_TRAIN}, {N_SEEDS} seeds")
    print(f"{'='*90}")
    snr_hdr = "  ".join(f"SNR={s:>2}dB" for s in EVAL_SNR_LIST)
    print(f"\n{'Model':<36}  {snr_hdr}")
    print("-" * 88)

    def _row(key, label):
        arr   = np.array(seed_pds[key])
        means = arr.mean(axis=0)
        stds  = arr.std(axis=0)
        vals  = "  ".join(f"{m:.3f}±{s:.3f}" for m, s in zip(means, stds))
        print(f"{label:<36}  {vals}")
        return means, stds

    print("\n--- Oracles ---")
    _row("oracle_amf",    "Oracle AMF   (true Q)")
    _row("oracle_t_rao",  "Oracle t-Rao (true Q, nu)")
    _row("oracle_t_glrt", "Oracle t-GLRT(true Q, nu)")

    for mtype, label in [("two_branch", "Two-Branch"), ("mlp_score", "Unconstrained MLP")]:
        print(f"\n--- {label} ---")
        for sigma in SIGMA_LIST:
            key = f"{mtype}_s{sigma}"
            _row(key, f"  sigma={sigma:<5}")

    # ---------------------------------------------------------------------------
    # Save CSV
    # ---------------------------------------------------------------------------
    run_id   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = RESULTS_DIR / f"sigma_sweep_{run_id}_d{N_DIM}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        mean_hdrs = [f"pd_mean_snr{s}" for s in EVAL_SNR_LIST]
        std_hdrs  = [f"pd_std_snr{s}"  for s in EVAL_SNR_LIST]
        writer.writerow(["model", "sigma", "n_dim", "nu", "n_train",
                         "n_seeds", "pfa"] + mean_hdrs + std_hdrs)
        for key in all_keys:
            arr   = np.array(seed_pds[key])
            means = arr.mean(axis=0)
            stds  = arr.std(axis=0)
            sigma = key.split("_s")[-1] if "_s" in key else "oracle"
            writer.writerow([key, sigma, N_DIM, NU, N_TRAIN, N_SEEDS, PFA]
                            + [f"{m:.6f}" for m in means]
                            + [f"{s:.6f}" for s in stds])
    print(f"\nSaved -> {csv_path}")
