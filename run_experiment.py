/"""
Main experiment runner.

Trains all three DSM models, runs all classical baselines, and produces
ROC curves and Pd-vs-SNR plots.

Usage:
    python run_experiment.py
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

from data.generate import EllipticalDataset, make_dataloaders
from models.score_models import build_model
from models.train import train
from baselines.classical import CLASSICAL_DETECTORS
from eval.detector import NeuralDetector
from eval.metrics import compute_roc, pd_at_pfa

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

N_DIM        = 16         # observation dimension
N_TRAIN      = 50_000
N_VAL        = 10_000
N_TEST       = 50_000
N_CALIB      = 5_000      # noise samples for neural detector calibration
BATCH_SIZE   = 512
N_EPOCHS     = 60
LR           = 1e-3
SIGMA_DSM    = 0.1        # DSM perturbation noise level (default; overridden by sweep)
HUBER_DELTA  = 1.0        # Huber loss delta (default; overridden by sweep)

SIGMA_GRID   = [0.05, 0.1, 0.2, 0.5]   # sweep candidates (used only if RUN_SWEEP=True)
DELTA_GRID   = [0.5, 1.0, 2.0, 5.0]    # sweep candidates (huber only)
SWEEP_EPOCHS = 20                        # reduced epochs for sweep

N_SEEDS      = 5                         # independent training runs for variance estimate
SWEEP_SEED   = 0                         # seed used for hyperparameter sweep

RUN_SWEEP    = False                     # set True to re-run hyperparameter sweep
WARMUP_EPOCHS = 12                       # epochs to freeze u-branch in Two-Branch

BEST_PARAMS  = {
    # No global grad_clip for any model — each model's robustness comes from its
    # own mechanism: per_sample_clip (linear), Huber loss (linear_huber), warmup (two_branch).
    # For n=16, inlier MSE grad_norm ≈ 20, outlier ≈ 20 000 → clip at 100 (5x inlier).
    # For Huber, inlier grad_norm ≈ 1 (Huber clips residuals to delta) → clip at 10 (10x inlier).
    "linear_mse":   {"sigma": 0.1, "delta": None, "warmup_epochs": 0,             "grad_clip": None, "per_sample_clip": 100.0},
    "linear_huber": {"sigma": 0.1, "delta": 1.0,  "warmup_epochs": 0,             "grad_clip": None, "per_sample_clip": 10.0},
    "two_branch":   {"sigma": 0.5, "delta": None, "warmup_epochs": WARMUP_EPOCHS, "grad_clip": 1.0,  "per_sample_clip": None},
}
SNR_DB_LIST  = [-10, -5, 0, 5, 10, 15]
SNR_TRAIN_DB = 10         # SNR used during training
PFA          = 1e-2       # fixed false-alarm rate for Pd curves

COV_TYPE     = "ar1"
COV_KWARGS   = {"rho": 0.9}
MU_DIST      = "spike"
MU_KWARGS    = {"p_outlier": 0.01, "outlier_scale": 1000.0}

DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_DIR     = Path("checkpoints")
FIG_DIR      = Path("figures")
FIG_DIR.mkdir(exist_ok=True)

DATASET_KWARGS = dict(
    n=N_DIM,
    snr_db=SNR_TRAIN_DB,
    cov_type=COV_TYPE,
    cov_kwargs=COV_KWARGS,
    mu_dist=MU_DIST,
    mu_kwargs=MU_KWARGS,
)

MODEL_TYPES = ["linear_mse", "linear_huber", "two_branch"]
MODEL_LABELS = {
    "linear_mse":   "DSM Linear (MSE)",
    "linear_huber": "DSM Linear (Huber)",
    "two_branch":   "DSM Two-Branch",
}


# ---------------------------------------------------------------------------
# Hyperparameter sweep
# ---------------------------------------------------------------------------

def sweep_hyperparams(train_dl, val_dl) -> dict:
    """
    Grid search over sigma (all models) and delta (linear_huber only).
    Trains each combo for SWEEP_EPOCHS, picks the combo with lowest val DSM loss.

    Returns dict: mtype -> {"sigma": float, "delta": float}
    """
    from models.train import LOSS_FNS

    best_params = {}

    for mtype in MODEL_TYPES:
        print(f"\n  Sweeping {MODEL_LABELS[mtype]}...")
        combos = (
            [(s, d) for s in SIGMA_GRID for d in DELTA_GRID]
            if mtype == "linear_huber"
            else [(s, None) for s in SIGMA_GRID]
        )

        best_val = float("inf")
        best_sigma, best_delta = SIGMA_DSM, HUBER_DELTA

        for sigma, delta in combos:
            model = build_model(mtype, n=N_DIM).to(DEVICE)
            optimizer = torch.optim.Adam(model.parameters(), lr=LR)
            loss_fn = LOSS_FNS[mtype]
            loss_kwargs = {"sigma": sigma}
            if delta is not None:
                loss_kwargs["delta"] = delta

            model.train()
            for _ in range(SWEEP_EPOCHS):
                for batch in train_dl:
                    w = batch["w"].to(DEVICE)
                    optimizer.zero_grad()
                    loss_fn(model, w, **loss_kwargs).backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    optimizer.step()

            model.eval()
            val_losses = []
            with torch.no_grad():
                for batch in val_dl:
                    w = batch["w"].to(DEVICE)
                    val_losses.append(loss_fn(model, w, **loss_kwargs).item())
            val_loss = np.mean(val_losses)

            tag = f"sigma={sigma}" + (f", delta={delta}" if delta is not None else "")
            print(f"    {tag:30s}  val={val_loss:.4f}")

            if val_loss < best_val:
                best_val = val_loss
                best_sigma = sigma
                best_delta = delta

        best_params[mtype] = {"sigma": best_sigma, "delta": best_delta}
        tag = f"sigma={best_sigma}" + (f", delta={best_delta}" if best_delta is not None else "")
        print(f"  -> Best: {tag}  val={best_val:.4f}")

    return best_params


# ---------------------------------------------------------------------------
# Train neural models
# ---------------------------------------------------------------------------

def train_all_models(train_dl, val_dl, best_params: dict | None = None, seed: int = 0):
    torch.manual_seed(seed)
    np.random.seed(seed)

    models = {}
    for mtype in MODEL_TYPES:
        params = (best_params or {}).get(mtype, {})
        sigma            = params.get("sigma", SIGMA_DSM)
        delta            = params.get("delta") or HUBER_DELTA
        warmup           = params.get("warmup_epochs", 0)
        grad_clip        = params.get("grad_clip", None)
        per_sample_clip  = params.get("per_sample_clip", None)

        print(f"\n{'='*50}")
        label = (f"Training: {MODEL_LABELS[mtype]}  (sigma={sigma}"
                 + (f", delta={delta}" if mtype == "linear_huber" else "")
                 + (f", warmup={warmup}" if warmup else "")
                 + (f", psc={per_sample_clip}" if per_sample_clip is not None else "")
                 + ")")
        print(label)
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
# Evaluate at a single SNR
# ---------------------------------------------------------------------------

def evaluate_all(models, snr_db: float, s: np.ndarray, Q: np.ndarray,
                 calib_noise: np.ndarray):
    """
    Returns dict: method_name -> (fpr, tpr, auc, pd_at_pfa)
    """
    # Generate test data at this SNR
    test_ds = EllipticalDataset(
        N_TEST, seed=99, snr_db=snr_db, **{k: v for k, v in DATASET_KWARGS.items()
                                            if k not in ("snr_db",)},
    )
    y_np      = test_ds.y.numpy()
    labels_np = test_ds.labels.numpy()
    W_np      = calib_noise   # training noise samples for classical methods

    # Signal amplitude at this SNR — used by Oracle LRT
    snr_linear = 10 ** (snr_db / 10.0)
    Q_inv_true = np.linalg.inv(Q)
    a_snr      = float(np.sqrt(snr_linear / (s @ Q_inv_true @ s)))

    results = {}

    # --- classical detectors ---
    for name, det_fn in CLASSICAL_DETECTORS.items():
        if name == "oracle":
            kwargs = {"Q_true": Q}
        elif name == "oracle_lrt":
            kwargs = {
                "Q_true":       Q,
                "a":            a_snr,
                "p_outlier":    MU_KWARGS["p_outlier"],
                "outlier_scale": MU_KWARGS["outlier_scale"],
            }
        else:
            kwargs = {}
        scores = det_fn(y_np, s, W_np, **kwargs)
        fpr, tpr, _, auc_val = compute_roc(scores, labels_np)
        pd = pd_at_pfa(scores, labels_np, pfa=PFA)
        results[name] = (fpr, tpr, auc_val, pd)

    # --- neural detectors ---
    for mtype, model in models.items():
        detector = NeuralDetector(model, s, device=DEVICE)
        calib_t  = torch.tensor(calib_noise, dtype=torch.float32)
        detector.calibrate(calib_t)
        scores = detector.score(y_np)
        fpr, tpr, _, auc_val = compute_roc(scores, labels_np)
        pd = pd_at_pfa(scores, labels_np, pfa=PFA)
        results[mtype] = (fpr, tpr, auc_val, pd)

    return results, labels_np


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

COLORS = {
    "oracle":       "black",
    "oracle_lrt":   "purple",
    "scm":          "gray",
    "tyler":        "blue",
    "huber":        "cyan",
    "linear_mse":   "red",
    "linear_huber": "orange",
    "two_branch":   "green",
}

ALL_LABELS = {
    "oracle":       "Oracle AMF",
    "oracle_lrt":   "Oracle LRT (true upper bound)",
    "scm":          "SCM AMF",
    "tyler":        "Tyler AMF",
    "huber":        "Huber AMF",
    **MODEL_LABELS,
}


def plot_roc(results: dict, snr_db: float):
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, (fpr, tpr, auc_val, _) in results.items():
        label = f"{ALL_LABELS.get(name, name)}  (AUC={auc_val:.3f})"
        ax.plot(fpr, tpr, color=COLORS.get(name, "purple"), label=label)
    ax.set_xlabel("False Alarm Rate (Pfa)")
    ax.set_ylabel("Detection Rate (Pd)")
    ax.set_title(f"ROC Curves — SNR = {snr_db} dB")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"roc_snr{snr_db:+.0f}dB.png", dpi=150)
    plt.close(fig)
    print(f"Saved ROC plot for SNR={snr_db} dB")


def plot_pd_vs_snr(pd_curves: dict):
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, pd_list in pd_curves.items():
        ax.plot(SNR_DB_LIST, pd_list, marker="o",
                color=COLORS.get(name, "purple"),
                label=ALL_LABELS.get(name, name))
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel(f"Pd @ Pfa={PFA}")
    ax.set_title("Detection Probability vs SNR")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "pd_vs_snr.png", dpi=150)
    plt.close(fig)
    print("Saved Pd-vs-SNR plot")


def plot_weight_function(model, label: str):
    """Plot the learned u(d) function for TwoBranchScore."""
    from models.score_models import TwoBranchScore
    if not isinstance(model, TwoBranchScore):
        return
    d_vals = torch.linspace(0, 200, 500)
    with torch.no_grad():
        u_vals = model.get_weight_function(d_vals).numpy()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(d_vals.numpy(), u_vals)
    ax.set_xlabel("Mahalanobis proxy  d = -y'Wy  ≈  y'Q⁻¹y")
    ax.set_ylabel("u(d)  [scalar weight]")
    ax.set_title(f"Learned weight function — {label}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "weight_function_two_branch.png", dpi=150)
    plt.close(fig)
    print("Saved weight function plot")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Device: {DEVICE}")

    # --- data ---
    train_dl, val_dl, test_dl, train_ds = make_dataloaders(
        n_train=N_TRAIN, n_val=N_VAL, n_test=N_TEST,
        batch_size=BATCH_SIZE, **DATASET_KWARGS,
    )
    s  = train_ds.s
    Q  = train_ds.Q

    # calibration noise: pure noise samples (label=0)
    calib_ds = EllipticalDataset(
        N_CALIB, seed=10, snr_db=0,
        n=N_DIM, cov_type=COV_TYPE, cov_kwargs=COV_KWARGS,
        mu_dist=MU_DIST, mu_kwargs=MU_KWARGS,
    )
    calib_noise = calib_ds.w.numpy()

    # --- hyperparameter selection ---
    if RUN_SWEEP:
        print("\n--- Hyperparameter sweep ---")
        torch.manual_seed(SWEEP_SEED)
        np.random.seed(SWEEP_SEED)
        best_params = sweep_hyperparams(train_dl, val_dl)
    else:
        best_params = BEST_PARAMS
        print("\n--- Using fixed hyperparameters ---")
        for mtype, p in best_params.items():
            print(f"  {MODEL_LABELS[mtype]:30s}  sigma={p['sigma']}"
                  + (f"  delta={p['delta']}" if p["delta"] else ""))

    # --- multi-seed training + evaluation ---
    # classical detectors are deterministic; run once and reuse
    classical_results, _ = evaluate_all({}, SNR_TRAIN_DB, s, Q, calib_noise)

    # seed_results[name] -> list of (auc, pd) across seeds
    seed_results = {name: [] for name in list(classical_results) + MODEL_TYPES}
    for name, (_, _, auc_val, pd) in classical_results.items():
        seed_results[name] = [(auc_val, pd)] * N_SEEDS  # same value every seed

    pd_curves_seeds = {name: [] for name in seed_results}  # list-of-lists over seeds
    last_models = None

    for seed in range(N_SEEDS):
        print(f"\n{'#'*60}")
        print(f"# Seed {seed + 1}/{N_SEEDS}")
        print(f"{'#'*60}")
        models = train_all_models(train_dl, val_dl, best_params=best_params, seed=seed)
        last_models = models

        neural_results, _ = evaluate_all(models, SNR_TRAIN_DB, s, Q, calib_noise)
        for mtype in MODEL_TYPES:
            _, _, auc_val, pd = neural_results[mtype]
            seed_results[mtype].append((auc_val, pd))

        # Pd vs SNR for this seed
        snr_pd = {mtype: [] for mtype in MODEL_TYPES}
        for snr in SNR_DB_LIST:
            res, _ = evaluate_all(models, snr, s, Q, calib_noise)
            for mtype in MODEL_TYPES:
                snr_pd[mtype].append(res[mtype][3])
        for mtype in MODEL_TYPES:
            pd_curves_seeds[mtype].append(snr_pd[mtype])

    # classical Pd vs SNR (run once)
    for name in classical_results:
        snr_pds = []
        for snr in SNR_DB_LIST:
            res, _ = evaluate_all({}, snr, s, Q, calib_noise)
            snr_pds.append(res[name][3])
        pd_curves_seeds[name] = [snr_pds] * N_SEEDS

    # --- summarise ---
    print(f"\n--- Results at SNR={SNR_TRAIN_DB} dB  (mean ± std over {N_SEEDS} seeds) ---")
    for name, vals in seed_results.items():
        aucs = [v[0] for v in vals]
        pds  = [v[1] for v in vals]
        print(f"  {ALL_LABELS.get(name, name):30s}  "
              f"AUC={np.mean(aucs):.4f}±{np.std(aucs):.4f}  "
              f"Pd={np.mean(pds):.4f}±{np.std(pds):.4f}")

    # --- ROC using last-seed models ---
    results_last, _ = evaluate_all(last_models, SNR_TRAIN_DB, s, Q, calib_noise)
    plot_roc(results_last, SNR_TRAIN_DB)

    # --- Pd vs SNR (mean across seeds) ---
    pd_curves_mean = {
        name: list(np.mean(pd_curves_seeds[name], axis=0))
        for name in seed_results
    }
    plot_pd_vs_snr(pd_curves_mean)

    # --- weight function (last seed) ---
    if last_models and "two_branch" in last_models:
        plot_weight_function(last_models["two_branch"], MODEL_LABELS["two_branch"])

    print("\nDone. Figures saved to", FIG_DIR)
