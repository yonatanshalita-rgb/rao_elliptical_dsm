"""
MGGD Detection Experiment
=========================
Rao-test detection on MGGD(p=64, beta=0.2) data.
Compares 7 detectors across N_train and SNR sweeps.

Rao statistic:  T(x) = s^T score(x)
Tyler/Gaussian AMF:  T(x) = (s^T C^{-1} x) / sqrt(s^T C^{-1} s)
H1 model:  x = theta * s + noise,  theta = sqrt(10^(SNR_dB/10) / C)

Methods
-------
  Oracle Rao            -- true MGGD score
  MLE Rao               -- Pascal MLE parameters
  TwoBranch Rao         -- trained two_branch model
  Linear Rao            -- trained linear_mse model
  MGGD Constrained Rao  -- trained mggd_constrained model
  Tyler AMF             -- Tyler scatter, AMF statistic
  Oracle Gaussian AMF   -- true M, Gaussian AMF

Usage
-----
  python mggd_detection_exp.py
  python mggd_detection_exp.py --quick
  python mggd_detection_exp.py --eval_only
  python mggd_detection_exp.py --snr_list 3 5 10
"""

import argparse
import pickle
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, roc_curve

from baselines.classical import (
    fit_mggd_mle, score_mggd_mle,
    fit_tyler_safe, score_tyler_linear,
)
from data.generate import ar1_covariance, sample_mggd, mggd_true_score, compute_m_norm
from models.score_models import build_model
from models.train import train

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

P          = 64
BETA       = 0.5
RHO        = 0.8
M_PARAM    = compute_m_norm(BETA, P)   # E[tau^2] = p; keeps score magnitude O(1)
SIGMA_DSM  = 0.3
N_TEST     = 50_000
PFA        = 0.01
BATCH_SIZE = 512
LR         = 1e-3
WARMUP_TB  = 20
GRAD_CLIP  = 1.0

N_TRAIN_FULL  = [20, 50, 100, 200, 500, 1000, 2000, 5000, 20_000]
N_TRAIN_QUICK = [50, 200, 1000, 5000]
N_MC_FULL     = 5
N_MC_QUICK    = 2

# With m_norm, Var[x] = M (same as Gaussian), so the matched-filter SNR
# maps to detection probability on the same dB scale as the Gaussian AMF.
# Gaussian AMF Pd@Pfa=0.01: SNR=7dB->0.46, SNR=11dB->0.89, SNR=13dB->0.98.
EVAL_SNR_LIST   = [5, 7, 9, 11, 13, 15]
FIXED_SNR       = 11         # Oracle Pd ~0.85-0.90 here
FIXED_N_VALS    = [200, 1000, 5000, 20_000]
ROC_N           = 200
ROC_SNR         = 11

METHOD_NAMES = [
    "Oracle Rao",
    "MLE Rao",
    "TwoBranch Rao",
    "Linear Rao",
    "MGGD Constrained Rao",
    "Tyler AMF",
    "Oracle Gaussian AMF",
]

COLORS = {
    "Oracle Rao":            "black",
    "MLE Rao":               "#1f77b4",
    "TwoBranch Rao":         "#d62728",
    "Linear Rao":            "#2ca02c",
    "MGGD Constrained Rao":  "#ff7f0e",
    "Tyler AMF":             "#9467bd",
    "Oracle Gaussian AMF":   "#7f7f7f",
}

_LS = {   # dashed/dotted for oracle lines
    "Oracle Rao":           "--",
    "Oracle Gaussian AMF":  ":",
}

_DEVICE     = "cpu"
CKPT_DIR    = Path("checkpoints") / "det"
RESULTS_PKL = Path("results") / "mggd_det_results.pkl"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _n_epochs(n_train: int) -> int:
    return max(100, 6_000 // max(1, n_train // BATCH_SIZE))


class _DictDS(torch.utils.data.Dataset):
    def __init__(self, arr: np.ndarray):
        self.t = torch.tensor(arr, dtype=torch.float32)
    def __len__(self):  return len(self.t)
    def __getitem__(self, i):  return {"w": self.t[i]}


def _loader(X: np.ndarray) -> DataLoader:
    return DataLoader(_DictDS(X), batch_size=min(len(X), BATCH_SIZE), shuffle=True)


def _train_dsm(mtype: str, X_train: np.ndarray) -> torch.nn.Module:
    """Train DSM model; no val set — pass train_dl as val (no early stopping)."""
    ne = _n_epochs(len(X_train))
    model = build_model(mtype, P)
    dl = _loader(X_train)
    warmup = min(WARMUP_TB, ne // 5) if mtype == "two_branch" else 0
    train(
        model, mtype, dl, dl,
        n_epochs=ne, lr=LR, sigma=SIGMA_DSM,
        warmup_epochs=warmup,
        grad_clip=GRAD_CLIP if mtype == "mggd_constrained" else None,
        device=_DEVICE,
        verbose=False,
    )
    return model


def _infer(model: torch.nn.Module, X: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(X, dtype=torch.float32).to(_DEVICE)).cpu().numpy()


def _rao(score: np.ndarray, s: np.ndarray) -> np.ndarray:
    """T(x) = s^T score(x), shape (N,). s is unit-norm."""
    return score @ s


def _amf(X: np.ndarray, C_inv: np.ndarray, s: np.ndarray) -> np.ndarray:
    """AMF statistic: (s^T C^{-1} x) / sqrt(s^T C^{-1} s)."""
    Cinv_s = C_inv @ s
    return (X @ Cinv_s) / float(np.sqrt(s @ Cinv_s))


def _roc_metrics(T_h0: np.ndarray, T_h1: np.ndarray):
    """Return (auc, pd_at_pfa). Flips sign if AUC < 0.5."""
    y_true  = np.concatenate([np.zeros(len(T_h0)), np.ones(len(T_h1))])
    y_score = np.concatenate([T_h0, T_h1])
    auc = float(roc_auc_score(y_true, y_score))
    if auc < 0.5:
        auc = 1.0 - auc
        y_score = -y_score
    fpr, tpr, _ = roc_curve(y_true, y_score, drop_intermediate=False)
    pd = float(np.interp(PFA, fpr, tpr))
    return auc, pd, fpr, tpr


# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------

def run_detection(N_TRAIN_LIST: list, N_MC: int,
                  snr_list: list | None = None,
                  eval_only: bool = False,
                  resume: bool = False) -> dict:
    snr_list = snr_list or EVAL_SNR_LIST

    M     = ar1_covariance(P, RHO)
    M_inv = np.linalg.inv(M)
    s     = np.ones(P) / np.sqrt(float(P))
    C     = float(s @ M_inv @ s)

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_PKL.parent.mkdir(parents=True, exist_ok=True)

    # Load existing results (for eval_only or resume)
    results    = {}   # (method, n_train, snr_db) -> [(auc, pd), ...]
    roc_curves = {}   # (method, mc)               -> (fpr, tpr) at ROC_N, ROC_SNR
    if (eval_only or resume) and RESULTS_PKL.exists():
        with open(RESULTS_PKL, "rb") as f:
            saved = pickle.load(f)
        results    = saved.get("results",    {})
        roc_curves = saved.get("roc_curves", {})
        print(f"Loaded {len(results)} result entries from {RESULTS_PKL}")
        if eval_only:
            return {"results": results, "roc_curves": roc_curves}

    for n_train in N_TRAIN_LIST:
        ne = _n_epochs(n_train)
        print(f"\n  N_train={n_train} (epochs={ne}):", flush=True)

        for mc in range(N_MC):
            # Skip seeds already fully evaluated (crash-resume support)
            if resume:
                oracle_done = results.get(("Oracle Rao", n_train, snr_list[0]), [])
                if len(oracle_done) > mc:
                    print(f"    mc {mc+1}/{N_MC} already done, skipping", flush=True)
                    continue
            X_train = sample_mggd(n_train, P, M, M_PARAM, BETA, seed=mc * 10)

            # ----- classical baselines -----
            mle_result = None
            if n_train >= P + 2:
                mle_result = fit_mggd_mle(X_train)
            tyler_C = fit_tyler_safe(X_train)
            tyler_Cinv = np.linalg.inv(tyler_C) if tyler_C is not None else None

            # ----- DSM models -----
            dsm = {}   # label -> model
            for mtype, label in [("linear_mse",      "Linear Rao"),
                                  ("two_branch",       "TwoBranch Rao"),
                                  ("mggd_constrained", "MGGD Constrained Rao")]:
                ckpt = CKPT_DIR / f"{mtype}_N{n_train}_mc{mc}.pt"
                if eval_only and ckpt.exists():
                    m = build_model(mtype, P).to(_DEVICE)
                    m.load_state_dict(torch.load(ckpt, map_location=_DEVICE))
                    dsm[label] = m
                else:
                    m = _train_dsm(mtype, X_train)
                    torch.save(m.state_dict(), ckpt)
                    dsm[label] = m

            # ----- evaluate across SNR list -----
            for snr_db in snr_list:
                theta = float(np.sqrt(10 ** (snr_db / 10.0) / C))
                seed_h0 = mc * 10 + 100
                seed_h1 = mc * 10 + 200
                X_h0 = sample_mggd(N_TEST, P, M, M_PARAM, BETA, seed=seed_h0)
                X_h1 = (sample_mggd(N_TEST, P, M, M_PARAM, BETA, seed=seed_h1)
                        + theta * s[None, :])

                def _record(method, T0, T1):
                    auc, pd, fpr, tpr = _roc_metrics(T0, T1)
                    results.setdefault((method, n_train, snr_db), []).append((auc, pd))
                    if n_train == ROC_N and snr_db == ROC_SNR:
                        roc_curves[(method, mc)] = (fpr, tpr)

                # Oracle Rao
                _record("Oracle Rao",
                        _rao(mggd_true_score(X_h0, M_inv, M_PARAM, BETA), s),
                        _rao(mggd_true_score(X_h1, M_inv, M_PARAM, BETA), s))

                # Oracle Gaussian AMF
                _record("Oracle Gaussian AMF",
                        _amf(X_h0, M_inv, s),
                        _amf(X_h1, M_inv, s))

                # MLE Rao
                if mle_result is not None:
                    Mh, mh, bh = mle_result
                    _record("MLE Rao",
                            _rao(score_mggd_mle(X_h0, Mh, mh, bh), s),
                            _rao(score_mggd_mle(X_h1, Mh, mh, bh), s))

                # Tyler AMF
                if tyler_Cinv is not None:
                    _record("Tyler AMF",
                            _amf(X_h0, tyler_Cinv, s),
                            _amf(X_h1, tyler_Cinv, s))

                # DSM Rao methods
                for label, model in dsm.items():
                    _record(label,
                            _rao(_infer(model, X_h0), s),
                            _rao(_infer(model, X_h1), s))

            # ----- per-seed progress -----
            def _mean_pd(method):
                v = results.get((method, n_train, FIXED_SNR), [])
                return np.mean([x[1] for x in v]) if v else float("nan")

            print(
                f"    mc {mc+1}/{N_MC} [SNR={FIXED_SNR}dB]"
                f"  Oracle={_mean_pd('Oracle Rao'):.3f}"
                f"  MLE={_mean_pd('MLE Rao'):.3f}"
                f"  Two={_mean_pd('TwoBranch Rao'):.3f}"
                f"  MGGD={_mean_pd('MGGD Constrained Rao'):.3f}"
                f"  Tyler={_mean_pd('Tyler AMF'):.3f}"
                f"  GaussAMF={_mean_pd('Oracle Gaussian AMF'):.3f}",
                flush=True,
            )

        # Checkpoint after each N_train
        with open(RESULTS_PKL, "wb") as f:
            pickle.dump({"results": results, "roc_curves": roc_curves}, f)
        print(f"  -> saved {RESULTS_PKL}", flush=True)

    return {"results": results, "roc_curves": roc_curves}


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _gather(results: dict, method: str, n_list: list, snr_db: float):
    ns, means, stds = [], [], []
    for n in n_list:
        vals = results.get((method, n, snr_db), [])
        if not vals:
            continue
        pds = [v[1] for v in vals]
        ns.append(n);  means.append(np.mean(pds));  stds.append(np.std(pds))
    return np.array(ns), np.array(means), np.array(stds)


def _gather_snr(results: dict, method: str, snr_list: list, n_train: int):
    snrs, means, stds = [], [], []
    for snr in snr_list:
        vals = results.get((method, n_train, snr), [])
        if not vals:
            continue
        pds = [v[1] for v in vals]
        snrs.append(snr);  means.append(np.mean(pds));  stds.append(np.std(pds))
    return np.array(snrs), np.array(means), np.array(stds)


def _method_line(ax, xs, means, stds, method):
    if len(xs) == 0:
        return
    c  = COLORS[method]
    ls = _LS.get(method, "-")
    ax.plot(xs, means, ls, color=c, marker="o", ms=4, label=method)
    ax.fill_between(xs, means - stds, means + stds, alpha=0.15, color=c)


# ---------------------------------------------------------------------------
# Plot: Pd vs N_train  (one figure per SNR value)
# ---------------------------------------------------------------------------

def plot_pd_vs_N(results: dict, N_TRAIN_LIST: list, snr_list: list,
                 out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    for snr_db in snr_list:
        fig, ax = plt.subplots(figsize=(7, 4))
        for method in METHOD_NAMES:
            ns, means, stds = _gather(results, method, N_TRAIN_LIST, snr_db)
            _method_line(ax, ns, means, stds, method)
        ax.axhline(PFA, color="k", lw=0.8, ls=":", alpha=0.4, label=f"Pfa={PFA}")
        ax.set_xscale("log")
        ax.set_xlabel("N_train")
        ax.set_ylabel(f"Pd @ Pfa={PFA}")
        ax.set_title(f"Detection Pd vs N_train  (SNR={snr_db} dB, p={P}, beta={BETA})")
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, which="both", alpha=0.3)
        ax.set_ylim(-0.02, 1.05)
        fig.tight_layout()
        fname = out_dir / f"mggd_det_pd_vs_N_snr{snr_db}_p{P}_beta{BETA}.png"
        fig.savefig(fname, dpi=150);  plt.close(fig)
        print(f"  saved {fname}")


# ---------------------------------------------------------------------------
# Plot: Pd vs SNR  (one subplot per N value from FIXED_N_VALS)
# ---------------------------------------------------------------------------

def plot_pd_vs_snr(results: dict, snr_list: list, n_vals: list, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    ncols = min(len(n_vals), 4)
    nrows = (len(n_vals) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5 * ncols, 4 * nrows),
                             sharey=True, squeeze=False)
    axes_flat = [axes[r][c] for r in range(nrows) for c in range(ncols)]

    for ax, n_train in zip(axes_flat, n_vals):
        for method in METHOD_NAMES:
            snrs, means, stds = _gather_snr(results, method, snr_list, n_train)
            _method_line(ax, snrs, means, stds, method)
        ax.axhline(PFA, color="k", lw=0.8, ls=":", alpha=0.4)
        ax.set_title(f"N_train={n_train}")
        ax.set_xlabel("SNR (dB)")
        ax.set_ylabel(f"Pd @ Pfa={PFA}")
        ax.legend(fontsize=6, ncol=2)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.02, 1.05)

    # hide unused axes
    for ax in axes_flat[len(n_vals):]:
        ax.set_visible(False)

    fig.suptitle(f"Detection Pd vs SNR  (p={P}, beta={BETA})", fontsize=12)
    fig.tight_layout()
    fname = out_dir / f"mggd_det_pd_vs_snr_p{P}_beta{BETA}.png"
    fig.savefig(fname, dpi=150);  plt.close(fig)
    print(f"  saved {fname}")


# ---------------------------------------------------------------------------
# Plot: ROC curves  (N=ROC_N, SNR=ROC_SNR)
# ---------------------------------------------------------------------------

def plot_roc(roc_curves: dict, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))

    for method in METHOD_NAMES:
        c  = COLORS[method]
        ls = _LS.get(method, "-")
        curves = [(fpr, tpr) for (m, mc_), (fpr, tpr) in roc_curves.items()
                  if m == method]
        if not curves:
            continue
        # thin lines per seed + mean
        fprs_interp = np.logspace(-3, 0, 200)
        tprs = [np.interp(fprs_interp, fpr, tpr) for fpr, tpr in curves]
        mean_tpr = np.mean(tprs, axis=0)
        for fpr, tpr in curves:
            ax.plot(fpr, tpr, ls, color=c, alpha=0.2, lw=0.8)
        ax.plot(fprs_interp, mean_tpr, ls, color=c, lw=1.8, label=method)

    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.4)
    ax.axvline(PFA, color="k", lw=0.8, ls=":", alpha=0.4)
    ax.set_xscale("log")
    ax.set_xlim(1e-3, 1.0)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("False alarm rate (Pfa)")
    ax.set_ylabel("Detection probability (Pd)")
    ax.set_title(f"ROC  N_train={ROC_N}, SNR={ROC_SNR} dB  (p={P}, beta={BETA})")
    ax.legend(fontsize=7)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fname = out_dir / f"mggd_det_roc_N{ROC_N}_snr{ROC_SNR}_p{P}_beta{BETA}.png"
    fig.savefig(fname, dpi=150);  plt.close(fig)
    print(f"  saved {fname}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick",     action="store_true",
                        help="small N grid, N_MC=2, no N=20000")
    parser.add_argument("--eval_only", action="store_true",
                        help="skip training; load checkpoints and replot")
    parser.add_argument("--resume",    action="store_true",
                        help="load saved pkl and skip already-completed seeds")
    parser.add_argument("--snr_list",  nargs="+", type=float, default=None,
                        help="override SNR list (dB)")
    parser.add_argument("--n_mc",      type=int, default=None)
    parser.add_argument("--device",    type=str, default=None)
    args = parser.parse_args()

    global _DEVICE
    if args.device:
        _DEVICE = args.device
    elif torch.cuda.is_available():
        _DEVICE = "cuda"
        print(f"GPU detected: {torch.cuda.get_device_name(0)}")
    else:
        _DEVICE = "cpu"
    print(f"Using device: {_DEVICE}")

    N_TRAIN = N_TRAIN_QUICK if args.quick else N_TRAIN_FULL
    N_MC    = args.n_mc or (N_MC_QUICK if args.quick else N_MC_FULL)
    snr_list = [float(s) for s in args.snr_list] if args.snr_list else EVAL_SNR_LIST

    out_dir = Path("figures")

    print(f"\n--- MGGD Detection Experiment ---")
    print(f"p={P}, beta={BETA}, rho={RHO}, sigma_dsm={SIGMA_DSM}")
    print(f"N_MC={N_MC}, N_train={N_TRAIN}, SNR={snr_list}")

    data = run_detection(N_TRAIN, N_MC, snr_list=snr_list,
                         eval_only=args.eval_only, resume=args.resume)
    results    = data["results"]
    roc_curves = data["roc_curves"]

    print("\nPlotting...")
    plot_pd_vs_N(results, N_TRAIN, snr_list, out_dir)
    n_vals_for_snr = [n for n in FIXED_N_VALS if n in N_TRAIN]
    if n_vals_for_snr:
        plot_pd_vs_snr(results, snr_list, n_vals_for_snr, out_dir)
    if roc_curves:
        plot_roc(roc_curves, out_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()
