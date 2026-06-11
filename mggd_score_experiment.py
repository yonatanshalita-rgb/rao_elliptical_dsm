"""
MGGD Score Experiment
=====================
Compares score estimators on MGGD-distributed data (Pascal et al. 2013).

Metric: MSE and cosine similarity between estimated and true score on a
fixed test set, as a function of training-set size N.

Methods
-------
  Pascal MLE        — fit_mggd_mle, analytic score
  Tyler AMF         — Tyler scatter, linear score -C^{-1}x
  Linear DSM        — LinearScore trained with DSM
  TwoBranch DSM     — TwoBranchScore trained with DSM
  MGGD Constrained  — MGGDConstrainedScore trained with DSM

Usage
-----
  python mggd_score_experiment.py                         # p=3 and p=64
  python mggd_score_experiment.py --dim 64               # p=64 only
  python mggd_score_experiment.py --dim 3 64             # both (default)
  python mggd_score_experiment.py --quick
  python mggd_score_experiment.py --synthetic_only
  python mggd_score_experiment.py --vistex_only
  python mggd_score_experiment.py --beta 0.2 0.5
"""

import argparse
import os
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from baselines.classical import (
    fit_mggd_mle, score_mggd_mle,
    score_tyler_linear, tyler_estimator, fit_tyler_safe,
)
from data.generate import ar1_covariance, sample_mggd, mggd_true_score
from models.score_models import build_model
from models.train import train

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DIM_LIST  = [3, 64]       # default --dim values
RHO_AR1   = 0.8
M_PARAM   = 1.0
BETA_LIST = [0.2, 0.5, 0.8]

N_MC_FULL  = 10
N_MC_QUICK = 3
N_TEST     = 5000


def _n_values(p: int, quick: bool) -> list[int]:
    """Return N sweep appropriate for dimension p.

    For small p (<=10): 2000 is enough to see MLE dominate.
    For larger p: extend to 5000 and include values that straddle N=p
    (the rank-deficiency threshold where MLE first becomes viable).
    """
    if p <= 10:
        full  = [20, 50, 100, 200, 500, 1000, 2000]
        short = [20, 100, 500, 2000]
    else:
        full  = [20, 50, 100, 200, 500, 1000, 2000, 5000]
        short = [50, 200, 1000, 5000]
    return short if quick else full

SIGMA_LIST       = [0.05, 0.1, 0.3, 0.5]
SIGMA_LIST_QUICK = [0.1]               # quick mode: main σ only
SIGMA_MAIN       = 0.1                 # σ used for the main MSE/cosine figures

DSM_EPOCHS       = 300
DSM_EPOCHS_QUICK = 50                  # quick mode: 1/6 epochs
LR         = 1e-3
WARMUP_TB  = 20
GRAD_CLIP  = 1.0     # for mggd_constrained only

COLORS = {
    "Pascal MLE":        "#1f77b4",
    "Tyler AMF":         "#ff7f0e",
    "Linear DSM":        "#2ca02c",
    "TwoBranch DSM":     "#9467bd",
    "MGGD Constrained":  "#d62728",
}
METHOD_NAMES = list(COLORS.keys())

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def score_mse(s_hat: np.ndarray, s_true: np.ndarray) -> float:
    return float(((s_hat - s_true) ** 2).mean())


def score_cosine(s_hat: np.ndarray, s_true: np.ndarray) -> float:
    num   = (s_hat * s_true).sum(axis=-1)
    denom = (np.linalg.norm(s_hat, axis=-1) *
             np.linalg.norm(s_true, axis=-1) + 1e-12)
    return float((num / denom).mean())


def _make_loader(X: np.ndarray, batch_size: int) -> DataLoader:
    t = torch.tensor(X, dtype=torch.float32)
    ds = TensorDataset(t)
    # train() expects batches to be dicts with key 'w'
    class _DictDS(torch.utils.data.Dataset):
        def __init__(self, arr): self.t = torch.tensor(arr, dtype=torch.float32)
        def __len__(self): return len(self.t)
        def __getitem__(self, i): return {"w": self.t[i]}
    return DataLoader(_DictDS(X), batch_size=batch_size, shuffle=True)


def _make_val_loader(X: np.ndarray, batch_size: int) -> DataLoader:
    class _DictDS(torch.utils.data.Dataset):
        def __init__(self, arr): self.t = torch.tensor(arr, dtype=torch.float32)
        def __len__(self): return len(self.t)
        def __getitem__(self, i): return {"w": self.t[i]}
    return DataLoader(_DictDS(X), batch_size=batch_size)


_DEVICE = "cpu"   # overridden by --device flag in main()


def _train_dsm(model_type: str, X_train: np.ndarray, X_val: np.ndarray,
               sigma: float, n_epochs: int = DSM_EPOCHS) -> torch.nn.Module:
    n = X_train.shape[1]
    model = build_model(model_type, n)
    bs = min(len(X_train), 256)   # larger batches on GPU
    train_dl = _make_loader(X_train, bs)
    val_dl   = _make_val_loader(X_val, bs)
    warmup = min(WARMUP_TB, n_epochs // 5) if model_type == "two_branch" else 0
    train(
        model, model_type, train_dl, val_dl,
        n_epochs=n_epochs, lr=LR, sigma=sigma,
        warmup_epochs=warmup,
        grad_clip=GRAD_CLIP if model_type == "mggd_constrained" else None,
        device=_DEVICE,
        verbose=False,
    )
    return model


def _eval_dsm(model: torch.nn.Module, X_test: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        t = torch.tensor(X_test, dtype=torch.float32).to(_DEVICE)
        return model(t).cpu().numpy()


# results[key] = {'mse': list, 'cos': list}
# key = (beta_or_label, sigma_or_'mle_tyler', method_name, n)
def _record(results: dict, key: tuple, s_hat: np.ndarray,
            s_true: np.ndarray) -> None:
    if key not in results:
        results[key] = {"mse": [], "cos": []}
    results[key]["mse"].append(score_mse(s_hat, s_true))
    results[key]["cos"].append(score_cosine(s_hat, s_true))


# ---------------------------------------------------------------------------
# Synthetic experiment
# ---------------------------------------------------------------------------

def run_synthetic(N_VALUES, N_MC, beta_list, p: int,
                  sigma_list=None, n_epochs: int = DSM_EPOCHS):
    if sigma_list is None:
        sigma_list = SIGMA_LIST

    M = ar1_covariance(p, RHO_AR1)
    M_inv = np.linalg.inv(M)

    results = {}

    for beta in beta_list:
        group = (p, beta)
        print(f"\n=== p={p}, beta={beta} ===")
        X_test = sample_mggd(N_TEST, p, M, M_PARAM, beta, seed=9999)
        s_true = mggd_true_score(X_test, M_inv, M_PARAM, beta)

        for n in N_VALUES:
            print(f"  n={n}:", flush=True)

            # precompute MLE and Tyler for all MC (σ-independent)
            mle_cache   = {}
            tyler_cache = {}
            for mc in range(N_MC):
                X_tr = sample_mggd(n, p, M, M_PARAM, beta, seed=mc * 10)
                result = fit_mggd_mle(X_tr)
                mle_cache[mc] = score_mggd_mle(X_test, *result) if result else None
                C_ty = fit_tyler_safe(X_tr)
                tyler_cache[mc] = (score_tyler_linear(X_test, C_ty)
                                   if C_ty is not None else None)

            for mc in range(N_MC):
                if mle_cache[mc] is not None:
                    _record(results, (group, "baseline", "Pascal MLE", n),
                            mle_cache[mc], s_true)
                if tyler_cache[mc] is not None:
                    _record(results, (group, "baseline", "Tyler AMF", n),
                            tyler_cache[mc], s_true)
            mle_vals = results.get((group, "baseline", "Pascal MLE", n), [])
            if mle_vals:
                print(f"  [baselines] MLE MSE={np.mean(mle_vals):.4f}", flush=True)

            # DSM methods — retrain at each sigma
            for sigma in sigma_list:
                for mc in range(N_MC):
                    X_tr  = sample_mggd(n, p, M, M_PARAM, beta, seed=mc * 10)
                    X_val = sample_mggd(max(n // 5, p + 1), p, M, M_PARAM, beta,
                                        seed=mc * 10 + 1)
                    for mtype, label in [("linear_mse",       "Linear DSM"),
                                         ("two_branch",        "TwoBranch DSM"),
                                         ("mggd_constrained",  "MGGD Constrained")]:
                        model = _train_dsm(mtype, X_tr, X_val, sigma, n_epochs)
                        s_hat = _eval_dsm(model, X_test)
                        _record(results, (group, sigma, label, n), s_hat, s_true)
                    # running average after each seed
                    lin_v  = results.get((group, sigma, "Linear DSM",       n), [])
                    two_v  = results.get((group, sigma, "TwoBranch DSM",    n), [])
                    mgg_v  = results.get((group, sigma, "MGGD Constrained", n), [])
                    print(
                        f"  [sigma={sigma} mc {mc+1}/{N_MC}]"
                        f" Lin={np.mean(lin_v):.4f}"
                        f" Two={np.mean(two_v):.4f}"
                        f" MGGD={np.mean(mgg_v):.4f}",
                        flush=True,
                    )

            print(f"  n={n} done", flush=True)

    return results


# ---------------------------------------------------------------------------
# VisTex experiment
# ---------------------------------------------------------------------------

def _load_vistex_vectors(path: Path) -> np.ndarray | None:
    try:
        import pywt
        from PIL import Image
    except ImportError:
        print("  pywt or Pillow not installed — skipping VisTex")
        return None

    if not path.exists():
        print(f"  Missing: {path}")
        print(f"  Download from: http://vismod.media.mit.edu/pub/VisTex/")
        return None

    img = np.array(Image.open(path)).astype(np.float32)
    if img.ndim == 2:
        img = img[:, :, None]

    channels = []
    for c in range(img.shape[2]):
        coeffs = pywt.swt2(img[:, :, c], "db4", level=1)
        # first level: (cA, (cH, cV, cD)); take cH (horizontal detail)
        cH = coeffs[0][1][0]
        channels.append(cH.ravel())

    X = np.stack(channels, axis=1).astype(np.float32)
    X -= X.mean(axis=0)
    return X


def run_vistex(N_VALUES, N_MC, vistex_dir: Path):
    images = {
        "Bark":   vistex_dir / "Bark.0000.ppm",
        "Leaves": vistex_dir / "Leaves.0008.ppm",
    }
    results = {}

    for label, path in images.items():
        print(f"\n=== VisTex: {label} ===")
        X_all = _load_vistex_vectors(path)
        if X_all is None:
            continue

        p = X_all.shape[1]
        M_inv_gt = None

        # fixed 80/20 split
        rng = np.random.default_rng(0)
        idx = rng.permutation(len(X_all))
        n_test_vt = max(int(0.2 * len(X_all)), 500)
        X_test = X_all[idx[:n_test_vt]]
        X_pool = X_all[idx[n_test_vt:]]

        # pseudo-ground-truth: MLE on full dataset
        print(f"  Fitting pseudo-GT MLE on {len(X_all)} samples...", end="", flush=True)
        gt = fit_mggd_mle(X_all)
        if gt is None:
            print(" failed — skipping")
            continue
        M_gt, m_gt, beta_gt = gt
        print(f" beta_gt={beta_gt:.3f}")
        s_true = score_mggd_mle(X_test, M_gt, m_gt, beta_gt)

        for n in N_VALUES:
            if n > len(X_pool):
                continue
            print(f"  n={n}", end="", flush=True)

            mle_cache   = {}
            tyler_cache = {}
            for mc in range(N_MC):
                rng_mc = np.random.default_rng(mc * 10)
                idx_tr = rng_mc.choice(len(X_pool), size=n, replace=False)
                X_tr = X_pool[idx_tr]
                result = fit_mggd_mle(X_tr)
                mle_cache[mc] = score_mggd_mle(X_test, *result) if result else None
                C_ty = fit_tyler_safe(X_tr)
                tyler_cache[mc] = (score_tyler_linear(X_test, C_ty)
                                   if C_ty is not None else None)

            for mc in range(N_MC):
                if mle_cache[mc] is not None:
                    _record(results, (label, "baseline", "Pascal MLE", n),
                            mle_cache[mc], s_true)
                _record(results, (label, "baseline", "Tyler AMF", n),
                        tyler_cache[mc], s_true)

            for sigma in SIGMA_LIST:
                for mc in range(N_MC):
                    rng_mc = np.random.default_rng(mc * 10)
                    idx_tr = rng_mc.choice(len(X_pool), size=n, replace=False)
                    X_tr  = X_pool[idx_tr]
                    n_val = max(n // 5, 20)
                    rng_v = np.random.default_rng(mc * 10 + 1)
                    idx_v = rng_v.choice(len(X_pool), size=n_val, replace=False)
                    X_val = X_pool[idx_v]

                    for mtype, mlabel in [("linear_mse",       "Linear DSM"),
                                          ("two_branch",        "TwoBranch DSM"),
                                          ("mggd_constrained",  "MGGD Constrained")]:
                        model = _train_dsm(mtype, X_tr, X_val, sigma)
                        s_hat = _eval_dsm(model, X_test)
                        _record(results, (label, sigma, mlabel, n), s_hat, s_true)

            print(" done")

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _gather(results: dict, group, sigma_or_base, method, N_VALUES):
    means_mse, stds_mse = [], []
    means_cos, stds_cos = [], []
    ns_used = []
    for n in N_VALUES:
        key = (group, sigma_or_base, method, n)
        if key not in results or not results[key]["mse"]:
            continue
        ns_used.append(n)
        means_mse.append(np.mean(results[key]["mse"]))
        stds_mse.append(np.std(results[key]["mse"]))
        means_cos.append(np.mean(results[key]["cos"]))
        stds_cos.append(np.std(results[key]["cos"]))
    return (np.array(ns_used),
            np.array(means_mse), np.array(stds_mse),
            np.array(means_cos), np.array(stds_cos))


def _slug(group) -> str:
    """File-name-safe slug for a group key."""
    if isinstance(group, tuple):
        p, beta = group
        return f"p{p}_beta{beta}"
    return str(group).replace(" ", "_")


def print_summary(results: dict, groups, group_labels, N_VALUES):
    """Print MSE table: rows = N, cols = method, one block per group."""
    methods_ordered = METHOD_NAMES
    for group, glabel in zip(groups, group_labels):
        print(f"\n--- {glabel} (MSE @ sigma={SIGMA_MAIN}) ---")
        header = f"{'N':>6}  " + "  ".join(f"{m[:12]:>12}" for m in methods_ordered)
        print(header)
        for n in N_VALUES:
            row = f"{n:>6}  "
            for method in methods_ordered:
                sk = "baseline" if method in ("Pascal MLE", "Tyler AMF") else SIGMA_MAIN
                key = (group, sk, method, n)
                if key in results and results[key]["mse"]:
                    val = np.mean(results[key]["mse"])
                    row += f"  {val:>12.4f}"
                else:
                    row += f"  {'—':>12}"
            print(row)


def plot_main(results: dict, groups, group_labels, N_VALUES, out_dir: Path):
    """One MSE + one cosine figure per group (beta value or VisTex image)."""
    out_dir.mkdir(parents=True, exist_ok=True)

    for group, glabel in zip(groups, group_labels):
        for metric, ylabel, suffix in [("mse", "Score MSE", "mse"),
                                        ("cos", "Cosine similarity", "cos")]:
            fig, ax = plt.subplots(figsize=(6, 4))

            for method in METHOD_NAMES:
                sigma_key = "baseline" if method in ("Pascal MLE", "Tyler AMF") \
                            else SIGMA_MAIN
                ns, mm, ms, mc_, mcs_ = _gather(results, group, sigma_key,
                                                 method, N_VALUES)
                if len(ns) == 0:
                    continue
                vals  = mm if metric == "mse" else mc_
                stds  = ms if metric == "mse" else mcs_
                color = COLORS[method]
                ax.plot(ns, vals, marker="o", color=color, label=method)
                ax.fill_between(ns, vals - stds, vals + stds,
                                alpha=0.15, color=color)

            ax.set_xscale("log")
            if metric == "mse":
                ax.set_yscale("log")
            ax.set_xlabel("N (training samples)")
            ax.set_ylabel(ylabel)
            ax.set_title(glabel)
            ax.legend(fontsize=8)
            ax.grid(True, which="both", alpha=0.3)
            fig.tight_layout()
            fname = out_dir / f"mggd_{suffix}_{_slug(group)}.png"
            fig.savefig(fname, dpi=150)
            plt.close(fig)
            print(f"  saved {fname}")


def plot_sigma_sweep(results: dict, groups, group_labels, N_VALUES, out_dir: Path):
    """For each DSM method, four σ curves on one plot per group."""
    out_dir.mkdir(parents=True, exist_ok=True)
    dsm_methods = ["Linear DSM", "TwoBranch DSM", "MGGD Constrained"]

    for group, glabel in zip(groups, group_labels):
        fig, axes = plt.subplots(1, len(dsm_methods),
                                 figsize=(5 * len(dsm_methods), 4), sharey=True)
        if len(dsm_methods) == 1:
            axes = [axes]

        sigma_colors = {0.05: "#1f77b4", 0.1: "#2ca02c",
                        0.3: "#ff7f0e", 0.5: "#d62728"}

        for ax, method in zip(axes, dsm_methods):
            # MLE reference
            ns, mm, ms, *_ = _gather(results, group, "baseline", "Pascal MLE", N_VALUES)
            if len(ns):
                ax.fill_between(ns, mm - ms, mm + ms, alpha=0.1, color="grey")
                ax.plot(ns, mm, "--", color="grey", lw=1, label="Pascal MLE")

            for sigma in SIGMA_LIST:
                ns, mm, ms, *_ = _gather(results, group, sigma, method, N_VALUES)
                if len(ns) == 0:
                    continue
                c = sigma_colors.get(sigma, "black")
                ax.plot(ns, mm, marker="o", color=c, label=f"σ={sigma}")
                ax.fill_between(ns, mm - ms, mm + ms, alpha=0.15, color=c)

            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_title(method)
            ax.set_xlabel("N")
            ax.legend(fontsize=7)
            ax.grid(True, which="both", alpha=0.3)

        axes[0].set_ylabel("Score MSE")
        fig.suptitle(f"σ sweep — {glabel}", fontsize=11)
        fig.tight_layout()
        fname = out_dir / f"mggd_sigma_sweep_{_slug(group)}.png"
        fig.savefig(fname, dpi=150)
        plt.close(fig)
        print(f"  saved {fname}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick",          action="store_true")
    parser.add_argument("--synthetic_only", action="store_true")
    parser.add_argument("--vistex_only",    action="store_true")
    parser.add_argument("--beta", nargs="+", type=float, default=None)
    parser.add_argument("--dim",    nargs="+", type=int,   default=DIM_LIST,
                        help="dimension(s) for AR(1) synthetic experiment")
    parser.add_argument("--n_mc",   type=int,  default=None,
                        help="override number of MC seeds (e.g. 1 for a quick preview)")
    parser.add_argument("--device", type=str,  default=None,
                        help="torch device: cpu | cuda (default: auto)")
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

    N_MC       = args.n_mc if args.n_mc else (N_MC_QUICK if args.quick else N_MC_FULL)
    n_epochs   = DSM_EPOCHS_QUICK  if args.quick else DSM_EPOCHS
    sigma_list = SIGMA_LIST_QUICK  if args.quick else SIGMA_LIST
    beta_list  = args.beta if args.beta else BETA_LIST
    dim_list   = args.dim

    out_dir = Path("figures")
    results_all = {}

    if not args.vistex_only:
        print("\n--- Synthetic MGGD experiment ---")
        if args.quick:
            print(f"(quick mode: {n_epochs} epochs, sigma={sigma_list}, N_MC={N_MC})")
        for p in dim_list:
            N_VALUES = _n_values(p, args.quick)
            r = run_synthetic(N_VALUES, N_MC, beta_list, p,
                              sigma_list=sigma_list, n_epochs=n_epochs)
            results_all.update(r)

        # collect all (p, beta) groups and their N_VALUES for plotting
        syn_groups = []
        for p in dim_list:
            for beta in beta_list:
                syn_groups.append((p, beta))
        syn_labels = [f"p={p}, beta={b}" for (p, b) in syn_groups]

        all_n = sorted(set(
            n for p in dim_list for n in _n_values(p, args.quick)
        ))
        print_summary(results_all, syn_groups, syn_labels, all_n)
        print("\nPlotting synthetic results...")
        plot_main(results_all, syn_groups, syn_labels, all_n, out_dir)
        plot_sigma_sweep(results_all, syn_groups, syn_labels, all_n, out_dir)

    if not args.synthetic_only:
        vistex_dir = Path("data/vistex")
        print("\n--- VisTex experiment ---")
        r = run_vistex(N_VALUES, N_MC, vistex_dir)
        results_all.update(r)

        vt_groups  = [g for g in results_all
                      if isinstance(g[0], str) and g[0] in ("Bark", "Leaves")]
        # deduplicate group labels
        seen = set()
        vt_unique = []
        for g in vt_groups:
            if g[0] not in seen:
                seen.add(g[0])
                vt_unique.append(g[0])
        if vt_unique:
            print("\nPlotting VisTex results...")
            plot_main(results_all, vt_unique, vt_unique, N_VALUES, out_dir)
            plot_sigma_sweep(results_all, vt_unique, vt_unique, N_VALUES, out_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
