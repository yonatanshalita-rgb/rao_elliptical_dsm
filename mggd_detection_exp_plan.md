# MGGD Detection Experiment — Implementation Plan

## Goal

Show that DSM-based Rao detectors beat classical methods at low training-sample counts
on MGGD data (p=64, beta=0.5), while MLE Rao and Oracle AMF dominate asymptotically.
The key thesis claim: even without knowing the true MGGD parameters, a DSM-trained score
model gives a near-optimal detector once N is large enough, and beats MLE when N < p.

---

## Detector family: Rao test

For H0: x ~ MGGD(0, M, m, β),  H1: x ~ MGGD(α·s, M, m, β) with α > 0,
the Rao statistic is the gradient of the log-likelihood w.r.t. α at α=0:

    T(x) = s^T · ∇_x log p(x|H0)  =  s^T · score(x)

Each method supplies a different score estimator; we evaluate T on held-out test
samples under H0 and H1, then compute the ROC.

---

## Methods

| Label | score estimator | notes |
|---|---|---|
| Oracle Rao | true MGGD score (formula, true M/m/β) | upper bound |
| MLE Rao | score from Pascal MLE parameters | returns None when N<p+2 — skip |
| TwoBranch Rao | trained two_branch model | main DSM method |
| Linear Rao | trained linear_mse model | DSM ablation |
| MGGD Constrained Rao | trained mggd_constrained model | structured DSM |
| Tyler AMF | (s^T C_Tyler^{-1} x) / sqrt(s^T C_Tyler^{-1} s) | returns None when N<p+1 — skip |
| Oracle Gaussian AMF | (s^T M_true^{-1} x) / sqrt(s^T M_true^{-1} s) | mismatched model (Gaussian) |

All Rao statistics are normalized: T(x) = (s^T score(x)) / ||s|| so that the
threshold is on the same scale regardless of signal direction.

The Oracle Gaussian AMF is included as a reference for "what Gaussian assumption buys you"
— it will be suboptimal because the data is genuinely MGGD.

---

## Fixed parameters

```
p       = 64
beta    = 0.5       # Gaussian-like but MGGD; change to 0.2 to show heavier-tail gap
m       = 1.0
M       = AR(1) covariance with rho=0.3  (reuse ar1_covariance from score exp)
sigma   = 0.3       (DSM noise level; fixed across all runs)
s       = M e_1 / ||M e_1||   (signal in first eigenvector direction of M)
SNR     = 3.0       (H1 samples: x = snr * s + noise, where noise ~ MGGD(0,M,m,beta))
N_test  = 10 000   (H0 + H1 balanced for ROC; reuse same test set across N_train)
Pfa     = 0.01      (threshold for Pd@Pfa plot)
```

---

## Varying axis: N_train

```
N_VALUES = [20, 50, 100, 200, 500, 1000, 2000, 5000]
N_MC     = 5   (seeds 0..4; mean + 1-sigma band in plots)
```

For each N_train, N_MC seed pairs:
1. Draw X_train (N_train × p) from H0
2. Draw X_val (max(N_train//5, p+1) × p) from H0 (DSM validation only)
3. Fit classical baselines on X_train
4. Train DSM models on X_train / X_val (sigma=0.3, adaptive epochs)
5. Evaluate all detectors on X_test_h0, X_test_h1

---

## H1 sample generation

```python
# Draw background sample from MGGD, then shift by SNR*s
x_h1 = sample_mggd(N_test, p, M, m, beta, seed=...) + SNR * s[None, :]
```

The score is evaluated at the raw test point x (background model), so the Rao test
is always T(x) = s^T · score_h0(x).  The threshold is swept over T values to get ROC.

---

## Metrics

1. **AUC** vs N_train (main plot): mean ± std over N_MC seeds, all methods
2. **Pd @ Pfa=0.01** vs N_train: same layout
3. **ROC curve** at N_train=200: all methods on one figure (log-scale Pfa axis)

---

## Figures

```
figures/mggd_det_auc_p64_beta0.5.png
figures/mggd_det_pd_p64_beta0.5.png
figures/mggd_det_roc_N200_p64_beta0.5.png
```

---

## File structure

```
mggd_detection_exp.py
  run_detection(N_VALUES, N_MC, snr, sigma, n_epochs) -> dict
    results[(method_label, n)] = list of (auc, pd) tuples
  plot_detection(results, ...)
  main() with argparse (--quick, --dim, --snr, --n_mc, --device)
```

Reuse from score experiment:
- `ar1_covariance`, `sample_mggd`, `mggd_true_score` from `data.generate`
- `fit_mggd_mle`, `score_mggd_mle`, `fit_tyler_safe`, `score_tyler_linear` from `baselines.classical`
- `_train_dsm`, `_eval_dsm` (copy/adapt from score experiment)

---

## Key implementation details

### Tyler and MLE None handling
Both can fail at low N. Use the same skip-and-don't-record pattern as the score experiment.
When Tyler returns None, Tyler AMF is simply absent from that (seed, N) point.
When MLE returns None, MLE Rao is absent.

### Oracle Gaussian AMF
Does not need training. At each seed, compute:
```python
T_amf = (X_test @ M_inv @ s) / np.sqrt(s @ M_inv @ s)   # (N_test,)
```
This is deterministic (uses true M) so result is the same across seeds.
Still include in the mc loop to keep code uniform; record auc/pd once per n.

### DSM training
Reuse `_train_dsm` from `mggd_score_experiment.py` verbatim.
For quick mode: 50 epochs. Full mode: adaptive (max(100, 6000 // max(1, N//BATCH_SIZE))).

### ROC computation
```python
from sklearn.metrics import roc_auc_score, roc_curve
y_true = np.concatenate([np.zeros(N_test), np.ones(N_test)])
y_score = np.concatenate([T_h0, T_h1])
auc = roc_auc_score(y_true, y_score)
fpr, tpr, _ = roc_curve(y_true, y_score)
pd_at_pfa = np.interp(Pfa, fpr, tpr)
```

### Plot style
- AUC and Pd plots: same color/marker scheme as score experiment
  (Oracle = dashed black, MLE = solid blue, TwoBranch = solid red, Linear = green,
   MGGD Constrained = orange, Tyler AMF = purple, Oracle Gaussian AMF = dashed gray)
- Fill ±1 std band; mark points where method was absent (None) as gaps in the curve
- ROC curves: log-scale x-axis (Pfa), linear y-axis (Pd)

---

## Potential issues

- **N_train < p for Tyler at small N**: Tyler needs N≥p+1. At N=20 (p=64), Tyler is always
  None. That is expected and the gap illustrates the proposed advantage.
- **MLE needs N≥p+2**: at N=20, 50 (< 66), MLE is always None. Same comment.
- **DSM sigma=0.3**: chosen to be non-negligible relative to signal scale but not so large
  that DSM score is entirely dominated by noise. The score experiment showed σ=0.3 gives
  a visible bias floor at large N; for detection that's fine because all methods share the
  same evaluation — relative ranking is what matters.
- **SNR calibration**: SNR=3.0 may be too easy or too hard depending on beta. Quick check:
  if Oracle Rao AUC ≈ 1.0 at N_test=10000, reduce SNR to 1.0 to widen the gap between methods.
  Add `--snr` flag to make this tunable without code edits.
