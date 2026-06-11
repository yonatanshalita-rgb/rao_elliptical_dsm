# MGGD Detection Experiment — Implementation Plan (v2)

## Goal

Show that DSM-based Rao detectors beat classical methods at low N on MGGD
data (β=0.5, ρ=0.8), while MLE Rao and Oracle recover asymptotically.
Two sweep axes: N_train (can we learn from few samples?) and SNR (does the gap depend
on signal strength?).

---

## Detector family: Rao test

H0: x ~ MGGD(0, M, m=1, β),   H1: x = θ·s + noise,  noise ~ MGGD(0, M, m=1, β)

Rao statistic: T(x) = s^T · score(x)   (gradient of log-likelihood w.r.t. θ at θ=0)

---

## Methods

| Label | score estimator | notes |
|---|---|---|
| Oracle Rao | true MGGD score (formula, true M/m/β) | upper bound |
| MLE Rao | Pascal MLE parameters | skip when fit returns None |
| TwoBranch Rao | trained two_branch model | main DSM method |
| Linear Rao | trained linear_mse model | DSM ablation |
| MGGD Constrained Rao | trained mggd_constrained model | structured DSM |
| Tyler AMF | (s^T C_Tyler^{-1} x) / sqrt(s^T C_Tyler^{-1} s) | skip when fit returns None |
| Oracle Gaussian AMF | (s^T M_true^{-1} x) / sqrt(s^T M_true^{-1} s) | mismatched Gaussian model |

---

## Fixed parameters

```
p       = 64
beta    = 0.5          # moderate heavy-tails; tau~Gamma(64,2) keeps noise at manageable scale
                       # beta=0.5 makes noise amplitude ~1e6, drowning any reasonable signal
m       = 1.0
rho     = 0.8          # AR(1) covariance, matching score experiment
M       = ar1_covariance(p, rho=0.8)
s       = ones(p) / sqrt(p)          # matches make_test_fixed_theta in existing code
sigma   = 0.3          # DSM noise level (fixed)
N_test  = 50_000       # H0 + H1 balanced; reused across all N_train
Pfa     = 0.01
```

---

## Varying axis 1: N_train

```python
N_TRAIN_LIST = [20, 50, 100, 200, 500, 1000, 2000, 5000, 20_000]
N_MC         = 5        # mean ± 1-std band
```

**No val set.** All N_train samples go to DSM training. Dropping the val set keeps the
N budget identical for DSM and the classical baselines (both use exactly N_train samples),
and avoids giving DSM a hidden disadvantage at small N. Use fixed adaptive epochs without
early stopping: `epochs = max(100, 6_000 // max(1, N_train // BATCH_SIZE))`.

---

## Varying axis 2: SNR

```python
EVAL_SNR_LIST = [20, 25, 30, 35, 40, 45]  # dB
# For beta=0.5, m=1, p=64, rho=0.8: tau~Gamma(64,2) -> E[tau]=128.
# Rao test SNR ~ theta*C*E[1/tau] ~ theta*0.178*(1/126).
# Oracle reaches Pd=0.9 only at ~35 dB; interesting spread is 25-40 dB.
```

Both sweep axes produce separate plots (see Figures below).

---

## H1 sample generation

SNR in dB defines the signal amplitude via the matched-filter normalisation:

```python
C = s @ M_inv @ s                                  # scalar, s^T M^{-1} s
theta = np.sqrt(10 ** (snr_db / 10.0) / C)        # matches make_test_fixed_theta
x_h1 = sample_mggd(N_test, p, M, m, beta, seed=...) + theta * s[None, :]
```

The score is always evaluated under H0 (background model).

---

## ROC and metrics

```python
from sklearn.metrics import roc_auc_score, roc_curve

y_true  = np.concatenate([np.zeros(N_test), np.ones(N_test)])
y_score = np.concatenate([T_h0, T_h1])             # T(x) = s^T score(x)
auc     = roc_auc_score(y_true, y_score)
fpr, tpr, _ = roc_curve(y_true, y_score)
pd_at_pfa   = np.interp(Pfa, fpr, tpr)
```

---

## Figures

```
figures/mggd_det_pd_vs_N_snrXX_p64_beta0.2.png   # Pd@Pfa=0.01 vs N_train, one per SNR
figures/mggd_det_pd_vs_snr_NXX_p64_beta0.2.png   # Pd@Pfa=0.01 vs SNR, one per N_train
figures/mggd_det_roc_N200_snr5_p64_beta0.2.png    # ROC curve at N=200, SNR=5 dB
```

Primary plot: Pd vs N_train at SNR=35 dB (Oracle Pd~0.9; methods spread across 0.1–0.9).
Secondary plot: Pd vs SNR at N_train ∈ {200, 1000, 5000, 20000}.

---

## File structure

```
mggd_detection_exp.py
  run_detection(N_TRAIN_LIST, EVAL_SNR_LIST, N_MC, sigma, n_epochs_fn) -> dict
    results[(method, n_train, snr_db)] = list of (auc, pd) over N_MC seeds
  plot_detection_vs_N(results, fixed_snr, ...)
  plot_detection_vs_snr(results, n_values, ...)
  main() — argparse: --quick, --snr_list, --n_mc, --device, --eval_only
```

`--eval_only`: skip training, load saved checkpoints from `checkpoints/det_<mtype>_N<n>_mc<mc>.pt`,
replot from saved results dict (`results/mggd_det_results.pkl`). Training 9 N values × 5 seeds × 3 DSM
models is expensive; this flag allows re-running the plots or changing SNR eval without retraining.

Reuse:
- `ar1_covariance`, `sample_mggd`, `mggd_true_score` from `data.generate`
- `fit_mggd_mle`, `score_mggd_mle`, `fit_tyler_safe`, `score_tyler_linear` from `baselines.classical`
  (all four already exist — added by the score experiment)
- `build_model`, `dsm_mse_loss` from `models.*`; mirror `_train_dsm` / `_eval_dsm` from score experiment

---

## Key implementation details

### Adaptive epochs (no val set)
```python
BATCH_SIZE = 512
def n_epochs(n_train):
    return max(100, 6_000 // max(1, n_train // BATCH_SIZE))
```
No `X_val` is passed to training. Training runs to the fixed epoch count.

### Oracle Gaussian AMF
Deterministic (uses true M). Compute once per n_train/SNR pair; record the same value
for all N_MC seeds (or just compute inside the loop — it is free).

### Tyler / MLE None handling
Same skip-and-don't-record pattern as the score experiment. At N_train < p+1 (Tyler)
or N_train < p+2 (MLE), those methods simply have no data point on the curve.

### Inner loop order
```
for n_train in N_TRAIN_LIST:
    fit/train all methods once on X_train (N_MC seeds)
    for snr_db in EVAL_SNR_LIST:
        generate X_test_h0, X_test_h1  (same seed for all methods at same snr)
        evaluate T(x) for each method → record (auc, pd)
```
Training is outer, SNR evaluation is inner — avoids re-training for each SNR point.

### Per-seed progress print
After each mc seed finishes all SNR evaluations, print:
```
  [N=200 mc 3/5 snr=5] Oracle pd=0.943 MLE pd=0.812 Two pd=0.743 ...
```

---

## Potential issues

- **β=0.2 heavy tails at small N**: MLE will fail (return None) for most seeds when
  N_train < ~p+10 due to numerical instability beyond the p+2 guard. This is expected
  and is the point — DSM fills the gap.
- **No val set overfitting risk**: at N_train=20, DSM has very few samples. With fixed
  epochs and no early stopping, overfitting is possible. If Pd for DSM is worse than
  Oracle Gaussian AMF at N=20, consider re-introducing a small fixed val set (e.g.,
  N_val=20 held out separately, not subtracted from N_train budget).
- **SNR=1 dB is very low**: all methods may have Pd ≈ Pfa = 0.01. That is fine — it
  shows the floor. SNR=20 dB may saturate all methods at Pd=1. The interesting range
  is 3–10 dB.
- **N_test=50k for β=0.2**: MGGD with β=0.2 has heavy tails; 50k samples gives stable
  ROC estimates even at Pfa=0.01 (500 false-alarm samples in expectation).
