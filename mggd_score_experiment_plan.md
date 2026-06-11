# MGGD Score Experiment — Implementation Plan

## Goal
Compare score estimators on MGGD-distributed
data. Metric: MSE between estimated and true
score on a fixed test set, as a function of training-set size N. Five methods compared
across β ∈ {0.2, 0.5, 0.8}, p=3, AR(1) scatter (ρ=0.8).

---

## Files to touch (in dependency order)

| File | Action | What changes |
|---|---|---|
| `data/generate.py` | add | `sample_mggd()` sampler |
| `baselines/classical.py` | add | `fit_mggd_mle()`, `score_mggd_mle()`, `score_tyler_linear()` |
| `models/score_models.py` | add | `MGGDConstrainedScore`, update `build_model` |
| `models/train.py` | add | `'mggd_constrained'` entry in `LOSS_FNS` |
| `mggd_score_experiment.py` | create | main experiment loop, VisTex block, plotting |
| `requirements.txt` | maybe | add `pywt` if missing |

---

## Step 1 — MGGD sampler (`data/generate.py`)

Add function `sample_mggd(n, p, M, m, beta, seed)`.

**Stochastic representation:** `x = τ · M^{1/2} · u`
- `u`: uniform on unit sphere — draw `z ~ N(0, I_p)`, set `u = z / ||z||`
- `τ^{2β} ~ Γ(p/(2β), 2m^β)` — draw `g ~ Γ(p/(2β), 2m^β)`, set `τ = g^{1/(2β)}`
- `M^{1/2}`: Cholesky of M (lower triangular L, so `x = τ · L · u`)

**Normalization check:** AR(1) with ρ=0.8 already has Tr(M)=p (all diagonal entries = 1). ✓

Also add `mggd_true_score(x, M_inv, m, beta)` returning the analytic score:
```
∇_x log p = -(β / m^β) · (x^T M^{-1} x)^{β-1} · M^{-1} x
```
This is the ground-truth against which all methods are evaluated.

---

## Step 2 — Pascal MLE (`baselines/classical.py`)

Add `fit_mggd_mle(X, max_iter=100, tol=1e-6)` returning `(M, m, beta)`.

**Algorithm 1 of Pascal et al. 2013:**

1. Initialize: `M = (p/N) · X^T X` (sample scatter, normalized), `beta = 0.5`
2. **Fixed-point for M** (eq. 9) — iterate until convergence:
   ```
   y_i = x_i^T M^{-1} x_i                    # Mahalanobis distance (scalar)
   S   = sum_i y_i^beta                       # normalizing sum
   M_new = (p / S) · sum_i [ y_i^{beta-1} · x_i x_i^T ]
   M_new = M_new * p / trace(M_new)           # enforce Tr(M) = p
   ```
   Note: the weight on each outer product is y_i^{β−1} (not y_i^{−β}).
   Sanity check: at β=1 this reduces to (1/N)·sum x_i x_i^T (Gaussian MLE ✓).
   At β→0 this approaches Tyler's estimator (weight ∝ 1/y_i ✓).

3. **Closed form for m** (eq. 8):
   ```
   m = [ (beta / (N·p)) · sum_i y_i^beta ]^{1/beta}
   ```
   Derived by setting d/dm of the log-likelihood to zero; note the β prefactor
   and the 1/β exponent (not 1/(2β)).
4. **Newton-Raphson for β** (eq. 13, set = 0 and iterate):
   ```
   S = sum_i y_i^beta                  # reuse from M-step

   f(beta) = (pN / (2S)) * sum_i y_i^beta * ln(y_i)
           - (pN / (2*beta)) * [ digamma(p / (2*beta)) + ln(2) ]
           - N
           - (pN / (2*beta)) * ln( (beta / (pN)) * S )
   ```
   This is eq (13) of Pascal et al. 2013, the score of the profile log-likelihood
   in β after substituting the optimal M and m.  Set f(β)=0, solve with one
   Newton-Raphson step per outer iteration:
   ```
   df = (f(beta + 1e-4) - f(beta - 1e-4)) / 2e-4   # central finite difference
   beta = beta - f(beta) / df
   beta = clip(beta, 0.05, 2.0)
   ```
   `digamma` = `scipy.special.digamma`.
5. Alternate steps 2→3→4 until `||M_new - M||_F / ||M||_F < tol`

Add `score_mggd_mle(X_test, M, m, beta)` that applies the analytic score formula with
fitted parameters.

Add `score_tyler_linear(X_test, M_tyler)` returning `-M_tyler^{-1} x` (linear Gaussian
score using Tyler scatter estimate). Tyler fitting is already in `baselines/classical.py`;
wrap it here.

**Numerical guard:** if N < p+2 or M is near-singular (min eigenvalue < 1e-8 * max),
return None and skip that MC trial for MLE.

---

## Step 3 — MGGD-constrained model (`models/score_models.py`)

Add class `MGGDConstrainedScore(nn.Module)`:

```python
# Parameters: A (n×n), log_alpha_shifted (scalar), log_c (scalar)
# Score: exp(log_c) * clamp(||Ax||^2, 1e-4)^(exp(log_alpha_shifted) - 1) * (-A^T A x)
#
# Parameterization:
#   alpha = exp(log_alpha_shifted) - 1    so alpha in (-1, ∞)
#   which covers β ∈ (0, 2) since alpha = β-1
#
# Why this parameterization:
#   - alpha must be > -1 for the MGGD score to be integrable (β > 0)
#   - exp(.) + (-1) keeps alpha > -1 hard, no clipping needed
#   - clamp at 1e-4 (not 1e-6) keeps d^alpha finite for negative alpha
```

**Initialization:**
- `A = (1/sqrt(n)) * I`  (same as other models)
- `log_alpha_shifted = 0`  → alpha = exp(0) - 1 = 0  → β=1 (Gaussian start)
- `log_c = 0`  → exp(log_c) = 1

**Forward:**
```python
z = x @ A.T                          # Ax
v = -(z @ A)                         # -A^T A x
d = (z*z).sum(-1, keepdim=True)      # ||Ax||^2, shape (..., 1)
d_safe = d.clamp(min=1e-4)
alpha = torch.exp(self.log_alpha_shifted) - 1.0   # scalar
c = torch.exp(self.log_c)                          # scalar
weight = c * d_safe.pow(alpha)                     # (..., 1)
return weight * v                                  # (..., n)
```

Update `build_model` factory: add `'mggd_constrained': MGGDConstrainedScore`.

Update `LOSS_FNS` in `train.py`: add `'mggd_constrained': dsm_mse_loss`.

The warmup mechanic in `train()` checks `hasattr(model, 'u_mlp')` — MGGD-constrained has
no `u_mlp`, so no warmup needed and existing code handles it correctly. ✓

---

## Step 4 — Main experiment (`mggd_score_experiment.py`)

### Constants
```python
P          = 3
RHO_AR1    = 0.8
M_PARAM    = 1.0
BETA_LIST  = [0.2, 0.5, 0.8]
N_VALUES   = [20, 50, 100, 200, 500, 1000, 2000]
N_MC       = 10          # --quick: 3
N_TEST     = 5000
SIGMA_LIST = [0.05, 0.1, 0.3, 0.5]   # outer loop; main reported curve uses 0.1
DSM_EPOCHS = 300
LR         = 1e-3
WARMUP_TB  = 20          # two-branch only
GRAD_CLIP  = 1.0         # applied to mggd_constrained to control d^alpha explosion
```

**σ is a first-class loop**, not an afterthought. Train all three DSM variants at each σ
value. MLE and Tyler AMF are σ-independent so computed once and reused across σ values.

The main paper figures show the σ=0.1 curves. A supplementary figure shows MSE vs. N for
all four σ values on a single plot per method, revealing the bias-variance trade-off.

### Score metrics
```python
def score_mse(s_hat, s_true):
    # s_hat, s_true: (N_TEST, p)  — raw, no scale normalization
    return ((s_hat - s_true)**2).mean().item()

def score_cosine(s_hat, s_true):
    # mean over test points of cos(angle between estimated and true score vector)
    num = (s_hat * s_true).sum(axis=-1)                       # (N_TEST,)
    denom = np.linalg.norm(s_hat, axis=-1) * np.linalg.norm(s_true, axis=-1) + 1e-12
    return (num / denom).mean().item()
```

MSE is the primary metric — it penalizes both wrong direction and wrong scale. Cosine
similarity is reported alongside it to separate the two failure modes: a method with
low cosine similarity has wrong direction (fundamentally wrong model); a method with
high cosine but high MSE has wrong scale only (e.g., DSM σ-bias at large N).

### MC loop (synthetic)
```python
# results[beta][sigma][method] = {'mse': [mc0, mc1, ...], 'cos': [mc0, mc1, ...]}
results = {}

for beta in BETA_LIST:
    M = ar1_covariance(P, RHO_AR1)
    M_inv = np.linalg.inv(M)
    X_test = sample_mggd(N_TEST, P, M, M_PARAM, beta, seed=9999)   # fixed test set
    s_true = mggd_true_score(X_test, M_inv, M_PARAM, beta)          # (N_TEST, P)

    for n in N_VALUES:
        # MLE and Tyler: σ-independent — compute once per (beta, n, mc)
        mle_scores   = {}   # mc -> s_mle or None
        tyler_scores = {}   # mc -> s_tyler

        for mc in range(N_MC):
            X_train = sample_mggd(n, P, M, M_PARAM, beta, seed=mc*10)
            X_val   = sample_mggd(max(n//5, 20), P, M, M_PARAM, beta, seed=mc*10+1)

            result = fit_mggd_mle(X_train)
            mle_scores[mc] = score_mggd_mle(X_test, *result) if result else None

            C_tyler = fit_tyler(X_train)
            tyler_scores[mc] = score_tyler_linear(X_test, C_tyler)

        for sigma in SIGMA_LIST:
            for mc in range(N_MC):
                X_train = sample_mggd(n, P, M, M_PARAM, beta, seed=mc*10)
                X_val   = sample_mggd(max(n//5, 20), P, M, M_PARAM, beta, seed=mc*10+1)

                # record σ-independent baselines once (at first sigma)
                if sigma == SIGMA_LIST[0]:
                    if mle_scores[mc] is not None:
                        _record(results, beta, 'mle_only', 'Pascal MLE',
                                mle_scores[mc], s_true)
                    _record(results, beta, 'mle_only', 'Tyler AMF',
                            tyler_scores[mc], s_true)

                # DSM models — retrain at each sigma
                for mtype, mkwargs in DSM_CONFIGS:
                    model = build_model(mtype, P, **mkwargs)
                    train_dl = make_loader(X_train, batch_size=min(n, 64))
                    val_dl   = make_loader(X_val,   batch_size=min(n, 64))
                    train(model, mtype, train_dl, val_dl,
                          n_epochs=DSM_EPOCHS, lr=LR, sigma=sigma,
                          warmup_epochs=WARMUP_TB if mtype=='two_branch' else 0,
                          grad_clip=GRAD_CLIP if mtype=='mggd_constrained' else None,
                          verbose=False)
                    with torch.no_grad():
                        s_hat = model(torch.tensor(X_test, dtype=torch.float32)).numpy()
                    _record(results, beta, sigma, mtype, s_hat, s_true)

# _record helper stores score_mse and score_cosine into results dict
```

### VisTex block
1. Load `data/vistex/Bark.0000.ppm` and `data/vistex/Leaves.0008.ppm`
2. Apply `pywt.swt2(channel, 'db4', level=1)` per RGB channel, take LH detail subband
3. Stack subbands → `(H*W, 3)`, zero-mean each column
4. Fixed 80/20 train-test split (same split every run)
5. Fit MLE on FULL dataset → pseudo-ground-truth score (M_gt, m_gt, beta_gt)
6. **Same N sweep and σ sweep as synthetic** — DSM methods retrained at each (N, σ);
   MLE and Tyler refit at each N only; pseudo-ground-truth score fixed throughout
7. Same MSE and cosine metrics; same `_record` / plotting infrastructure
8. If files missing, print download URL and skip with a warning (don't crash)

### Plotting
**Main figures** — one per beta value (synthetic) + one per VisTex image.
Each: log-log axes, MSE on y, N on x, shaded ±1 std bands, σ=0.1 for DSM methods.
MLE and Tyler shown on every panel regardless of σ.

**Supplementary σ-sweep figure** — for each DSM method, four curves (one per σ) on a
single plot; MLE shown as a horizontal reference at each N. Reveals bias-variance
trade-off: large σ = low variance but high bias at large N; small σ = low bias but
noisy at small N.

**Cosine similarity figures** — mirror layout of MSE figures. Shared color/style so
reader can compare MSE and cosine panels side by side.

```
figures/mggd_mse_beta_0.2.png    figures/mggd_cos_beta_0.2.png
figures/mggd_mse_beta_0.5.png    figures/mggd_cos_beta_0.5.png
figures/mggd_mse_beta_0.8.png    figures/mggd_cos_beta_0.8.png
figures/mggd_sigma_sweep.png     # supplementary
figures/vistex_bark.png
figures/vistex_leaves.png
```

Colors (consistent with `dsm_vs_amf_exp.py` style):
```python
COLORS = {
    'Pascal MLE':       '#1f77b4',
    'Tyler AMF':        '#ff7f0e',
    'Linear DSM':       '#2ca02c',
    'TwoBranch DSM':    '#9467bd',
    'MGGD Constrained': '#d62728',
}
```

### CLI flags
```
--quick          N_MC=3, N_VALUES=[20,100,500,2000]
--synthetic_only skip VisTex block
--vistex_only    skip synthetic block
--beta           one or more values, default all three
```

---

## Validation checkpoints (run before full experiment)

1. **Sampler sanity:** `sample_mggd(10000, 3, M, 1.0, 0.5, seed=0)` — check
   `(1/N) X^T X ≈ M` and `E[||x||^2] ≈ p` (for m=1, β=1 this is exact).

2. **MLE recovery:** fit on N=2000, β=0.5 — check `||M_hat - M||_F / ||M||_F < 0.05`
   and `|beta_hat - 0.5| < 0.05`.

3. **MGGD-constrained convergence:** train on N=500, β=0.5, 300 epochs —
   check alpha converges toward −0.5 (i.e., exp(log_alpha_shifted)−1 ≈ −0.5) and
   MSE decreases below Linear DSM.

4. **Gradient explosion guard:** train MGGD-constrained on N=20, β=0.2 with
   grad_clip=1.0 — confirm no NaN loss after epoch 1.

---

## Known issues / risks

| Risk | Mitigation |
|---|---|
| MLE fixed-point doesn't converge at small N | Cap at 100 iters, return best iterate; skip trial if M singular |
| DSM bias at σ=0.1 inflates DSM MSE vs MLE at large N | Report MSE at multiple σ values in appendix; σ=0.1 is the main curve |
| VisTex wavelet subbands not truly MGGD | Label the VisTex block "semi-synthetic", MLE score is proxy only |
| `train()` cosine LR scheduler with N_MC×len(N_VALUES) calls is slow | Each model is small (p=3); 300 epochs on N≤2000 is <1s on CPU |
| pywt not installed | Add to requirements.txt; graceful skip if import fails |
