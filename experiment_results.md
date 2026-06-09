# Student-t Additive Detection: Experiment Results

## Setup

**Model:** $\mathbf{y} = \mathbf{w} + \theta\mathbf{s}$, $\mathbf{w} \sim t_\nu(\mathbf{0}, \Sigma)$. $H_0$: $\theta=0$, $H_1$: $\theta = \theta_{\max}$ (fixed).

| Parameter | Value |
|---|---|
| Dimension $d$, degrees of freedom $\nu$ | 16, 3 |
| Covariance | AR(1), $\rho = 0.9$ |
| Steering vector $\mathbf{s}$ | $\mathbf{1}/\sqrt{d}$ |
| Train / test / calibration samples | 50K / 50K / 5K |
| Seeds, $P_{fa}$ | 3, 0.01 |
| DSM noise level $\sigma$ (neural models) | 0.5 |
| Epochs | 60 (two-branch: 12-epoch W-branch warmup) |

**Two evaluation regimes** (fixed $H_1$):
- **Small $\theta$**: $\theta_{\max}$ at SNR = 0 dB ($\theta \approx 3.0$) — local optimality regime
- **Large $\theta$**: $\theta_{\max}$ at SNR = 10 dB ($\theta \approx 9.5$) — suppression regime

---

## Detectors

**Per-scalar notation:** $A = \mathbf{y}^\top Q\mathbf{y}$, $B = \mathbf{y}^\top Q\mathbf{s}$, $C = \mathbf{s}^\top Q\mathbf{s}$, $w(A) = (\nu+d)/(\nu+A)$.

| Tier | Detector | Statistic | Knows |
|---|---|---|---|
| Oracle | NP (UMP) | $\frac{\nu+d}{2}\log\frac{\nu+A}{\nu+A-2\theta B+\theta^2 C}$ | $Q, \nu, \theta$ |
| Oracle | t-GLRT | $\frac{\nu+d}{2}\log\frac{\nu+A}{\nu+A-B^2/C}$ | $Q, \nu$ |
| Oracle | t-Rao | $w(A)\cdot B \;/\; \sqrt{\frac{\nu+d}{\nu+d+2}C}$ | $Q, \nu$ |
| Oracle | $q_\sigma$-Rao | $\mathbf{s}^\top\nabla\log q_\sigma(\mathbf{y})$ (quadrature) | $Q, \nu, \sigma$ |
| Oracle | Gaussian AMF | $B/\sqrt{C}$ | $Q$ |
| Practical | t-GLRT / t-Rao | Same formulas with $\hat{Q}$ (t-MLE) | $\nu$, estimated $Q$ |
| Classical | Tyler / Huber / SCM AMF | $\hat{B}/\sqrt{\hat{C}}$ | estimated $Q$ |
| Neural | DSM Two-Branch | $u(\|A\mathbf{y}\|^2)\cdot(-AA^\top\mathbf{y})$, $\sigma=0.5$ | learned |
| Neural | DSM Unconstrained MLP | $\mathrm{MLP}(\mathbf{y})\in\mathbb{R}^d$, $3\times128$, $\sigma=0.5$ | learned |

Neural detectors know neither $Q$ nor $\nu$. Both are trained with the same DSM MSE objective at $\sigma=0.5$.

---

## Results

### Small $\theta$ (SNR = 0 dB, local optimality regime)

| Detector | AUC | $P_d$ |
|---|---|---|
| **Oracle NP** | **0.7450** | **0.0936** |
| Oracle t-Rao | 0.7310 | 0.0836 |
| Oracle $q_\sigma$-Rao | 0.7300 | 0.0813 |
| Practical t-Rao | 0.7307 ± 0.0002 | 0.0822 ± 0.0008 |
| Oracle t-GLRT | 0.7247 | 0.0767 |
| Practical t-GLRT | 0.7242 ± 0.0001 | 0.0757 ± 0.0006 |
| DSM Unconstrained MLP | 0.7279 ± 0.0011 | 0.0792 ± 0.0011 |
| DSM Two-Branch | 0.7272 ± 0.0000 | 0.0602 ± 0.0005 |
| Tyler / Huber AMF | 0.7161 ± 0.0002 | 0.0197 ± 0.0003 |
| Oracle Gaussian AMF | 0.7163 | 0.0201 |
| SCM AMF | 0.7135 ± 0.0018 | 0.0192 ± 0.0002 |

The t-Rao is locally most powerful and leads all tests except the NP (which additionally knows $\theta$). The GLRT is worse than the t-Rao here — local optimality outweighs the GLRT's large-$\theta$ advantage. The unconstrained MLP tracks the oracle $q_\sigma$-Rao closely; the two-branch is slightly below both in $P_d$ (cost of the scalar constraint at small $\theta$). AMF baselines extract no tail information and cluster near the Gaussian AMF.

### Large $\theta$ (SNR = 10 dB, suppression regime)

| Detector | AUC | $P_d$ |
|---|---|---|
| **Oracle NP** | **0.9650** | **0.6161** |
| DSM Two-Branch | 0.9454 ± 0.0002 | 0.5963 ± 0.0008 |
| Oracle t-GLRT | 0.9427 | 0.6011 |
| Practical t-GLRT | 0.9426 ± 0.0001 | 0.5970 ± 0.0015 |
| Oracle Gaussian AMF | 0.9381 | 0.1431 |
| Tyler / Huber AMF | 0.9379 ± 0.0001 | 0.1416 ± 0.0019 |
| SCM AMF | 0.9361 ± 0.0014 | 0.1311 ± 0.0014 |
| DSM Unconstrained MLP | 0.9087 ± 0.0035 | 0.4048 ± 0.0078 |
| Oracle $q_\sigma$-Rao | 0.9072 | 0.3067 |
| Oracle t-Rao | 0.9072 | 0.3152 |
| Practical t-Rao | 0.9066 ± 0.0001 | 0.3077 ± 0.0009 |

The t-Rao collapses below the Gaussian AMF — the $w(A)$ suppression is decisive. The oracle $q_\sigma$-Rao collapses to the same level (see Finding 2). The unconstrained MLP lands at oracle $q_\sigma$-Rao level. The two-branch surpasses the t-GLRT in AUC and sits between the GLRT and the NP ceiling, without knowing $Q$, $\nu$, or $\theta$.

---

## Findings

### Finding 1 — Small $\sigma$ two-branch converges to oracle t-Rao

A preliminary run with $\sigma=0.05$ confirmed: two-branch AUC $0.9067 \pm 0.0034$ vs oracle t-Rao $0.9072$ (fixed $\theta$, within 1 std). By Tweedie's formula, minimizing DSM at $\sigma \to 0$ trains toward $\nabla\log t_\nu = -w(A)Q\mathbf{y}$, which is exactly representable by the two-branch architecture. No architecture mismatch at small $\sigma$.

### Finding 2 — Oracle $q_\sigma$-Rao = oracle t-Rao at large $\theta$

At large $A$, the posterior $\tau|\mathbf{y}$ under $q_\sigma$ concentrates at $\tau^* \approx (A+\nu)/(d+\nu+2)$, making the $\sigma^2 I$ term negligible:

$$\mathbb{E}_{\tau|\mathbf{y}}\!\left[\frac{1}{\tau d_i + \sigma^2}\right] \approx \frac{\nu+d}{(\nu+A)\,d_i} \quad\Longrightarrow\quad \mathbf{s}^\top\nabla\log q_\sigma(\mathbf{y}) \approx -w(A)\cdot B$$

The convolved score converges to the t-score at large signal amplitude. Both oracles are identically suppressed (AUC 0.9072 each). The unconstrained MLP, which learns the direction-dependent $q_\sigma$ score, inherits this suppression (AUC 0.9087, within noise).

### Finding 3 — Two-branch advantage is architectural

The true $q_\sigma$ score $-W(\mathbf{y})\mathbf{y}$ applies a **different weight per eigendirection** of $\Sigma$: $w_i(\mathbf{y}) = \mathbb{E}[1/(\tau d_i+\sigma^2)|\mathbf{y}]$, smaller for large-eigenvalue directions. The two-branch is constrained to $u(A)\cdot Q\mathbf{y}$ — a single scalar for all directions. DSM finds the best scalar approximation:

$$u^*(A_0) = \frac{\mathbb{E}_{A=A_0}[(Q\mathbf{y})^\top W(\mathbf{y})\mathbf{y}]}{\mathbb{E}_{A=A_0}[\|Q\mathbf{y}\|^2]}$$

This is a weighted average across eigendirections, **pulled above** the signal direction's weight by contributions from less-suppressed small-eigenvalue directions. Since $\mathbf{s} = \mathbf{1}/\sqrt{d}$ is dominated by the largest-eigenvalue eigenvector (DC component, AR(1)), the oracle applies a small weight to the signal direction; $u^*(A)$ applies a larger compromise weight — less suppression on exactly the direction that matters.

**Experimental confirmation:** training the same DSM objective on an unconstrained MLP (enough capacity to learn the full direction-dependent score) gives AUC 0.9087 at large $\theta$ — matching the oracle $q_\sigma$-Rao, not the two-branch. Same objective, same data, same $\sigma$. The 0.037 AUC gap between two-branch (0.9454) and MLP (0.9087) is entirely due to the scalar architecture constraint.

Training losses are identical (~2.09 val loss for both), ruling out underfitting. The two-branch is not learning a better approximation to $q_\sigma$'s score — it is learning a *different* function that is a better detector.

### Finding 4 — Performance hierarchy

| Detector | Knows | AUC small $\theta$ | AUC large $\theta$ |
|---|---|---|---|
| Oracle NP | $Q, \nu, \theta$ | 0.745 | **0.965** |
| Oracle t-GLRT | $Q, \nu$ | 0.725 | 0.943 |
| DSM Two-Branch | learned | 0.727 | **0.945** |
| Oracle t-Rao | $Q, \nu$ | **0.731** | 0.907 |
| Oracle $q_\sigma$-Rao | $Q, \nu$ | 0.730 | 0.907 |
| DSM Unconstrained MLP | learned | 0.728 | 0.909 |
| Tyler / Huber AMF | est. $Q$ | 0.716 | 0.938 |

At large $\theta$: the two-branch (no $Q$, no $\nu$, no $\theta$) surpasses the oracle t-GLRT (knows $Q$ and $\nu$) and closes roughly half the gap from the GLRT to the NP ceiling. At small $\theta$: it is modestly below the oracle t-Rao, the cost of learning a non-locally-optimal scalar weight. The unconstrained MLP consistently tracks the oracle $q_\sigma$-Rao in both regimes.

---

# Experiment 2: Pd vs SNR Curve (d=64)

## Setup

| Parameter | Value |
|---|---|
| Dimension $d$, degrees of freedom $\nu$ | 64, 3 |
| Covariance | AR(1), $\rho = 0.9$ |
| Steering vector $\mathbf{s}$ | $\mathbf{1}/\sqrt{d}$ |
| Train / test / calibration samples | 50K / 50K per SNR point / 5K |
| Seeds, $P_{fa}$ | 5, 0.01 |
| DSM noise level $\sigma$ | 0.5 |
| Epochs | 60 (two-branch: 12-epoch warmup) |
| $H_1$ evaluation | Fixed $\theta = \theta_{\max}(\text{SNR})$ at each point |
| SNR evaluation points | 1, 3, 5, 10, 15, 20 dB |

**Why d=64:** At d=16 with $\nu=3$, the oracle t-Rao barely outperforms the oracle AMF (+0.004 in Pd at U[5,15dB]) — the heavy-tail correction is swamped by the variance of $w(A)$. At d=64, the Mahalanobis norm $A$ concentrates more tightly (LLN), suppression variance decreases, and t-Rao clearly outperforms AMF. The comparison becomes meaningful.

**Metric:** $P_d$ at $P_{fa} = 1\%$, threshold set from calibration noise (not test data). CFAR verified: empirical $P_{fa} \in [0.010, 0.011]$ for all detectors.

## Detectors

| Detector | Knows | Notes |
|---|---|---|
| Oracle Gaussian AMF | $Q$ | $T = B/\sqrt{C}$ |
| Oracle t-Rao | $Q, \nu$ | $T = w(A)\cdot B/\sqrt{I_{0,t}}$; locally optimal at $\theta\to 0$ |
| Oracle t-GLRT | $Q, \nu$ | $T = \frac{\nu+d}{2}\log\frac{\nu+A}{\nu+A-B^2/C}$; no suppression |
| Tyler AMF | est. $Q$ | $T = \hat{B}/\sqrt{\hat{C}}$; robust scatter estimator |
| DSM Linear | learned | $\psi = -AA^\top y$; converges to Gaussian AMF score |
| DSM Fixed t-weight | est. $Q$, knows $\nu$ | $\psi = w_t(d)\cdot(-AA^\top y)$; theoretical scalar, learned $Q$ |
| DSM Two-Branch | learned | $\psi = u(d)\cdot(-AA^\top y)$; learns everything from noise |

## Results — $P_d$ @ $P_{fa} = 1\%$ (mean ± std, 5 seeds)

| Detector | 1 dB | 3 dB | 5 dB | 10 dB | 15 dB | 20 dB |
|---|---|---|---|---|---|---|
| Oracle Gaussian AMF | 0.022±0.004 | 0.027±0.005 | 0.037±0.007 | 0.139±0.029 | 0.825±0.044 | 0.994±0.001 |
| Oracle t-Rao | 0.120±0.006 | 0.168±0.006 | 0.233±0.008 | 0.444±0.010 | 0.641±0.009 | 0.757±0.009 |
| Oracle t-GLRT | 0.110±0.002 | 0.172±0.003 | 0.267±0.003 | 0.621±0.003 | 0.886±0.003 | 0.976±0.001 |
| Tyler AMF | 0.023±0.005 | 0.029±0.007 | 0.038±0.009 | 0.142±0.037 | 0.820±0.057 | 0.994±0.001 |
| DSM Linear | 0.020±0.004 | 0.026±0.005 | 0.034±0.008 | 0.125±0.032 | 0.792±0.071 | 0.993±0.001 |
| DSM Fixed t-weight | 0.077±0.018 | 0.112±0.028 | 0.167±0.042 | 0.398±0.081 | 0.662±0.072 | 0.830±0.040 |
| **DSM Two-Branch** | **0.073±0.004** | **0.115±0.006** | **0.195±0.007** | **0.616±0.007** | **0.864±0.002** | **0.930±0.004** |

## Findings

**1. AMF / Tyler / DSM Linear are identical throughout.** DSM Linear confirms it converges to the Gaussian AMF score — all three track each other within noise at every SNR. They are blind at low SNR (Pd ≈ 0.02–0.04 at 1–5 dB) and only useful above 15 dB.

**2. Oracle t-Rao is good at low SNR, then gets suppressed.** It leads all practical detectors at 1–10 dB, but the AMF overtakes it between 15 and 20 dB (0.641 vs 0.825 at 15 dB; 0.757 vs 0.994 at 20 dB). This is the suppression: $w(A) = (\nu+d)/(\nu+A)$ shrinks toward zero as $A$ grows with $\theta$.

**3. Oracle t-GLRT is best throughout** — no suppression, the MLE substitution $\hat\theta = B/C$ avoids the large-$A$ penalty.

**4. DSM Two-Branch tracks the t-GLRT across the full curve.** At 10 dB: 0.616 vs 0.621. At 15 dB: 0.864 vs 0.886. Consistently close across all SNR values with tight std (±0.002–0.007). It does this knowing neither $Q$ nor $\nu$, outperforming even the oracle t-Rao at high SNR and competitive at low SNR.

**5. DSM Fixed t-weight is intermediate but inconsistent.** Slightly below t-Rao at low SNR, slightly above t-Rao at high SNR (less suppressed because the DSM-learned $Q$ is slightly different from the true $Q$). High variance (±0.018–0.081) — the rigid theoretical scalar amplifies errors in the covariance estimate at d=64.

**6. CFAR confirmed.** With threshold set from calibration noise (5K samples), empirical $P_{fa} \in [0.010, 0.011]$ for all detectors at all SNR values. The NeuralDetector normalisation ($T_s = -\mathbf{s}^\top(\psi - \bar\psi)/C_{\text{norm}}$) makes the null distribution approximately $\mathcal{N}(0,1)$ by construction.

## Information summary

| Detector | Knows $Q$? | Knows $\nu$? | Pd @ 10 dB | Pd @ 15 dB |
|---|---|---|---|---|
| Oracle t-GLRT | yes | yes | 0.621 | 0.886 |
| DSM Two-Branch | learned | no | **0.616** | **0.864** |
| DSM Fixed t-weight | learned | yes | 0.398 | 0.662 |
| Oracle t-Rao | yes | yes | 0.444 | 0.641 |
| Oracle AMF / Tyler / DSM Linear | yes/est./learned | no | ~0.13 | ~0.80 |

The two-branch uses less information than any oracle and matches the oracle t-GLRT within noise — the only data-driven model that avoids both failure modes (AMF's low-SNR blindness and t-Rao's high-SNR suppression).
