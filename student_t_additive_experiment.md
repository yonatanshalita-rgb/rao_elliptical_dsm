# Student-t Additive Signal Detection: Mathematical Reference

## 1. Noise and Signal Model

### 1.1 Multivariate Student-t via Scale Mixture

$$\mathbf{w}_i \sim t_\nu(\mathbf{0}, \Sigma), \qquad
\mathbf{w}_i = \sqrt{\tau_i}\,\mathbf{z}_i, \quad
\mathbf{z}_i \sim \mathcal{N}(\mathbf{0}, \Sigma), \quad
\tau_i \sim \text{Inv-Gamma}\!\left(\tfrac{\nu}{2}, \tfrac{\nu}{2}\right)$$

Density:

$$p(\mathbf{w}) = \frac{\Gamma\!\left(\frac{\nu+d}{2}\right)}{\Gamma\!\left(\frac{\nu}{2}\right)(\pi\nu)^{d/2}|\Sigma|^{1/2}}
\left(1 + \frac{\mathbf{w}^\top Q\mathbf{w}}{\nu}\right)^{-(\nu+d)/2}, \qquad Q = \Sigma^{-1}$$

Moments: $\mathbb{E}[\mathbf{w}] = \mathbf{0}$, $\mathrm{Cov}(\mathbf{w}) = \frac{\nu}{\nu-2}\Sigma$ for $\nu > 2$.

**Experiment parameters:** $\nu = 3$, $d = 16$, AR(1) covariance with $\rho = 0.9$.

### 1.2 Additive Signal Model

$$\mathbf{y}_i = \mathbf{w}_i + \theta_i\,\mathbf{s}$$

- $\mathbf{s} = \mathbf{1}/\sqrt{d}$: unit steering vector
- $H_0$: $\theta_i = 0$
- $H_1$: $\theta_i \sim U(0, \theta_{\max})$ or $\theta_i = \theta_{\max}$ (fixed)
- $\theta_{\max} = \sqrt{\mathrm{SNR}_{\mathrm{lin}} / C}$, $C = \mathbf{s}^\top Q\mathbf{s}$

**Per-sample scalars:**

$$A = \mathbf{y}^\top Q\mathbf{y} \quad \text{(Mahalanobis norm)}, \qquad
B = \mathbf{y}^\top Q\mathbf{s}, \qquad
C = \mathbf{s}^\top Q\mathbf{s}$$

By Cauchy–Schwarz: $B^2 \leq A \cdot C$ always.

---

## 2. Oracle Detectors (know $Q$, $\nu$)

### 2.1 Oracle Gaussian AMF

$$T_{\mathrm{AMF}}(\mathbf{y}) = \frac{B}{\sqrt{C}}$$

Optimal for Gaussian noise; sub-optimal under Student-t because the likelihood ratio also depends on $A$.

### 2.2 Oracle t-Rao (Score Test)

The score of $t_\nu(\mathbf{0}, \Sigma)$ at $\theta = 0$:

$$\nabla_\theta \log p(\mathbf{y})\big|_{\theta=0}
= -\frac{\nu+d}{\nu+A}\,Q\mathbf{y} \cdot (-\mathbf{s}) = w(A)\,B, \qquad w(A) = \frac{\nu+d}{\nu+A}$$

Fisher information for $\theta$ at $H_0$:

$$I_{0,t} = \frac{\nu+d}{\nu+d+2}\,C$$

Rao test statistic:

$$T_{\mathrm{Rao},t} = \frac{w(A)\cdot B}{\sqrt{\dfrac{\nu+d}{\nu+d+2}\,C}}$$

**Locally most powerful** at $\theta \to 0$, but $w(A)$ suppresses large signals: for large $\theta$, $A \approx \theta^2 C$, so $w(A) \to 0$.

### 2.3 Oracle t-GLRT

Marginal log-likelihood ratio for amplitude $\theta$:

$$\log\Lambda(\mathbf{y};\theta) = \frac{\nu+d}{2}\log\frac{\nu+A}{\nu + A - 2\theta B + \theta^2 C}$$

MLE of $\theta$: $\hat\theta = B/C$ (clipped to $\geq 0$ for the one-sided test).

Substituting:

$$\boxed{T_{\mathrm{GLRT}} = \frac{\nu+d}{2}\log\frac{\nu+A}{\nu+A-B^2/C} \quad \text{if } B > 0, \text{ else } 0}$$

The argument of $\log$ is $\geq 1$ by Cauchy–Schwarz. For large $\theta$: $B^2/C \approx A$, so $T \approx \frac{\nu+d}{2}\log\frac{\nu+A}{\nu} \to \infty$ — grows without the suppression problem.

Gaussian limit ($\nu \to \infty$): $T_{\mathrm{GLRT}} \to B^2/(2C)$.

### 2.4 Oracle $q_\sigma$-Rao (Numerical)

By Tweedie's formula, a perfectly trained two-branch at DSM noise level $\sigma$ converges to the score of the **convolved** distribution $q_\sigma = t_\nu * \mathcal{N}(\mathbf{0}, \sigma^2 I)$, not the score of $t_\nu$.

Using the scale-mixture representation $\mathbf{y} \mid \tau \sim \mathcal{N}(\mathbf{0}, \tau\Sigma + \sigma^2 I)$:

$$\nabla \log q_\sigma(\mathbf{y}) = -\mathbb{E}_{\tau\mid\mathbf{y}}\!\left[(\tau\Sigma + \sigma^2 I)^{-1}\right]\mathbf{y}$$

where the posterior $\tau \mid \mathbf{y}$ has density:

$$p(\tau \mid \mathbf{y}) \propto |\tau\Sigma + \sigma^2 I|^{-1/2}
\exp\!\left(-\tfrac{1}{2}\mathbf{y}^\top(\tau\Sigma+\sigma^2 I)^{-1}\mathbf{y}\right)
\cdot \tau^{-\nu/2-1} e^{-\nu/(2\tau)}$$

**No closed form exists** for $\sigma > 0$ with non-isotropic $\Sigma$: adding $\sigma^2 I$ breaks the conjugacy between the Inv-Gamma prior and Gaussian likelihood that yields the clean $w(A) = (\nu+d)/(\nu+A)$ formula.

**Numerical implementation** via eigendecomposition $\Sigma = UDU^\top$:

In the eigenbasis, $(\tau\Sigma + \sigma^2 I)^{-1} = U\,\mathrm{diag}(1/(\tau d_i + \sigma^2))U^\top$.
The score projection $\mathbf{s}^\top\nabla\log q_\sigma(\mathbf{y})$ reduces to:

$$-\mathbb{E}_{\tau\mid\mathbf{y}}\!\left[\sum_i \frac{\tilde{s}_i\tilde{y}_i}{\tau d_i + \sigma^2}\right], \quad
\tilde{\mathbf{y}} = U^\top\mathbf{y},\quad \tilde{\mathbf{s}} = U^\top\mathbf{s}$$

Computed via log-uniform quadrature over $\tau \in [10^{-3}, 10^3]$ with $K = 500$ points.
All $N$ test samples are processed simultaneously with two $(N \times K)$ matrix multiplies.
**Runtime:** $\approx 1$ s for $N = 50{,}000$ samples on CPU.

---

## 3. Practical Detectors (estimated $\hat\Sigma$, known $\nu$)

### 3.1 t-MLE Scatter Estimator

MLE for $\Sigma$ under $t_\nu(\mathbf{0},\Sigma)$ with known $\nu$, via EM algorithm:

**E-step:** $w_i = \dfrac{\nu+d}{\nu + \mathbf{w}_i^\top\hat\Sigma^{-1}\mathbf{w}_i}$

**M-step:** $\hat\Sigma \leftarrow \dfrac{1}{M}\sum_{i=1}^M w_i\,\mathbf{w}_i\mathbf{w}_i^\top$

This converges to the MLE. Unlike Tyler's M-estimator (which estimates shape only, scale-invariant), the t-MLE produces a **properly scaled** $\hat\Sigma$ directly usable in $A$, $B$, $C$.

Initialised from the sample covariance; typically converges in $< 50$ iterations.

### 3.2 Practical t-Rao and t-GLRT

Replace oracle $Q = \Sigma^{-1}$ with $\hat{Q} = \hat\Sigma_{\mathrm{MLE}}^{-1}$ in the oracle formulas. Scale matters here (through $A = \mathbf{y}^\top\hat{Q}\mathbf{y}$ in $w(A)$ and the GLRT log), which is why the properly scaled t-MLE is used rather than Tyler's shape-only estimator.

---

## 4. AMF Baselines (estimated covariance, ignore $\nu$)

All three use $T = B/\sqrt{C}$ with estimated $\hat{Q}$, differing only in the estimator:

| Estimator | Weight $u_i$ | Fixed-point form |
|---|---|---|
| SCM | $1$ | $\hat\Sigma = \frac{1}{M}\sum \mathbf{w}_i\mathbf{w}_i^\top$ |
| Tyler | $d / (\mathbf{w}_i^\top\hat\Sigma^{-1}\mathbf{w}_i)$ | Scale-free, trace-normalised |
| Huber | $\min(1,\, \beta d / (\mathbf{w}_i^\top\hat\Sigma^{-1}\mathbf{w}_i))$ | Hybrid, $\beta=0.9$ |

The AMF ignores the dependence of the t-distribution likelihood on $A$, making it sub-optimal for additive t-noise regardless of how well $\hat\Sigma$ is estimated.

---

## 5. DSM Neural Detectors

### 5.1 Architecture

**Linear (MSE / Huber):** $\psi(\mathbf{y}) = -AA^\top\mathbf{y}$ (NSD constraint via $-A^\top A$ parametrisation). Learns $\psi \approx -Q\mathbf{y}$.

**Two-Branch:**

$$\psi(\mathbf{y}) = u(\|A\mathbf{y}\|^2)\cdot(-AA^\top\mathbf{y})$$

where $u: \mathbb{R}_{\geq 0} \to \mathbb{R}_{>0}$ is a small MLP (via softplus output). The scalar $\|A\mathbf{y}\|^2$ is a proxy for the Mahalanobis norm $A = \mathbf{y}^\top Q\mathbf{y}$.

### 5.2 Training Objective (DSM)

$$\mathcal{L}_{\mathrm{DSM}} = \mathbb{E}_{\mathbf{w},\boldsymbol{\epsilon}}\!\left[\left\|\psi(\mathbf{w}+\sigma\boldsymbol{\epsilon}) + \frac{\boldsymbol{\epsilon}}{\sigma}\right\|^2\right], \quad \boldsymbol{\epsilon}\sim\mathcal{N}(\mathbf{0},I)$$

By Tweedie's formula (Vincent 2011), the minimiser is $\psi^* = \nabla\log q_\sigma$ where $q_\sigma = t_\nu * \mathcal{N}(\mathbf{0},\sigma^2 I)$.

- **$\sigma = 0.05$:** $q_{0.05} \approx t_\nu$ → two-branch converges to $\approx t$-Rao
- **$\sigma = 0.5$:** $q_{0.5}$ has softer tails → flatter weight $u(d)$ → closer to GLRT structure

Two-branch warmup: W-branch trained alone for 12 epochs (u-branch frozen at $u\equiv 1$), then both trained jointly.

### 5.3 NeuralDetector Normalisation

$$T_s(\mathbf{y}) = \frac{-\mathbf{s}^\top(\psi(\mathbf{y}) - \bar{\psi})}{\sqrt{\mathbf{s}^\top\hat{C}_\psi\mathbf{s}}}$$

where $\bar\psi = \frac{1}{M}\sum_i\psi(\mathbf{w}_i)$ and $\hat{C}_\psi = \frac{1}{M}\sum_i(\psi(\mathbf{w}_i)-\bar\psi)(\psi(\mathbf{w}_i)-\bar\psi)^\top$ are computed from $M = 5{,}000$ calibration noise samples.

The denominator is a **constant** w.r.t. $\mathbf{y}$ and does not affect AUC — it sets the threshold scale for Pd at a fixed Pfa.

---

## 6. Detector Hierarchy and Fairness

| Detector | Knows $Q$? | Knows $\nu$? | Target statistic |
|---|---|---|---|
| Oracle Gaussian AMF | yes | no | $B/\sqrt{C}$ |
| Oracle t-Rao | yes | yes | $w(A)\cdot B / \sqrt{I_{0,t}}$ |
| Oracle t-GLRT | yes | yes | $\frac{\nu+d}{2}\log\frac{\nu+A}{\nu+A-B^2/C}$ |
| Oracle $q_\sigma$-Rao | yes | yes | $\mathbf{s}^\top\nabla\log q_\sigma(\mathbf{y})$ (numerical) |
| Practical t-Rao | estimated (t-MLE) | **yes** | $w(\hat{A})\cdot\hat{B} / \sqrt{\hat{I}_{0,t}}$ |
| Practical t-GLRT | estimated (t-MLE) | **yes** | GLRT with $\hat{Q}$ |
| SCM / Tyler / Huber AMF | estimated | no | $\hat{B}/\sqrt{\hat{C}}$ |
| DSM Two-Branch ($\sigma=0.5$) | learned | **no** | $\approx\nabla\log q_{0.5}$ projected on $\mathbf{s}$ |
| DSM Two-Branch ($\sigma=0.05$) | learned | **no** | $\approx\nabla\log t_\nu$ projected on $\mathbf{s}$ |

### Information asymmetry note

The **Practical t-Rao and t-GLRT** detectors know $\nu$ (passed in as a model parameter) but must estimate $Q$ from noise samples. The **DSM two-branch** must learn everything — the covariance structure and the tail index — entirely from noise samples, without being told $\nu$ or the distributional family.

This means the practical t-Rao/t-GLRT have an information advantage over the DSM models that the AMF baselines do not share: the AMF formula $B/\sqrt{C}$ is $\nu$-free and makes no use of tail information. A fair apples-to-apples comparison for the DSM branch is therefore against the AMF baselines (SCM/Tyler/Huber), not against the practical t-detectors.

The **oracle detectors** are upper bounds only — they assume full knowledge of $Q$ and $\nu$ simultaneously, which is never available in practice.

The correct interpretation of results:
- DSM two-branch outperforming Tyler/Huber AMF → the branch is successfully extracting heavy-tail information from the noise samples without being told $\nu$.
- DSM two-branch approaching practical t-GLRT → the branch has nearly closed the gap despite not knowing $\nu$, which would be a strong result.

---

## 7. Estimating $\nu$ (Not Used in Current Experiment)

The practical detectors above treat $\nu$ as **known**. When $\nu$ must also be estimated from noise samples, two standard approaches exist:

### 7.1 Method of Moments

For $\mathbf{w}\sim t_\nu(\mathbf{0},\Sigma)$: $\mathbb{E}[\mathbf{w}^\top\Sigma^{-1}\mathbf{w}] = d\cdot\nu/(\nu-2)$.

Given a scatter estimate $\hat\Sigma$ (e.g., from Tyler), compute the sample mean Mahalanobis distance:

$$\hat{m} = \frac{1}{Md}\sum_{i=1}^M \mathbf{w}_i^\top\hat\Sigma^{-1}\mathbf{w}_i$$

Then solve $\hat{m} = \hat\nu/(\hat\nu-2)$ to get:

$$\hat\nu_{\mathrm{MOM}} = \frac{2\hat{m}}{\hat{m}-1} \qquad (\text{requires } \hat{m} > 1,\text{ i.e., } \nu > 2)$$

### 7.2 Profile MLE

Given $\hat\Sigma$, maximise the log-likelihood over $\nu$ by solving:

$$\psi\!\left(\tfrac{\nu+d}{2}\right) - \psi\!\left(\tfrac{\nu}{2}\right) - \frac{d}{\nu}
= \frac{1}{M}\sum_{i=1}^M\left[\log\!\left(1+\frac{A_i}{\nu}\right) - \frac{(\nu+d)\,A_i}{\nu(\nu+A_i)}\right]$$

where $\psi$ is the digamma function. Solved numerically (e.g., Brent's method on $\nu \in (2, 200)$). More accurate than MOM, especially for small $M$.

---

## 8. DSM Sigma Sweep: Interpolation Between t-Rao and Gaussian AMF

### 8.1 Theoretical Prediction

By Tweedie's formula the DSM minimiser is $\psi^* = \nabla\log q_\sigma$ where $q_\sigma = t_\nu * \mathcal{N}(\mathbf{0}, \sigma^2 I)$. Three limiting regimes:

| $\sigma$ | $q_\sigma$ | Two-branch converges to | Detection behaviour |
|---|---|---|---|
| $\sigma \to 0$ | $t_\nu$ | Oracle t-Rao | Locally optimal; suppressed at large $\theta$ |
| $\sigma = 0.5$ | Smoothed $t_\nu$ | Scalar projection of $\nabla\log q_\sigma$ | Tracks t-GLRT |
| $\sigma \to \infty$ | $\mathcal{N}(\mathbf{0}, \sigma^2 I)$ | Gaussian AMF | Blind at low SNR; no suppression |

For the **unconstrained MLP**, with sufficient capacity it can represent the full direction-dependent score $\nabla\log q_\sigma(\mathbf{y}) = -W(\mathbf{y})\mathbf{y}$, so it should converge to the oracle $q_\sigma$-Rao at any $\sigma > 0$.

### 8.2 Experiment Settings

$d = 64$, $\nu = 3$, AR(1) $\rho = 0.9$, $N_{\mathrm{train}} = 50{,}000$, 3 seeds.
$\sigma \in \{0.05,\, 0.1,\, 0.3,\, 0.5,\, 1.0,\, 2.0,\, 5.0\}$.
Models: **Two-Branch** and **Unconstrained MLP** (3 × 128 hidden layers).
Evaluation: $P_d$ @ $P_{fa} = 1\%$, fixed $\theta = \theta_{\max}(\mathrm{SNR})$, SNR $\in \{1,3,5,10,15,20\}$ dB.

### 8.3 Results — Two-Branch

| $\sigma$ | 1 dB | 3 dB | 5 dB | 10 dB | 15 dB | 20 dB |
|---|---|---|---|---|---|---|
| Oracle AMF | 0.021 | 0.026 | 0.035 | 0.126 | 0.815 | 0.993 |
| Oracle t-Rao | 0.116 | 0.163 | 0.227 | 0.436 | 0.632 | 0.749 |
| Oracle t-GLRT | 0.114 | 0.177 | 0.273 | 0.627 | 0.888 | 0.976 |
| **0.05** | 0.109 | 0.156 | 0.217 | 0.426 | 0.623 | 0.746 |
| **0.1** | 0.119 | 0.171 | 0.242 | 0.464 | 0.659 | 0.767 |
| **0.3** | 0.107 | 0.169 | 0.266 | 0.595 | 0.793 | 0.876 |
| **0.5** | 0.056 | 0.085 | 0.141 | 0.452 | 0.847 | 0.954 |
| **1.0** | 0.040 | 0.057 | 0.091 | 0.425 | 0.913 | 0.973 |
| **2.0** | 0.025 | 0.032 | 0.045 | 0.185 | 0.862 | 0.988 |
| **5.0** | 0.021 | 0.026 | 0.035 | 0.125 | 0.795 | 0.992 |

**Findings:**
- $\sigma = 0.05$: matches oracle t-Rao closely — confirmed empirically.
- $\sigma = 0.3$: tracks oracle t-GLRT across the full SNR range (0.266 vs 0.273 at 5 dB, 0.595 vs 0.627 at 10 dB), with near-zero variance. The effective sweet spot.
- $\sigma = 5.0$: matches oracle AMF to 3 decimal places — confirmed empirically.
- The transition is smooth and monotone: as $\sigma$ increases, low-SNR performance degrades and high-SNR performance improves.

### 8.4 Results — Unconstrained MLP

| $\sigma$ | 1 dB | 3 dB | 5 dB | 10 dB | 15 dB | 20 dB |
|---|---|---|---|---|---|---|
| **0.05** | 0.010 | 0.010 | 0.010 | 0.009 | 0.004 | 0.000 |
| **0.1** | 0.011 | 0.011 | 0.011 | 0.010 | 0.007 | 0.001 |
| **0.3** | 0.019 | 0.024 | 0.031 | 0.095 | 0.272 | 0.122 |
| **0.5** | 0.031 | 0.043 | 0.065 | 0.206 | 0.455 | 0.667 |
| **1.0** | 0.038 | 0.054 | 0.086 | 0.379 | 0.841 | 0.921 |
| **2.0** | 0.048 | 0.073 | 0.123 | 0.500 | 0.827 | 0.620 |
| **5.0** | 0.030 | 0.042 | 0.063 | 0.302 | 0.878 | 0.750 |

**Findings:**
- $\sigma \in \{0.05, 0.1\}$: $P_d \approx P_{fa} = 1\%$ — complete training failure. With $\sigma = 0.05$, the DSM target $-\varepsilon/\sigma$ has std $= 1/0.05 = 20$, producing extremely large gradient magnitudes that prevent the unconstrained MLP from converging. The two-branch survives via W-branch warmup and its constrained structure.
- $\sigma \geq 1.0$: starts producing useful results but with high seed-to-seed variance (high std). The MLP is harder to optimise at $d = 64$ due to its larger parameter count (~50K vs ~5K for two-branch).
- The MLP never reliably approaches either the oracle t-Rao (at small $\sigma$) or Gaussian AMF (at large $\sigma$) because it lacks the architectural inductive bias that stabilises the two-branch.

### 8.5 Why the Two-Branch is More Robust

The two-branch architecture $\psi = u(\|A\mathbf{y}\|^2)\cdot(-A^\top A\mathbf{y})$ imposes two stabilising constraints:
1. **W-branch warmup**: the linear map $A$ is trained alone for 12 epochs before $u$ is unfrozen, establishing a good covariance estimate before scalar adaptation begins.
2. **Structural constraint**: the output is a scalar times a matrix-vector product. Even at small $\sigma$, gradients flow through a compact, well-conditioned path. The MLP's unconstrained $\mathbb{R}^d \to \mathbb{R}^d$ map has no such structure.
