# Mathematics: Additive Signal Detection under Student-t Noise

## 1. Noise Model

### Multivariate Student-t as a Scale Mixture

The multivariate Student-t distribution $t_\nu(\mathbf{0}, \Sigma)$ is defined via the scale mixture:

$$\mathbf{w} = \sqrt{\tau}\, \mathbf{z}, \qquad \mathbf{z} \sim \mathcal{N}(\mathbf{0}, \Sigma), \quad \tau = \frac{\nu}{\chi^2_\nu}$$

so $\tau \sim \text{Inv-Gamma}(\nu/2,\, \nu/2)$, and marginalising over $\tau$ gives the density:

$$p(\mathbf{w}) = \frac{\Gamma\!\left(\tfrac{\nu+d}{2}\right)}{\Gamma\!\left(\tfrac{\nu}{2}\right)(\pi\nu)^{d/2}|\Sigma|^{1/2}} \left(1 + \frac{\mathbf{w}^\top Q \mathbf{w}}{\nu}\right)^{-(\nu+d)/2}$$

where $Q = \Sigma^{-1}$ is the precision matrix and $d$ is the dimension.

**Moments:** $\mathbb{E}[\mathbf{w}] = \mathbf{0}$, $\text{Cov}(\mathbf{w}) = \tfrac{\nu}{\nu-2}\Sigma$ for $\nu > 2$.  
As $\nu \to \infty$ the distribution converges to $\mathcal{N}(\mathbf{0}, \Sigma)$.

---

## 2. Observation Model

$$\mathbf{y}_i = \mathbf{w}_i + \theta_i\, \mathbf{s}$$

- $\mathbf{s} \in \mathbb{R}^d$: known steering vector
- $\theta_i$: unknown scalar signal amplitude
- $\mathbf{w}_i \sim t_\nu(\mathbf{0}, \Sigma)$: noise, i.i.d.

**Hypotheses:**

$$H_0: \theta = 0 \qquad H_1: \theta > 0$$

**Per-observation scalars** (with $\mathbf{z} = \mathbf{y}$ since $\mu = \mathbf{0}$):

$$A = \mathbf{y}^\top Q \mathbf{y} \quad\text{(Mahalanobis norm)}, \qquad B = \mathbf{y}^\top Q \mathbf{s}, \qquad C = \mathbf{s}^\top Q \mathbf{s}$$

By Cauchy–Schwarz in the $Q$-metric: $B^2 \leq A \cdot C$, so $B^2/C \leq A$ always.

---

## 3. Score Function

The score of $p(\mathbf{w})$ with respect to a location shift $\theta\mathbf{s}$ at $\theta = 0$ is:

$$\nabla_\mathbf{y} \log p(\mathbf{y})\big|_{\theta=0} = -\frac{\nu+d}{\nu + A}\, Q\mathbf{y} = -w(A)\, Q\mathbf{y}$$

where the **adaptive weight** is:

$$w(A) = \frac{\nu + d}{\nu + A}$$

This is the core of robust detection: observations with large Mahalanobis norm $A$ (outliers) are **down-weighted**. As $\nu \to \infty$, $w(A) \to 1$ and the score reduces to $-Q\mathbf{y}$, the Gaussian score.

---

## 4. Oracle t-Rao (Score Test)

The Rao (score) test statistic for $H_0: \theta = 0$ is the score projected onto $\mathbf{s}$, divided by the square root of the Fisher information:

$$T_{\text{Rao},t} = \frac{w(A) \cdot B}{\sqrt{I_{0,t}}}$$

**Fisher information** for the scalar amplitude $\theta$ at $\theta = 0$:

$$I_{0,t} = \mathbf{s}^\top F_\mu\, \mathbf{s} = \frac{\nu+d}{\nu+d+2}\, C$$

where $F_\mu = \tfrac{\nu+d}{\nu+d+2}\, Q$ is the Fisher information matrix for the location parameter of $t_\nu(\mu, \Sigma)$.

Therefore:

$$\boxed{T_{\text{Rao},t} = \frac{w(A) \cdot B}{\sqrt{\dfrac{\nu+d}{\nu+d+2}\, C}}}$$

**Interpretation:** The t-Rao is the **locally most powerful** test near $\theta = 0$. It is the optimal score test but is not globally optimal — for large $\theta$, the large signal inflates $A$, causing $w(A) \to 0$, which paradoxically down-weights a strong signal.

---

## 5. Oracle t-GLRT (Generalised Likelihood Ratio Test)

### Likelihood Ratio

For a known $\theta$, the log-likelihood ratio is:

$$\log \Lambda(\mathbf{y};\, \theta) = \frac{\nu+d}{2}\log\frac{\nu + A}{\nu + A - 2\theta B + \theta^2 C}$$

### MLE of $\theta$ under $H_1$

Minimise the denominator with respect to $\theta$:

$$\frac{d}{d\theta}(\nu + A - 2\theta B + \theta^2 C) = -2B + 2\theta C = 0 \implies \hat{\theta} = \frac{B}{C}$$

For the one-sided test ($\theta \geq 0$), clip: $\hat{\theta} = \max(0,\, B/C)$.

### GLRT Statistic

Substituting $\hat{\theta} = B/C$ into the denominator:

$$\nu + A - 2\frac{B}{C}\cdot B + \left(\frac{B}{C}\right)^2 C = \nu + A - \frac{B^2}{C}$$

Therefore:

$$\boxed{T_{\text{GLRT}} = \frac{\nu+d}{2}\log\frac{\nu + A}{\nu + A - B^2/C} \quad \text{when } B > 0, \text{ else } 0}$$

Since $B^2/C \leq A$ (Cauchy–Schwarz), the denominator satisfies $\nu + A - B^2/C \geq \nu > 0$, so the argument of $\log$ is always $\geq 1$ and $T_{\text{GLRT}} \geq 0$.

**Gaussian limit** ($\nu \to \infty$): using $\log(1+x) \approx x$ for small $x$:

$$T_{\text{GLRT}} \to \frac{B^2}{2C} = \frac{T_{\text{AMF}}^2}{2}$$

so the GLRT reduces to the squared matched filter under Gaussian noise, as expected.

---

## 6. Why the Gaussian AMF is Suboptimal for Additive t-Noise

The Gaussian AMF is $T_{\text{AMF}} = B / \sqrt{C}$, using only $B$.

The likelihood ratio for a known shift $\theta_0$ is:

$$\Lambda(\mathbf{y};\,\theta_0) = \left[\frac{\nu+A}{\nu + A - 2\theta_0 B + \theta_0^2 C}\right]^{(\nu+d)/2}$$

This depends on **both $A$ and $B$**, not on $B$ alone. The AMF discards the information in $A$.

**Why a location family does not imply AMF-optimality.** The t-distribution is a location family in $\theta$, but UMP (uniformly most powerful) tests based on a scalar sufficient statistic exist only for **exponential families** (via the monotone likelihood ratio property). The t-distribution is not an exponential family — its tails are polynomial, not exponential. Consequently, the likelihood ratio is **not monotone in $B$ alone**:

$$\frac{\partial \log \Lambda}{\partial B} = \frac{(\nu+d)\,\theta_0}{\nu + A - 2\theta_0 B + \theta_0^2 C} > 0 \quad \text{(for fixed } A\text{)}$$

but $A$ itself varies with $\mathbf{y}$, and the joint rejection region $\{\Lambda(\mathbf{y}) > \eta\}$ is **not** of the form $\{B > c\}$.

**Contrast with the replacement/scaling model.** For compound-Gaussian clutter where the entire observation scales with texture $\tau$:

$$H_0: \mathbf{y} = \sqrt{\tau}\,\mathbf{z}, \qquad H_1: \mathbf{y} = \sqrt{\tau}(a\mathbf{s} + \mathbf{z})$$

the signal scales with $\tau$, making the GLRT scale-invariant. It reduces to the AMF (which is invariant to $\tau$), and the AMF is UMPI (uniformly most powerful invariant) regardless of the texture distribution. This is the setting behind the classical result that the AMF is optimal for compound-Gaussian clutter — it does **not** apply to the additive model.

---

## 7. Optimality Summary

| Model | Optimal test | Depends on |
|---|---|---|
| Additive Gaussian | AMF: $B/\sqrt{C}$ | $B$ only |
| Additive $t_\nu$, known $\theta$ | NP-LRT: $\Lambda(\mathbf{y};\theta)$ | $A$ and $B$ |
| Additive $t_\nu$, unknown $\theta$ | t-GLRT (oracle) | $A$ and $B$ |
| Replacement/scaling, any texture | AMF (UMPI) | $B$ and $A$ (via ratio) |

---

## 8. Connection to DSM and the Two-Branch Model

The true score of $t_\nu(\mathbf{0}, \Sigma)$ is:

$$\nabla_\mathbf{y} \log p(\mathbf{y}) = -w(A)\, Q\mathbf{y}, \qquad w(A) = \frac{\nu+d}{\nu+A}$$

This is exactly the **two-branch factorisation**:

- **Linear branch:** $\mathbf{v} = W\mathbf{y} \approx -Q\mathbf{y}$, learned by DSM
- **Scalar branch:** $u(d) \approx w(d) = (\nu+d)/(\nu+d_0)$, learned by the u-MLP

where $d_0 = \|\!A\mathbf{y}\|^2$ is the Mahalanobis proxy. If the model is well-trained, the two-branch score approximates $-w(A)\,Q\mathbf{y}$ without ever being told $\nu$ or $Q$.

The resulting neural test statistic $T_s(\mathbf{y}) = -\mathbf{s}^\top(\psi(\mathbf{y}) - \bar{z})/\sqrt{\mathbf{s}^\top \hat{C}_\psi \mathbf{s}}$ then approximates $w(A)\cdot B$ up to normalisation — behaviorally similar to the t-Rao, but learned entirely from noise samples. The empirical advantage over the t-Rao seen in AUC suggests the learned statistic better handles the composite $H_1: \theta \sim U(0, \theta_{\max})$ by adapting its weighting across the full range of $\theta$, rather than being locally optimal only at $\theta = 0$.
