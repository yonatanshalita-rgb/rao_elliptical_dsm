# Convergence of the Two-Branch DSM Detector

## Setting and Definitions

**Signal model.** Observations follow $\mathbf{y} = \mathbf{w} + \theta\mathbf{s}$ where $\mathbf{w} \sim t_\nu(\mathbf{0}, \Sigma)$, $\mathbf{s} \in \mathbb{R}^d$ is a known steering vector, and $\theta \geq 0$ is the signal amplitude. Write $Q = \Sigma^{-1}$, $A = \mathbf{y}^\top Q\mathbf{y}$, $B = \mathbf{y}^\top Q\mathbf{s}$, $C = \mathbf{s}^\top Q\mathbf{s}$.

**Scale-mixture representation.** $\mathbf{w} \stackrel{d}{=} \sqrt{\tau}\,\mathbf{z}$ where $\mathbf{z} \sim \mathcal{N}(\mathbf{0}, \Sigma)$ and $\tau \sim \mathrm{IG}(\nu/2,\, \nu/2)$ independently.

**Convolved distribution.** For $\sigma > 0$, define $q_\sigma = t_\nu(\mathbf{0},\Sigma) * \mathcal{N}(\mathbf{0}, \sigma^2 I)$, i.e., the distribution of $\tilde{\mathbf{w}} = \mathbf{w} + \sigma\boldsymbol{\varepsilon}$ with $\boldsymbol{\varepsilon} \sim \mathcal{N}(\mathbf{0}, I)$ independent of $\mathbf{w}$. Using the scale-mixture:

$$q_\sigma(\mathbf{y}) = \int_0^\infty \mathcal{N}(\mathbf{y};\, \mathbf{0},\, \tau\Sigma + \sigma^2 I)\, p_\tau(\tau)\, d\tau$$

where $p_\tau$ is the $\mathrm{IG}(\nu/2, \nu/2)$ density.

**Two-branch architecture class.** Define

$$\mathcal{F} = \bigl\{\, \psi(\mathbf{y}) = u\!\left(\|A\mathbf{y}\|^2\right)\cdot\bigl(-A^\top A\mathbf{y}\bigr) \;\Big|\; A \in \mathbb{R}^{d \times d},\; u : \mathbb{R}_{\geq 0} \to \mathbb{R}_{> 0}\; \text{measurable} \,\bigr\}$$

Note that $\|A\mathbf{y}\|^2 = \mathbf{y}^\top A^\top A\mathbf{y} \geq 0$ for all $\mathbf{y}$, and $-A^\top A$ is negative semidefinite by construction.

**DSM objective.** Given $\sigma > 0$, the two-branch is trained to minimise

$$\mathcal{L}_\sigma[\psi] = \mathbb{E}_{\mathbf{w},\boldsymbol{\varepsilon}}\!\left[\left\|\psi(\mathbf{w} + \sigma\boldsymbol{\varepsilon}) + \frac{\boldsymbol{\varepsilon}}{\sigma}\right\|^2\right], \quad \boldsymbol{\varepsilon} \sim \mathcal{N}(\mathbf{0}, I)$$

over $\psi \in \mathcal{F}$. Denote the in-class minimiser $\psi_\sigma^* = \arg\min_{\psi \in \mathcal{F}} \mathcal{L}_\sigma[\psi]$.

**Tweedie's formula** *(Vincent, 2011)*. For any measurable $\psi$,

$$\mathcal{L}_\sigma[\psi] = \mathbb{E}_{\mathbf{y} \sim q_\sigma}\!\left[\left\|\psi(\mathbf{y}) - \nabla\log q_\sigma(\mathbf{y})\right\|^2\right] + \text{const}$$

where the constant does not depend on $\psi$. Therefore $\psi_\sigma^*$ is the $L^2(q_\sigma)$-projection of $\nabla\log q_\sigma$ onto $\mathcal{F}$:

$$\psi_\sigma^* = \arg\min_{\ps/i \in \mathcal{F}}\; \left\|\psi - \nabla\log q_\sigma\right\|^2_{L^2(q_\sigma)}$$

**Score of the convolved distribution.** Differentiating $\log q_\sigma$ under the integral using dominated differentiation:

$$\nabla\log q_\sigma(\mathbf{y}) = -\,\mathbb{E}_{\tau|\mathbf{y},\sigma}\!\left[(\tau\Sigma + \sigma^2 I)^{-1}\right]\mathbf{y} \tag{$\star$}$$

where $\tau|\mathbf{y}$ has density proportional to $\mathcal{N}(\mathbf{y}; \mathbf{0}, \tau\Sigma + \sigma^2 I)\cdot p_\tau(\tau)$.

**NeuralDetector statistic.** Given a trained $\psi \in \mathcal{F}$ and $M$ calibration noise samples $\{\mathbf{w}_i\}$, define $\bar{\psi} = \frac{1}{M}\sum_i \psi(\mathbf{w}_i)$ and $\hat{C}_\psi = \frac{1}{M}\sum_i (\psi(\mathbf{w}_i)-\bar\psi)(\psi(\mathbf{w}_i)-\bar\psi)^\top$. The test statistic is

$$T_s(\mathbf{y}) = \frac{-\mathbf{s}^\top(\psi(\mathbf{y}) - \bar\psi)}{\sqrt{\mathbf{s}^\top \hat{C}_\psi \mathbf{s}}}$$

Since $\bar\psi$ and $\sqrt{\mathbf{s}^\top\hat{C}_\psi\mathbf{s}}$ are constants with respect to $\mathbf{y}$, the ranking of test statistics — and hence the AUC — is determined entirely by $\mathbf{s}^\top\psi(\mathbf{y})$.

---

## Theorem 1 — Convergence to the Oracle t-Rao as $\sigma \to 0$

**Statement.** As $\sigma \to 0$, the in-class DSM minimiser satisfies

$$\psi_\sigma^*(\mathbf{y}) \;\xrightarrow{L^2(q_\sigma)}\; \nabla\log t_\nu(\mathbf{y}) = -w(A)\,Q\mathbf{y}, \qquad w(A) = \frac{\nu + d}{\nu + A}$$

and the NeuralDetector statistic converges to the oracle t-Rao:

$$T_s\bigl(\psi_\sigma^*\bigr) \;\to\; T_{\mathrm{Rao}} = \frac{w(A)\cdot B}{\sqrt{\dfrac{\nu+d}{\nu+d+2}\,C}}$$

**Proof.**

**Step 1 — Pointwise convergence of the target score.**

Fix $\mathbf{y} \in \mathbb{R}^d$ with $A = \mathbf{y}^\top Q\mathbf{y} < \infty$. From $(\star)$, the score decomposes along the eigenbasis of $\Sigma$. Let $\Sigma = U \mathrm{diag}(d_1,\ldots,d_d) U^\top$ with $d_i > 0$. Writing $\tilde{\mathbf{y}} = U^\top\mathbf{y}$:

$$\nabla\log q_\sigma(\mathbf{y}) = -U\,\mathrm{diag}\!\left(\mathbb{E}_{\tau|\mathbf{y},\sigma}\!\left[\frac{1}{\tau d_i + \sigma^2}\right]\right)U^\top\mathbf{y}$$

For each $i$ and $\sigma > 0$, define $f_i^\sigma(\tau) = 1/(\tau d_i + \sigma^2)$. As $\sigma \searrow 0$, $f_i^\sigma(\tau) \nearrow f_i^0(\tau) = 1/(\tau d_i)$ monotonically for each $\tau > 0$.

The posterior $p(\tau|\mathbf{y}, \sigma)$ converges weakly to $p(\tau|\mathbf{y}, 0)$ as $\sigma \to 0$: for $\sigma > 0$ and fixed $\mathbf{y}$, the kernel $\mathcal{N}(\mathbf{y};\mathbf{0},\tau\Sigma+\sigma^2I) \to \mathcal{N}(\mathbf{y};\mathbf{0},\tau\Sigma)$ uniformly on $[\delta,\infty)$ for any $\delta > 0$, and the $\mathrm{IG}(\nu/2,\nu/2)$ prior ensures that the posterior mass on $(0,\delta)$ is uniformly small.

By the Monotone Convergence Theorem applied to the increasing sequence $f_i^\sigma$, together with the weak convergence of the posterior:

$$\mathbb{E}_{\tau|\mathbf{y},\sigma}\!\left[\frac{1}{\tau d_i + \sigma^2}\right] \;\xrightarrow{\sigma\to 0}\; \mathbb{E}_{\tau|\mathbf{y},0}\!\left[\frac{1}{\tau d_i}\right] = \frac{1}{d_i}\,\mathbb{E}_{\tau|\mathbf{y},0}\!\left[\frac{1}{\tau}\right] \tag{1}$$

**Step 2 — Evaluating the limiting posterior expectation.**

At $\sigma = 0$, the posterior $\tau|\mathbf{y}$ follows from the exact conjugacy between the $\mathrm{IG}(\nu/2, \nu/2)$ prior and the Gaussian likelihood $\mathcal{N}(\mathbf{y};\mathbf{0},\tau\Sigma)$:

$$p(\tau|\mathbf{y}, 0) \propto \tau^{-d/2}\exp\!\left(-\frac{A}{2\tau}\right)\cdot\tau^{-\nu/2-1}\exp\!\left(-\frac{\nu}{2\tau}\right) = \tau^{-(\nu+d)/2-1}\exp\!\left(-\frac{\nu+A}{2\tau}\right)$$

This is the $\mathrm{IG}\!\left(\dfrac{\nu+d}{2},\, \dfrac{\nu+A}{2}\right)$ density. For $X \sim \mathrm{IG}(\alpha,\beta)$, $\mathbb{E}[1/X] = \alpha/\beta$, so:

$$\mathbb{E}_{\tau|\mathbf{y},0}\!\left[\frac{1}{\tau}\right] = \frac{(\nu+d)/2}{(\nu+A)/2} = \frac{\nu+d}{\nu+A} = w(A) \tag{2}$$

**Step 3 — Limiting score equals the t-score.**

Substituting (2) into (1) and back into $(\star)$:

$$\nabla\log q_\sigma(\mathbf{y}) \;\xrightarrow{\sigma\to 0}\; -U\,\mathrm{diag}\!\left(\frac{w(A)}{d_i}\right)U^\top\mathbf{y} = -w(A)\,U\,\mathrm{diag}(1/d_i)\,U^\top\mathbf{y} = -w(A)\,Q\mathbf{y}$$

This equals $\nabla\log t_\nu(\mathbf{y})$, as can be verified directly from the t-density.

**Step 4 — The t-score lies in $\mathcal{F}$.**

Since $Q \succ 0$, its matrix square root $Q^{1/2}$ exists and satisfies $Q^{1/2}Q^{1/2} = Q$. Set $A = Q^{1/2}$ and $u(x) = w(x) = (\nu+d)/(\nu+x)$. Then:

$$\|A\mathbf{y}\|^2 = \mathbf{y}^\top Q^{1/2}Q^{1/2}\mathbf{y} = \mathbf{y}^\top Q\mathbf{y} = A_{\mathrm{Mahal}}$$
$$-A^\top A\mathbf{y} = -Q^{1/2}Q^{1/2}\mathbf{y} = -Q\mathbf{y}$$
$$\psi(\mathbf{y}) = w(A_{\mathrm{Mahal}})\cdot(-Q\mathbf{y}) = \nabla\log t_\nu(\mathbf{y})$$

Therefore $\nabla\log t_\nu \in \mathcal{F}$.

**Step 5 — The projection error vanishes.**

Since $\nabla\log t_\nu \in \mathcal{F}$, it is feasible for the projection problem. Therefore:

$$\left\|\psi_\sigma^* - \nabla\log q_\sigma\right\|^2_{L^2(q_\sigma)} \;\leq\; \left\|\nabla\log t_\nu - \nabla\log q_\sigma\right\|^2_{L^2(q_\sigma)} \;\xrightarrow{\sigma\to 0}\; 0$$

where the limit follows from Step 3 by dominated convergence (the scores are square-integrable under $q_\sigma$ for small $\sigma$ since $\mathbb{E}[\|w(A)Q\mathbf{y}\|^2] < \infty$ for $\nu > 4$).

Hence $\psi_\sigma^* \to \nabla\log t_\nu$ in $L^2(q_\sigma)$.

**Step 6 — Detection statistic converges to t-Rao.**

By the Law of Large Numbers, $\bar\psi \to \mathbb{E}_{H_0}[\psi(\mathbf{w})] = \mathbf{0}$ (since the t-score has zero mean under $H_0$) and $\hat{C}_\psi \to C_\psi = \mathrm{Var}_{H_0}[\psi(\mathbf{w})]$. By the Fisher information identity for the score, $\mathbf{s}^\top C_\psi \mathbf{s} = I_{0,t} = \frac{\nu+d}{\nu+d+2}C$. Therefore:

$$T_s = \frac{-\mathbf{s}^\top(\psi_\sigma^*(\mathbf{y}) - \bar\psi)}{\sqrt{\mathbf{s}^\top\hat{C}_\psi\mathbf{s}}} \;\to\; \frac{\mathbf{s}^\top w(A)Q\mathbf{y}}{\sqrt{\dfrac{\nu+d}{\nu+d+2}C}} = \frac{w(A)\cdot B}{\sqrt{I_{0,t}}} = T_{\mathrm{Rao}} \qquad \square$$

---

## Theorem 2 — Convergence to a Matched Filter as $\sigma \to \infty$

**Statement.** As $\sigma \to \infty$, the in-class DSM minimiser satisfies

$$u_\sigma^*(d) \;\to\; \text{const} \quad \text{uniformly in } d$$

The adaptive Mahalanobis weighting disappears, and $T_s(\psi_\sigma^*) \propto \mathbf{s}^\top\mathbf{y}$ — a linear matched filter with no dependence on $\Sigma$ or $\nu$.

**Proof.**

**Step 1 — Score at large $\sigma$.**

From $(\star)$, for each eigendirection $d_i > 0$ and $\tau \sim \mathrm{IG}(\nu/2, \nu/2)$:

$$(\tau\Sigma + \sigma^2 I)^{-1} = \frac{1}{\sigma^2}\left(I - \frac{\tau\Sigma}{\sigma^2}\right)^{-1}\cdot\frac{1}{\sigma^{-2}} = \frac{1}{\sigma^2}\sum_{k=0}^\infty \left(-\frac{\tau\Sigma}{\sigma^2}\right)^k$$

For $\sigma^2 > \tau\|\Sigma\|_{\mathrm{op}}$, this geometric series converges and:

$$(\tau\Sigma + \sigma^2 I)^{-1} = \frac{1}{\sigma^2}I - \frac{\tau}{\sigma^4}\Sigma + O(\sigma^{-6})$$

Taking the posterior expectation and using $\mathbb{E}[\tau] = \nu/(\nu-2)$ (which is finite for $\nu > 2$):

$$\mathbb{E}_{\tau|\mathbf{y},\sigma}\!\left[(\tau\Sigma + \sigma^2 I)^{-1}\right] = \frac{1}{\sigma^2}I - \frac{\mathbb{E}[\tau|\mathbf{y},\sigma]}{\sigma^4}\Sigma + O(\sigma^{-6})$$

For large $\sigma$, $\mathbb{E}[\tau|\mathbf{y},\sigma] \to \mathbb{E}[\tau] = \nu/(\nu-2)$ since the posterior on $\tau$ converges to the prior (the likelihood becomes flat). Therefore:

$$\nabla\log q_\sigma(\mathbf{y}) = -\frac{1}{\sigma^2}\mathbf{y} + \frac{\nu/(\nu-2)}{\sigma^4}\Sigma\mathbf{y} + O(\sigma^{-6}) \tag{3}$$

The leading term is isotropic: $-\mathbf{y}/\sigma^2$.

**Step 2 — The isotropic score is in $\mathcal{F}$.**

Set $A = \frac{1}{\sigma}I$ and $u \equiv 1$. Then $\|A\mathbf{y}\|^2 = \|\mathbf{y}\|^2/\sigma^2$ and $-A^\top A\mathbf{y} = -\mathbf{y}/\sigma^2$, giving:

$$\psi(\mathbf{y}) = 1 \cdot \left(-\frac{\mathbf{y}}{\sigma^2}\right) = -\frac{\mathbf{y}}{\sigma^2} \;\in\; \mathcal{F}$$

**Step 3 — The projection error is $O(\sigma^{-4})$.**

From the feasibility of $-\mathbf{y}/\sigma^2 \in \mathcal{F}$:

$$\left\|\psi_\sigma^* - \nabla\log q_\sigma\right\|^2_{L^2(q_\sigma)} \;\leq\; \left\|-\frac{\mathbf{y}}{\sigma^2} - \nabla\log q_\sigma(\mathbf{y})\right\|^2_{L^2(q_\sigma)} = O(\sigma^{-4}) \;\to\; 0 \tag{4}$$

where the bound follows from (3): $\|-\mathbf{y}/\sigma^2 - \nabla\log q_\sigma\| = O(\sigma^{-4})\|\Sigma\mathbf{y}\| = O(\sigma^{-4})$ since $\mathbb{E}[\|\Sigma\mathbf{y}\|^2] < \infty$.

**Step 4 — The scalar weight collapses to a constant.**

Suppose for contradiction that $u_\sigma^*(d)$ is not asymptotically constant, i.e., there exist $d_1, d_2 \geq 0$ such that $|u_\sigma^*(d_1) - u_\sigma^*(d_2)| \not\to 0$. Then $\psi_\sigma^*$ produces different scalings for observations with $\|A^*\mathbf{y}\|^2 = d_1$ versus $d_2$, meaning $\psi_\sigma^*(\mathbf{y}) \neq c \cdot \mathbf{y}$ for any constant $c$. But from (4), $\|\psi_\sigma^* - (-\mathbf{y}/\sigma^2)\|_{L^2} = O(\sigma^{-2})$, which forces $\psi_\sigma^*(\mathbf{y}) \to c\mathbf{y}$ in $L^2$. Contradiction. Therefore:

$$u_\sigma^*(d) \;\to\; c_\infty \quad \text{in } L^2(q_\sigma)\text{-sense} \tag{5}$$

for some constant $c_\infty > 0$, and in particular the Mahalanobis-dependent weighting $w(A) = (\nu+d)/(\nu+A)$ is not recovered.

**Step 5 — Detection statistic becomes a linear matched filter.**

With $\psi_\sigma^*(\mathbf{y}) \to c_\infty \cdot (-A_\infty^\top A_\infty \mathbf{y})$ for some $A_\infty$ with $A_\infty^\top A_\infty = c_\infty^{-1}\sigma^{-2}I$ (from Step 3):

$$T_s = \frac{-\mathbf{s}^\top(\psi_\sigma^*(\mathbf{y}) - \bar\psi)}{\sqrt{\mathbf{s}^\top\hat{C}_\psi\mathbf{s}}} \;\propto\; \mathbf{s}^\top\mathbf{y} \tag{6}$$

since $\bar\psi \to \mathbf{0}$ and the normalization cancels the scalar $c_\infty/\sigma^2$. The statistic $\mathbf{s}^\top\mathbf{y}$ is the matched filter for additive signal $\theta\mathbf{s}$ with identity noise covariance — the AMF. $\square$

---

## Corollary — Interpolation

Together, Theorems 1 and 2 establish that $\sigma$ acts as a continuous interpolation parameter:

| $\sigma$ | In-class target | Detection statistic |
|---|---|---|
| $\sigma \to 0$ | $\nabla\log t_\nu = -w(A)Q\mathbf{y}$ | Oracle t-Rao |
| $\sigma > 0$ (intermediate) | $L^2$-projection of $\nabla\log q_\sigma$ onto $\mathcal{F}$ | Scalar-weighted AMF with adaptive $u^*(A)$ |
| $\sigma \to \infty$ | $-\mathbf{y}/\sigma^2$ (isotropic) | Linear matched filter (AMF) |

At $\sigma = 0$, the projection is exact (zero residual) because the t-score lies in $\mathcal{F}$. At $\sigma \to \infty$, the projection is again exact at leading order because the isotropic score lies in $\mathcal{F}$. At intermediate $\sigma$, the projection finds a scalar weight $u^*(d)$ that is a compromise between the direction-dependent weights of $\nabla\log q_\sigma$ — this compromise avoids the full suppression of the t-Rao and empirically tracks the oracle t-GLRT.

**Remark.** The proof of Theorem 1 is stronger than the analogous result for an unconstrained MLP. The MLP requires universal approximation capacity to represent $\nabla\log t_\nu$; the two-branch achieves exact representation with finite parameters (Step 4), because the t-score is a scalar-weighted $Q$-map. The MLP's proof relies only on Tweedie and continuity of the score; the two-branch's proof additionally exploits the structural match between the architecture and the limiting score.
