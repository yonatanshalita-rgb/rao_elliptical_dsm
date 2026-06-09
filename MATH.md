# Mathematical Details

## 1. Signal Model

We consider the binary hypothesis test:

$$H_0: \mathbf{y} = \mathbf{w} \qquad H_1: \mathbf{y} = a\mathbf{s} + \mathbf{w}$$

where:
- $\mathbf{y} \in \mathbb{R}^n$ is the observation vector
- $\mathbf{s} \in \mathbb{R}^n$ is the known target steering vector (unit norm)
- $a > 0$ is the (unknown) target amplitude, set by the SNR
- $\mathbf{w}$ is the noise vector

**SNR definition.** We define the signal-to-noise ratio as:

$$\mathrm{SNR} = a^2 \, \mathbf{s}^\top Q^{-1} \mathbf{s}$$

so that $a = \sqrt{\mathrm{SNR} / (\mathbf{s}^\top Q^{-1} \mathbf{s})}$.

---

## 2. Elliptical Noise Model

The noise is drawn from an **elliptical distribution**:

$$\mathbf{w}_i = \mu_i \, \mathbf{v}_i$$

where:
- $\mathbf{v}_i \overset{\text{i.i.d.}}{\sim} \mathcal{N}(\mathbf{0}, Q)$ is the Gaussian component with covariance $Q \succ 0$
- $\mu_i \geq 0$ is an i.i.d. scalar **texture variable**, independent of $\mathbf{v}_i$

The marginal distribution of $\mathbf{w}_i$ is elliptical with scatter matrix $Q$ and density generator determined by the distribution of $\mu_i$.

### 2.1 Texture Distributions

**Spike mixture (adversarial outliers):**
$$\mu_i = \begin{cases} 1 & \text{with probability } 1 - p \\ c & \text{with probability } p \end{cases}$$

Default: $p = 0.01$, $c = 1000$. The sample covariance is dominated by the rare large-$\mu$ samples.

**Compound Gaussian / K-distribution:**
$$\tau_i \sim \Gamma(\alpha, \beta), \quad \mu_i = \sqrt{\tau_i}$$

The marginal $\mathbf{w}_i$ follows a K-distribution. Widely used in sea clutter modelling.

**Multivariate-$t$ distribution:**
$$\mu_i = \sqrt{\nu / \chi^2_\nu}, \quad \chi^2_\nu \sim \chi^2(\nu)$$

The marginal $\mathbf{w}_i$ is multivariate-$t$ with $\nu$ degrees of freedom.

### 2.2 Covariance Structure

The noise covariance is:

$$\mathbb{E}[\mathbf{w}_i \mathbf{w}_i^\top] = \mathbb{E}[\mu_i^2] \cdot Q$$

so $Q$ governs the spatial correlation. We use the **AR(1) model**:

$$Q_{jk} = \rho^{|j-k|}, \quad \rho = 0.9$$

which is standard in array signal processing and yields a highly correlated, ill-conditioned covariance.

---

## 3. Optimal Detector and AMF

Under **Gaussian** noise $\mathbf{w} \sim \mathcal{N}(\mathbf{0}, Q)$, the Neyman–Pearson optimal detector is the **Adaptive Matched Filter (AMF)**:

$$T_\mathrm{AMF}(\mathbf{y}) = \frac{\mathbf{s}^\top Q^{-1} \mathbf{y}}{\sqrt{\mathbf{s}^\top Q^{-1} \mathbf{s}}}$$

Under $H_0$: $T_\mathrm{AMF} \sim \mathcal{N}(0, 1)$.  
Under $H_1$: $T_\mathrm{AMF} \sim \mathcal{N}(\sqrt{\mathrm{SNR}}, 1)$.

Crucially, the AMF is **also optimal under elliptical noise** (up to a monotone transformation) because the likelihood ratio depends on $\mathbf{y}$ only through $\mathbf{s}^\top Q^{-1} \mathbf{y}$. The difficulty is that $Q$ is unknown in practice.

---

## 4. Classical Covariance Estimators

### 4.1 Sample Covariance Matrix (SCM)

$$\widehat{Q}_\mathrm{SCM} = \frac{1}{M} \sum_{i=1}^M \mathbf{w}_i \mathbf{w}_i^\top$$

This is the MLE under Gaussian noise. Under elliptical noise with outliers (large $\mu_i$), the SCM is dominated by the high-$\mu$ samples and becomes nearly singular, destroying detector performance.

### 4.2 Tyler's M-Estimator

Tyler's estimator (Tyler, 1987) is the **maximum likelihood estimator of scatter** over all elliptical distributions with unknown density generator. It solves the fixed-point equation:

$$\widehat{C} = \frac{n}{M} \sum_{i=1}^M \frac{\mathbf{w}_i \mathbf{w}_i^\top}{\mathbf{w}_i^\top \widehat{C}^{-1} \mathbf{w}_i}$$

normalised so that $\mathrm{tr}(\widehat{C}) = n$. The fixed-point iteration converges under mild conditions.

Key property: Tyler's estimator **reweights each sample by the inverse of its estimated Mahalanobis distance**, completely suppressing outliers (large $\mu_i$ increases $\mathbf{w}_i^\top C^{-1} \mathbf{w}_i$, reducing the weight to near zero).

### 4.3 Huber M-Estimator

The Huber estimator (Kent & Tyler, 1991) interpolates between SCM and Tyler via the weight function:

$$u_\beta(t) = \frac{\beta}{\max(\beta, \, t/n)}, \quad t = \mathbf{w}^\top C^{-1} \mathbf{w}$$

$$\widehat{C} = \frac{1}{M} \sum_{i=1}^M u_\beta(\mathbf{w}_i^\top \widehat{C}^{-1} \mathbf{w}_i) \cdot \mathbf{w}_i \mathbf{w}_i^\top$$

- $\beta = 1$: reduces to SCM
- $\beta \to 0$: approaches Tyler

Default: $\beta = 0.9$.

---

## 5. Paper's Neural Test Statistic

Following the paper, the score-based test statistic is:

$$T_s(\mathbf{y}) = \frac{-\mathbf{s}^\top \bigl(\widehat{\psi}(\mathbf{y}) - \bar{\mathbf{z}}\bigr)}{\sqrt{\mathbf{s}^\top \widehat{C}_\psi \mathbf{s}}}$$

where:
- $\widehat{\psi}(\mathbf{y})$ is the learned score function (network output)
- $\bar{\mathbf{z}} = \frac{1}{N_c} \sum_{i=1}^{N_c} \widehat{\psi}(\mathbf{w}_i)$ is the empirical mean of scores over calibration noise samples
- $\widehat{C}_\psi = \frac{1}{N_c} \sum_{i=1}^{N_c} \widehat{\psi}(\mathbf{w}_i)\widehat{\psi}(\mathbf{w}_i)^\top - \bar{\mathbf{z}}\bar{\mathbf{z}}^\top$ is the empirical score covariance

The sign is chosen so that $T_s > 0$ under $H_1$.

**Connection to AMF.** If $\widehat{\psi}(\mathbf{y}) = -Q^{-1}\mathbf{y}$ (the true Gaussian score), then $\bar{\mathbf{z}} = \mathbf{0}$ (under $H_0$) and $T_s$ reduces exactly to the AMF.

---

## 6. Denoising Score Matching (DSM)

### 6.1 Score Matching Objective

The goal is to learn $\widehat{\psi}$ such that $\widehat{\psi}(\mathbf{w}) \approx \nabla_\mathbf{w} \log p(\mathbf{w})$.

The **implicit score matching** objective (Hyvärinen, 2005) is:

$$\mathcal{L}(\psi) = \mathbb{E}_\mathbf{w}\left[ \|\psi(\mathbf{w})\|^2 + 2\,\mathrm{div}\,\psi(\mathbf{w}) \right]$$

This avoids computing the true score but requires the divergence $\mathrm{div}\,\psi$.

### 6.2 Perturbation-Based DSM

We use the **perturbation DSM** (Vincent, 2011), which is simpler to implement:

Given a noise level $\sigma > 0$, sample $\widetilde{\mathbf{w}} = \mathbf{w} + \sigma \varepsilon$ with $\varepsilon \sim \mathcal{N}(\mathbf{0}, I)$.

The noising kernel score is $\nabla_{\widetilde{\mathbf{w}}} \log q(\widetilde{\mathbf{w}} | \mathbf{w}) = -\varepsilon/\sigma$.

**DSM MSE loss:**

$$\mathcal{L}_\mathrm{MSE}(\psi) = \mathbb{E}_{\mathbf{w}, \varepsilon}\left[ \left\| \psi(\mathbf{w} + \sigma\varepsilon) + \frac{\varepsilon}{\sigma} \right\|^2 \right]$$

### 6.3 Huber DSM Loss

Replace the squared norm with the Huber pseudo-norm applied element-wise:

$$\ell_\delta(r) = \begin{cases} \frac{1}{2} r^2 & |r| \leq \delta \\ \delta\bigl(|r| - \frac{\delta}{2}\bigr) & |r| > \delta \end{cases}$$

$$\mathcal{L}_\mathrm{Huber}(\psi) = \mathbb{E}_{\mathbf{w}, \varepsilon}\left[ \sum_{j=1}^n \ell_\delta\!\left( \psi_j(\widetilde{\mathbf{w}}) + \frac{\varepsilon_j}{\sigma} \right) \right]$$

For large residuals (outlier noise realisations), the gradient is capped at $\delta$, making the loss robust to extreme perturbations induced by large $\mu_i$.

Default: $\delta = 1.0$, $\sigma = 0.1$.

---

## 7. Score Function Architectures

### Model 1 — Linear DSM (MSE loss)

$$\psi(\mathbf{y}) = W\mathbf{y}, \quad W \in \mathbb{R}^{n \times n}$$

Trained with $\mathcal{L}_\mathrm{MSE}$.

**Motivation.** The true Gaussian score is $\nabla \log p(\mathbf{y}) = -Q^{-1}\mathbf{y}$, which is linear. A linear model is the natural parametrisation. Under elliptical noise, $W$ will converge toward $-\mathbb{E}[\mu_i^2]^{-1} Q^{-1}$, but the MSE loss gives equal weight to all samples including outliers (large $\mu_i$), causing $W$ to be biased by those few samples. **Expected to fail.**

Initialisation: $W = \frac{1}{n} I$ (gentle whitening).

**Implementation note — per-sample gradient clipping.** In the spike texture model, outlier samples ($\mu_i = 1000$) produce gradient contributions $\approx 1000\times$ larger than inlier samples. With a batch of 512 samples and $p_\text{outlier} = 0.01$, the $\sim 5$ outlier samples per batch dominate the gradient entirely, causing high training variance and biased parameter estimates. Fix: each sample's contribution to the loss is scaled by

$$s_i = \min\!\left(1,\; \frac{C}{\frac{2}{n}\|\mathbf{r}_i\|\|\widetilde{\mathbf{w}}_i\|}\right)$$

where $\mathbf{r}_i = \psi(\widetilde{\mathbf{w}}_i) - \mathbf{t}_i$ is the residual and $\frac{2}{n}\|\mathbf{r}_i\|\|\widetilde{\mathbf{w}}_i\|$ is the estimated Frobenius norm of the per-sample gradient for a linear model. The scale $s_i$ is detached (treated as a constant), so gradients flow only through the loss, not through the norm estimate. Threshold $C = 100$ for MSE (about $5\times$ the typical inlier gradient norm $\approx 20$). After applying this fix the training variance collapsed from std $\approx 0.34$ to std $< 0.001$ in AUC across 5 seeds, and the model reached AUC $= 0.978 \pm 0.0001$, Pd $= 0.718 \pm 0.004$ at SNR $= 10\,\mathrm{dB}$ — essentially matching the Oracle AMF.

---

### Model 2 — Linear DSM (Huber loss)

Identical architecture to Model 1:

$$\psi(\mathbf{y}) = W\mathbf{y}$$

Trained with $\mathcal{L}_\mathrm{Huber}$ (delta = 1.0).

**Motivation.** The Huber loss down-weights large residuals. When $\mu_i$ is large, $\varepsilon/\sigma$ is small relative to $\psi(\widetilde{\mathbf{w}})$, but the noisy sample $\widetilde{\mathbf{w}}$ is far from the typical set. The Huber loss limits the gradient contribution of such samples, steering $W$ toward $-Q^{-1}$ estimated from the inlier population. The network implicitly learns a robust linear score.

**Implementation note — per-sample gradient clipping.** Applied analogously to Model 1, using the Huber-clipped gradient $h_i = \text{clip}(\mathbf{r}_i, -\delta, \delta)$ as the gradient proxy:

$$s_i = \min\!\left(1,\; \frac{C}{\frac{1}{n}\|\mathbf{h}_i\|\|\widetilde{\mathbf{w}}_i\|}\right)$$

Threshold $C = 10$ (about $10\times$ the typical inlier gradient norm $\approx 1$, since Huber clips element-wise residuals to $\delta = 1$). After the fix the model achieves AUC $= 0.975 \pm 0.0004$, Pd $= 0.673 \pm 0.005$ at SNR $= 10\,\mathrm{dB}$. Notably this is slightly *below* Model 1 (MSE + per-sample clip): the double robustness mechanism (Huber element-wise clipping combined with per-sample norm clipping) appears to over-regularize the gradient signal, leaving less effective learning signal than per-sample clipping alone.

---

### Model 3 — Two-Branch DSM (MSE loss)

The score function is factored as:

$$\psi(\mathbf{y}) = u(d(\mathbf{y})) \cdot \mathbf{v}(\mathbf{y})$$

**v-branch (linear):**
$$\mathbf{v}(\mathbf{y}) = W\mathbf{y}, \quad W \in \mathbb{R}^{n \times n}$$

This estimates the direction $-Q^{-1}\mathbf{y}$.

**Mahalanobis proxy:**
$$d(\mathbf{y}) = -\mathbf{y}^\top \mathbf{v}(\mathbf{y}) = -\mathbf{y}^\top W \mathbf{y}$$

Once $W \approx -Q^{-1}$ (symmetric), this gives:
$$d(\mathbf{y}) \approx \mathbf{y}^\top Q^{-1} \mathbf{y} = d_Q^2(\mathbf{y}) > 0$$

We clip $d$ to $[0, \infty)$ before feeding the u-branch to guard against early training when $W$ has not yet converged.

**u-branch (nonlinear MLP):**
$$u = \mathrm{softplus}\!\bigl(\mathrm{MLP}(\max(d, 0))\bigr)$$

The MLP has input dimension 1, two hidden layers of width 32 with Tanh activations, and scalar output. `softplus` ensures $u > 0$.

Initialisation: the final MLP layer is zero-initialised so $u \approx \log 2 \approx 0.693$ at the start (neutral scaling).

**Why this architecture?** Every M-estimator has the form:

$$\nabla \log p_\mathrm{elliptical}(\mathbf{w}) = -u(\mathbf{w}^\top Q^{-1} \mathbf{w}) \cdot Q^{-1} \mathbf{w}$$

where $u(t) = -g'(t) / g(t)$ is determined by the density generator $g$.
- Tyler: $u(t) = n/t$
- Gaussian: $u(t) = 1$ (constant)
- Huber: $u(t) = \beta \cdot \min(1, n/(\beta t))$

The two-branch architecture is a **learned M-estimator** — the network can discover any of these weight functions, or an optimal one for the specific noise distribution, purely from data. After training, plotting $u(d)$ vs $d$ reveals which classical estimator the network has converged to.

**Training:** standard $\mathcal{L}_\mathrm{MSE}$ loss. Robustness comes entirely from the architecture, not the loss.

**Implementation note — branch warm-up.** Without special treatment, the u-branch collapses to a near-zero output early in training (a lazy local minimum: setting $u \approx 0$ makes $\psi \approx 0$, which is a low-loss solution before $W$ has converged). This kills all gradient flow into the u-branch permanently. Fix: freeze the u-branch parameters for the first 12 epochs so the W-branch first learns a reasonable linear filter $W \approx -Q^{-1}$. Once unfrozen, the u-branch immediately receives meaningful Mahalanobis proxy values $d \approx \mathbf{y}^\top Q^{-1}\mathbf{y}$ and can learn the down-weighting function. With this warm-up the training loss drops from $\sim\!100$ to $\sim\!1.77$ at epoch 13 reliably across all random seeds, and the model achieves AUC $= 0.982 \pm 0.001$, Pd $= 0.782 \pm 0.006$ at SNR $= 10\,\mathrm{dB}$.

**Implementation note — sign of $d$.** A subtle sign error was found and corrected in the implementation. The original code computed $d = \mathbf{y}^\top W\mathbf{y}$ (positive sign). However, $W$ is trained to approximate $-Q^{-1}$ (the score is $-Q^{-1}\mathbf{y}$, so $W \to -Q^{-1}$), which is negative definite. Therefore $\mathbf{y}^\top W\mathbf{y} \approx -\mathbf{y}^\top Q^{-1}\mathbf{y} < 0$ throughout training. The clamp to $[0, \infty)$ then zeroed out $d$ entirely (~54% of samples were negative, the rest were outlier-dominated transients), the u-branch received a constant zero input, and its gradients were killed. The fix is to negate: $d = -\mathbf{y}^\top W\mathbf{y} \approx \mathbf{y}^\top Q^{-1}\mathbf{y} > 0$, which is the correct positive Mahalanobis proxy. After this fix the Two-Branch model reached AUC $\approx 0.976$, matching the classical robust detectors.

---

## 8. Summary Table

Results at SNR = 10 dB, mean ± std over 5 seeds.

| Method | AUC | Pd (@PFA=0.01) | Notes |
| --- | --- | --- | --- |
| Oracle AMF | 0.9784 ± 0.0000 | 0.722 ± 0.000 | Upper bound (knows $Q$) |
| SCM AMF | 0.9600 ± 0.0000 | 0.518 ± 0.000 | Baseline, fails with outliers |
| Tyler AMF | 0.9782 ± 0.0000 | 0.723 ± 0.000 | Classical robust |
| Huber AMF | 0.9782 ± 0.0000 | 0.723 ± 0.000 | Interpolates SCM↔Tyler |
| DSM Linear (MSE) | 0.9781 ± 0.0001 | 0.718 ± 0.004 | Matches Oracle after per-sample clip fix |
| DSM Linear (Huber) | 0.9745 ± 0.0004 | 0.673 ± 0.005 | Double robustness reduces learning signal |
| DSM Two-Branch | 0.9821 ± 0.0006 | 0.784 ± 0.007 | Beats Oracle — learned M-estimator |

---

## 9. References

- Tyler, D.E. (1987). A distribution-free M-estimator of multivariate scatter. *Annals of Statistics*.
- Kent, J.T. & Tyler, D.E. (1991). Redescending M-estimates of multivariate location and scatter. *Annals of Statistics*.
- Hyvärinen, A. (2005). Estimation of non-normalized statistical models by score matching. *JMLR*.
- Vincent, P. (2011). A connection between score matching and denoising autoencoders. *Neural Computation*.
