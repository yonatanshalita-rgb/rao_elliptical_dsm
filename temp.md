# Two-Branch Convergence: t-Rao vs t-GLRT

## What the Two-Branch Should Converge To

DSM trains $\psi$ to approximate the score of the **noise** distribution.
For $t_\nu(\mathbf{0}, \Sigma)$, the true score is:

$$\nabla_{\mathbf{w}} \log p(\mathbf{w}) = -w(A) \cdot Q\mathbf{w}, \qquad w(A) = \frac{\nu + d}{\nu + A}$$

A perfectly trained two-branch gives $\psi(\mathbf{y}) \approx -w(A)\,Q\mathbf{y}$, and the NeuralDetector computes:

$$T_s(\mathbf{y}) = \frac{-\mathbf{s}^\top(\psi(\mathbf{y}) - \bar{z})}{\sqrt{\mathbf{s}^\top \hat{C}_\psi \mathbf{s}}} \;\approx\; \frac{w(A)\cdot B}{\sqrt{\mathrm{Var}_{H_0}[w(A)\cdot B]}}$$

By the score-variance identity, $\mathrm{Var}_{H_0}[\text{score}] = I_{0,t}$, so the denominator equals $\sqrt{I_{0,t}}$.
Therefore **a perfect two-branch converges to the t-Rao**, not the t-GLRT:

$$T_s(\mathbf{y}) \;\xrightarrow{\text{perfect training}}\; T_{\text{Rao},t} = \frac{w(A)\cdot B}{\sqrt{\dfrac{\nu+d}{\nu+d+2}\,C}}$$

---

## Why the Two-Branch Can Surpass the t-Rao

The t-Rao is the **locally most powerful** test — optimal only for infinitesimal $\theta$ near zero.
The evaluation uses $H_1: \theta \sim U(0,\,\theta_{\max})$ with $\theta_{\max} \approx 9.5$ at SNR $= 10\,\text{dB}$.
That is far from local.

**The suppression problem for large $\theta$.**
Under $H_1$ with large $\theta$:

$$A = \mathbf{y}^\top Q\mathbf{y} = (\mathbf{w} + \theta\mathbf{s})^\top Q(\mathbf{w} + \theta\mathbf{s}) \approx \theta^2 C + 2\theta B_{\text{noise}} + A_{\text{noise}}$$

The t-Rao weight $w(A) = (\nu+d)/(\nu+A)$ then suppresses the observation because large $A$ looks like a noise outlier.
For $\theta = 9.5$, $C = 0.112$:

$$A \approx 9.5^2 \times 0.112 \approx 10 \qquad \Rightarrow \qquad w(A) \approx \frac{19}{13} \approx 1.46$$

compared to the noise-only value $w(A_{\text{noise}}) \approx 19/3 \approx 6.3$.
The t-Rao loses sensitivity precisely where the signal is strongest.

**Why the learned $u(d)$ can be better.**
The two-branch learns $u(d)$ via finite-data DSM with sigma-perturbed noise.
It is not guaranteed to exactly reproduce $w(A) = (\nu+d)/(\nu+A)$.
If the learned weight is slightly less aggressive in suppressing large $d$, it naturally handles the large-$\theta$ portion of $H_1$ better.
The calibrated denominator $\sqrt{\mathbf{s}^\top \hat{C}_\psi \mathbf{s}}$ also adapts to the actual learned $\psi$, not the theoretical $\sqrt{I_{0,t}}$.

---

## The Correct Upper Bound

The **t-GLRT** avoids the suppression problem by substituting the MLE $\hat\theta = B/C$:

$$T_{\text{GLRT}} = \frac{\nu+d}{2}\log\frac{\nu + A}{\nu + A - B^2/C} \qquad (B > 0)$$

For large $\theta$: $A \approx \theta^2 C$ and $B \approx \theta C$, so $B^2/C \approx A$, giving:

$$T_{\text{GLRT}} \approx \frac{\nu+d}{2}\log\frac{\nu + A}{\nu} \;\to\; \infty \quad \text{as } \theta \to \infty$$

The t-GLRT grows (logarithmically) with signal strength rather than shrinking.
In $P_d$ at $1\%$ $P_{fa}$, the two-branch ($0.215$) sits **below** the t-GLRT ($0.241$), which is the expected ordering.

The marginal AUC advantage of two-branch ($0.7876$) over t-GLRT ($0.7845$) is $\approx 0.003$, within 1–2 standard errors for $N=50\,000$ test samples — treat as within noise.

---

## Summary

| Detector | Designed for | Handles large $\theta$? | AUC | $P_d$ @ $1\%$ |
|---|---|---|---|---|
| t-Rao | Local $H_1$ ($\theta \to 0$) | No — $w(A)$ suppresses strong signals | $0.7747$ | $0.161$ |
| t-GLRT (oracle) | Any $\theta$ via MLE | Yes — grows with signal | $0.7845$ | $0.241$ |
| Two-Branch (DSM) | Score matching on $H_0$ | Partially — depends on learned $u(d)$ | $0.7876$ | $0.215$ |
| Gaussian AMF | Gaussian noise | N/A (wrong model) | $0.7777$ | $0.046$ |

The two-branch approaches the t-GLRT from below in $P_d$, which is the correct expected ordering.
Beating the t-Rao in AUC is real but unsurprising: the composite $H_1$ exposes the t-Rao's local-optimality limitation, and the learned $u(d)$ avoids the worst of the suppression problem.

---

## Why Two-Branch Does Not Converge to t-Rao (Empirical Finding)

### Exact consequence of perfect convergence

The NeuralDetector normalisation $\sqrt{\mathbf{s}^\top \hat{C}_\psi \mathbf{s}}$ is a **constant** with respect to $\mathbf{y}$ — it does not affect the ranking of test statistics across samples.
Therefore if the two-branch perfectly learned $u(d) = (\nu+d)/(\nu+d_0)$, the **AUC must equal the t-Rao's AUC exactly**. No constant rescaling changes rankings.

Empirically: two-branch AUC $= 0.946$ vs t-Rao AUC $= 0.907$ (fixed $\theta$ case). This gap is too large to be noise. The two-branch learned a **different weight function** than $(\nu+d)/(\nu+A)$.

### Why DSM produces a different weight function

DSM with noise level $\sigma > 0$ trains the model to approximate the score of the **convolved** distribution:

$$q_\sigma(\tilde{\mathbf{w}}) = \int p(\mathbf{w})\,\mathcal{N}(\tilde{\mathbf{w}};\,\mathbf{w},\,\sigma^2 I)\,d\mathbf{w} = t_\nu * \mathcal{N}(\mathbf{0},\,\sigma^2 I)$$

The score of $q_\sigma$ is not the t-score; the Gaussian convolution softens the heavy tails and produces a **flatter** weight function — less aggressive suppression at large $d$.

This is not "imperfect training accidentally improving things." It is training on a **different objective** that produces a systematically different estimator. The convolved score at $\sigma = 0.5$ is a well-defined function; it just differs from the exact t-score at $\sigma = 0$.

### Validation via fixed-$\theta$ experiment

| Detector | AUC (uniform $\theta$) | AUC (fixed $\theta$) | $P_d$ (fixed $\theta$) |
|---|---|---|---|
| t-GLRT | $0.7845$ | $0.9427$ | $0.601$ |
| Two-Branch | $0.7876$ | $0.9456$ | $0.595$ |
| t-Rao | $0.7747$ | $0.9072$ | $0.315$ |
| Gaussian AMF | $0.7777$ | $0.9381$ | $0.143$ |

Under fixed $\theta = \theta_{\max}$ (simple $H_1$), the t-Rao collapses below the Gaussian AMF in AUC — the suppression of large signals is severe.
The two-branch tracks the t-GLRT closely in both AUC and $P_d$, confirming that the learned $u(d)$ is a flatter function that avoids the suppression problem.

### Open question

The learned weight function should be visible in the $u(d)$ vs theoretical $(\nu+d)/(\nu+d_0)$ plot.
At low SNR (small $\theta$), the t-Rao should dominate (its local optimality advantage), while the two-branch's flatter weight may be suboptimal there. The Pd-vs-SNR curves distinguish these regimes.

---

## Why a "Wrong" Score Can Produce a Better Detector

### The score is not the detection objective

The test statistic is $T_s(\mathbf{y}) = \mathbf{s}^\top \psi(\mathbf{y}) / C_{\text{norm}}$.
Its AUC depends only on how well $\mathbf{s}^\top \psi(\mathbf{y})$ **ranks** $H_0$ samples below $H_1$ samples.
The question is therefore not "does $\psi$ approximate the true score well" but "does $\mathbf{s}^\top\psi(\mathbf{y})$ separate $H_0$ from $H_1$ well."

### The true t-score is not the optimal detection statistic

The true t-score produces the t-Rao statistic $w(A) \cdot B$, which is the **locally most powerful** test at $\theta \to 0$.
For finite — especially large — $\theta$, the Neyman–Pearson optimal test is the full likelihood ratio, approximated by the t-GLRT:

$$T_{\text{GLRT}} = \frac{\nu+d}{2}\log\frac{\nu+A}{\nu+A-B^2/C}$$

Both are strictly better than the t-Rao at large $\theta$.
Therefore if the learned $\psi$ produces a projection $\mathbf{s}^\top\psi(\mathbf{y})$ that is closer in structure to the GLRT than to the t-Rao, it will outperform the true-score-based test.
This is not a contradiction — the t-Rao being derived from the true score does not make it the best detector.

### Why DSM with $\sigma > 0$ produces something closer to the GLRT

The DSM objective with noise level $\sigma$ trains $\psi$ to approximate the score of the **convolved** distribution $q_\sigma = t_\nu * \mathcal{N}(\mathbf{0}, \sigma^2 I)$.
At a point $\mathbf{y}$ with large Mahalanobis norm $A$, the smoothed score averages over a Gaussian neighbourhood of $\mathbf{y}$, many of whose points have smaller $A$.
The result is a **less suppressed** output than the exact t-score would give at that $\mathbf{y}$.

Concretely, the projection $\mathbf{s}^\top\psi_\sigma(\mathbf{y})$ is some interpolation between $w(A) \cdot B$ (t-Rao, suppresses large $A$) and $B$ (Gaussian AMF, ignores $A$).
This interpolation empirically tracks the GLRT better than the t-Rao does, particularly for large-amplitude $H_1$ signals.

### Summary

| Claim | Correct? |
|---|---|
| True t-score → t-Rao (locally optimal test) | Yes |
| t-Rao = optimal detection test for all $\theta$ | **No** — only locally optimal at $\theta \to 0$ |
| Any deviation from true score → worse detection | **No** — depends on whether the deviation moves toward or away from the GLRT |
| DSM with $\sigma > 0$ learns a different, valid estimator | Yes — score of $q_\sigma$, not of $t_\nu$ |
| That estimator can beat the t-Rao in AUC | Yes, because the GLRT beats the t-Rao and the learned test tracks the GLRT |



You're exactly right — Tweedie's formula (Vincent 2011) does guarantee this. Let me state it precisely.

  The DSM objective $\mathbb{E}_{w,\varepsilon}\left[|\psi(\mathbf{w}+\sigma\varepsilon) + \varepsilon/\sigma|^2\right]$ is equivalent (up to a constant) to:

  $$\mathbb{E}{\tilde{w} \sim q\sigma}\left[|\psi(\tilde{\mathbf{w}}) - \nabla_{\tilde{\mathbf{w}}} \log q_\sigma(\tilde{\mathbf{w}})|^2\right]$$
  
  So minimizing DSM is exactly minimizing the score matching objective for $q_\sigma = t_\nu * \mathcal{N}(\mathbf{0}, \sigma^2 I)$, and the guaranteed minimizer is $\nabla \log
  q_\sigma$ — not $\nabla \log t_\nu$.

  There is no contradiction. The two statements are:
  1. Tweedie guarantees perfect convergence to $\nabla \log q_\sigma$ ✓
  2. $q_\sigma \neq t_\nu$ for $\sigma > 0$ ✓

  So we perfectly learn the score of the wrong distribution. It's not imperfect learning — it's exact learning of a different target. Only as $\sigma \to 0$ do $q_\sigma \to t_\nu$
   and the two scores agree. With $\sigma = 0.5$, the Gaussian convolution meaningfully changes the distribution and its score.

  This is actually a well-known issue in score-based generative models: the noise schedule determines which distribution's score you learn. Small $\sigma$ recovers the true score
  but gives high-variance gradients with finite data; large $\sigma$ is stable but biased away from the true score.

  The practical implication for your experiment: if you want the two-branch to converge to the t-Rao, you need $\sigma \to 0$. With $\sigma = 0.5$, you're deliberately (or
  inadvertently) learning the score of a smoother distribution, whose weight function suppresses outliers less aggressively — which happens to be closer to the GLRT for large
  $\theta$. This is a real effect with a clean theoretical cause, not a coincidence.
  



● Not quite an upper bound in the usual sense — it's actually the exact limit the two-branch converges to, not a ceiling above it.
  
  Here's the distinction:

  What a perfectly trained two-branch gives:
  By Tweedie, it learns $\psi^* = \nabla \log q_\sigma$ exactly (given enough data and capacity). The NeuralDetector then computes:

  $$T_s = \frac{\mathbf{s}^\top \nabla \log q_\sigma(\mathbf{y})}{C_\text{norm}}$$

  where $C_\text{norm} = \sqrt{\mathbf{s}^\top \hat{C}\psi \mathbf{s}}$ is a constant the calibration step estimates. The oracle $q\sigma$-Rao would use
  $\sqrt{\text{Var}{H_0}[\mathbf{s}^\top \nabla \log q\sigma]}$ — which is exactly what calibration estimates. So they're the same statistic. The oracle $q_\sigma$-Rao is not above
   the perfect two-branch; it is the perfect two-branch.

  The two-branch is bounded by it from below (finite data, finite network capacity), not from above.

  Why it's not the detection upper bound either:
  The score of $q_\sigma$ under $H_0$ is not the NP-optimal detection statistic for the true model $t_\nu$. For large θ, the t-GLRT is strictly better in Pd. Our empirical results
  confirm this: σ=0.5 two-branch has Pd=0.595 vs t-GLRT Pd=0.601. The two-branch approaches the GLRT but doesn't surpass it in Pd.

  The practical problem: $\nabla \log q_\sigma$ for $t_\nu$ has no closed form — it requires evaluating $\mathbb{E}[\mathbf{w} \mid \tilde{\mathbf{w}}]$ under the posterior, which
  is an intractable integral. That's precisely why we use the neural network to approximate it via DSM rather than computing it analytically.

  So the hierarchy is:

  $$\text{trained two-branch} \leq \underbrace{\text{oracle } q_\sigma\text{-Rao}}_{\text{= perfect two-branch}} \leq \text{t-GLRT (for large } \theta\text{)}$$

  The oracle $q_\sigma$-Rao tells you how much room the neural model has to improve with more training/capacity, not how much room is left before hitting a fundamental detection
  limit.
  


  Estimating $\nu$ — two standard approaches:

  Method of moments (simple, no optimization): For $t_\nu(\mathbf{0},\Sigma)$, $\mathbb{E}[\mathbf{w}^\top\Sigma^{-1}\mathbf{w}/d] = \nu/(\nu-2)$. Compute the mean Mahalanobis
  distance under $\hat\Sigma$ (e.g., Tyler's estimate), then:

  $$\hat\nu = \frac{2\hat{m}}{\hat{m}-1}, \qquad \hat{m} = \frac{1}{Md}\sum_i \mathbf{w}_i^\top\hat\Sigma^{-1}\mathbf{w}_i$$

  This is closed-form and fast. Requires $\hat{m} > 1$ (i.e., $\nu > 2$).

  Profile MLE (more accurate): Given $\hat\Sigma$, solve a scalar equation involving the digamma function $\psi$ numerically with Brent's method. More accurate for small $M$.

  For our experiment $\nu = 3$ is known (we chose the model). But if you want to include a fully blind practical detector that estimates both $\Sigma$ and $\nu$, the natural choice
   is:
  1. Tyler estimate for $\hat\Sigma$ (shape, no $\nu$ needed)
  2. Method-of-moments $\hat\nu$ from the Mahalanobis distances under $\hat\Sigma$
  3. Plug both into the t-Rao and t-GLRT formulas

  Both approaches are documented in Section 7 of student_t_additive_experiment.md. Want me to add a fully blind variant once the current run finishes?
_____
● This is a sharp observation and the answer is genuinely interesting. Let me work through it.
  
  Why oracle q_σ-Rao = oracle t-Rao at large θ

  At large A (which occurs at large θ), the posterior τ|y concentrates at τ* ≈ (A+ν)/(d+ν+2). For large τ, the σ²I term becomes negligible relative to τΣ, so:
  
  $$\mathbb{E}_{\tau|y}\left[\frac{1}{\tau d_i + \sigma^2}\right] \approx \frac{1}{d_i} \cdot \mathbb{E}\left[\frac{1}{\tau}\bigg|y\right] \approx \frac{1}{d_i} \cdot
  \frac{\nu+d}{\nu+A}$$

  Plugging into the oracle q_σ-Rao formula: $s^\top \nabla \log q_\sigma(y) \approx -\frac{\nu+d}{\nu+A} \cdot B = -w(A) \cdot B$ — exactly the t-Rao. The Gaussian convolution
  becomes irrelevant at large amplitude because σ²I ≪ τΣ. So oracle q_σ-Rao and oracle t-Rao suffer the same suppression at large θ. This is why they coincide in the large-θ
  results.

  ---
  Why the two-branch is NOT converging to the oracle q_σ-Rao

  The true score $\nabla \log q_\sigma(y) = -\mathbb{E}_{\tau|y}[(\tau\Sigma + \sigma^2 I)^{-1}], y$ is a matrix-valued function of y — in the eigenbasis of Σ, each eigendirection
  i gets its own weight $\mathbb{E}[1/(\tau d_i + \sigma^2)|y]$.

  The two-branch is constrained to the form $\psi(y) = u(|Ay|^2) \cdot Qy$, which applies a single scalar to all eigendirections. This function class cannot represent the oracle
  q_σ-Rao for non-isotropic Σ (and your Σ is very non-isotropic — AR(1), ρ=0.9).

  What DSM actually finds is the L2-projection of $\nabla \log q_\sigma$ onto the scalar-weight function class:
  $$u^* = \arg\min_u; \mathbb{E}{y \sim q\sigma}\left[|u(|Ay|^2), Qy - \nabla \log q_\sigma(y)|^2\right]$$

  This is a well-defined but genuinely different function. It's the best scalar approximation to the true (direction-dependent) score, trained from noise samples in the moderate-A
  regime, then extrapolated to large A at test time.

  Why this projection beats the oracle q_σ-Rao for detection

  The oracle q_σ-Rao applies stronger down-weighting to large-eigenvalue eigendirections. For your steering vector s = 1/√d (all-ones normalized), s is aligned with the principal
  eigenvector of Σ (the DC component for AR(1)) — the direction with the largest eigenvalue and therefore the most suppressed direction in the oracle score.

  The two-branch's scalar u*(d) doesn't add this direction-dependent suppression. Its detection statistic is just $u^*(d) \cdot B$, where B = s⊤Qy already has the $1/d_i$ weighting
  from Q baked in. The extra per-direction suppression from the matrix W(y) in the oracle score actively hurts detection when s aligns with the most suppressed direction.

  ---
  Summary: what is ψ actually learning?

  The two-branch learns the optimal scalar weight function within its architecture class — not the score of any named distribution, and not something that converges to the oracle
  q_σ-Rao with more data or capacity. The architecture misspecification is not incidental; it's the reason it outperforms both the oracle t-Rao and the oracle q_σ-Rao. The scalar
  constraint strips out the direction-dependent suppression that hurts both oracle tests at large θ, while the learned u*(d) still adapts to the overall outlierness of y.

  The correct hierarchy is:

  $$\text{trained two-branch} \lesssim \underbrace{\text{oracle }u^*\text{-Rao}}_{\text{best scalar-weight test}} \leq \text{t-GLRT}$$

  where the "oracle u*-Rao" is the scalar-weight test with the theoretically optimal $u^*(d)$, which is what a perfectly trained two-branch would converge to — and this is neither
  the oracle q_σ-Rao nor any classical test.
_________



● Yes, it's just the likelihood ratio — and it's closed form.
  
  Under $H_0$: $\mathbf{y} \sim t_\nu(\mathbf{0}, \Sigma)$ and under $H_1$: $\mathbf{y} \sim t_\nu(\theta\mathbf{s}, \Sigma)$. The log-LRT is:

  $$T_{NP}(\theta) = \frac{\nu+d}{2}\log\frac{\nu + A}{\nu + (\mathbf{y} - \theta\mathbf{s})^\top Q(\mathbf{y} - \theta\mathbf{s})} = \frac{\nu+d}{2}\log\frac{\nu + A}{\nu + A -
  2\theta B + \theta^2 C}$$

  This requires knowing $\theta$ exactly. By the NP lemma it's UMP for this simple vs simple problem — highest $P_d$ at every $P_{fa}$, so highest AUC.

  Relationship to the GLRT: completing the square gives $A - 2\theta B + \theta^2 C = (\theta - B/C)^2 C + A - B^2/C \geq A - B^2/C$, so the NP denominator is always $\geq$ the
  GLRT denominator, meaning $T_{NP}(\theta_{\text{true}}) \leq T_{GLRT}$ as raw numbers. The GLRT inflates its statistic by optimizing over $\theta$ — but they rank differently and
  the NP test wins in AUC.

  At large $\theta$: under $H_1$, $(\mathbf{y} - \theta\mathbf{s})^\top Q(\mathbf{y} - \theta\mathbf{s}) = \mathbf{w}^\top Q\mathbf{w} = A_w$, so:
  $$T_{NP} = \frac{\nu+d}{2}\log\frac{\nu + A}{\nu + A_w} \approx \frac{\nu+d}{2}\log\frac{\nu + \theta^2 C + A_w}{\nu + A_w}$$
  This grows with $\theta$ — no suppression, because the test measures distance from $\theta\mathbf{s}$, not from the origin.

  This is a proper oracle upper bound that we're currently missing from the experiment. Want me to add it?

