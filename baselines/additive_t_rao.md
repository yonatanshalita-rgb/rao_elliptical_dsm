i # Rao (Score) Test — Additive Signal Model

## Model

The observation is:

```
y_i = w_i + theta * s
```

- `s`     : known signal vector (d x 1)
- `theta` : unknown scalar amplitude  (H0: theta = 0, H1: theta > 0)
- `w_i`   : i.i.d. noise from p_w, with mean mu and covariance Sigma

Work with the centered observation `z = y - mu` and let `Q = Sigma^{-1}`.

**Per-observation scalars** (same as the replacement model):

```
A = z'Qz        (Mahalanobis norm of z)
B = z'Qs        (match of z with signal)
C = s'Qs        (signal energy, fixed)
```

---

## Gaussian Background  (p_w = N(mu, Sigma))

**Score at theta = 0:**

```
a(y) = B
```

**Fisher information:**

```
I0 = C
```

**Rao statistic:**

```
T_Rao = B / sqrt(C)
```

This is the Q-weighted matched filter — a classical result.
For a Gaussian model linear in theta, Rao = Wald = GLRT, so there
is no gain from the more complex detectors.

---

## t-Distributed Background  (p_w = t_nu(mu, Sigma))

### Score at theta = 0

Define the adaptive weight (same form as in the replacement model):

```
w(A) = (nu + d) / (nu + A)
```

Large A means the observation is an outlier — w(A) downweights it.

The score is:

```
a_t(y) = w(A) * B
```

Compare with the replacement model score `d - w(A)*(A - B)`.
The additive score is simpler: just a reweighted matched filter.
The extra `d - w(A)*A` term in the replacement model comes from the
fact that the covariance itself shrinks with theta there ((1-theta)^2 * Sigma),
adding a "variance channel" to the score. In the additive model the
covariance is fixed, so that term vanishes.

### Fisher Information

The Fisher information matrix for the location parameter of t_nu(mu, Sigma) is:

```
F_mu = [(nu + d) / (nu + d + 2)] * Q
```

The scalar Fisher information for theta follows by projection onto s:

```
I0_t = s' F_mu s = [(nu + d) / (nu + d + 2)] * C
```

As nu -> inf this recovers the Gaussian result I0_t -> C.

### t-Rao Statistic

```
T_{Rao,t} = w(A) * B / sqrt( [(nu+d)/(nu+d+2)] * C )
```

---

## Side-by-Side Comparison

```
                  Replacement model       Additive model
                  -----------------       --------------
Gaussian score    d - A + B               B
Gaussian I0       2d + C                  C
t score           d - w(A)*(A - B)        w(A) * B
t I0              [2d*nu + (nu+d)*C]      [(nu+d)/(nu+d+2)] * C
                  / (nu+d+2)
```

The additive Fisher information depends only on C (the signal's SNR
in the Q-metric). The replacement model mixes in d from the variance
channel.

---

## Notes on Novelty

The additive model t-Rao is **not new**. Its components are standard:

- The score `w(A)*B` is a classical M-estimator applied to detection,
  present in compound-Gaussian radar literature (Gini, Greco, Farina;
  Pascal, Forster, Ovarlez) and traceable to Huber's robust statistics.
- The Fisher information formula for t_nu location is a textbook result
  in elliptical distribution theory.

By contrast, the Rao and t-Rao for the **replacement model** are not
previously published — the replacement score structure is genuinely novel.

The additive t-Rao is useful as a **baseline** in comparisons, and as
a sanity check (it reduces to the matched filter under Gaussian noise).
