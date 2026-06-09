"""
Quick oracle-only sweep — no training.
Find the SNR range where oracle t-Rao hits Pd ~ 0.7 @ Pfa=1%
with theta ~ U[snr_min, snr_max].
"""
import numpy as np
from data.generate import ar1_covariance, sample_mu_t
from eval.metrics import pd_at_pfa, compute_roc

NU       = 3.0
RHO      = 0.9
N_TEST   = 100_000
PFA      = 0.01
SEED     = 42

def theta_from_snr(snr_db, C):
    return np.sqrt(10 ** (snr_db / 10.0) / C)

def make_test(snr_min_db, snr_max_db, rng, nu=NU, d=16):
    Sigma = ar1_covariance(d, rho=RHO)
    Q     = np.linalg.inv(Sigma)
    L     = np.linalg.cholesky(Sigma)
    s     = np.ones(d) / np.sqrt(d)
    C     = float(s @ Q @ s)

    theta_min = theta_from_snr(snr_min_db, C) if snr_min_db is not None else 0.0
    theta_max = theta_from_snr(snr_max_db, C)

    mu    = sample_mu_t(N_TEST, df=nu, rng=rng)
    noise = mu[:, None] * (rng.standard_normal((N_TEST, d)) @ L.T)

    labels   = rng.integers(0, 2, size=N_TEST)
    theta_h1 = rng.uniform(theta_min, theta_max, size=N_TEST)
    theta    = np.where(labels == 1, theta_h1, 0.0)
    y        = noise + theta[:, None] * s[None, :]

    return y, labels, s, Q, C

def scores(y, s, Q, nu):
    d    = y.shape[1]
    A    = np.einsum("ni,ij,nj->n", y, Q, y)
    B    = y @ (Q @ s)
    C    = float(s @ Q @ s)
    amf  = B / np.sqrt(C)
    w_A  = (nu + d) / (nu + A)
    rao  = w_A * B / np.sqrt((nu + d) / (nu + d + 2) * C)
    glrt = np.where(B > 0, 0.5*(nu+d)*np.log((nu+A)/(nu+A-B**2/C)), 0.0)
    return amf, rao, glrt

rng = np.random.default_rng(SEED)

ranges = [
    (None, 10), (None, 15), (None, 20), (None, 25),
    (5,   15),  (5,   20),
    (8,   15),  (8,   20),
]

# --- SNR sweep for d = 16, 32, 64 ---
for d in [16, 32, 64]:
    print(f"\nSNR sweep  (nu={NU}, d={d})")
    print(f"{'SNR range':>16}  {'C':>7}  {'AMF Pd':>8}  {'t-Rao Pd':>10}  {'t-GLRT Pd':>11}")
    print("-" * 62)
    for (lo, hi) in ranges:
        y, labels, s, Q, C = make_test(lo, hi, rng, nu=NU, d=d)
        amf, rao, glrt = scores(y, s, Q, NU)
        lo_str = f"{lo}" if lo is not None else "0"
        pd_amf  = pd_at_pfa(amf,  labels, PFA)
        pd_rao  = pd_at_pfa(rao,  labels, PFA)
        pd_glrt = pd_at_pfa(glrt, labels, PFA)
        marker = " <---" if 0.65 <= pd_rao <= 0.80 else ""
        print(f"U[{lo_str},{hi}dB]{'':<6}  {C:7.4f}  {pd_amf:8.3f}  {pd_rao:10.3f}  {pd_glrt:11.3f}{marker}")
