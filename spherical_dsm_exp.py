"""
spherical_dsm_exp.py
--------------------
Theoretical experiment: DSM on directional (spherical) data.

Setup:
  u ~ N(0, Sigma)  in R^d,  x = u / ||u||  (projection onto unit sphere)
  y = x + sigma*eps,  eps ~ N(0, I)
  Model: psi(y) = Ay / ||Ay||   (A learnable, initialized as Identity)
  Loss:  E[ || psi(y) - (x - y) / sigma^2 ||^2 ]

The DSM target (x-y)/sigma^2 = -eps/sigma is the score of the Gaussian
kernel p(y|x) = N(y; x, sigma^2 I).  By Tweedie's formula the minimiser
over all measurable psi is nabla log q_sigma(y), and the direction of this
score involves the precision Q = Sigma^{-1}.

Hypothesis: the learned P = A^T A is proportional to Q = Sigma^{-1},
matching (up to scale) the inverse of Tyler's M-estimator on x.
"""

import numpy as np
import torch
import torch.utils.data
from baselines.classical import tyler_estimator

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
D         = 5
N         = 10_000
SIGMA     = 0.1        # noise level for y = x + sigma*eps
N_EPOCHS  = 300
LR        = 5e-3
BATCH_SIZE = 512

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# ---------------------------------------------------------------------------
# Ground-truth covariance Sigma and precision Q
# Use a random positive-definite Sigma so the result is non-trivial
# ---------------------------------------------------------------------------
rng = np.random.default_rng(SEED)
M_rand   = rng.standard_normal((D, D))
Sigma    = M_rand @ M_rand.T / D + np.eye(D)   # Sigma = covariance (PD)
Q_true   = np.linalg.inv(Sigma)                # Q     = precision

print(f"d={D}  N={N}  sigma={SIGMA}  epochs={N_EPOCHS}")
print(f"\nGround-truth Sigma (covariance):\n{np.round(Sigma, 3)}")
print(f"\nGround-truth Q = Sigma^-1 (precision):\n{np.round(Q_true, 3)}")

# ---------------------------------------------------------------------------
# Generate directional data  x = u / ||u||
# ---------------------------------------------------------------------------
u = rng.multivariate_normal(np.zeros(D), Sigma, size=N)   # (N, D)
x = u / np.linalg.norm(u, axis=1, keepdims=True)          # (N, D), ||x||=1
print(f"\nData: {N} samples projected onto S^{D-1}")
print(f"  mean ||x||: {np.linalg.norm(x, axis=1).mean():.6f}  (should be 1.0)")

# ---------------------------------------------------------------------------
# Tyler's M-estimator on directional data
# Tyler estimates the shape matrix Sigma_tyler ~ Sigma (up to scale)
# Its inverse Q_tyler = Sigma_tyler^{-1} ~ Q_true (up to scale)
# ---------------------------------------------------------------------------
Sigma_tyler = tyler_estimator(x)
Q_tyler     = np.linalg.inv(Sigma_tyler)
print(f"\nTyler scatter estimate (Sigma_tyler, trace-normalised):\n{np.round(Sigma_tyler, 3)}")

# ---------------------------------------------------------------------------
# DSM training
# ---------------------------------------------------------------------------
x_t   = torch.tensor(x, dtype=torch.float32)
eps_t = torch.randn_like(x_t) * SIGMA
y_t   = x_t + eps_t                              # noisy observations
tgt_t = (x_t - y_t) / (SIGMA ** 2)              # DSM target: -eps/sigma

ds     = torch.utils.data.TensorDataset(y_t, tgt_t)
loader = torch.utils.data.DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

# Learnable A, initialised to identity
A         = torch.nn.Parameter(torch.eye(D, dtype=torch.float32))
optimizer = torch.optim.Adam([A], lr=LR)

print(f"\nTraining psi(y) = Ay/||Ay||  with DSM MSE loss ...")
for epoch in range(1, N_EPOCHS + 1):
    epoch_loss = 0.0
    for y_b, t_b in loader:
        optimizer.zero_grad()
        Ay  = y_b @ A.T                                    # (B, D)
        psi = Ay / (Ay.norm(dim=1, keepdim=True) + 1e-8)  # (B, D)  unit vectors
        loss = ((psi - t_b) ** 2).mean()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    if epoch % 50 == 0 or epoch == 1:
        print(f"  Epoch {epoch:>3}/{N_EPOCHS}  loss = {epoch_loss / len(loader):.6f}")

# Learned precision  P = A^T A
A_np     = A.detach().numpy()
P_learned = A_np.T @ A_np

print(f"\nLearned P = A^T A:\n{np.round(P_learned, 3)}")

# ---------------------------------------------------------------------------
# Comparison utilities
# ---------------------------------------------------------------------------

def normalise(M: np.ndarray) -> np.ndarray:
    """Divide by Frobenius norm."""
    return M / np.linalg.norm(M, 'fro')

def frob_dist(M1: np.ndarray, M2: np.ndarray) -> float:
    """Frobenius distance after normalising both matrices to unit norm."""
    return float(np.linalg.norm(normalise(M1) - normalise(M2), 'fro'))

def cosine_sim(M1: np.ndarray, M2: np.ndarray) -> float:
    """Cosine similarity between vectorised (flattened) matrices."""
    v1, v2 = M1.flatten(), M2.flatten()
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
comparisons = [
    ("P_learned", "Q_tyler^{-1}",  P_learned,  Q_tyler),
    ("P_learned", "Q_true",         P_learned,  Q_true),
    ("Q_tyler^{-1}", "Q_true",      Q_tyler,    Q_true),
    ("Sigma_tyler",  "Sigma_true",   Sigma_tyler, Sigma),
]

print(f"\n{'='*60}")
print(f"Comparison  (frob_dist = ||M1/||M1|| - M2/||M2||||_F,  lower is better)")
print(f"            (cosine_sim = <vec(M1), vec(M2)> / norms,  higher is better)")
print(f"{'='*60}")
print(f"{'Pair':<36}  {'frob_dist':>9}  {'cosine_sim':>10}")
print("-" * 60)
for name1, name2, M1, M2 in comparisons:
    fd = frob_dist(M1, M2)
    cs = cosine_sim(M1, M2)
    print(f"{name1 + ' vs ' + name2:<36}  {fd:>9.4f}  {cs:>10.4f}")

print(f"\nInterpretation:")
print(f"  cosine_sim → 1.0  means the matrices are proportional (same shape)")
print(f"  frob_dist  → 0.0  means the matrices are identical after normalisation")
print(f"\n  If P_learned ≈ Q_tyler^{{-1}}: DSM recovers the precision structure")
print(f"  from directional data without access to the covariance or Q_true.")
