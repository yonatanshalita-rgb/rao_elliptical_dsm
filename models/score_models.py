"""
Score function models psi(y) for the DSM-based detector.

All three models produce an estimate of the score ∇_y log p(y).
Under Gaussian noise N(0,Q), the true score is -Q^{-1} y.

NSD constraint
--------------
The linear layer in every model is parameterised as W = -A^T A, where A is
the learnable (n x n) matrix.  This hard-constrains W to be NSD (negative
semi-definite) for all values of A:

    y^T W y = -y^T A^T A y = -||Ay||^2 <= 0

Benefits:
- The Mahalanobis proxy d = -y^T W y = ||Ay||^2 >= 0 always, without clamping.
- The u-branch in TwoBranchScore receives meaningful positive input from
  the very first epoch, removing the dead-gradient problem at initialisation.

Initialisation: A = (1/sqrt(n)) * I, so W = -(1/n) * I at the start --
a gentle isotropic negative-definite initialisation consistent with the
previous (1/n)*I initialisation of the old nn.Linear weight.

Efficient batch computation for y (..., n):
    z = y @ A.T        -- shape (..., n), row-vector form of Ay
    v = -(z @ A)       -- shape (..., n), row-vector form of -A^T Ay = Wy
    d = (z*z).sum(-1)  -- shape (...),   = ||Ay||^2 >= 0

Model 1 — LinearScore (MSE loss)
    psi(y) = W y = -A^T A y
    Trained with standard DSM MSE loss.

Model 2 — LinearScore (Huber loss)
    Identical architecture to Model 1.
    Trained with Huber DSM loss.

Model 3 — TwoBranchScore
    v(y) = W y = -A^T A y                linear branch
    d(y) = -y^T v = ||Ay||^2 >= 0        Mahalanobis proxy  (no clamp needed)
    u(y) = softplus(MLP(d(y)))            scalar branch, u > 0
    psi(y) = u(y) * v(y)
"""

import torch
import torch.nn as nn
from torch import Tensor


# ---------------------------------------------------------------------------
# Model 1 & 2: Linear score (architecture is identical; loss differs at train)
# ---------------------------------------------------------------------------

class LinearScore(nn.Module):
    """
    psi(y) = W y = -A^T A y,   A in R^{n x n} learnable.

    W is NSD by construction.  Initialised so A = (1/sqrt(n))*I,
    giving W = -(1/n)*I at the start.
    """

    def __init__(self, n: int):
        super().__init__()
        self.A = nn.Parameter(torch.eye(n) / n ** 0.5)

    def forward(self, y: Tensor) -> Tensor:
        """y: (..., n)  ->  psi: (..., n)"""
        z = y @ self.A.T      # (..., n)  =  Ay  (row-vector form)
        return -(z @ self.A)  # (..., n)  =  -A^T Ay = Wy


# ---------------------------------------------------------------------------
# Model 3: Two-branch score
# ---------------------------------------------------------------------------

class TwoBranchScore(nn.Module):
    """
    v-branch:  v = W y = -A^T A y        (NSD linear map)
    proxy:     d = -y^T v = ||Ay||^2     (always >= 0, no clamp needed)
    u-branch:  u = softplus(MLP(d))      (scalar, u > 0)
    output:    psi = u * v

    The u-MLP learns an arbitrary positive function of the estimated
    Mahalanobis distance.
    """

    def __init__(self, n: int, u_hidden_dims: list[int] = [32, 32]):
        super().__init__()

        # --- v-branch: NSD linear map W = -A^T A ---
        self.A = nn.Parameter(torch.eye(n) / n ** 0.5)

        # --- u-branch ---
        layers = []
        in_dim = 1
        for h in u_hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.Tanh()]
            in_dim = h
        layers += [nn.Linear(in_dim, 1)]
        self.u_mlp = nn.Sequential(*layers)

        # initialise u-MLP to output ≈ log 2 ≈ 0.693 at start (softplus(0))
        nn.init.zeros_(self.u_mlp[-1].weight)
        nn.init.zeros_(self.u_mlp[-1].bias)

    def forward(self, y: Tensor) -> Tensor:
        """y: (..., n)  ->  psi: (..., n)"""
        z   = y @ self.A.T                              # (..., n)  Ay
        v   = -(z @ self.A)                             # (..., n)  -A^T Ay = Wy
        d   = (z * z).sum(dim=-1, keepdim=True)         # (..., 1)  ||Ay||^2 >= 0
        u   = nn.functional.softplus(self.u_mlp(d))     # (..., 1)  > 0
        return u * v                                    # (..., n)

    def get_weight_function(self, d_values: Tensor) -> Tensor:
        """
        Evaluate u(d) for a range of scalar d values.
        d_values: (K,)  ->  u_values: (K,)
        """
        d = d_values.unsqueeze(-1)                      # (K, 1)  d >= 0 always
        return nn.functional.softplus(self.u_mlp(d)).squeeze(-1)


# ---------------------------------------------------------------------------
# Model 4: Fixed-weight score  (known scalar, learned covariance)
# ---------------------------------------------------------------------------

class FixedWeightScore(nn.Module):
    """
    psi(y) = w_t(||Ay||^2) * (-AA^T y)

    The scalar is FIXED to the theoretical t-weight:
        w_t(d) = (nu + n) / (nu + d)
    Only the linear map A (i.e. the covariance estimate) is learned via DSM.

    This is the "DSM-estimated t-Rao" baseline: it learns Q through score
    matching but uses the known nu rather than a data-driven scalar.
    Requires nu to be known at construction time.
    """

    def __init__(self, n: int, nu: float):
        super().__init__()
        self.A  = nn.Parameter(torch.eye(n) / n ** 0.5)
        self.nu = nu
        self.n  = n

    def forward(self, y: Tensor) -> Tensor:
        """y: (..., n)  ->  psi: (..., n)"""
        z = y @ self.A.T                                # (..., n)  Ay
        v = -(z @ self.A)                               # (..., n)  -A^T Ay
        d = (z * z).sum(dim=-1, keepdim=True)           # (..., 1)  ||Ay||^2
        w = (self.nu + self.n) / (self.nu + d)          # (..., 1)  fixed t-weight
        return w * v                                     # (..., n)


# ---------------------------------------------------------------------------
# Model 5: Unconstrained MLP score
# ---------------------------------------------------------------------------

class MLPScore(nn.Module):
    """
    Unconstrained MLP: psi: R^n -> R^n with no structural constraints.

    Used as a control against the two-branch: a sufficiently wide MLP trained
    with DSM should converge to the full direction-dependent score of q_sigma,
    not a scalar-weighted approximation.  If its detection AUC at large theta
    matches the oracle q_sigma-Rao (and not the two-branch), that confirms the
    two-branch's advantage comes from its scalar-weight architecture constraint.
    """

    def __init__(self, n: int, hidden_dims: list[int] = [128, 128, 128]):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = n
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.SiLU()]
            in_dim = h
        layers.append(nn.Linear(in_dim, n))
        self.net = nn.Sequential(*layers)

    def forward(self, y: Tensor) -> Tensor:
        """y: (..., n)  ->  psi: (..., n)"""
        return self.net(y)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_model(model_type: str, n: int, **kwargs) -> nn.Module:
    """
    model_type: 'linear_mse' | 'linear_huber' | 'two_branch' | 'mlp_score'
    Both linear variants return a LinearScore; the loss choice is
    handled in the training module.
    """
    if model_type in ("linear_mse", "linear_huber"):
        return LinearScore(n, **kwargs)
    elif model_type == "two_branch":
        return TwoBranchScore(n, **kwargs)
    elif model_type == "fixed_weight":
        return FixedWeightScore(n, **kwargs)
    elif model_type == "mlp_score":
        return MLPScore(n, **kwargs)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
