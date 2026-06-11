"""
DSM training for all three score models.

Denoising Score Matching (DSM) loss
------------------------------------
For a noise distribution p(w), DSM minimises:

    L_DSM(psi) = E_{w} [ || psi(w) - score(w) ||^2 ]

where score(w) = ∇_w log p(w).

We use the implicit DSM objective (Vincent 2011), which avoids computing
the true score explicitly.  For a noising kernel q(w_tilde | w) with known
score ∇_{w_tilde} log q(w_tilde | w), the objective becomes:

    L_DSM(psi) = E_{w, epsilon} [ || psi(w + sigma*epsilon) + epsilon/sigma ||^2 ]

where epsilon ~ N(0, I) and sigma is a small noise level.

In practice we use the finite-difference / direct formulation:

    L_DSM(psi) = E_w [ || psi(w) ||^2 + 2 * div psi(w) ]

which requires computing the divergence.  Because our models are either
linear (div = trace(W)) or structured (two-branch), we use the
Hutchinson trace estimator for the divergence in general:

    div psi(w) ≈ v^T (dpsi/dw) v,   v ~ Rademacher

Loss variants
-------------
MSE (standard DSM):
    L = E [ || psi(w) + sigma^{-2} w_tilde - w) ||^2 ]   (noisy DSM)
    -- implemented via the perturbation approach below.

Huber DSM:
    Replace the squared norm with the Huber pseudo-norm element-wise:
    L_Huber = E [ sum_i  huber_delta( [psi(w)]_i - target_i ) ]
    This down-weights large residuals (outlier noise realisations).

Implementation note
-------------------
We use the *perturbation-based* DSM objective throughout, which is
more stable numerically:

    Given w, sample w_tilde = w + sigma * eps,  eps ~ N(0,I)
    Target score: t = -eps / sigma  (= ∇ log q(w_tilde | w))
    Loss: || psi(w_tilde) - t ||^2  (MSE)  or  Huber(psi(w_tilde) - t)
"""

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from typing import Literal


# ---------------------------------------------------------------------------
# DSM loss functions
# ---------------------------------------------------------------------------

def dsm_mse_loss(psi_net: nn.Module, w: Tensor, sigma: float = 0.1,
                 per_sample_clip: float | None = None) -> Tensor:
    """
    Perturbation-based DSM with MSE.
    w: (B, n)

    per_sample_clip: if set, scales each sample's loss contribution so its
    estimated gradient Frobenius norm does not exceed this value.  For a
    linear model psi=Wy the per-sample gradient norm is (2/n)*||r_i||*||w~_i||.
    Outliers (large ||w~||) are down-weighted; inliers are unaffected.
    """
    eps = torch.randn_like(w)
    w_tilde = w + sigma * eps
    target = -eps / sigma                       # (B, n)
    psi = psi_net(w_tilde)                      # (B, n)
    residual = psi - target                     # (B, n)

    if per_sample_clip is not None:
        per_sample_loss = (residual ** 2).mean(dim=-1)          # (B,)
        with torch.no_grad():
            n = w.shape[-1]
            grad_norm = (2.0 / n) * residual.norm(dim=-1) * w_tilde.norm(dim=-1)
            scale = (per_sample_clip / grad_norm.clamp(min=1e-8)).clamp(max=1.0)
        return (scale * per_sample_loss).mean()

    return (residual ** 2).mean()


def dsm_huber_loss(psi_net: nn.Module, w: Tensor, sigma: float = 0.1,
                   delta: float = 1.0,
                   per_sample_clip: float | None = None) -> Tensor:
    """
    Perturbation-based DSM with Huber loss.
    Huber(r) = 0.5 r^2            if |r| <= delta
             = delta(|r| - 0.5*delta)  otherwise
    delta controls the transition from quadratic to linear.

    per_sample_clip: same scheme as dsm_mse_loss.  For Huber the per-sample
    gradient norm is (1/n)*||huber_grad_i||*||w~_i|| where huber_grad clips
    each element to [-delta, delta].
    """
    eps = torch.randn_like(w)
    w_tilde = w + sigma * eps
    target = -eps / sigma
    psi = psi_net(w_tilde)

    if per_sample_clip is not None:
        per_sample_loss = nn.functional.huber_loss(
            psi, target, delta=delta, reduction="none"
        ).mean(dim=-1)                                          # (B,)
        with torch.no_grad():
            residual = psi - target
            huber_grad = torch.where(
                residual.abs() <= delta, residual, delta * residual.sign()
            )
            n = w.shape[-1]
            grad_norm = (1.0 / n) * huber_grad.norm(dim=-1) * w_tilde.norm(dim=-1)
            scale = (per_sample_clip / grad_norm.clamp(min=1e-8)).clamp(max=1.0)
        return (scale * per_sample_loss).mean()

    return nn.functional.huber_loss(psi, target, delta=delta, reduction="mean")


LOSS_FNS = {
    "linear_mse":        dsm_mse_loss,
    "linear_huber":      dsm_huber_loss,
    "two_branch":        dsm_mse_loss,
    "fixed_weight":      dsm_mse_loss,
    "mlp_score":         dsm_mse_loss,
    "mggd_constrained":  dsm_mse_loss,
}


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    model: nn.Module,
    model_type: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    n_epochs: int = 50,
    lr: float = 1e-3,
    sigma: float = 0.1,
    huber_delta: float = 1.0,
    device: str = "cpu",
    save_dir: Path | None = None,
    verbose: bool = True,
    warmup_epochs: int = 0,
    grad_clip: float | None = None,
    per_sample_clip: float | None = None,
) -> dict:
    """
    Train a score model with DSM.

    warmup_epochs: for TwoBranchScore, freeze the u-branch for this many epochs
                   so the linear W-branch first learns the spatial filter before
                   the scalar branch is allowed to learn Mahalanobis down-weighting.
    grad_clip: global gradient norm clipping via clip_grad_norm_.  None = disabled.
               For linear models, per_sample_clip is the preferred robustness
               mechanism; global clipping is not needed.
    per_sample_clip: if set, each sample's gradient contribution is scaled so
                     its estimated Frobenius norm does not exceed this threshold.
                     Passed to the loss function (dsm_mse_loss / dsm_huber_loss).

    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    loss_fn = LOSS_FNS[model_type]
    loss_kwargs = {"sigma": sigma}
    if model_type == "linear_huber":
        loss_kwargs["delta"] = huber_delta
    if per_sample_clip is not None:
        loss_kwargs["per_sample_clip"] = per_sample_clip

    has_u_branch = hasattr(model, "u_mlp")

    history = {"train_loss": [], "val_loss": []}
    best_val = float("inf")

    for epoch in range(1, n_epochs + 1):
        # --- branch warm-up: freeze / unfreeze u_mlp ---
        if has_u_branch:
            warming_up = epoch <= warmup_epochs
            for p in model.u_mlp.parameters():
                p.requires_grad = not warming_up
            if verbose and epoch == 1 and warmup_epochs > 0:
                print(f"[{model_type}] Warming up W-branch for {warmup_epochs} epochs "
                      f"(u-branch frozen)")
            if verbose and epoch == warmup_epochs + 1 and warmup_epochs > 0:
                print(f"[{model_type}] Epoch {epoch}: unfreezing u-branch")

        # --- train ---
        model.train()
        train_losses = []
        for batch in train_loader:
            w = batch["w"].to(device)
            optimizer.zero_grad()
            loss = loss_fn(model, w, **loss_kwargs)
            loss.backward()
            if grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()
            train_losses.append(loss.item())

        # --- validate ---
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                w = batch["w"].to(device)
                val_losses.append(loss_fn(model, w, **loss_kwargs).item())

        train_loss = np.mean(train_losses)
        val_loss   = np.mean(val_losses)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        scheduler.step()

        if verbose and epoch % 10 == 0:
            print(f"[{model_type}] Epoch {epoch:3d}/{n_epochs}  "
                  f"train={train_loss:.4f}  val={val_loss:.4f}")

        if save_dir is not None and val_loss < best_val:
            best_val = val_loss
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), save_dir / f"{model_type}_best.pt")

    return history
