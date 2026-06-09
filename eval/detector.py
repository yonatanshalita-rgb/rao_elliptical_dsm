"""
Neural detector: applies the paper's test statistic T_s using a trained score model.

The test statistic (from the paper) is:

    T_s(y) = -s^T (psi_hat(y) - z_bar) / sqrt(s^T C_hat_psi s)

where:
    psi_hat(y)  = trained score network evaluated at y
    z_bar       = (1/N) sum_i psi_hat(y_i)     (empirical mean of scores)
    C_hat_psi   = (1/N) sum_i psi_hat(y_i) psi_hat(y_i)^T - z_bar z_bar^T
                  (empirical covariance of scores)
    s           = target steering vector

The statistics z_bar and C_hat_psi are estimated from a held-out calibration
set (noise-only samples) to ensure CFAR behaviour.
"""

import torch
import torch.nn as nn
import numpy as np
from torch import Tensor


class NeuralDetector:
    """
    Wraps a trained score model and computes T_s(y).

    Usage:
        detector = NeuralDetector(model, s)
        detector.calibrate(noise_samples)   # estimate z_bar, C_hat_psi
        scores = detector.score(y)          # compute T_s for test observations
    """

    def __init__(self, model: nn.Module, s: np.ndarray, device: str = "cpu"):
        self.model = model.to(device).eval()
        self.s = torch.tensor(s, dtype=torch.float32, device=device)
        self.device = device
        self.z_bar = None
        self.denom = None          # sqrt(s^T C_hat_psi s)

    @torch.no_grad()
    def calibrate(self, noise_samples: Tensor | np.ndarray,
                  batch_size: int = 1024):
        """
        Estimate z_bar and C_hat_psi from noise-only samples.
        noise_samples: (M, n)
        """
        if isinstance(noise_samples, np.ndarray):
            noise_samples = torch.tensor(noise_samples, dtype=torch.float32)

        psi_list = []
        for i in range(0, len(noise_samples), batch_size):
            batch = noise_samples[i:i+batch_size].to(self.device)
            psi_list.append(self.model(batch).cpu())
        psi_all = torch.cat(psi_list, dim=0)                  # (M, n)

        z_bar = psi_all.mean(dim=0)                            # (n,)
        psi_centered = psi_all - z_bar.unsqueeze(0)
        C_psi = (psi_centered.T @ psi_centered) / len(psi_all)  # (n, n)

        # add small ridge for numerical stability
        n = C_psi.shape[0]
        C_psi += 1e-6 * torch.eye(n)

        self.z_bar = z_bar.to(self.device)
        C_psi = C_psi.to(self.device)
        s = self.s
        self.denom = torch.sqrt(s @ (C_psi @ s)).item()       # sqrt(s^T C_psi s)

    @torch.no_grad()
    def score(self, y: Tensor | np.ndarray, batch_size: int = 1024) -> np.ndarray:
        """
        Compute T_s(y) for test observations.
        y: (N, n)
        Returns: (N,) numpy array of test statistics
        """
        assert self.z_bar is not None, "Call calibrate() first."

        if isinstance(y, np.ndarray):
            y = torch.tensor(y, dtype=torch.float32)

        scores = []
        for i in range(0, len(y), batch_size):
            batch = y[i:i+batch_size].to(self.device)
            psi = self.model(batch)                            # (B, n)
            diff = psi - self.z_bar.unsqueeze(0)              # (B, n)
            # T_s = -s^T (psi(y) - z_bar) / denom
            t = -(diff @ self.s) / self.denom                  # (B,)
            scores.append(t.cpu().numpy())

        return np.concatenate(scores)
