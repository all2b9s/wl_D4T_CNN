"""
QuadGauss — Gaussian-weighted second-moment ellipticity measurement
====================================================================

A parameter-free, differentiable ``nn.Module`` that computes (e1, e2) from a
deconvolved galaxy image via Gaussian-weighted quadrupole moments.

Intended as a **classical baseline** to compare against CNN-based shape
measurement models.  The interface exactly mirrors the CNN models:

    Input:  [B, 1, H, W]   (the "m" / deconvolved component from Q-images)
    Output: [B, 2]          (e1, e2)

All operations are pure PyTorch → fully differentiable → compatible with
``shape_pixel_gradients()`` and the ``Calibrator`` response pipeline.

Formula
-------
.. math::

    w(x,y) &= \\exp\\!\\left(-\\frac{(x-c_x)^2 + (y-c_y)^2}{2\\sigma^2}\\right)  \\\\
    Q_{ij}  &= \\sum_{x,y} w(x,y)\\, I(x,y)\\, (x_i-c_i)(x_j-c_j)                \\\\
    e_1     &= \\frac{Q_{xx} - Q_{yy}}{Q_{xx} + Q_{yy} + \\epsilon}               \\\\
    e_2     &= \\frac{2\\,Q_{xy}}{Q_{xx} + Q_{yy} + \\epsilon}

Notes
-----
- No trainable parameters — the model is a fixed analytical estimator.
- No input normalisation — ellipticity is scale-invariant (ratio of moments).
- The Gaussian width *σ* is the **only** hyper-parameter; it controls the
  effective weight radius in pixels.
- By default the weight is centred at the array centre (no centroid iteration).

References
----------
- Kaiser, Squires & Broadhurst (1995), "A Method for Weak Lensing Observations"
- Hirata & Seljak (2003), "Shear calibration biases in weak-lensing surveys"
"""

import torch
from torch import nn

from src.architecture.CNN_toolkit import gaussian_weight_2d


class QuadGauss(nn.Module):
    """Gaussian-weighted second-moment ellipticity estimator.

    Uses ``gaussian_weight_2d`` (same weight function as the CNN pooling)
    to form the spatial weight, then computes second moments explicitly
    via :math:`\\sum I(x,y) \\cdot (x_i-c_i)(x_j-c_j)`.

    Parameters
    ----------
    sigma : float
        Gaussian weight width in *pixels*.
    eps : float
        Stabiliser added to the denominator to avoid division by zero.
    """

    def __init__(self, sigma: float = 4.0, eps: float = 1e-8):
        super().__init__()
        self.sigma = sigma
        self.eps = eps

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:  deconvolved image, shape ``[B, 1, H, W]``.

        Returns:
            ellipticity ``[B, 2]``  —  ``(e1, e2)``.
        """
        B, _, H, W = x.shape
        dtype = x.dtype
        device = x.device

        # ---- coordinate grid (centred, same convention as gaussian_weight_2d) ----
        yy = torch.arange(H, device=device, dtype=dtype) - (H - 1) / 2.0
        xx = torch.arange(W, device=device, dtype=dtype) - (W - 1) / 2.0
        dy, dx = torch.meshgrid(yy, xx, indexing="ij")       # [H, W] each

        # ---- Gaussian weight (reuse CNN_toolkit) ----
        w = gaussian_weight_2d((H, W), std=self.sigma,
                               device=device, dtype=dtype)    # [H, W]
        w = w.unsqueeze(0).unsqueeze(0)                       # [1, 1, H, W]

        # ---- weighted image ----
        xw = x * w                                            # [B, 1, H, W]

        # ---- second moments (scalar per image) ----
        Qxx = (xw * dx * dx).sum(dim=(-2, -1))               # [B, 1]
        Qyy = (xw * dy * dy).sum(dim=(-2, -1))               # [B, 1]
        Qxy = (xw * dx * dy).sum(dim=(-2, -1))               # [B, 1]

        # ---- ellipticity ----
        T = Qxx.squeeze(-1) + Qyy.squeeze(-1) + self.eps     # [B]
        e1 = (Qxx.squeeze(-1) - Qyy.squeeze(-1)) / T
        e2 = (2.0 * Qxy.squeeze(-1)) / T

        return torch.stack([e1, e2], dim=-1)                  # [B, 2]

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------
    def extra_repr(self) -> str:
        return f"sigma={self.sigma}, eps={self.eps}"


# ------------------------------------------------------------------
# Quick test (runs when executed directly)
# ------------------------------------------------------------------
if __name__ == "__main__":
    print("=== QuadGauss smoke test ===")
    m = QuadGauss(sigma=4.0)
    print(m)

    # synthetic Gaussian blob with known ellipticity
    H, W = 64, 64
    yy = torch.arange(H, dtype=torch.float32) - (H - 1) / 2
    xx = torch.arange(W, dtype=torch.float32) - (W - 1) / 2
    dy, dx = torch.meshgrid(yy, xx, indexing="ij")
    # elliptical Gaussian: sigma_x=6, sigma_y=4  →  e1≈(36-16)/(36+16)=0.385
    a, b = 6.0, 4.0
    blob = torch.exp(-0.5 * (dx ** 2 / a ** 2 + dy ** 2 / b ** 2))
    blob = blob.unsqueeze(0).unsqueeze(0)     # [1,1,H,W]

    with torch.no_grad():
        e = m(blob).squeeze(0)
    print(f"Input: elliptical Gaussian (σ_x={a}, σ_y={b})")
    print(f"Expected e1 ≈ {(a**2 - b**2)/(a**2 + b**2):.4f},  e2 ≈ 0.0")
    print(f"Measured:  e1 = {e[0].item():.4f},  e2 = {e[1].item():.4f}")

    # gradient check
    print("\n=== Gradient check ===")
    x = blob.clone().detach().requires_grad_(True)
    pred = m(x)
    g = torch.autograd.grad(pred.sum(), x)[0]
    print(f"pred shape: {pred.shape}")
    print(f"grad shape: {g.shape}")
    print(f"grad has NaN: {torch.isnan(g).any().item()}")
    print("OK — QuadGauss is differentiable.")
