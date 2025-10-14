import os
import math
import numpy as np
import pandas as pd
from typing import Literal, Tuple

import torch
from torch import nn
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.models.resnet import resnet18, ResNet18_Weights

from src.architecture.CNN_toolkit import D4_eq_weight, center_crop_to, gaussian_weight_2d
from src.architecture.CNN_module import R180Inv_Conv2d, D4Inv_Conv2d, ConvGELU, BiasFreeMLP, ConvGELU_Res, asinh_norm_torch

# ----------------------------
# Basic modules:
# ----------------------------


class Forward8_base(nn.Module):
    """
    Base class for D4-equivariant forward models.

    Input:
        x: [B, in_dim, H, W]

    Output:
        [B, 2]  -> (e1, e2), D4-equivariant

    This class provides:
      - D4 group transformations (rotations/reflections)
      - D4 inverse mapping
      - shared sign buffers for (e1, e2)
      - trunk feature extraction helper
    """

    def __init__(
        self,
    ):
        super().__init__()

        # --- D4 group signs for e1/e2 ---
        # Transform order: [r0, r90, r180, r270, r0·sx, r90·sx, r180·sx, r270·sx]
        self.register_buffer(
            "signs_e1",
            torch.tensor([+1, -1, +1, -1, +1, -1, +1, -1], dtype=torch.float32),
        )
        self.register_buffer(
            "signs_e2",
            torch.tensor([+1, -1, +1, -1, -1, +1, -1, +1], dtype=torch.float32),
        )

    # ============================================================
    # D4 spatial transformations
    # ============================================================

    @staticmethod
    def _rot90_k(x: torch.Tensor, k: int) -> torch.Tensor:
        """Rotate feature map by 90° * k."""
        return torch.rot90(x, k % 4, dims=(-2, -1))

    @staticmethod
    def _reflect_x(x: torch.Tensor) -> torch.Tensor:
        """Reflect feature map along the x-axis (vertical flip)."""
        return torch.flip(x, dims=[-2])

    def _d4_augment(self, x: torch.Tensor) -> list[torch.Tensor]:
        """
        Generate D4 orbit:
        [r0, r90, r180, r270, r0·sx, r90·sx, r180·sx, r270·sx]
        Returns a list of 8 transformed tensors, each of shape [B, C, H, W].
        """
        xs = [self._rot90_k(x, k) for k in range(4)]
        xs += [self._reflect_x(self._rot90_k(x, k)) for k in range(4)]
        return xs

    def _d4_inverse(self, x: torch.Tensor, idx: int) -> torch.Tensor:
        """
        Apply the inverse of the D4 transform with index `idx`.
        Inverse order ensures the feature maps are restored to the canonical frame.
        """
        if idx < 4:  # r^k
            return self._rot90_k(x, -idx)
        else:  # r^k * sx
            k = idx - 4
            return self._rot90_k(self._reflect_x(x), -k)

    # ============================================================
    # Trunk feature extraction
    # ============================================================

    def _trunk_feature(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extracts the feature map from the CNN trunk and crops
        to match the original spatial extent (minus convolutional shrinkage).
        """
        H = x.size(-2) - 6
        W = x.size(-1) - 6

        # Apply zero padding before convolution blocks
        #x = self.inpad(x)

        # Sequentially apply Conv blocks
        for blk in self.blocks:
            x = blk(x)

        # Crop back to the central region
        h_, w_ = x.size(-2), x.size(-1)
        x = x[
            :,
            :,
            (h_ // 2 - H // 2):(h_ // 2 + H // 2),
            (w_ // 2 - W // 2):(w_ // 2 + W // 2),
        ]
        return x

# ------------------------------------------------------------
# Child model: SmoothCNN_GeLU
# ------------------------------------------------------------
class Forward8_fixW_CNN(Forward8_base):
    """
    D4-equivariant CNN model with Gaussian smoothing and weighted pooling.
    Input:  [B, in_dim, H, W]
    Output: [B, 2]  (e1, e2)
    """

    def __init__(
        self,
        in_dim=1,
        base_channels=32,
        head_hidden=128,
        out_dim=2,
        num_layers=5,
        res_factor=0.1,
    ):
        super().__init__()
        assert out_dim == 2, "This D4-equivariant head assumes two shape components."
        C = base_channels
        #self.inpad = nn.ConstantPad2d(10, 0.0)

        # --- CNN trunk ---
        self.blocks = nn.ModuleList(
            [ConvGELU_Res(in_dim, C, res_factor)] + [ConvGELU_Res(C, C, res_factor) for _ in range(max(0, num_layers - 1))]
        )

        # --- Heads ---
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.head_e1 = BiasFreeMLP(C, hidden=head_hidden)
        self.head_e2 = BiasFreeMLP(C, hidden=head_hidden)

    # ============================================================
    # Weighted Global Average Pool
    # ============================================================

    @staticmethod
    def _weighted_gap(feat: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """
        Weighted global average pooling.
        feat: [B,C,H,W]
        w:    [B,1,H,W]
        """
        ws = w.sum(dim=(-2, -1), keepdim=True)
        v = (feat * w).sum(dim=(-2, -1), keepdim=True) / ws
        return v.squeeze(-1).squeeze(-1)  # [B,C]

    # ============================================================
    # Forward pass
    # ============================================================

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # --- Normalization ---
        w = gaussian_weight_2d((x.size(-2), x.size(-1))).to(x)
        m = (x*w).sum(dim=(1, 2, 3), keepdim=True)*100 # mean brightness
        norm = 0.5 + F.softplus(m - 0.5, beta=10.0, threshold=20)
        x = x / norm
        #x = asinh_norm_torch(x)

        # --- D4 orbit and inverse aggregation ---
        xs = self._d4_augment(x)
        feats = []
        for idx, xi in enumerate(xs):
            f = self._trunk_feature(xi)
            f_inv = self._d4_inverse(f, idx)
            feats.append(f_inv)
        feats = torch.stack(feats, dim=0)  # [8,B,C,H,W]

        # --- Sign-weighted mean for e1/e2 ---
        s1 = self.signs_e1.view(8, 1, 1, 1, 1).to(feats)
        s2 = self.signs_e2.view(8, 1, 1, 1, 1).to(feats)

        self.f_mean_e1 = (feats * s1).mean(dim=0)  # [B,C,H,W]
        self.f_mean_e2 = (feats * s2).mean(dim=0)  # [B,C,H,W]

        # --- Align weight map to feature map size (center crop) ---
        Hf, Wf = self.f_mean_e1.shape[-2:]
        w_f = gaussian_weight_2d((Hf, Wf), 16).to(self.f_mean_e1)  # [Hf,Wf]

        # --- Weighted pooling ---
        v1 = self._weighted_gap(self.f_mean_e1, w_f)
        v2 = self._weighted_gap(self.f_mean_e2, w_f)

        # --- Heads ---
        e1 = self.head_e1(v1)
        e2 = self.head_e2(v2)
        e = torch.cat([e1, e2], dim=-1)  # [B,2]

        return e