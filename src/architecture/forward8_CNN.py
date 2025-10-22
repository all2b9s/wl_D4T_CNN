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
from src.architecture.CNN_module import  ConvGELU, BiasFreeMLP, ConvGELU_Res, ConvGELU_layernorm

# ----------------------------
# Basic modules:
# ----------------------------


class Forward8_base(nn.Module):
    """
    Optimized base for D4-equivariant models:
    - Stacked D4 orbit (one trunk pass)
    - Bulk inverse mapping
    - Pre-shaped sign buffers
    """

    def __init__(self):
        super().__init__()
        # Transform order: [r0, r90, r180, r270, r0·sx, r90·sx, r180·sx, r270·sx]
        self.register_buffer("signs_e1", torch.tensor([+1, -1, +1, -1, +1, -1, +1, -1], dtype=torch.float32))
        self.register_buffer("signs_e2", torch.tensor([+1, -1, +1, -1, -1, +1, -1, +1], dtype=torch.float32))

        # Pre-shaped for fast broadcast in [8,1,1,1,1]
        #self.register_buffer("signs_e1_5d", self.signs_e1.view(8, 1, 1, 1, 1))
        #self.register_buffer("signs_e2_5d", self.signs_e2.view(8, 1, 1, 1, 1))

    # ----- D4 transforms (vectorized) -----

    @staticmethod
    def _rot90_k(x: torch.Tensor, k: int) -> torch.Tensor:
        # NOTE: rot90+flip can create negative strides; we’ll call .contiguous() later once, globally.
        return torch.rot90(x, k % 4, dims=(-2, -1))

    @staticmethod
    def _reflect_x(x: torch.Tensor) -> torch.Tensor:
        return torch.flip(x, dims=[-2])

    def _d4_orbit_stack(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns stacked orbit as [8, B, C, H, W].
        We call .contiguous() once after stacking to avoid hidden copies in convs.
        """
        x0 = x
        x1 = self._rot90_k(x, 1)
        x2 = self._rot90_k(x, 2)
        x3 = self._rot90_k(x, 3)

        # reflections
        xr0 = self._reflect_x(x0)
        xr1 = self._reflect_x(x1)
        xr2 = self._reflect_x(x2)
        xr3 = self._reflect_x(x3)

        xs = torch.stack([x0, x1, x2, x3, xr0, xr1, xr2, xr3], dim=0)  # [8,B,C,H,W]
        return xs.contiguous()  # ensure positive strides for downstream convs

    def _d4_inverse_stack(self, f8: torch.Tensor) -> torch.Tensor:
        """
        Bulk inverse on stacked features.
        Input:  f8 [8, B, C, H, W] in transform order above
        Output: inv_f8 [8, B, C, H, W], each map restored to canonical frame.
        """
        assert f8.dim() == 5 and f8.size(0) == 8
        # Inverses: for r^k -> r^{-k}; for r^k*sx -> r^{-k}*sx
        g0, g1, g2, g3, g4, g5, g6, g7 = f8.unbind(dim=0)

        # rotate back
        g0i = self._rot90_k(g0, 0)
        g1i = self._rot90_k(g1, -1)
        g2i = self._rot90_k(g2, -2)
        g3i = self._rot90_k(g3, -3)

        # reflect then rotate back
        g4i = self._rot90_k(self._reflect_x(g4), 0)
        g5i = self._rot90_k(self._reflect_x(g5), -1)
        g6i = self._rot90_k(self._reflect_x(g6), -2)
        g7i = self._rot90_k(self._reflect_x(g7), -3)

        inv = torch.stack([g0i, g1i, g2i, g3i, g4i, g5i, g6i, g7i], dim=0).contiguous()
        return inv

    # ----- Trunk feature extraction (works on [N,C,H,W]) -----

    def _trunk_feature(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies conv blocks. If spatial shrink is constant (=6), we crop once.
        Works with x shaped [N,C,H,W] (N=B or 8B).
        """
        H = x.size(-2) 
        W = x.size(-1) 

        for blk in self.blocks:  # keep it simple; torch.compile will fuse this nicely
            x = blk(x)

        h_, w_ = x.size(-2), x.size(-1)
        x = x[:, :, (h_//2 - H//2):(h_//2 + H//2), (w_//2 - W//2):(w_//2 + W//2)]
        return x.contiguous()


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
        self._w_cache = {}

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


    def _get_cached_weight(self, hw, sigma, device, dtype):
        key = (hw[0], hw[1], sigma, device.type, str(dtype))
        w = self._w_cache.get(key)
        if w is None:
            w = gaussian_weight_2d(hw, sigma).to(device=device, dtype=dtype)  # [H,W]
            w = w.unsqueeze(0).unsqueeze(0)  # [1,1,H,W] for cheap broadcast
            self._w_cache[key] = w
        return self._w_cache[key]

    # ============================================================
    # Forward pass
    # ============================================================

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # --- Normalization ---
        w = self._get_cached_weight(x.shape[-2:], sigma=16, device=x.device, dtype=x.dtype)
        m = (x*w).sum(dim=(1, 2, 3), keepdim=True)*100 # mean brightness
        norm = 1 + F.softplus(m - 1, beta=10.0, threshold=20)
        x = x / norm
        #x = asinh_norm_torch(x)

        # --- D4 orbit and inverse aggregation ---
        xs = self._d4_orbit_stack(x)              # [8,B,C,H,W]
        B = x.size(0)
        x8 = xs.view(-1, xs.size(2), xs.size(3), xs.size(4))  # [8B,C,H,W]

        # 2) Single trunk pass
        f8 = self._trunk_feature(x8)              # [8B,C,Hf,Wf]
        f8 = f8.view(8, B, f8.size(1), f8.size(2), f8.size(3))  # [8,B,C,Hf,Wf]

        # 3) Bulk inverse back to canonical frame
        feats = self._d4_inverse_stack(f8)        # [8,B,C,Hf,Wf]

        # --- Sign-weighted mean for e1/e2 --- 
        s1 = self.signs_e1.view(8, 1, 1, 1, 1).to(feats) 
        s2 = self.signs_e2.view(8, 1, 1, 1, 1).to(feats)
        # --- Sign-weighted mean for e1/e2 ---
        self.f_mean_e1 = (feats * s1).mean(dim=0)  # [B,C,Hf,Wf]
        self.f_mean_e2 = (feats * s2).mean(dim=0)  # [B,C,Hf,Wf]

        # --- Align weight map to feature map size (center crop) ---
        Hf, Wf = self.f_mean_e1.shape[-2:]
        w_f = self._get_cached_weight((Hf, Wf), sigma=16, device=self.f_mean_e1.device, dtype=self.f_mean_e1.dtype) # [Hf,Wf]

        # --- Weighted pooling ---
        v1 = self._weighted_gap(self.f_mean_e1, w_f)
        v2 = self._weighted_gap(self.f_mean_e2, w_f)

        # --- Heads ---
        e1 = self.head_e1(v1)
        e2 = self.head_e2(v2)
        e = torch.cat([e1, e2], dim=-1)  # [B,2]
        return e
    
class Forward8_fixW_nores_CNN(Forward8_base):
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
    ):
        super().__init__()
        assert out_dim == 2, "This D4-equivariant head assumes two shape components."
        C = base_channels
        self.inpad = nn.ConstantPad2d(10, 0.0)
        # --- CNN trunk ---
        self.blocks = nn.ModuleList(
            [ConvGELU_layernorm(in_dim, C)] + [ConvGELU_layernorm(C, C) for _ in range(max(0, num_layers - 1))]
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

    def _trunk_feature(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extracts the feature map from the CNN trunk and crops
        to match the original spatial extent (minus convolutional shrinkage).
        """
        H = x.size(-2) - 6
        W = x.size(-1) - 6

        # Apply zero padding before convolution blocks
        x = self.inpad(x)

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
    # ==========================================
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
    
class SmoothCNN_GeLU_mocked(Forward8_base):
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
    ):
        super().__init__()
        assert out_dim == 2, "This D4-equivariant head assumes two shape components."
        C = base_channels
        self.inpad = nn.ConstantPad2d(10, 0.0)
        # --- CNN trunk ---
        self.blocks = nn.ModuleList(
            [ConvGELU(in_dim, C)] + [ConvGELU(C, C) for _ in range(max(0, num_layers - 1))]
        )
        self.smooth = T.GaussianBlur(kernel_size=5, sigma=0.7)
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


    def _trunk_feature(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extracts the feature map from the CNN trunk and crops
        to match the original spatial extent (minus convolutional shrinkage).
        """
        H = x.size(-2) - 6
        W = x.size(-1) - 6

        # Apply zero padding before convolution blocks
        x = self.inpad(x)

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
    # ============================================================
    # Forward pass
    # ============================================================

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # --- Normalization ---
        self.m   = x.sum(dim=(1, 2, 3), keepdim=True)         # [B,1,1,1]
        self.eps = x.std(dim=(1, 2, 3), keepdim=True)
        x = x / (self.m + self.eps.clamp(min=1e-6))          
        self.x = self.smooth(x)                                    # [B,in_dim,H,W]
        norm = torch.sqrt(self.x.pow(2).sum(dim=(-1, -2), keepdim=True) + 1e-6)  # [B,in_dim,1,1]
        w = self.x / norm
        w = w.mean(dim=1, keepdim=True)                 

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

        Hf, Wf = self.f_mean_e1.shape[-2], self.f_mean_e1.shape[-1]
        Hw, Ww = w.shape[-2], w.shape[-1]
        if (Hw != Hf) or (Ww != Wf):
            dh = (Hw - Hf) // 2
            dw = (Ww - Wf) // 2
            w = w[:, :, dh:dh + Hf, dw:dw + Wf]   # center crop


        # --- Weighted pooling ---
        v1 = self._weighted_gap(self.f_mean_e1, w)
        v2 = self._weighted_gap(self.f_mean_e2, w)

        # --- Heads ---
        e1 = self.head_e1(v1)
        e2 = self.head_e2(v2)
        e = torch.cat([e1, e2], dim=-1)  # [B,2]

        return e