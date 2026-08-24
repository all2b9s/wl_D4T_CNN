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

from src.architecture.CNN_toolkit import D4_eq_weight, center_crop_to, quad_gaussian_2d, gaussian_weight_2d
from src.architecture.CNN_module import R180Inv_Conv2d, D4Inv_Conv2d, ConvGELU, BiasFreeMLP, ResConvBNGELU, ConvGELU_Res
#######################################################################################################
# CNN models:
#######################################################################################################


class SmoothCNN_GeLU(nn.Module):
    """
      input:  [B, in_dim, H, W]
            output: [B, 2]  (e1, e2), D4-equivariant
    """
    def __init__(self, 
                 in_dim=1, 
                 base_channels=32, 
                 head_hidden=128, 
                 out_dim=2, 
                 num_layers: int = 5,):
        super().__init__()
        assert out_dim == 2, "This D4-equivariant head assumes two shape components."
        C = base_channels
        self.inpad = nn.ConstantPad2d(10, 0.0)


        # ---------- NEW: depthwise Gaussian smoothing layer ----------
        self.smooth = T.GaussianBlur(kernel_size=5, sigma=0.7)

        # CNN backbone
        layers = [ConvGELU(in_dim, C)]
        layers += [ConvGELU(C, C) for _ in range(max(0, num_layers - 1))]
        self.blocks = nn.ModuleList(layers)

        self.gap = nn.AdaptiveAvgPool2d(1)  # -> [B, C, 1, 1]

        self.head_e1 = BiasFreeMLP(C, hidden=head_hidden)
        self.head_e2 = BiasFreeMLP(C, hidden=head_hidden)

        # 8 elements from _d4_augment r0, r90, r180, r270, r0·sx, r90·sx, r180·sx, r270·sx
        self.register_buffer('signs_e1', torch.tensor([+1, -1, +1, -1, +1, -1, +1, -1], dtype=torch.float32))
        self.register_buffer('signs_e2', torch.tensor([+1, -1, +1, -1, -1, +1, -1, +1], dtype=torch.float32))

    # ----------------- D4 group action -----------------
    @staticmethod
    def _rot90_k(x, k: int):
        return torch.rot90(x, k % 4, dims=(-2, -1))

    @staticmethod
    def _reflect_x(x):
        return torch.flip(x, dims=[-2]) 

    def _d4_augment(self, x):
        xs = []
        for k in range(4):
            xs.append(self._rot90_k(x, k))
        for k in range(4):
            xs.append(self._reflect_x(self._rot90_k(x, k)))
        return xs  # list of 8 tensors [B,C,H,W]

    def _d4_inverse(self, x, idx: int):
        if idx < 4:      # r^k
            return self._rot90_k(x, -idx)
        else:            # r^k * sx
            k = idx - 4
            return self._rot90_k(self._reflect_x(x), -k)

    # ----------------- trunk: extract feature maps -----------------
    def _trunk_feature(self, x):
        H = x.size(-2) - 6
        W = x.size(-1) - 6
        x = self.inpad(x)
        for blk in self.blocks:
            x = blk(x)
        h_, w_ = x.size(-2), x.size(-1)
        x = x[:, :,
              (h_ // 2 - H // 2):(h_ // 2 + H // 2),
              (w_ // 2 - W // 2):(w_ // 2 + W // 2)]  # [B,C,H,W]
        return x

    # ----------------- forward -----------------
    def forward(self, x):
        self.m   = x.sum(dim=(1, 2, 3), keepdim=True)         # [B,1,1,1]
        self.eps = x.std(dim=(1, 2, 3), keepdim=True)
        x = x / (self.m + self.eps.clamp(min=1e-6))          
        self.x = self.smooth(x)                                    # [B,in_dim,H,W]
        norm = torch.sqrt(self.x.pow(2).sum(dim=(-1, -2), keepdim=True) + 1e-6)  # [B,in_dim,1,1]
        w = self.x / norm                                                          # [B,in_dim,H,W]

        # If in_dim>1, compress to a single weight map; if in_dim==1 this is a no-op
        w = w.mean(dim=1, keepdim=True)                                            # [B,1,H,W]

        # 1) D4 Orbit with features as you already have...
        xs = self._d4_augment(x)
        feats = []
        for idx, xi in enumerate(xs):
            f = self._trunk_feature(xi)      # [B,C,H,W]
            f_inv = self._d4_inverse(f, idx) # [B,C,H,W]
            feats.append(f_inv)
        feats = torch.stack(feats, dim=0)     # [8,B,C,H,W]

        s1 = self.signs_e1.view(8, 1, 1, 1, 1).to(feats.dtype).to(feats.device)
        s2 = self.signs_e2.view(8, 1, 1, 1, 1).to(feats.dtype).to(feats.device)

        self.f_mean_e1 = (feats * s1).mean(dim=0)  # [B,C,H,W]
        self.f_mean_e2 = (feats * s2).mean(dim=0)  # [B,C,H,W]


        Hf, Wf = self.f_mean_e1.shape[-2], self.f_mean_e1.shape[-1]
        Hw, Ww = w.shape[-2], w.shape[-1]
        if (Hw != Hf) or (Ww != Wf):
            dh = (Hw - Hf) // 2
            dw = (Ww - Wf) // 2
            w = w[:, :, dh:dh + Hf, dw:dw + Wf]   # center crop

        # 2) Weighted GAP helper
        def weighted_gap(feat, w):
            # feat: [B,C,H,W], w: [B,1,H,W]
            ws = w.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-6)  # [B,1,1,1]
            v = (feat * w).sum(dim=(-2, -1), keepdim=True) / ws     # [B,C,1,1]
            return v.squeeze(-1).squeeze(-2)                        # [B,C]

        # 3) Compute weighted pooled vectors
        v1 = weighted_gap(self.f_mean_e1, w)  # [B,C]
        v2 = weighted_gap(self.f_mean_e2, w)  # [B,C]

        # 4) Heads
        e1 = self.head_e1(v1)
        e2 = self.head_e2(v2)  # [B,1]
        e  = torch.cat([e1, e2], dim=-1)
        return e

class R180Inv_CNN_GeLU(nn.Module):
    """
      input:  [B, in_dim, H, W]  ('in_dim' bands)
      output:  [B, 2]        (e1, e2)
    """


    def __init__(self, in_dim=1, base_channels=32, head_hidden=256, out_dim=2, num_layers=5):
        super().__init__()
        C = base_channels
        self.inpad = nn.ConstantPad2d(10, 0.0)

        self.blocks = nn.ModuleList([R180Inv_Conv2d(in_dim if i == 0 else C, C) for i in range(num_layers)])

        self.gap = nn.AdaptiveAvgPool2d(1)  # -> [B, C, 1, 1]

        # MLP head
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(C, head_hidden),
            nn.GELU(),
            nn.Linear(head_hidden, out_dim),
        )

        # MLP initialization
        for m in self.head:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # x: [B,in_dim,H,W]
        H = x.size(-2)
        W = x.size(-1)
        x = self.inpad(x)                       # [B,in_dim,H+20,W+20]
        for block in self.blocks:
            x = block(x)
        
        # Crop out the center [H, W]
        h_, w_ = x.size(-2), x.size(-1)
        #x = x[:, :, (h_ // 2 - H // 2):(h_ // 2 + H // 2), (w_ // 2 - W // 2):(w_ // 2 + W // 2)] # [B,C,H,W]
        x = self.gap(x).squeeze(-1).squeeze(-2)  # [B, C]
        
        y = self.head(x)                         # [B, 2]
        return y

class D4T_CNN_GeLU(nn.Module):
    def __init__(self, in_dim=1,
                kernel_size=3, 
                base_channels=64, 
                head_hidden=128, 
                num_layers=5):
        super().__init__()
        C = base_channels
        padding_size = int((kernel_size-1)/2*num_layers+4)
        self._w_cache = {}
        self.inpad = nn.ConstantPad2d(padding_size, 0.0)
        self.blocks = nn.ModuleList([D4Inv_Conv2d(in_dim if i == 0 else C, C, k = kernel_size) for i in range(num_layers)])

        # MLP head
        self.head0 = BiasFreeMLP(C, head_hidden)
        self.head1 = BiasFreeMLP(C, head_hidden)

    def _get_cached_weight(self, hw, sigma, device, dtype, mode):
        key = (hw[0], hw[1], sigma, device.type, str(dtype), str(mode))
        w = self._w_cache.get(key)
        if w is None:
            w = quad_gaussian_2d(hw, sigma, device=device, mode = mode).to(dtype=dtype)  # [H,W]
            w = w.unsqueeze(0).unsqueeze(0)  # [1,1,H,W] for cheap broadcast
            self._w_cache[key] = w
        return self._w_cache[key]

    def forward(self, x):
        B, C, H, W = x.shape

        # Gaussian Normalization
        w = self._get_cached_weight(x.shape[-2:], sigma=16, device=x.device, dtype=x.dtype, mode="Gaussian")
        m = (x*w).sum(dim=(1, 2, 3), keepdim=True)*100 # mean brightness
        norm = 1 + F.softplus(m - 1, beta=10.0, threshold=20)
        x = x / norm

        # --- run feature extraction after inpad ---
        y = self.inpad(x)  # [B, C, H+12, W+12]
        for block in self.blocks:
            y = block(y)   # expected to keep the spatial size: still approx [B, C, H+12, W+12]

        self.y  = center_crop_to(y,  (H, W))  # [B, C, H+8, W+8]
        self.w0 = self._get_cached_weight([H,W], sigma=4, device=x.device, dtype=x.dtype, mode="x2-y2")
        self.w1 = self._get_cached_weight([H,W], sigma=4, device=x.device, dtype=x.dtype, mode="2xy")

        self.y0 = self.y * self.w0
        self.y1 = self.y * self.w1

        z0 = self.y0.sum(dim=(-1, -2))  # [B, C]
        z1 = self.y1.sum(dim=(-1, -2))  # [B, C]

        shape0 = self.head0(z0)  # [B, 1]
        shape1 = self.head1(z1)  # [B, 1]

        out = torch.cat([shape0, shape1], dim=-1)  # [B, 2]
        return out




