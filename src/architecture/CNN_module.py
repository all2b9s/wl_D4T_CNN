import os
import math
import numpy as np
import pandas as pd
from typing import Literal, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from src.datasets.single_gal_dataset import make_loaders, rotate_spin2, rotate_image_90, flip_image_y, flip_spin2
# ----------------------------
# Related modules:
# ----------------------------

class Rot90EquivariantWrapper(nn.Module):
    """
    Hard-code 90 rotation into the model
    """
    def __init__(self, base_model: nn.Module, mode: str = "equal"):
        super().__init__()
        assert mode in ("equal", "precision")
        self.base = base_model
        self.mode = mode

    def _forward_one(self, x, k, flipped):
        """
        Apply: optional flip, then k*90° rotation -> base -> undo rotation -> undo flip on spin-2.
        """
        # 1) flip (if any) then rotate input
        x_tf = flip_image_y(x) if flipped else x
        x_tf = rotate_image_90(x_tf, k)  # your helper; torch.rot90 under the hood

        # 2) predict
        y = self.base(x_tf)  # [B,2] or [B,3]

        if self.mode == "precision":
            mean = y[..., :2]
            logvar = y[..., 2:3]  # [B,1], isotropic predictive variance
        else:
            mean = y

        # 3) map predictions back to the original frame
        #    Inverse order: undo rotation first, then undo mirror via conjugation.
        y_back = rotate_spin2(mean, k, inverse=True)
        if flipped:
            y_back = flip_spin2(y_back)

        if self.mode == "precision":
            # Scalar logvar is invariant to rotation/conjugation for isotropic case
            return y_back, logvar
        else:
            return y_back, None

    def forward(self, x):
        preds = []
        vars_ = []  # only used in precision mode

        for flipped in (False, True):      # no flip, mirror
            for k in range(2):             # 0, 90 deg
                yk, lv = self._forward_one(x, k, flipped)
                preds.append(yk)
                if self.mode == "precision":
                    vars_.append(lv)

        Y = torch.stack(preds, dim=0)      # [8, B, 2]

        if self.mode == "equal":
            return Y.mean(dim=0)           # [B,2]

        # precision-weighted averaging (scalar precision per transform)
        LOGVAR = torch.stack(vars_, dim=0)     # [8, B, 1]
        W = torch.exp(-LOGVAR)                 # [8, B, 1] precisions
        W2 = W.expand(-1, -1, 2)               # match [8,B,2]
        out = (W2 * Y).sum(dim=0) / (W2.sum(dim=0) + 1e-12)   # [B,2]
        return out

class R180Inv_Conv2d(nn.Module):
    """
    Rot 180 degree invariant convolution
    """
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, bias=True):
        super().__init__()
        self.P = nn.Parameter(torch.randn(out_ch, in_ch, k, k) * (2.0/(in_ch*k*k))**0.5)
        self.bias = nn.Parameter(torch.zeros(out_ch)) if bias else None
        self.s, self.p = s, p
        self.act = nn.GELU()

    @staticmethod
    def _rot180(W):  # 180°旋转 = 上下+左右翻转
        return torch.flip(W, dims=(-2, -1))

    def forward(self, x):
        W_even = 0.5 * (self.P + self._rot180(self.P))
        x = F.conv2d(x, W_even, self.bias, stride=self.s, padding=self.p)
        return self.act(x)
    
class D4Inv_Conv2d(nn.Module):
    """
    Rot 180 degree invariant convolution
    """
    def __init__(self, in_ch, out_ch, k=3, s=1, p=0, bias=True):
        super().__init__()
        self.P = nn.Parameter(torch.randn(out_ch, in_ch, k, k) * (2.0/(in_ch*k*k))**0.5)
        self.bias = nn.Parameter(torch.zeros(out_ch)) if bias else None
        self.s, self.p = s, p
        self.act = nn.GELU()
    
    @staticmethod
    def _rot90(W):
        return torch.rot90(W, 1, dims=(-2,-1))
    
    @staticmethod
    def _mirror(W) -> torch.Tensor:
        return torch.flip(W, dims=(-1,))  # horizontal flip
    
    def _D4_avg(self, W):
        Ws = [W]
        for _ in range(3):
            W = self._rot90(W)
            Ws.append(W)
        Wm = self._mirror(W)
        Ws.append(Wm)
        for _ in range(3):
            Wm = self._rot90(Wm)
            Ws.append(Wm)
        W_avg = torch.stack(Ws, dim=0).mean(dim=0)
        return W_avg

    def forward(self, x):
        self.W_D4 = self._D4_avg(self.P)
        x = F.conv2d(x, self.W_D4, self.bias, stride=self.s, padding=self.p)
        return self.act(x)

class ConvGELU(nn.Module):
    """ReflectionPad2d + Conv2d(3x3, stride=1) + GeLU"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=0, bias=True)
        self.act  = nn.GELU()

        # Kaiming init
        nn.init.kaiming_normal_(self.conv.weight, nonlinearity='relu')
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        x = self.conv(x)
        x = self.act(x)
        return x

class BiasFreeMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_dim, hidden, bias=False),
            nn.Tanh(),
            nn.Linear(hidden, 1, bias=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

def hann_window(H, W, margin, device, dtype):
    if margin <= 0:
        return torch.ones(1, 1, H, W, device=device, dtype=dtype)
    # 1D Hann 余弦缓入
    def ramp(n):
        v = torch.ones(n, device=device, dtype=dtype)
        m = int(margin)
        if m > 0:
            t = torch.linspace(0, 1, m+1, device=device, dtype=dtype)
            c = 0.5*(1 - torch.cos(torch.pi*t))
            v[:m+1] = c
            v[-(m+1):] = c.flip(0)
        return v
    wx = ramp(W); wy = ramp(H)
    win = wy.view(H,1) * wx.view(1,W)
    return win.view(1,1,H,W)
