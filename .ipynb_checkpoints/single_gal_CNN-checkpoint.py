import os
import math
import numpy as np
import pandas as pd
from typing import Literal, Tuple

import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet18, ResNet18_Weights

from single_gal_dataset import make_loaders, rotate_spin2, rotate_image_90, flip_image_y, flip_spin2
from CNN_toolkit import D4_eq_weight, d4_variants, center_crop_to, gaussian_blur
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

#######################################################################################################
# CNN models:
#######################################################################################################

class SmoothCNN_GeLU(nn.Module):
    """
      input:  [B, in_dim, H, W]  ('in_dim' bands)
      output:  [B, 2]        (e1, e2)
    """


    def __init__(self, in_dim = 1,  base_channels=64, head_hidden=256, out_dim=2):
        super().__init__()
        C = base_channels
        self.inpad = nn.ConstantPad2d(10, 0.0)
        self.block1 = ConvGELU(in_dim,   C)
        self.block2 = ConvGELU(C,   C)
        self.block3 = ConvGELU(C,   C)
        self.block4 = ConvGELU(C,   C)
        self.block5 = ConvGELU(C,   C)


        self.gap = nn.AdaptiveAvgPool2d(1)  # -> [B, C, 1, 1]

        # MLP head
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(C, head_hidden),
            nn.GELU(),
            nn.Linear(head_hidden, out_dim),
        )

        # 线性层初始化
        for m in self.head:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # x: [B,in_dim,H,W]
        H = x.size(-2)-6
        W = x.size(-1)-6
        x = self.inpad(x)                       # [B,in_dim,H+20,W+20]
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x) 
        
        # Crop out the center [H, W]
        h_, w_ = x.size(-2), x.size(-1)
        x = x[:, :, (h_ // 2 - H // 2):(h_ // 2 + H // 2), (w_ // 2 - W // 2):(w_ // 2 + W // 2)] # [B,C,H,W]
        x = self.gap(x).squeeze(-1).squeeze(-2)  # [B, C]
        
        y = self.head(x)                         # [B, 2]
        return y

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
                base_channels=32, 
                head_hidden=256, 
                num_layers=5):
        super().__init__()
        C = base_channels
        padding_size = int((kernel_size-1)/2*num_layers+4)
        self.inpad = nn.ConstantPad2d(padding_size, 0.0)
        self.blocks = nn.ModuleList([D4Inv_Conv2d(in_dim if i == 0 else C, C, k = kernel_size) for i in range(num_layers)])

        # MLP head
        self.head0 = BiasFreeMLP(C, head_hidden)
        self.head1 = BiasFreeMLP(C, head_hidden)

    def forward(self, x):
        B, C, H, W = x.shape
        self.m = x.sum(dim=(1,2,3), keepdim=True)   # [B,1,1,1]
        self.eps = x.std(dim=(1,2,3), keepdim=True)
        x = x / (self.m + self.eps)
        
        y = self.inpad(x)  # [B, C, H+12, W+12]
        w0, w1 = D4_eq_weight(y)
        for block in self.blocks:
            y = block(y)
        # Crop out the center [H, W]
        #win = hann_window(H+4, W+4, margin=6, device=w0.device, dtype=w0.dtype)
        self.y = center_crop_to(y, (H+4, W+4))  # [B, C, H, W]
        self.w0 = center_crop_to(w0, (H+4, W+4))
        self.w1 = center_crop_to(w1, (H+4, W+4))

        norm_0 = torch.sqrt((self.w0**2+1e-6).sum(dim=(-1,-2)))
        norm_1 = torch.sqrt((self.w1**2+1e-6).sum(dim=(-1,-2))) 

        self.y0 = self.y*self.w0
        self.y1 = self.y*self.w1

        z0 = self.y0.sum(dim = (-1,-2))/norm_0 # [B,C]
        z1 = self.y1.sum(dim = (-1,-2))/norm_1 # [B,C]

        shape0 = self.head0(z0)  # [B, 1]
        shape1 = self.head1(z1)  # [B, 1]

        out = torch.cat([shape0, shape1], dim=-1)  # [B,2], order: [shape_0, shape_1]
        return out

class ShpaeResNet_GeLU(nn.Module):
    def __init__(self, out_dim: int = 2, pretrained: bool = False):
        super().__init__()

        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = resnet18(weights=weights)

        # 1. First conv 改成 1-channel
        old_conv = self.backbone.conv1
        self.backbone.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False
        )

        # 2. 将所有 ReLU 改成 GELU（遍历替换）
        def replace_relu_with_gelu(module: nn.Module):
            for name, child in module.named_children():
                if isinstance(child, nn.ReLU):
                    setattr(module, name, nn.GELU())
                else:
                    replace_relu_with_gelu(child)

        replace_relu_with_gelu(self.backbone)

        # 3. 自定义全连接 head（用 GELU）
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, out_dim),
        )

    def forward(self, x):
        return self.backbone(x)

from training import train_model

if __name__ == "__main__":
    # Example run
    _ = train_model(
        images_path="/projects/bdsp/wenyinli/datasets/simple_gal_images.npy",
        csv_path="/projects/bdsp/wenyinli/datasets/simple_gal_info.csv",
        target="e",          # or "g" if you want to train on shear labels
        epochs=50,
        batch_size=256,
        num_workers=8,
        lr=1e-3,
        use_amp=True,
    )
