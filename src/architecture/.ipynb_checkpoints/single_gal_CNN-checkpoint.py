import os
import math
import numpy as np
import pandas as pd
from typing import Literal, Tuple

import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet18, ResNet18_Weights

from src.architecture.CNN_toolkit import D4_eq_weight, center_crop_to
from src.architecture.CNN_module import R180Inv_Conv2d, D4Inv_Conv2d, ConvGELU, BiasFreeMLP
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




