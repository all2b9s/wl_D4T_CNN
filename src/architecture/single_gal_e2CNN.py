import os
import math
import numpy as np
import pandas as pd
from typing import Literal, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from e2cnn import gspaces, nn as enn

from src.datasets.single_gal_dataset import make_loaders, rotate_spin2, rotate_image_90

# --------------------------
# GeLU 激活
class GeLU(enn.EquivariantModule):
    def __init__(self, field_type):
        super().__init__()
        self.in_type = field_type  
        self.out_type = field_type

    def forward(self, x):
        assert x.type == self.in_type
        return enn.GeometricTensor(F.gelu(x.tensor), self.out_type)

    def evaluate_output_shape(self, input_shape):
        return input_shape

# ---------- gspace ----------
gspace_D4 = gspaces.FlipRot2dOnR2(N=4)

def trivial_type(gspace, C=1):
    return enn.FieldType(gspace, [gspace.trivial_repr] * C)

def regular_type(gspace, C):
    return enn.FieldType(gspace, [gspace.regular_repr] * C)

E2_D4 = gspace_D4.fibergroup.irreps['irrep_1,1']
spin2_type_D4 = enn.FieldType(gspace_D4, [E2_D4])

# --------------------------
# 等变 block，添加了 BatchNorm 和 GeLU
class D4Block(enn.EquivariantModule):
    def __init__(self, in_type, out_type, k=3, p=1):
        super().__init__()
        conv = enn.R2Conv(in_type, out_type, kernel_size=k, padding=p, bias=False)
        bn   = enn.InnerBatchNorm(conv.out_type)
        act  = GeLU(conv.out_type)
        self.block = enn.SequentialModule(conv, bn, act)

        self.in_type = conv.in_type
        self.out_type = act.out_type

    def forward(self, x):
        return self.block(x)

    def evaluate_output_shape(self, input_shape):
        return self.block.evaluate_output_shape(input_shape)

# --------------------------
# 主体网络，包含 5 层等变 block
class D4ShapeNet(nn.Module):
    def __init__(self, in_ch=1, C=32):
        super().__init__()
        self.in_type = trivial_type(gspace_D4, in_ch)
        self.reg_type = regular_type(gspace_D4, C)

        self.pad = nn.ReflectionPad2d(1)

        self.lift = enn.R2Conv(self.in_type, self.reg_type, kernel_size=5, padding=2, bias=False)
        self.bn0  = enn.InnerBatchNorm(self.reg_type)
        self.act0 = GeLU(self.reg_type)

        self.b1 = D4Block(self.reg_type, self.reg_type, k=3, p=1)
        self.b2 = D4Block(self.reg_type, self.reg_type, k=3, p=1)
        self.b3 = D4Block(self.reg_type, self.reg_type, k=3, p=1)
        #self.b4 = D4Block(self.reg_type, self.reg_type, k=3, p=1)
        #self.b5 = D4Block(self.reg_type, self.reg_type, k=3, p=1)

        self.head = enn.R2Conv(self.reg_type, spin2_type_D4, kernel_size=1, bias=True)

    def forward(self, x):  # x: [B, 1, H, W]
        x = self.pad(x)
        x = enn.GeometricTensor(x, self.in_type)

        y = self.lift(x)
        y = self.bn0(y)
        y = self.act0(y)

        y = self.b1(y)
        y = self.b2(y)
        y = self.b3(y)
        #y = self.b4(y)
        #y = self.b5(y)

        z = self.head(y)
        out = z.tensor.mean(dim=(-1, -2))  # Global average pooling -> [B, 2]
        return out

def train_model(
    images_path="images.npy",
    csv_path="gt_info.csv",
    target: Literal["e","g"]="e",
    epochs: int = 10,
    batch_size: int = 256,
    num_workers: int = 8,
    lr: float = 1e-3,
    device: str = None,
    pt_path: str = None,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader = make_loaders(
        images_path, csv_path, batch_size, num_workers, target=target, augment=False,
    )
    model = D4ShapeNet().to(device)
    if pt_path is not None and os.path.isfile(pt_path):
        model.load_state_dict(torch.load(pt_path, map_location=device))
        print(f"Loaded model weights from {pt_path}")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.MSELoss()  

    def run_epoch(loader, train=True):
        model.train(train)
        total_loss, n = 0.0, 0
        for imgs, y in loader:
            imgs = imgs.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            pred = model(imgs)
            loss = loss_fn(pred, y)

            if train:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

            total_loss += loss.item() * imgs.size(0)
            n += imgs.size(0)
        return total_loss / max(n, 1)

    for ep in range(1, epochs + 1):
        tr = run_epoch(train_loader, train=True)
        va = run_epoch(val_loader, train=False)
        print(f"Epoch {ep:03d} | train {tr:.6f} | val {va:.6f}")

    # quick test evaluation
    test_loss = run_epoch(test_loader, train=False)
    print(f"Test loss: {test_loss:.6f}")

    return model