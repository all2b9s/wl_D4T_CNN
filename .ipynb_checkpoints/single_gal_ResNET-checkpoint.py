import os
import math
import numpy as np
import pandas as pd
from typing import Literal, Tuple

import torch
from torch import nn
from torchvision.models.resnet import resnet18, ResNet18_Weights

from single_gal_dataset import make_loaders, rotate_spin2, rotate_image_90

# ----------------------------
# ResNet regressor (e1,e2)
# ----------------------------
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

class Rot90EquivariantWrapper(nn.Module):
    """
    Hard-code 90 rotation into the model
    """
    def __init__(self, base_model: nn.Module, mode: str = "equal"):
        super().__init__()
        assert mode in ("equal", "precision")
        self.base = base_model
        self.mode = mode

    def forward(self, x):
        preds = []
        vars_ = []  # 仅在 precision 模式使用
        for k in range(4):
            xk = rotate_image_90(x, k)
            yk = self.base(xk)               # shape: [B, 2] 或 [B, 3]
            if self.mode == "precision":
                mean = yk[..., :2]
                logvar = yk[..., 2:3]        # [B,1]
                vars_.append(logvar)
                yk = mean
            yk_rotback = rotate_spin2(yk, k, inverse=True)
            preds.append(yk_rotback)         # [B,2]

        Y = torch.stack(preds, dim=0)        # [4, B, 2]

        if self.mode == "equal":
            out = Y.mean(dim=0)              # [B,2]
            return out

        # precision-weighted averaging
        LOGVAR = torch.stack(vars_, dim=0)   # [4, B, 1]
        W = torch.exp(-LOGVAR)               # [4, B, 1]
        W2 = W.expand(-1, -1, 2)             # [4, B, 2]
        out = (W2 * Y).sum(dim=0) / (W2.sum(dim=0) + 1e-12)   # [B,2]
        return out

    
    
# ----------------------------
# Training loop (MSE / Huber)
# ----------------------------
def train_model(
    images_path="images.npy",
    csv_path="gt_info.csv",
    target: Literal["e","g"]="e",
    epochs: int = 10,
    batch_size: int = 256,
    num_workers: int = 8,
    lr: float = 1e-3,
    use_amp: bool = True,
    huber_delta: float = 0.02,
    device: str = None,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader = make_loaders(
        images_path, csv_path, batch_size, num_workers, target=target, augment=True
    )

    base = ShpaeResNet_GeLU(out_dim=2, pretrained=False)
    model = model = Rot90EquivariantWrapper(base, mode="equal").to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.SmoothL1Loss(beta=huber_delta)  # Huber; or use nn.MSELoss()

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    def run_epoch(loader, train=True):
        model.train(train)
        total_loss, n = 0.0, 0
        for imgs, y in loader:
            imgs = imgs.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                pred = model(imgs)
                loss = loss_fn(pred, y)

            if train:
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()

            total_loss += loss.item() * imgs.size(0)
            n += imgs.size(0)
        return total_loss / max(n, 1)

    for ep in range(1, epochs + 1):
        tr = run_epoch(train_loader, train=True)
        va = run_epoch(val_loader, train=False)
        scheduler.step()
        print(f"Epoch {ep:03d} | train {tr:.6f} | val {va:.6f} | lr {scheduler.get_last_lr()[0]:.2e}")

    # quick test evaluation
    test_loss = run_epoch(test_loader, train=False)
    print(f"Test loss: {test_loss:.6f}")

    return model


if __name__ == "__main__":
    # Example run
    _ = train_model(
        images_path="images.npy",
        csv_path="gt_info.csv",
        target="e",          # or "g" if you want to train on shear labels
        epochs=10,
        batch_size=256,
        num_workers=8,
        lr=1e-3,
        use_amp=True,
    )
