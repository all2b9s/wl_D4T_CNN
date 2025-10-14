import os
import math
import numpy as np
import pandas as pd
from typing import Literal, Tuple

import torch
from torch.utils.data import get_worker_info
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.transforms import GaussianBlur

# ----------------------------
# Dataset (memmap + CSV)
# ----------------------------

def flip_image_y(x):
    """
    Reflect across the y-axis (horizontal flip).
    x: [B, C, H, W]
    """
    return torch.flip(x, dims=(-1,))  # flip width

def flip_image_x(x):
    """
    Reflect across the x-axis (vertical flip).
    x: [B, C, H, W]
    """
    return torch.flip(x, dims=(-2,))  # flip width

def flip_spin2(y):
    """
    Spin-2 conjugation: (e1, e2) -> (e1, -e2).
    y: [..., 2]
    """
    y = y.clone()
    y[..., 1] = -y[..., 1]
    return y

def rotate_spin2(y: torch.Tensor, k: int, inverse = True):
    """
    y: [B, 2]  (e1,e2) 或 (g1,g2)
    旋转角 theta = k * 90°；spin-2 需要旋转 2*theta。
    inverse=True 表示把输出旋回到原坐标系（用 -2*theta）
    """
    if k == 0:
        return y
    theta = k * (math.pi/2.0) * 2.0
    if inverse:
        theta = -theta
    c, s = math.cos(theta), math.sin(theta)
    e1, e2 = y[..., 0], y[..., 1]
    y1 = c*e1 - s*e2
    y2 = s*e1 + c*e2
    return torch.stack([y1, y2], dim=-1)

def rotate_image_90(img: torch.Tensor, k: int) -> torch.Tensor:
    # img: [1,H,W] or [H,W]
    if k == 0:
        return img
    # torch.rot90 works on [*, H, W]
    if img.ndim == 2:
        return torch.rot90(img, k, dims=(0,1))
    return torch.rot90(img, k, dims=(-2,-1))

def add_resmoothed_noise(img, max_sigma=0.05, rng=None):
    """
    Works with [H,W], [C,H,W], or [B,C,H,W]. CPU-safe inside Dataset/__getitem__.
    """
    # Ensure CPU inside DataLoader workers
    if img.device.type != "cpu":
        img = img.cpu()

    # Ensure float dtype
    if not img.is_floating_point():
        img = img.float()

    # Worker-safe RNG
    if rng is None:
        rng = torch.Generator(device="cpu")
        wi = get_worker_info()
        rng.manual_seed(wi.seed if wi is not None else torch.initial_seed())

    # Leading dims before H,W
    assert img.dim() >= 2, "img must have at least H,W"
    lead = img.shape[:-2]

    # Per-(leading dims) sigma
    if len(lead) == 0:
        sigma = torch.rand((), generator=rng, device="cpu", dtype=img.dtype) * max_sigma
    else:
        sigma = torch.rand(lead, generator=rng, device="cpu", dtype=img.dtype) * max_sigma
        sigma = sigma.view(*lead, 1, 1)

    # Noise (use randn because randn_like may not accept generator)
    noise = torch.randn(
        img.shape, generator=rng, device=img.device, dtype=img.dtype
    )
    return img + noise * sigma



# ----------------------------
# Dataset class
# ----------------------------

class SingleGalaxyDataset(Dataset):
    """
    Loads images from a single .npy (via memmap) and labels from a CSV.
    Supports spin-2–correct 90° rotations as augmentation.
    """
    def __init__(
        self,
        images_path: str,
        csv_path: str,
        split: Literal["train","val","test"] = "train",
        target: Literal["e","g"] = "e",
        augment: bool = True,
        normalize: Literal["per_image","none"] = "per_image",
        channel_first: bool = True,
    ):
        super().__init__()
        assert os.path.exists(images_path), images_path
        assert os.path.exists(csv_path), csv_path

        # Load metadata/labels
        df = pd.read_csv(csv_path)
        df = df[df["split"] == split].copy().sort_values("id")
        self.ids = df["id"].to_numpy().astype(np.int64)

        if target == "e":
            self.targets = df[["e1","e2"]].to_numpy().astype(np.float32)
        else:
            raise ValueError("target must be 'e' or 'g'")

        # Keep fields if you want more complex losses later
        #self.extra = df[["flux","hlr_arcsec","sersic_n","psf_fwhm_arcsec","noise_std_ADU"]].to_numpy()

        # Open memmap (read-only, zero RAM)
        self.images = np.load(images_path, mmap_mode="r").astype(np.float32)  # (N,H,W) float32
        self.augment = augment and (split == "train")
        self.normalize = normalize
        self.channel_first = channel_first

        # Basic checks
        H, W = self.images.shape[1], self.images.shape[2]
        self.shape_hw = (H, W)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        i = self.ids[idx]
        # load one image; copy to torch
        img_np = self.images[i]  # (H,W) float32
        img = torch.tensor(img_np)  # [H,W]

        # per-image normalization (robust if mean varies with flux/noise)
        if self.normalize == "per_image":
            m = img.mean()
            s = img.std()
            if s > 0:
                img = (img - m) / s
            else:
                img = img - m

        # add channel
        if self.channel_first:
            img = img.unsqueeze(0)  # [1,H,W]

        # target
        y = torch.tensor(self.targets[idx], dtype=torch.float32)  # [2]

        # data augmentation: random 0,90,180,270 rotation with proper spin-2 update
        if self.augment:
            k = torch.randint(low=0, high=4, size=(1,), dtype=torch.int64).item()
            img = rotate_image_90(img, k)
            y = rotate_spin2(y, k)

            do_flip_y = torch.rand(1).item() < 0.5
            if do_flip_y:
                img = flip_image_y(img)
                y = flip_spin2(y)
            
            do_flip_x = torch.rand(1).item() < 0.5
            if do_flip_x:
                img = flip_image_x(img)
                y = flip_spin2(y)

            add_noise = torch.rand(1).item() < 0.5
            if add_noise: 
                img = add_resmoothed_noise(img)  # max_sigma=0.05 with 50% prob

        return img, y


# ----------------------------
# Data loaders
# ----------------------------
def make_loaders(
    images_path: str,
    csv_path: str,
    batch_size: int = 128,
    num_workers: int = 4,
    target: Literal["e","g"] = "e",
    augment: bool = True,
    normalize: Literal["per_image","none"] = "none",
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_ds = SingleGalaxyDataset(images_path, csv_path, "train", target, augment, normalize)
    val_ds   = SingleGalaxyDataset(images_path, csv_path, "val",   target, False,   normalize)
    test_ds  = SingleGalaxyDataset(images_path, csv_path, "test",  target, False,   normalize)

    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin)
    return train_loader, val_loader, test_loader
