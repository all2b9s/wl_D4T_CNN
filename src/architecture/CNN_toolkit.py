import math
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
import torch.nn.functional as F
from torch import nn
import pandas as pd

# Testing plotting and data loading

def load_test_item(
    images_path: str = "images.npy",
    csv_path: str = "gt_info.csv",
    index_range: list = None,   # e.g. [0, 10]
    target: str = "e",          # "e" -> (e1, e2), "g" -> (g1, g2)
    normalize: str = "none"     # or "per_image"
):
    if index_range is None or len(index_range) != 2:
        raise ValueError("index_range must be a list like [start, end].")

    # Load CSV and select test subset
    df = pd.read_csv(csv_path)
    df_test = df[df["split"] == "test"].sort_values("id").reset_index(drop=True)

    idx_min, idx_max = index_range
    if idx_min < 0 or idx_max > len(df_test):
        raise IndexError("index_range is out of bounds for test dataset.")

    # Load all images (mmap for memory efficiency)
    imgs = np.load(images_path, mmap_mode="r")

    img_list, y_list = [], []

    for index in range(idx_min, idx_max):
        row = df_test.iloc[index]
        img_id = int(row["id"])

        # read image
        img = imgs[img_id].astype(np.float64, copy=True)

        # normalization
        if normalize == "per_image":
            m, s = img.mean(), img.std()
            img = (img - m) / s if s > 0 else (img - m)

        # target
        if target == "e":
            y = np.array([row["e1"], row["e2"]], dtype=np.float64)
        elif target == "g":
            y = np.array([row["g1"], row["g2"]], dtype=np.float64)
        else:
            raise ValueError("target must be 'e' or 'g'")

        img_list.append(img)
        y_list.append(y)

    imgs_out = np.stack(img_list)   # shape [N, H, W]
    ys_out = np.stack(y_list)       # shape [N, 2]

    return imgs_out, ys_out

def plot_shape_bidirectional(
    img: np.ndarray,                 # [H, W]
    y_gt: np.ndarray,                # [2], (e1,e2) or (g1,g2)
    y_pred: np.ndarray,# optional [2]; if provided, it is plotted as well
    arrow_scale: float = 0.35,       # arrow length scale (times |e| unless fixed)
    fixed_length: bool = False,      # True: ignore |e| and use fixed length (direction only)
    percent_clip: float = 99.5,      # contrast clipping percentile
    title = None,
    fig_ax = None,     # pass (fig, ax) or create one automatically
):
    """
    - Treat (e1, e2) as a spin-2 shape: principal-axis angle φ = 0.5 * atan2(e2, e1) (π-periodic)
    - Draw one arrow in each of the ±φ directions from the image center
    - If y_pred is provided: blue=pred, orange=gt; with only y_gt, only gt is drawn
    """
    assert img.ndim == 2, "img should be [H, W]"
    H, W = img.shape
    cx, cy = (W - 1) / 2.0, (H - 1) / 2.0

    def _phi_amp(y):
        e1, e2 = float(y[0]), float(y[1])
        phi = 0.5 * math.atan2(e2, e1)
        amp = arrow_scale * (min(H, W) / 2.0)
        if not fixed_length:
            amp *= math.hypot(e1, e2)
        return phi, amp

    def _draw_bidirectional(ax, phi, amp, color, inverse = False):
        ux, uy = math.cos(phi), math.sin(phi)
        dx, dy = amp * ux, amp * uy
        if inverse:
            ax.quiver([cx], [cy], [-dx], [-dy], angles='xy', scale_units='xy', scale=1,
                      width=0.006, color=color)
        else:
            ax.quiver([cx], [cy], [ dx], [ dy], angles='xy', scale_units='xy', scale=1,
                  width=0.006, color=color)

    # Display image (robust contrast)
    #lo, hi = np.percentile(img, [100 - percent_clip, percent_clip])
    if fig_ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(3.5, 3.5))
    else:
        fig, ax = fig_ax

    ax.imshow(img, cmap="gray", origin="lower", interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])

    # Plot GT
    phi_gt, amp_gt = _phi_amp(y_gt)
    _draw_bidirectional(ax, phi_gt, amp_gt, color="orange")

    # Optional: plot prediction
    if y_pred is not None:
        phi_pr, amp_pr = _phi_amp(y_pred)
        _draw_bidirectional(ax, phi_pr, amp_pr, color="blue", inverse = True)

    if title is None:
        if y_pred is None:
            title = f"GT |e|={np.linalg.norm(y_gt):.2f}"
        else:
            title = f"pred |e|={np.linalg.norm(y_pred):.2f}  |gt|={np.linalg.norm(y_gt):.2f}"
    ax.set_title(title)
    if fig_ax is None:
        plt.tight_layout(); plt.show()
    return fig, ax

def D4_eq_weight(img):
    """
    Computes second-order Q/U with baked smoothing (no extra pass).
    Input : img [B, C, H, W]
    Output: w0 = Ixx - Iyy, w1 = 2 Ixy
    Notes : valid conv (no padding), 5-tap separable kernels
    """
    B, C, H, W = img.shape

    # 1D kernels (float, device/dtype follow img)
    # s5: smoothing; d5s: 1st-derivative (Scharr-like 5 tap); dd5: 2nd-derivative (5 tap)
    s5  = torch.tensor([1, 4, 6, 4, 1], device=img.device, dtype=img.dtype) / 16.0
    d5s = torch.tensor([1, -8, 0, 8, -1], device=img.device, dtype=img.dtype) / 12.0
    dd5 = torch.tensor([-1, 16, -30, 16, -1], device=img.device, dtype=img.dtype) / 12.0

    # reshape to conv2d friendly 2D separable kernels
    # x-axis kernel: [1,1,KW,1]; y-axis kernel: [1,1,1,KH]
    kx_s5  = s5.view(1, 1, -1, 1).repeat(C, 1, 1, 1)    
    ky_s5  = s5.view(1, 1,  1, -1).repeat(C, 1, 1, 1)
    kx_d5s = d5s.view(1, 1, -1, 1).repeat(C, 1, 1, 1)
    ky_d5s = d5s.view(1, 1,  1, -1).repeat(C, 1, 1, 1)
    kx_dd5 = dd5.view(1, 1, -1, 1).repeat(C, 1, 1, 1)
    ky_dd5 = dd5.view(1, 1,  1, -1).repeat(C, 1, 1, 1)

    # helper: separable valid conv (x then y), groups=C
    def sep_conv(x, kx, ky):
        x = F.conv2d(x, kx, groups=C, padding=0)  # along x (width)
        x = F.conv2d(x, ky, groups=C, padding=0)  # along y (height)
        return x

    # I_xx: dd5 along x, s5 along y
    ixx = sep_conv(img, kx_dd5, ky_s5)
    # I_yy: s5 along x, dd5 along y
    iyy = sep_conv(img, kx_s5, ky_dd5)
    # I_xy: d5s along x, d5s along y
    ixy = sep_conv(img, kx_d5s, ky_d5s)

    w0 = ixx - iyy
    w1 = 2.0 * ixy
    return w0, w1

@torch.no_grad()
def _to_device(x, device):
    # Grad is disabled only in this helper for faster transfer;
    # the main function enables it again where needed.
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=x.dtype, non_blocking=True)
    return torch.from_numpy(np.asarray(x, dtype=x.dtype)).to(device, non_blocking=True)

def shape_pixel_gradients(
    model: torch.nn.Module,
    img_np,                       # [H,W] or [B,H,W] or [B,1,H,W]
    device=None,
    normalize: str = "none",      # "per_image" or "none"
    mode: str = "memory",         # "memory" or "speed"
    eps: float = 1e-6,
    dtype = np.float32,
):
    """
    Returns:
      pred:     np.ndarray, [B, 2]
      grad_e1:  np.ndarray, [B,1,H,W]
      grad_e2:  np.ndarray, [B,1,H,W]
    """
    assert mode in ("memory", "speed")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # ---- 1) Move input to device (no grad tracking here to reduce overhead) ----
    x = _to_device(img_np, device)

    # Standardize to [B, H, W]
    if x.ndim == 2:
        x = x.unsqueeze(0)              # [1,H,W]
    elif x.ndim == 4 and x.shape[1] == 1:
        x = x.squeeze(1)                # [B,H,W]
    elif x.ndim != 3:
        raise ValueError("img_np must be [H,W], [B,H,W], or [B,1,H,W]")

    B, H, W = x.shape

    # ---- 2) Normalize (before requires_grad) ----
    if normalize == "per_image":
        m = x.mean(dim=(-1, -2), keepdim=True)
        s = x.std(dim=(-1, -2), keepdim=True).clamp_min(eps)
        x = (x - m) / s
    elif normalize != "none":
        raise ValueError("normalize must be 'per_image' or 'none'")

    # Add channel back -> [B,1,H,W]
    x = x.unsqueeze(1)

    # ---- 3) Model setup: disable parameter grads, use eval mode ----
    model = model.to(device).eval()
    prev_flags = [p.requires_grad for p in model.parameters()]
    for p in model.parameters():
        p.requires_grad_(False)

    try:
        if mode == "memory":
            # ---- 4A) Single forward, two backward passes (memory-saving) ----
            x.requires_grad_(True)
            pred = model(x)[:, :2]          # [B,2]

            # e1 gradient
            go1 = torch.zeros_like(pred)
            go1[:, 0] = 1.0
            (gx1,) = torch.autograd.grad(
                outputs=pred, inputs=x, grad_outputs=go1,
                retain_graph=True, create_graph=False, allow_unused=False
            )

            # e2 gradient
            go2 = torch.zeros_like(pred)
            go2[:, 1] = 1.0
            (gx2,) = torch.autograd.grad(
                outputs=pred, inputs=x, grad_outputs=go2,
                retain_graph=False, create_graph=False, allow_unused=False
            )

            grad_e1 = gx1      # [B,1,H,W]
            grad_e2 = gx2      # [B,1,H,W]

        else:
            x = x.detach()              
            x.requires_grad_(True)
            x2 = torch.cat([x, x], dim=0)   # [2B,1,H,W]

            pred2 = model(x2)[:, :2]    # [2B,2]
            pred = pred2[:B]

            go = torch.zeros_like(pred2)
            go[:B, 0] = 1.0
            go[B:, 1] = 1.0
            (gx2,) = torch.autograd.grad(
                outputs=pred2, inputs=x2, grad_outputs=go,
                retain_graph=False, create_graph=False, allow_unused=False
            )
            grad_e1 = gx2[:B]
            grad_e2 = gx2[B:]

        # ---- 5) Convert to numpy (keep [B,1,H,W]) ----
        pred_np = pred.detach().float().cpu().numpy()
        grad_e1_np = grad_e1.detach().float().cpu().numpy()
        grad_e2_np = grad_e2.detach().float().cpu().numpy()

    finally:
        for p, f in zip(model.parameters(), prev_flags):
            p.requires_grad_(f)

        # Explicitly free GPU tensors that may hold retained autograd graphs.
        # These may not all exist depending on the 'mode' branch taken.
        try:
            del go1
        except (NameError, UnboundLocalError):
            pass
        try:
            del go2
        except (NameError, UnboundLocalError):
            pass
        try:
            del go
        except (NameError, UnboundLocalError):
            pass
        try:
            del pred
        except (NameError, UnboundLocalError):
            pass
        try:
            del pred2
        except (NameError, UnboundLocalError):
            pass
        try:
            del gx1
        except (NameError, UnboundLocalError):
            pass
        try:
            del gx2
        except (NameError, UnboundLocalError):
            pass
        try:
            del x
        except (NameError, UnboundLocalError):
            pass
        try:
            del x2
        except (NameError, UnboundLocalError):
            pass

    return pred_np, grad_e1_np, grad_e2_np

def draw_grad(model, img):
    pred, grad_e1, grad_e2 = shape_pixel_gradients(model, img, normalize="none")

    print("pred (e1,e2):", pred)
    fig, axs = plt.subplots(1, 3, figsize=(10, 3))
    im0 = axs[0].imshow(img, cmap="gray", origin="lower")
    axs[0].set_title("input")
    im1 = axs[1].imshow(grad_e1[0][0], cmap="bwr", origin="lower", vmax = np.abs(grad_e1).max(), vmin = -np.abs(grad_e1).max())
    axs[1].set_title(r"$\partial e_1/\partial I$")
    im2 = axs[2].imshow(grad_e2[0][0], cmap="bwr", origin="lower", vmax = np.abs(grad_e2).max(), vmin = -np.abs(grad_e2).max())
    axs[2].set_title(r"$\partial e_2/\partial I$")

    fig.colorbar(im0, ax=axs[0])
    fig.colorbar(im1, ax=axs[1])
    fig.colorbar(im2, ax=axs[2])

    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    plt.show()
    return pred, grad_e1, grad_e2

# --------------------------
# Utils: geometry & sizing
# --------------------------

def d4_variants(x: torch.Tensor) -> torch.Tensor:
    """
    Given x: [B, 1, H, W], return 8 variants along a new dim V:
      V=0..3: rotations by k*90
      V=4..7: mirrored versions (horizontal flip) of those four
    Output: [B, 8, 1, H, W]
    """
    rots = [torch.rot90(x, k=k, dims=(-2, -1)) for k in range(4)]
    flips = [torch.flip(r, dims=(-1,)) for r in rots]  # horizontal mirror
    v = torch.stack(rots + flips, dim=1)
    return v

def center_crop_to(x: torch.Tensor, target_hw: Tuple[int, int]) -> torch.Tensor:
    """
    Center-crop tensor x: [B, C, H, W] to (target_H, target_W).
    """
    _, _, H, W = x.shape
    tH, tW = target_hw
    assert H >= tH and W >= tW, "Crop target must be <= current size"
    top = (H - tH) // 2
    left = (W - tW) // 2
    return x[:, :, top:top + tH, left:left + tW]

# --------------------------
# Gaussian blur (depthwise)
# --------------------------
def gaussian_kernel2d(ks: int, sigma: float, device=None, dtype=None):
    ax = torch.arange(ks, device=device, dtype=dtype) - (ks - 1) / 2
    g1d = torch.exp(-0.5 * (ax / sigma) ** 2)
    g1d = g1d / g1d.sum()
    g2d = torch.outer(g1d, g1d)
    g2d = g2d / g2d.sum()
    return g2d  # [ks, ks]

def gaussian_weight_2d(size: int | tuple[int, int], std: float = 16, device=None, normalize=True, dtype=torch.float32):
    """
    Generate a 2D Gaussian weight image centered in the middle.

    Args:
        size: int or (H, W)
        std: standard deviation (in pixels)
        device: torch device (cpu / cuda)
        normalize: if True, normalize so that sum = 1

    Returns:
        Tensor of shape [H, W] with dtype=float32
    """
    if isinstance(size, int):
        H = W = size
    else:
        H, W = size

    y = torch.arange(H, device=device, dtype=dtype) - (H-1)/2
    x = torch.arange(W, device=device, dtype=dtype) - (W-1)/2
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    
    
    g = torch.exp(-0.5 * (xx**2 + yy**2) / (std**2))/(2*np.pi*std**2)
    g = (g + g.flip([0, 1]) + g.flip([0]) + g.flip([1]))/4 # make sure center is max
    #if normalize:
    #    g /= g.sum()
    return g

import torch, numpy as np

def quad_gaussian_2d(
    size: int | tuple[int, int],
    std: float = 16,
    mode: str = "x2-y2",   # options: "x2-y2" or "2xy"
    device=None,
    normalize=True,
):
    """
    Generate (x^2 - y^2)*Gaussian or 2xy*Gaussian weight maps.

    Args:
        size: int or (H, W)
        std: Gaussian sigma in pixels
        mode: "x2-y2" or "2xy" or "Gaussian"
        device: torch device
        normalize: if True, normalize to unit RMS amplitude (not sum)

    Returns:
        Tensor [H, W] of dtype float32
    """
    if isinstance(size, int):
        H = W = size
    else:
        H, W = size

    y = torch.arange(H, device=device, dtype=torch.float32) - H // 2
    x = torch.arange(W, device=device, dtype=torch.float32) - W // 2
    yy, xx = torch.meshgrid(y, x, indexing="ij")

    g = torch.exp(-0.5 * (xx**2 + yy**2) / (std**2)) / (2 * np.pi * std**2)

    if mode == "x2-y2":
        out = (xx**2 - yy**2) * g
    elif mode == "2xy":
        out = (2 * xx * yy) * g
    elif mode == "Gaussian":
        out = g
    else:
        raise ValueError("mode must be 'x2-y2' or '2xy'")

    if normalize:
        out = out / torch.sqrt((out**2).sum())  # unit RMS

    return out


def gaussian_blur(x: torch.Tensor, ks: int = 3, sigma: float = 1.0) -> torch.Tensor:
    """
    x: [B, 1, H, W] or [B, C, H, W]; applies same kernel to all channels.
    """
    B, C, H, W = x.shape
    k = gaussian_kernel2d(ks, sigma, device=x.device, dtype=x.dtype)
    k = k.view(1, 1, ks, ks)
    k = k.repeat(C, 1, 1, 1)  # depthwise
    padding = ks // 2
    return F.conv2d(x, k, bias=None, stride=1, padding=padding, groups=C)

def predictor(model, img, normalize = "none"):
    if normalize == "per_image":
        m = img.mean()
        s = img.std()
        img = (img - m) / s if s > 0 else (img - m)
    elif normalize == "none":
        pass
    device = next(model.parameters()).device
    model.eval()
    if img.ndim == 2:
        img_t = torch.tensor(img).unsqueeze(0).unsqueeze(0).to(device)  # [1,1,H,W]
    elif img.ndim == 3:
        img_t = torch.tensor(img).unsqueeze(1).to(device)
    elif img.ndim == 4:
        img_t = torch.tensor(img).to(device)
    with torch.no_grad():
        y_pred_t = model(img_t)          # [1,2]
    y_pred = y_pred_t.squeeze(0).cpu().numpy() 
    return y_pred



class PairedShearWrapper(nn.Module):
    """
    Wrap a base shape model:
    - Base model: input [N, H, W] -> output [N, 2]
    - Wrapped model: input [N, 2, H, W] (two sheared images per sample)
        -> output (shape1, shape2), each with shape [N, 2]
    """
    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.base_model = base_model

    def forward(self, imgs_pair: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        imgs_pair: [N, 2, H, W]
            imgs_pair[:, 0, ...] is the first shear image
            imgs_pair[:, 1, ...] is the second shear image

        Returns:
            shape1: [N, 2]
            shape2: [N, 2]
        """
        img_p = imgs_pair[:,0]
        img_n = imgs_pair[:,1]
        
        shape_p = self.base_model(img_p)   # [N, 2]
        shape_n = self.base_model(img_n)   # [N, 2]
        
        e_mean = 0.5 * (shape_p + shape_n)
        delta_e = 0.5 * (shape_p - shape_n)
        return e_mean, delta_e