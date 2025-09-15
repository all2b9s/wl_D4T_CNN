import math
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
from src.datasets.single_gal_dataset import make_loaders
import torch.nn.functional as F
from torch import nn
import pandas as pd


def load_test_item(
    images_path: str = "images.npy",
    csv_path: str = "gt_info.csv",
    index: int = 0,
    target: str = "e",   # "e" -> (e1,e2), "g" -> (g1,g2)
    normalize: str = "per_image"  # or "none"
):

    df = pd.read_csv(csv_path)
    df_test = df[df["split"] == "test"].sort_values("id").reset_index(drop=True)
    assert 0 <= index < len(df_test), f"index out of range: 0..{len(df_test)-1}"
    row = df_test.iloc[index]
    img_id = int(row["id"])

    # read image
    imgs = np.load(images_path, mmap_mode="r")
    img = imgs[img_id].astype(np.float32, copy=True)  # [H, W]

    # target
    if target == "e":
        y = np.array([row["e1"], row["e2"]], dtype=np.float32)
    elif target == "g":
        y = np.array([row["g1"], row["g2"]], dtype=np.float32)
    else:
        raise ValueError("target must be 'e' or 'g'")

    # normalization
    if normalize == "per_image":
        m, s = img.mean(), img.std()
        img = (img - m) / s if s > 0 else (img - m)

    return img, y

def plot_shape_bidirectional(
    img: np.ndarray,                 # [H, W]
    y_gt: np.ndarray,                # [2], (e1,e2) 或 (g1,g2)
    y_pred: np.ndarray,# 可选 [2]，若传则一并画出
    arrow_scale: float = 0.35,       # 箭头长度系数（再乘 |e| 或固定长度）
    fixed_length: bool = False,      # True: 忽略|e|，固定长度只显示方向
    percent_clip: float = 99.5,      # 对比度截断
    title = None,
    fig_ax = None,     # 可传入 (fig, ax)，否则自动创建
):
    """
    - 将 (e1,e2) 视为 spin-2 形状：主轴角 φ = 0.5 * atan2(e2, e1)（π 周期）
    - 在图像中心沿 ±φ 方向各画一支箭头
    - y_pred 若提供，则：蓝色=pred，橙色=gt；只给 y_gt 就只画 gt
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

    # 显示图像（robust 对比度）
    lo, hi = np.percentile(img, [100 - percent_clip, percent_clip])
    if fig_ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(3.5, 3.5))
    else:
        fig, ax = fig_ax

    ax.imshow(np.clip(img, lo, hi), cmap="gray", origin="lower",
              vmin=lo, vmax=hi, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])

    # 画 GT
    phi_gt, amp_gt = _phi_amp(y_gt)
    _draw_bidirectional(ax, phi_gt, amp_gt, color="orange")

    # 可选：画预测
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
    Computes the second-order gradients of an image using Sobel operators.

    Args:
            img (torch.Tensor): Input image tensor of shape [B, C, H, W], 
                                                    where B is the batch size, C is the number of channels,
                                                    H is the height, and W is the width.

    Returns:
            tuple: A tuple containing two tensors:
                    - weight_0 (torch.Tensor): Ixx-Iyy, for shape0
                    - weight_1 (torch.Tensor): sIxy, for shape1
    """
    B,C,H,W = img.shape
    # Sobel 
    Kx = torch.tensor([[-1.,0.,1.],
                       [-2.,0.,2.],
                       [-1.,0.,1.]],
                    device=img.device, dtype=img.dtype).view(1,1,3,3)
    Ky = Kx.transpose(-1,-2).contiguous()
    Kx, Ky = Kx.repeat(C,1,1,1), Ky.repeat(C,1,1,1)

    # valid 卷积（不 pad）
    ix  = F.conv2d(img, Kx, groups=C, padding=0)
    iy  = F.conv2d(img, Ky, groups=C, padding=0)
    ixx = F.conv2d(ix, Kx, groups=C, padding=0)
    iyy = F.conv2d(iy, Ky, groups=C, padding=0)
    ixy = F.conv2d(ix, Ky, groups=C, padding=0)

    w0 = ixx - iyy  
    w1 = 2.0 * ixy  

    return w0, w1

def shape_pixel_gradients(
    model: torch.nn.Module,
    img_np,                       # [H,W] or [B,H,W] or [B,1,H,W]
    device=None,
    normalize: str = "none", # "per_image" or "none"
):
    """
    Returns:
      pred:     np.ndarray, [B, 2]
      grad_e1:  np.ndarray, [B,1,H,W]
      grad_e2:  np.ndarray, [B,1,H,W]
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    # ---- Prepare input tensor on device ----
    if isinstance(img_np, torch.Tensor):
        x = img_np
        # ensure float32 on device
        x = x.to(device=device, dtype=torch.float32)
    else:
        x = torch.from_numpy(np.asarray(img_np, dtype=np.float32)).to(device)

    # Accept [H,W], [B,H,W], or [B,1,H,W]
    if x.ndim == 2:
        x = x.unsqueeze(0)          # [1,H,W]
        single = True
    elif x.ndim == 3:
        single = False
    elif x.ndim == 4 and x.shape[1] == 1:
        # already [B,1,H,W] -> squeeze channel for normalization, will re-add
        single = (x.shape[0] == 1)
        x = x.squeeze(1)            # [B,H,W]
    else:
        raise ValueError("img_np must be [H,W], [B,H,W], or [B,1,H,W]")

    B, H, W = x.shape

    # ---- Normalization (per-image if requested) ----
    if normalize == "per_image":
        m = x.mean(dim=(-1, -2), keepdim=True)
        s = x.std(dim=(-1, -2), keepdim=True)
        x = torch.where(s > 0, (x - m) / s, x - m)
    elif normalize != "none":
        raise ValueError("normalize must be 'per_image' or 'none'")

    # Add channel dim -> [B,1,H,W]
    x = x.unsqueeze(1).contiguous()
    x.requires_grad_(True)

    # ---- Disable param grads: we only need ∂y/∂x ----
    prev_flags = [p.requires_grad for p in model.parameters()]
    for p in model.parameters():
        p.requires_grad_(False)

    # ---- One forward pass on duplicated batch ----
    # Make [2B,1,H,W]: first half used to backprop e1, second half for e2
    x2 = torch.cat([x, x], dim=0)                   # [2B,1,H,W]
    pred2 = model(x2)                               # [2B, 2]
    # Use the first B predictions as the output preds (inputs identical)
    pred = pred2[:B]

    # ---- One backward to get both grads at once ----
    # grad_outputs shape must match pred2: [2B, 2]
    go = torch.zeros_like(pred2)
    go[:B, 0] = 1.0   # select e1 for first copy
    go[B:, 1] = 1.0   # select e2 for second copy

    (gx2,) = torch.autograd.grad(
        outputs=pred2,
        inputs=x2,
        grad_outputs=go,
        retain_graph=False,
        create_graph=False,
        allow_unused=False,
    )
    # Split back into e1/e2 grads for the original batch
    grad_e1 = gx2[:B]   # [B,1,H,W]
    grad_e2 = gx2[B:]   # [B,1,H,W]

    # ---- Restore model flags ----
    for p, f in zip(model.parameters(), prev_flags):
        p.requires_grad_(f)

    # ---- To numpy with shape conventions ----
    pred_np = pred.detach().cpu().numpy()               # [B,2]
    grad_e1_np = grad_e1.detach().cpu().numpy()         # [B,H,W]
    grad_e2_np = grad_e2.detach().cpu().numpy()         # [B,H,W]

    return pred_np, grad_e1_np, grad_e2_np 

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