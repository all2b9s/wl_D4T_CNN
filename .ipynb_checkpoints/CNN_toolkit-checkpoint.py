import math
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
from single_gal_dataset import make_loaders
import torch.nn.functional as F


def load_test_item(
    images_path: str = "images.npy",
    csv_path: str = "gt_info.csv",
    index: int = 0,
    target: str = "e",   # "e" -> (e1,e2), "g" -> (g1,g2)
    normalize: str = "per_image"  # or "none"
):
    """
    返回：
      img: np.ndarray, shape [H, W], float32，已按需要做归一化
      y  : np.ndarray, shape [2], float32 （(e1,e2) 或 (g1,g2)）
    说明：
      - index 是 **test split 内部**的序号（不是全局 id）
      - 若你要用全局 id 直接读，请把下面的 iloc 改成按 id 筛选
    """
    df = pd.read_csv(csv_path)
    df_test = df[df["split"] == "test"].sort_values("id").reset_index(drop=True)
    assert 0 <= index < len(df_test), f"index out of range: 0..{len(df_test)-1}"
    row = df_test.iloc[index]
    img_id = int(row["id"])

    # 打开 .npy（memmap，低内存）
    imgs = np.load(images_path, mmap_mode="r")
    img = imgs[img_id].astype(np.float32, copy=True)  # [H, W]

    # 目标
    if target == "e":
        y = np.array([row["e1"], row["e2"]], dtype=np.float32)
    elif target == "g":
        y = np.array([row["g1"], row["g2"]], dtype=np.float32)
    else:
        raise ValueError("target must be 'e' or 'g'")

    # 归一化（与训练可保持一致）
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
        device, dtype = img.device, img.dtype

        # 3x3 Sobel kernels for pixel gradients
        Kx = torch.tensor([[-1., 0., 1.],
                            [-2., 0., 2.],
                            [-1., 0., 1.]], 
                            device=device, dtype=dtype).view(1,1,3,3)
        Ky = Kx.transpose(-1, -2).contiguous()

        # 1st order gradients:
        img_x = F.conv2d(img, Kx, padding=1, groups=img.shape[1])  # [B, C, H, W]
        img_y = F.conv2d(img, Ky, padding=1, groups=img.shape[1])  # [B, C, H, W]

        # 2nd order gradients:
        img_xx = F.conv2d(img_x, Kx, padding=1, groups=img.shape[1])  # [B, C, H, W]
        img_yy = F.conv2d(img_y, Ky, padding=1, groups=img.shape[1])  # [B, C, H, W]
        img_xy = F.conv2d(img_x, Ky, padding=1, groups=img.shape[1])  # [B, C, H, W]

        weight_0 = (img_xx - img_yy)  # [B, C, H, W]
        weight_1 = 2*img_xy  # [B, C, H, W]
        return weight_0, weight_1

def shape_pixel_gradients(
    model: torch.nn.Module,
    img_np: np.ndarray,            # [H, W], float-like
    device = None,
    normalize: str = "per_image",  # "per_image" or "none"
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      pred: np.ndarray, shape [2]   -> 模型的 (e1,e2) 预测
      grad_e1: np.ndarray, [H, W]   -> ∂e1 / ∂I
      grad_e2: np.ndarray, [H, W]   -> ∂e2 / ∂I
    """
    assert img_np.ndim == 2, "img_np must be [H, W]"
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    # 复制以免原数组被修改
    img = torch.from_numpy(np.array(img_np, copy=True)).float()

    # 与训练保持一致的归一化
    if normalize == "per_image":
        m = img.mean()
        s = img.std()
        img = (img - m) / s if s > 0 else (img - m)
    elif normalize == "none":
        pass
    else:
        raise ValueError("normalize must be 'per_image' or 'none'")

    # 准备计算图
    x = img.unsqueeze(0).unsqueeze(0).to(device)  # [1,1,H,W]
    x.requires_grad_(True)

    # 前向
    pred = model(x)[0]  # [2]

    # 分别对 e1/e2 回传，拿像素梯度
    grad_maps = []
    for j in range(2):
        # 清理历史梯度
        model.zero_grad(set_to_none=True)
        if x.grad is not None:
            x.grad.zero_()

        pred[j].backward(retain_graph=True)
        grad_maps.append(x.grad.detach().cpu().squeeze(0).squeeze(0).numpy())

    grad_e1, grad_e2 = grad_maps[0], grad_maps[1]
    return pred.detach().cpu().numpy(), grad_e1, grad_e2

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
    img_t = torch.tensor(img).unsqueeze(0).unsqueeze(0).to(device)  # [1,1,H,W]
    with torch.no_grad():
        y_pred_t = model(img_t)          # [1,2]
    y_pred = y_pred_t.squeeze(0).cpu().numpy() 
    return y_pred