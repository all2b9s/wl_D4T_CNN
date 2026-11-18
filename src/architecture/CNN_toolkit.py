import math
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Tuple
import torch.nn.functional as F


def load_test_item(
    images_path: str = "images.npy",
    csv_path: str = "gt_info.csv",
    index_range: list = None,
    target: str = "e",
    normalize: str = "none",
):
    if index_range is None or len(index_range) != 2:
        raise ValueError("index_range must be a list like [start, end].")

    df = pd.read_csv(csv_path)
    df_test = df[df["split"] == "test"].sort_values("id").reset_index(drop=True)

    idx_min, idx_max = index_range
    if idx_min < 0 or idx_max > len(df_test):
        raise IndexError("index_range is out of bounds for test dataset.")

    imgs = np.load(images_path, mmap_mode="r")
    img_list, y_list = [], []

    for index in range(idx_min, idx_max):
        row = df_test.iloc[index]
        img_id = int(row["id"])
        img = imgs[img_id].astype(np.float64, copy=True)

        if normalize == "per_image":
            m, s = img.mean(), img.std()
            img = (img - m) / s if s > 0 else (img - m)

        if target == "e":
            y = np.array([row["e1"], row["e2"]], dtype=np.float64)
        elif target == "g":
            y = np.array([row["g1"], row["g2"]], dtype=np.float64)
        else:
            raise ValueError("target must be 'e' or 'g'")

        img_list.append(img)
        y_list.append(y)

    return np.stack(img_list), np.stack(y_list)


def plot_shape_bidirectional(
    img: np.ndarray,                 # [H, W]
    y_gt: np.ndarray,                # [2], (e1,e2) or (g1,g2)
    y_pred: np.ndarray,# optional [2]; if provided, it is plotted as well
    arrow_scale: float = 0.35,       # arrow length scale (times |e| unless fixed)
    fixed_length: bool = False,      # True: ignore |e| and use fixed length (direction only)
    percent_clip: float = 99.5,      # contrast clipping percentile
    title = None,
    fig_ax = None,     # optional (fig, ax); auto-created if not given
):
    """
    - Treat (e1,e2) as a spin-2 shape: principal-axis angle phi = 0.5 * atan2(e2, e1) (pi-periodic)
    - Draw one arrow along each +/-phi direction at the image center
    - If y_pred is provided: blue = pred, orange = gt; if only y_gt is given, draw gt only
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

    # ---- 2) normalization (in-place, before requires_grad) ----
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
            # e1 gradient
            go1 = torch.zeros_like(pred)
            go1[:, 0] = 1.0
            (gx1,) = torch.autograd.grad(
                outputs=pred, inputs=x, grad_outputs=go1,
                retain_graph=True, create_graph=False, allow_unused=False
            )

            # e2 gradient
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

# --------------------------
# Utils: geometry & sizing
# --------------------------


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


def gaussian_weight_2d(size: int | tuple[int, int], std: float = 16, device=None, normalize=True, dtype=torch.float32,
                       suppress: bool = True, r_cut: float = 24.0, suppress_order: int = 8):
    """
    Generate a 2D Gaussian weight image centered in the middle.

    Args:
        size: int or (H, W)
        std: standard deviation (in pixels)
        device: torch device (cpu / cuda)
        normalize: if True, normalize so that sum = 1
        suppress: if True, multiply by a smooth radial super-Gaussian taper
            T(r) = exp(-(r^2 / r_cut^2)^(suppress_order/2)), which is ~1 inside
            r_cut and strongly suppresses r > r_cut regardless of std.
            (T(r_cut) = e^-1; for order 8: T(0.5 r_cut) ~ 0.996, T(1.33 r_cut) ~ 3e-4)
        r_cut: taper radius (in pixels)
        suppress_order: even integer; larger -> sharper edge, flatter inside

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
    if suppress:
        # smooth (C-inf, written in r^2 so no cusp at r=0), isotropic outer taper
        g = g * torch.exp(-((xx**2 + yy**2) / r_cut**2) ** (suppress_order / 2))
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


# Differentiable D4 model and training primitives.
from src.architecture.CNN_module import D4Inv_Conv2d


class D4IsotropicConv2d(D4Inv_Conv2d):
    """Fourth-order low-frequency isotropy, preserving DC and second moment."""

    def _D4_avg(self, weight):
        w = super()._D4_avg(weight)
        second = w[..., 0, 1] + 2 * w[..., 0, 0]
        edge = 2 * second / 3
        corner = second / 6
        center = w.sum((-2, -1)) - 4 * edge - 4 * corner
        return torch.stack(
            (corner, edge, corner, edge, center, edge, corner, edge, corner),
            -1
        ).reshape_as(w)


class D4Residual(nn.Module):

    def __init__(self, in_channels, channels, res_factor, isotropic4=False):
        super().__init__()
        self.block = (D4IsotropicConv2d if isotropic4 else D4Inv_Conv2d)(
            in_channels,
            channels,
            k=3,
            s=1,
            p=1
        )
        self.skip = nn.Conv2d(
            in_channels,
            channels,
            1,
            bias=False
        ) if in_channels != channels else nn.Identity()
        if isinstance(self.skip, nn.Conv2d):
            nn.init.kaiming_normal_(self.skip.weight, nonlinearity='linear')
        self.res_factor = res_factor

    def forward(self, x):
        return self.block(x) + self.res_factor * self.skip(x)


class CubicEllipticityReadout(nn.Module):
    """Fixed spin-2 radial readout, the cubic truncation of 2*epsilon(chi).

    2*epsilon = chi + |chi|^2*chi/4 + O(|chi|^5). This polynomial is smooth
    for all network outputs; it is not the exact reduced ellipticity map.
    The coefficient comes from that identity, not a fit to test galaxies.
    Calibration must differentiate this complete composed predictor.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        e = self.model(x)
        return e * (1 + 0.25 * e.square().sum(-1, keepdim=True))


class ExplicitLayerNorm(nn.LayerNorm):

    def forward(self, x):
        axes = tuple(range(x.ndim - len(self.normalized_shape), x.ndim))
        centered = x - x.mean(axes, keepdim=True)
        out = centered * torch.rsqrt(centered.square().mean(axes, keepdim=True) + self.eps)
        if self.weight is not None:
            out = out * self.weight
        if self.bias is not None:
            out = out + self.bias
        return out


def use_explicit_layernorm(model):
    names = []
    for name, module in model.named_modules():
        if type(module) is nn.LayerNorm:
            # Same module state, shapes, Parameter objects and state_dict names.
            module.__class__ = ExplicitLayerNorm
            names.append(name)
    return names


class RadialAmplitudeReadout(nn.Module):
    """Fixed linear gain; legacy class name and alpha argument load old models."""

    def __init__(self, model, alpha, gain=1.0):
        super().__init__()
        self.model = model
        self.radial_alpha = 0.0  # Compression retired for all subsequent experiments.
        self.radial_gain = float(gain)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(super().__getattr__('model'), name)

    def forward(self, x):
        e = self.model(x)
        return self.radial_gain * e


class InputCrop63(torch.nn.Module):
    """Align a 64-pixel cutout's integer detection center with an odd grid.

    Crop top/left once. Autograd supplies zeros on removed pixels, so the
    unchanged 64-pixel AnaCal shear tangents give the exact cropped response.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.input_crop = 1

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(super().__getattr__('model'), name)

    def forward(self, x):
        if x.shape[-2:] != (64, 64):
            raise ValueError('InputCrop63 expects native64 Q images')
        return self.model(x[..., 1:, 1:])


class ObservableCenter(nn.Module):
    """D4-vector centroid with a smooth radial bound, differentiated in full.

    The odd 63 grid and the separable symmetric cubic translation commute
    with exact D4 pixel permutations; fx/fy/pad buffers are kept only for
    checkpoint compatibility. Gaussian-weighted signed flux is stabilized smoothly; no
    detached centroid, per-object truth, or calibrated shear offset is used.
    """

    def __init__(self, model, sigma=4.0, iterations=3, max_shift=3.0, pad=127, enabled=True):
        super().__init__()
        self.model = model
        # enabled=False is an exact pass-through; the wrapper, its buffers and
        # state_dict keys stay so checkpoints and attribute paths are unchanged.
        self.enabled = bool(enabled)
        # The wrapped predictor retains original pixels [1:,1:].
        self.input_crop = 1
        self.sigma, self.iterations = (float(sigma), int(iterations))
        self.max_shift, self.pad = (float(max_shift), int(pad))
        if self.pad < 63 or self.pad % 2 != 1:
            raise ValueError('Require odd padding at least 63')
        coord = torch.arange(-31, 32, dtype=torch.float64)
        yy, xx = torch.meshgrid(coord, coord, indexing='ij')
        self.register_buffer('xx', xx.clone())
        self.register_buffer('yy', yy.clone())
        freq = torch.fft.fftfreq(self.pad, dtype=torch.float64)
        self.register_buffer('fx', freq[None, :])
        self.register_buffer('fy', freq[:, None])

    def centroid(self, image):
        cx = image.new_zeros(len(image), 1, 1)
        cy = torch.zeros_like(cx)
        for _ in range(self.iterations):
            weight = torch.exp(-((self.xx - cx).square() + (self.yy - cy).square()) / (2 * self.sigma ** 2))
            weighted = image[:, 0] * weight
            flux = weighted.sum((-2, -1), keepdim=True)
            denominator = (flux.square() + 1.0).sqrt()
            nx = (weighted * self.xx).sum((-2, -1), keepdim=True) / denominator
            ny = (weighted * self.yy).sum((-2, -1), keepdim=True) / denominator
            bound = (1 + ((nx.square() + ny.square()) / self.max_shift ** 2).square()).pow(0.25)
            cx, cy = (nx / bound, ny / bound)
        return (cx, cy)

    def forward(self, value):
        if not self.enabled:
            return self.model(value)
        image = value[..., 1:, 1:]
        cx, cy = self.centroid(image)
        self.last_center = torch.cat((cx.flatten(1), cy.flatten(1)), dim=1)
        shifted = fourier_recenter(image, cx, cy, self.fx, self.fy, self.pad)
        return self.model(F.pad(shifted, (1, 0, 1, 0)))


class OddGELU(nn.Sequential):
    """Odd part of the entire biased MLP; half preserves the linear scale."""

    def forward(self, x):
        return 0.5 * (super().forward(x) - super().forward(-x))


class OddGELUDifference(OddGELU):
    """Literal f(x)-f(-x), retaining the user's requested amplitude convention."""

    def forward(self, x):
        return 2 * super().forward(x)


class AnchoredOddGELU(OddGELU):
    """Diagnostic restoring the old zero-input gate, not the primary odd model."""

    def __init__(self, *modules, anchor):
        super().__init__(*modules)
        self.register_buffer('anchor', anchor)

    def forward(self, x):
        return self.anchor + super().forward(x)


class TrainableRadial(nn.Module):
    """Trainable linear gain; the legacy radial name is checkpoint-compatible."""

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.log_gain = nn.Parameter(torch.zeros(()))
        self.radial_alpha = 0.0

    def forward(self, x):
        e = self.model(x)
        return self.log_gain.exp() * e


def response_torch(model, q, create_graph=True):
    x = q[:, 0:1].detach().requires_grad_(True)
    e = model(x)
    rows = []
    for axis in (0, 1):
        j = torch.autograd.grad(
            e[:, axis].sum(),
            x,
            create_graph=create_graph,
            retain_graph=True
        )[0]
        rows.append((j * q[:, 1:3]).sum((-2, -1)))
    return (e, torch.stack(rows, 1))


def jacobian_jvp(model, x, tangents):
    """Forward-mode directional derivatives of model(x).

    x [B,1,H,W], tangents [B,K,H,W] -> e [B,2], J [B,2,K] with
    J[:, a, k] = sum_pixels d e_a / d x * tangents[:, k]. The K tangents share
    one primal pass (vmap over jvp). Differentiable w.r.t. the parameters by
    ordinary (first-order) backward, replacing reverse-over-reverse.
    Model attributes set during this call (e.g. scalar_pooled) are transform
    wrappers; read them only after a plain model(x) call.
    """
    from torch.func import jvp, vmap
    x = x.detach()
    tangents = tangents.detach().transpose(0, 1).unsqueeze(2)    # [K,B,1,H,W]
    e, de = vmap(lambda t: jvp(model, (x,), (t,)))(tangents)
    return e[0], de.permute(1, 2, 0)


def response_jvp(model, q):
    """Same (e, R) as response_torch(model, q, create_graph=True), by forward mode."""
    return jacobian_jvp(model, q[:, 0:1], q[:, 1:3])


def geometry_inputs(q, angle, dx=0.0, dy=0.0):
    # Fixed linear image map. Its input is data, not a trainable tensor.
    n = len(q)
    c = np.cos(angle)
    s = np.sin(angle)
    a = q.new_tensor([[c, s, -dx / 31], [-s, c, -dy / 31]])[None].expand(n, -1, -1)
    grid = F.affine_grid(a, (n, 3, 63, 63), align_corners=True)
    rotated = F.grid_sample(
        q[..., 1:, 1:],
        grid,
        mode='bicubic',
        padding_mode='zeros',
        align_corners=True
    )
    spin = q.new_tensor([
        [np.cos(2 * angle), -np.sin(2 * angle)],
        [np.sin(2 * angle), np.cos(2 * angle)]
    ])
    rotated = torch.cat([rotated[:, :1], torch.einsum('jk,bkhw->bjhw', spin, rotated[:, 1:3])], 1)
    return (F.pad(rotated, (1, 0, 1, 0)), spin)


def rotate_quarter_turns(x: torch.Tensor, k: int) -> torch.Tensor:
    # NOTE: rot90+flip can create negative strides; we’ll call .contiguous() later once, globally.
    return torch.rot90(x, k % 4, dims=(-2, -1))


def reflect_y(x: torch.Tensor) -> torch.Tensor:
    return torch.flip(x, dims=[-2])


def d4_orbit_stack(x: torch.Tensor) -> torch.Tensor:
    """
    Returns stacked orbit as [8, B, C, H, W].
    We call .contiguous() once after stacking to avoid hidden copies in convs.
    """
    x0 = x
    x1 = rotate_quarter_turns(x, 1)
    x2 = rotate_quarter_turns(x, 2)
    x3 = rotate_quarter_turns(x, 3)
    # reflections
    xr0 = reflect_y(x0)
    xr1 = reflect_y(x1)
    xr2 = reflect_y(x2)
    xr3 = reflect_y(x3)
    # [8,B,C,H,W]
    xs = torch.stack([x0, x1, x2, x3, xr0, xr1, xr2, xr3], dim=0)
    # ensure positive strides for downstream convs
    return xs.contiguous()


def d4_inverse_stack(f8: torch.Tensor) -> torch.Tensor:
    """
    Bulk inverse on stacked features.
    Input:  f8 [8, B, C, H, W] in transform order above
    Output: inv_f8 [8, B, C, H, W], each map restored to canonical frame.
    """
    assert f8.dim() == 5 and f8.size(0) == 8
    # Inverses: for r^k -> r^{-k}; for r^k*sx -> r^{-k}*sx
    g0, g1, g2, g3, g4, g5, g6, g7 = f8.unbind(dim=0)
    # rotate back
    g0i = rotate_quarter_turns(g0, 0)
    g1i = rotate_quarter_turns(g1, -1)
    g2i = rotate_quarter_turns(g2, -2)
    g3i = rotate_quarter_turns(g3, -3)
    # reflect then rotate back
    g4i = rotate_quarter_turns(reflect_y(g4), 0)
    g5i = rotate_quarter_turns(reflect_y(g5), -1)
    g6i = rotate_quarter_turns(reflect_y(g6), -2)
    g7i = rotate_quarter_turns(reflect_y(g7), -3)
    inv = torch.stack([g0i, g1i, g2i, g3i, g4i, g5i, g6i, g7i], dim=0).contiguous()
    return inv


def _keys_cubic(s, a=-0.5):
    """Keys cubic-convolution kernel (Catmull-Rom for a=-0.5), support |s|<2."""
    s = s.abs()
    inner = ((a + 2) * s - (a + 3)) * s * s + 1
    outer = ((a * s - 5 * a) * s + 8 * a) * s - 4 * a
    return torch.where(s <= 1, inner, torch.where(s < 2, outer, torch.zeros_like(s)))


def fourier_recenter(image, cx, cy, fx=None, fy=None, pad=None):
    """Shift a field so that output(y, x) = image(y + cy, x + cx).

    Compact-support (4-tap Keys cubic, separable) interpolation with zero
    outside the stamp. The former global Fourier shift spread truncated
    edge flux over the whole stamp as Nyquist ringing, which the e1 matched
    derivative kernel amplifies and the e2 kernel cannot see; a 4-tap kernel
    keeps edge flux local. The name and the unused fx/fy/pad arguments are
    retained so existing callers and checkpoint buffers stay compatible.

    Every operation is elementary, so cx/cy and the image remain in the
    autograd graph to all orders. The kernel is symmetric and separable,
    hence exactly D4-covariant about the integer grid centre.
    """
    if cx.shape != (len(image), 1, 1) or cy.shape != cx.shape:
        raise ValueError('cx, cy must have shape [B,1,1]')
    with torch.no_grad():
        radius = int(torch.ceil(torch.maximum(cx.abs().max(), cy.abs().max()))) + 2
    taps = torch.arange(-radius, radius + 1, device=image.device, dtype=image.dtype)
    # Output pixel i reads input i+m with weight K(m - c).
    wx = _keys_cubic(taps - cx.reshape(-1, 1))  # [B, T]
    wy = _keys_cubic(taps - cy.reshape(-1, 1))
    h, w = image.shape[-2:]
    padded = F.pad(image, (radius, radius, radius, radius))
    rows = padded.new_zeros(*padded.shape[:-1], w)
    for k, m in enumerate(range(-radius, radius + 1)):
        rows = rows + padded[..., :, radius + m:radius + m + w] * wx[:, k, None, None, None]
    out = image.new_zeros(image.shape)
    for k, m in enumerate(range(-radius, radius + 1)):
        out = out + rows[..., radius + m:radius + m + h, :] * wy[:, k, None, None, None]
    return out


def matched_spin_basis(reference, size):
    """Matched axial/mixed spin-2 derivatives; size is 3 or 5 pixels."""
    if size == 3:
        d1 = reference.new_tensor([-0.5, 0, 0.5])
        d2 = reference.new_tensor([1.0, -2.0, 1.0])
    elif size == 5:
        d1 = reference.new_tensor([1 / 12, -2 / 3, 0, 2 / 3, -1 / 12])
        d2 = reference.new_tensor([-1 / 12, 4 / 3, -2.5, 4 / 3, -1 / 12])
    else:
        raise ValueError('Matched spin kernel size must be 3 or 5')
    delta = torch.zeros_like(d1)
    delta[size // 2] = 1
    return torch.stack((torch.outer(delta, d2) - torch.outer(d2, delta), 2 * torch.outer(d1, d1)))
