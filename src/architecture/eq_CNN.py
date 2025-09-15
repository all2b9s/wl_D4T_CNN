import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from wenyinli.codes.single_galaxies.src.architecture.CNN_toolkit import d4_variants, center_crop_to, gaussian_blur


# --------------------------
# First layer with shared weights across D4 variants
# --------------------------
class FirstSharedD4Conv(nn.Module):
    """
    Applies the same Conv2d (in=1 -> out=C) to each of the 8 D4 variants,
    then averages across the 8 to get [B, C, H', W'].
    """
    def __init__(self, out_channels: int, kernel_size: int = 3, bias: bool = True):
        super().__init__()
        self.conv = nn.Conv2d(1, out_channels, kernel_size=kernel_size, bias=bias, padding=0)
        self.act = nn.GELU()  # optional activation, can be removed if not needed

    def forward(self, x1: torch.Tensor) -> torch.Tensor:
        """
        x1: [B, 1, H, W]  (already padded upstream)
        returns: [B, C, H_out, W_out]
        """
        v = d4_variants(x1)                           # [B, 8, 1, H, W]
        B, V, _, H, W = v.shape
        v = v.reshape(B * V, 1, H, W)                 # fuse variants into batch
        y = self.conv(v)                               # [B*8, C, H', W']
        C = y.shape[1]
        y = y.reshape(B, V, C, y.shape[-2], y.shape[-1]).mean(dim=1)  # avg over 8 variants
        y = self.act(y)                               # [B, C, H', W']
        return y  # [B, C, H', W']

# --------------------------
# Bias-free MLP with tanh
# --------------------------
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

# --------------------------
# Main model
# --------------------------
class D4_eq_CNN(nn.Module):
    """
    Input:  x in [B, 1, H, W]
    Output: [B, 2] = [shape_1, shape_0]
    """
    def __init__(
        self,
        base_channels: int = 32,
        num_layers: int = 5,
        kernel_size: int = 3,
        pad_pixels: int = 10,
        mlp_hidden: int = 128,
        blur_ks: int = 3,
        blur_sigma: float = 1.0,
    ):
        super().__init__()
        self.pad_pixels = pad_pixels
        self.blur_ks = blur_ks
        self.blur_sigma = blur_sigma

        # One-time padding before any convs
        self.inpad = nn.ConstantPad2d(pad_pixels, 0.0)

        # First layer: shared weights across the 8 D4 variants
        self.first = FirstSharedD4Conv(out_channels=base_channels, kernel_size=kernel_size, bias=True)

        # Subsequent conv blocks via nn.Sequential + for loop, GeLU activations
        blocks = []
        C = base_channels
        for _ in range(num_layers - 1):
            blocks += [
                nn.Conv2d(C, C, kernel_size=kernel_size, padding=0, bias=True),  # keep 'valid', we crop later
                nn.GELU(),
            ]
        self.backbone = nn.Sequential(*blocks)

        # Global average pooling to a vector
        self.gap = nn.AdaptiveAvgPool2d(1)

        # Two separate bias-free MLP heads (tanh inside)
        self.mlp_w1 = BiasFreeMLP(in_dim=C, hidden=mlp_hidden)  # -> shape_1
        self.mlp_w2 = BiasFreeMLP(in_dim=C, hidden=mlp_hidden)  # -> shape_0



    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, 1, H, W]
        returns: [B, 2] = [shape_1, shape_0]
        """
        B, _, H, W = x.shape

        # Build the two weights (on original size)
        self.w1, self.w2 = self._build_weights(x)  # [B,1,H,W] each

        # One-time padding
        xp = self.inpad(x)  # [B,1,H+2p,W+2p]
        y = self.first(xp)  # [B,C,H',W'] (smaller due to valid conv)
        y = self.backbone(y)  # still valid convs, shrinking further
        y = center_crop_to(y, (H, W))  # [B,C,H,W]

        y1 = y * self.w1  # [B,C,H,W]
        y2 = y * self.w2  # [B,C,H,W]

        # Global average pool -> [B,C,1,1] -> [B,C]
        #z1 = self.gap(y1).squeeze(-1).squeeze(-2)  # [B,C]
        #z2 = self.gap(y2).squeeze(-1).squeeze(-2)  # [B,C]
        z1 = y1.sum(dim = (-1,-2))/torch.abs(self.w1).sum(dim=(-1,-2))  # [B,C]
        z2 = y2.sum(dim = (-1,-2))/torch.abs(self.w2).sum(dim=(-1,-2))  # [B,C]

        # Two bias-free MLPs with tanh inside
        shape_0 = self.mlp_w1(z1)  # [B,1]
        shape_1 = self.mlp_w2(z2)  # [B,1]

        out = torch.cat([shape_0, shape_1], dim=-1)  # [B,2], order: [shape_0, shape_1]
        return out
    
    # ---- weight construction helpers ----
    @staticmethod
    def _rot90(x: torch.Tensor) -> torch.Tensor:
        return torch.rot90(x, k=1, dims=(-2, -1))

    @staticmethod
    def _mirror(x: torch.Tensor) -> torch.Tensor:
        return torch.flip(x, dims=(-1,))  # horizontal flip

    def _build_weights(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: [B, 1, H, W]  (un-padded original image)
        Returns blurred w1, w2 each shaped [B, 1, H, W]
        """
        xr = self._rot90(x)
        xrr = self._rot90(xr)
        xrrr = self._rot90(xrr)
        diff = (x - xr + xrr - xrrr)/4  # [B, 1, H, W] (shape0 - shape1)
        m_diff = self._mirror(diff)
        w0 = diff + m_diff # shape0 weight
        w1 = diff - m_diff # shape1 weight

        # Low-pass (Gaussian blur)
        w0 = gaussian_blur(w0, ks=self.blur_ks, sigma=self.blur_sigma)
        w1 = gaussian_blur(w1, ks=self.blur_ks, sigma=self.blur_sigma)
        return w0, w1

# --------------------------
# Example
# --------------------------
if __name__ == "__main__":
    B, H, W = 2, 64, 64
    x = torch.randn(B, 1, H, W)

    model = D4_eq_CNN(
        base_channels=32,
        num_layers=5,
        kernel_size=3,
        pad_pixels=10,
        mlp_hidden=128,
        blur_ks=9,
        blur_sigma=3.0,
    )

    y = model(x)
    print(y.shape)  # [2, 2] -> [shape_1, shape_0]
