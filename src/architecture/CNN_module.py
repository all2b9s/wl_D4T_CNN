import torch
from torch import nn
import torch.nn.functional as F


class ChannelLayerNorm2d(nn.Module):
    def __init__(self, num_channels, eps=1e-5, affine=True):
        super().__init__()
        self.eps = eps
        if affine:
            self.weight = nn.Parameter(torch.ones(1, num_channels, 1, 1))
            self.bias   = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        else:
            self.weight = self.bias = None

    def forward(self, x):
        # x: [N,C,H,W]
        mean = x.mean(dim=1, keepdim=True)
        var  = (x - mean).pow(2).mean(dim=1, keepdim=True)
        y = (x - mean) * torch.rsqrt(var + self.eps)
        if self.weight is not None:
            y = y * self.weight + self.bias
        return y



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
    def _rot180(W):  # 180-degree rotation = flip up-down + left-right
        return torch.flip(W, dims=(-2, -1))

    def forward(self, x):
        W_even = 0.5 * (self.P + self._rot180(self.P))
        x = F.conv2d(x, W_even, self.bias, stride=self.s, padding=self.p)
        return self.act(x)
    

class D4Inv_Conv2d(nn.Module):
    """
    Rot 180° invariant convolution with channel-only normalization (via nn.LayerNorm).
    """
    def __init__(self, in_ch, out_ch, k=3, s=1, p=0, bias=True):
        super().__init__()
        self.P = nn.Parameter(
            torch.randn(out_ch, in_ch, k, k) * (2.0 / (in_ch * k * k)) ** 0.5
        )
        self.bias = nn.Parameter(torch.zeros(out_ch)) if bias else None
        self.s, self.p = s, p

        # LayerNorm applied across channels only (for each spatial location)
        self.norm = nn.LayerNorm(out_ch, elementwise_affine=True)
        self.act = nn.GELU()

    @staticmethod
    def _rot90(W):
        return torch.rot90(W, 1, dims=(-2, -1))

    @staticmethod
    def _mirror(W):
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
        return torch.stack(Ws, dim=0).mean(dim=0)

    def forward(self, x):
        self.W_D4 = self._D4_avg(self.P)
        x = F.conv2d(x, self.W_D4, self.bias, stride=self.s, padding=self.p)

        # LayerNorm normalizes over the channel dimension only
        x = x.permute(0, 2, 3, 1)            # [B, H, W, C]
        x = self.norm(x)                     # normalize per pixel across C
        x = x.permute(0, 3, 1, 2).contiguous()  # back to [B, C, H, W]

        x = self.act(x)
        return x



class ConvGELU(nn.Module):
    """ReflectionPad2d + Conv2d(3x3, stride=1) + GeLU"""
    def __init__(self, in_ch, out_ch,
                bn_eps: float = 1e-5,
                bn_momentum: float = 0.1,):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=0, bias=True)
        self.bn = nn.BatchNorm2d(out_ch, eps=bn_eps, momentum=bn_momentum)
        self.act  = nn.GELU()

        # Kaiming init
        nn.init.kaiming_normal_(self.conv.weight, nonlinearity='relu')
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x
    
class ConvGELU_Res(nn.Module):
    """
    ReflectionPad2d + Conv2d(3×3) + LayerNorm(GroupNorm) + GeLU
    + 0.1 * residual connection

    Residual path helps preserve input information and stabilize training,
    while LayerNorm (implemented via GroupNorm(1, C)) prevents batch
    statistic coupling between low- and high-S/N samples.
    """

    def __init__(self, in_ch, out_ch, residual_scale: float = 1, kernel_size: int =3):
        super().__init__()
        self.residual_scale = residual_scale
        #self.pad = nn.ReflectionPad2d(1)
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=1, padding=kernel_size//2, bias=True)
        self.norm = ChannelLayerNorm2d(out_ch, eps=1e-5, affine=True)
        self.act = nn.GELU()

        # optional 1x1 skip if channel mismatch
        self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, bias=False) if in_ch != out_ch else None

        # initialization
        nn.init.kaiming_normal_(self.conv.weight, nonlinearity="relu")
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)
        if self.skip is not None:
            nn.init.kaiming_normal_(self.skip.weight, nonlinearity="linear")

    def forward(self, x):
        residual = self.skip(x) if self.skip is not None else x
        y = self.conv(x)
        y = self.norm(y)
        y = self.act(y)
        return y + self.residual_scale * residual

class ConvReLU_Res(nn.Module):
    """
    Conv2d(3×3) + LayerNorm(GroupNorm) + ReLU
    + residual connection

    Same as ConvGELU_Res but with ReLU activation.
    """

    def __init__(self, in_ch, out_ch, residual_scale: float = 1, kernel_size: int =3):
        super().__init__()
        self.residual_scale = residual_scale
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=1, padding=kernel_size//2, bias=True)
        self.norm = ChannelLayerNorm2d(out_ch, eps=1e-5, affine=True)
        self.act = nn.ReLU()

        # optional 1x1 skip if channel mismatch
        self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, bias=False) if in_ch != out_ch else None

        # initialization
        nn.init.kaiming_normal_(self.conv.weight, nonlinearity="relu")
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)
        if self.skip is not None:
            nn.init.kaiming_normal_(self.skip.weight, nonlinearity="linear")

    def forward(self, x):
        residual = self.skip(x) if self.skip is not None else x
        y = self.conv(x)
        y = self.norm(y)
        y = self.act(y)
        return y + self.residual_scale * residual

class ConvGELU_layernorm(nn.Module):
    """
    ReflectionPad2d + Conv2d(3×3) + LayerNorm(GroupNorm) + GeLU
    + 0.1 * residual connection

    Residual path helps preserve input information and stabilize training,
    while LayerNorm (implemented via GroupNorm(1, C)) prevents batch
    statistic coupling between low- and high-S/N samples.
    """

    def __init__(self, in_ch, out_ch, residual_scale: float = 0.1):
        super().__init__()
        self.residual_scale = residual_scale
        #self.pad = nn.ReflectionPad2d(1)
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=0, bias=True)
        self.norm = nn.LayerNorm(out_ch, elementwise_affine=True)  # per-channel normalization
        self.act = nn.GELU()


        # initialization
        nn.init.kaiming_normal_(self.conv.weight, nonlinearity="relu")
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        y = self.conv(x)

        # Apply normalization along channels only
        # LayerNorm expects [N, H, W, C], so we permute
        y = y.permute(0, 2, 3, 1)
        y = self.norm(y)
        y = y.permute(0, 3, 1, 2)

        y = self.act(y)
        return y 


class BiasFreeMLP(nn.Module):
    """
    Bias-free MLP head (3 hidden layers + Tanh) used to predict a single
    shape component from a pooled feature vector.
    """
    def __init__(self, in_dim: int, hidden: int = 128, activation: Literal['tanh', 'relu'] = 'tanh'):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_dim, hidden, bias=False),
            nn.ReLU() if activation == 'relu' else nn.Tanh(),
            nn.Linear(hidden, hidden, bias=False),
            nn.ReLU() if activation == 'relu' else nn.Tanh(),
            nn.Linear(hidden, hidden, bias=False),
            nn.ReLU() if activation == 'relu' else nn.Tanh(),
            nn.Linear(hidden, 1, bias=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
    
class BiasFreeMLP_2l(nn.Module):
    """
    Bias-free 2-layer MLP head (Tanh) used to predict a single shape component.
    """
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


class ResConvBNGELU(nn.Module):
    """
    Residual Conv block: Conv-BN-GELU -> Conv-BN, with skip connection.
    - Preserves H×W when stride=1 (same-padding).
    - Uses 1×1 projection if in/out channels or stride differ.
    - BatchNorm is fine for batch sizes ~1e2.

    Args:
        in_c (int): input channels
        out_c (int): output channels
        k (int): kernel size (odd recommended)
        stride (int): conv stride for both convs (projection uses same stride)
        dilation (int): dilation for both convs
        groups (int): conv groups (keep 1 for standard conv)
        bn_eps (float): BatchNorm eps
        bn_momentum (float): BatchNorm momentum
        dropout_p (float): optional Dropout2d after the second BN (default 0.0 = off)
    """
    def __init__(
        self,
        in_c: int,
        out_c: int,
        k: int = 3,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bn_eps: float = 1e-5,
        bn_momentum: float = 0.1,
        dropout_p: float = 0.0,
    ):
        super().__init__()
        assert k >= 1 and isinstance(k, int), "kernel size must be int >= 1"
        assert stride >= 1 and dilation >= 1
        p = (k // 2) * dilation  # 'same' padding for odd k

        self.conv1 = nn.Conv2d(
            in_c, out_c, kernel_size=k, stride=stride, padding=p,
            dilation=dilation, groups=groups, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_c, eps=bn_eps, momentum=bn_momentum)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(
            out_c, out_c, kernel_size=k, stride=1, padding=p,
            dilation=dilation, groups=groups, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_c, eps=bn_eps, momentum=bn_momentum)

        # Optional dropout after BN2 (kept off by default)
        self.drop = nn.Dropout2d(dropout_p) if dropout_p and dropout_p > 0.0 else nn.Identity()

        # Projection for skip path if shape changes
        needs_proj = (in_c != out_c) or (stride != 1)
        self.proj = (
            nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_c, eps=bn_eps, momentum=bn_momentum),
            )
            if needs_proj else nn.Identity()
        )

        #self._init_weights()

    def _init_weights(self):
        # Kaiming for GELU; BN to sensible defaults
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="gelu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.drop(out)

        out = out + self.proj(identity)
        out = self.act(out)
        return out

