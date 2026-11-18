import torch
from torch import nn
import torch.nn.functional as F
import torchvision.transforms as T

from src.architecture.CNN_toolkit import (
    gaussian_weight_2d, rotate_quarter_turns, reflect_y,
    d4_orbit_stack, d4_inverse_stack,
)
from src.architecture.CNN_module import ConvGELU, BiasFreeMLP, BiasFreeMLP_2l, ConvGELU_Res, ConvGELU_layernorm

# ----------------------------
# Basic modules:
# ----------------------------


class Forward8_base(nn.Module):
    """
    Optimized base for D4-equivariant models:
    - Stacked D4 orbit (one trunk pass)
    - Bulk inverse mapping
    - Pre-shaped sign buffers
    """

    def __init__(self):
        super().__init__()
        # Transform order: [r0, r90, r180, r270, r0·sx, r90·sx, r180·sx, r270·sx]
        self.register_buffer("signs_e1", torch.tensor([+1, -1, +1, -1, +1, -1, +1, -1], dtype=torch.float32))
        self.register_buffer("signs_e2", torch.tensor([+1, -1, +1, -1, -1, +1, -1, +1], dtype=torch.float32))

        # Pre-shaped for fast broadcast in [8,1,1,1,1]
        #self.register_buffer("signs_e1_5d", self.signs_e1.view(8, 1, 1, 1, 1))
        #self.register_buffer("signs_e2_5d", self.signs_e2.view(8, 1, 1, 1, 1))

    # Shared exact pixel operations; historical method names stay compatible.
    _rot90_k = staticmethod(rotate_quarter_turns)
    _reflect_x = staticmethod(reflect_y)
    _d4_orbit_stack = staticmethod(d4_orbit_stack)
    _d4_inverse_stack = staticmethod(d4_inverse_stack)

    # ----- Trunk feature extraction (works on [N,C,H,W]) -----

    def _trunk_feature(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies conv blocks. If spatial shrink is constant (=6), we crop once.
        Works with x shaped [N,C,H,W] (N=B or 8B).
        """
        H = x.size(-2) 
        W = x.size(-1) 

        for blk in self.blocks:  # keep it simple; torch.compile will fuse this nicely
            x = blk(x)

        h_, w_ = x.size(-2), x.size(-1)
        x = x[:, :, (h_//2 - H//2):(h_//2 + H//2), (w_//2 - W//2):(w_//2 + W//2)]
        return x.contiguous()


# ------------------------------------------------------------
# Child model: SmoothCNN_GeLU
# ------------------------------------------------------------
class Forward8_fixW_CNN(Forward8_base):
    """
    D4-equivariant CNN model with Gaussian smoothing and weighted pooling.
    Input:  [B, in_dim, H, W]
    Output: [B, 2]  (e1, e2)
    """

    def __init__(
        self,
        in_dim=1,
        base_channels=32,
        head_hidden=128,
        out_dim=2,
        num_layers=5,
        res_factor=0.1,
        nl_sigma=16,
        gw_sigma=16,
        shared_head=False,
        suppress=True,
        input_size=None,
    ):
        super().__init__()
        assert out_dim == 2, "This D4-equivariant head assumes two shape components."
        C = base_channels
        self._w_cache = {}

        # --- CNN trunk ---
        self.blocks = nn.ModuleList(
            [ConvGELU_Res(in_dim, C, res_factor)] + [ConvGELU_Res(C, C, res_factor) for _ in range(max(0, num_layers - 1))]
        )
        # --- Heads ---
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.shared_head = bool(shared_head)
        if self.shared_head:
            self.head_shared = BiasFreeMLP(C, hidden=head_hidden)
        else:
            self.head_e1 = BiasFreeMLP(C, hidden=head_hidden)
            self.head_e2 = BiasFreeMLP(C, hidden=head_hidden)

        self.nl_sigma = nl_sigma
        self.gw_sigma = gw_sigma
        self.suppress = bool(suppress)
        self._set_input_size(input_size)

    # ============================================================
    # Optional central crop with unpadded (valid) convolutions
    # ============================================================

    def _set_input_size(self, input_size):
        """input_size (odd) crops input_size pixels centred on pixel h//2 at the
        start of forward, so normalization, features and pooling see only that region;
        the 3x3 convolutions then run unpadded, giving an odd feature grid of
        input_size - 2*num_layers centred on the same pixel. D4 stays exact.
        None keeps the original zero-padded behaviour."""
        if input_size is not None:
            if (isinstance(input_size, bool) or not isinstance(input_size, int)
                    or input_size % 2 == 0 or input_size - 2 * len(self.blocks) < 1):
                raise ValueError('input_size must be an odd integer larger than 2*num_layers')
            if any(b.conv.kernel_size != (3, 3) for b in self.blocks):
                raise ValueError('valid convolution assumes 3x3 kernels')
        self.input_size = input_size

    def _crop_input(self, x):
        n = getattr(self, 'input_size', None)
        if n is None:
            return x
        # Centre on pixel h//2: the integer detection pixel of a 64 cutout
        # (32) and the centre of InputCrop63's 63 grid (31, i.e. the same pixel).
        h = x.shape[-1]
        start = h // 2 - n // 2
        if x.ndim != 4 or x.shape[-2] != h or start < 0 or start + n > h:
            raise ValueError(f'cropped F8 needs a square input of at least {n} pixels')
        return x[..., start:start + n, start:start + n]

    @staticmethod
    def _pre_lowpass(z):
        """Fixed separable binomial [1,4,6,4,1]/16 on pre-activations.

        GELU(x) = x/2 + x^2/sqrt(2 pi) - O(x^4) has no cubic term, so keeping
        pre-activations below half Nyquist (|k| < pi/2) keeps the dominant
        quadratic sum frequencies below Nyquist: sub-pixel shifts then do not
        alias through the pointwise nonlinearity. Replicate padding on the
        square grid with a symmetric kernel commutes with D4 exactly.
        Implemented as shifted adds (faster than depthwise conv or banded matmul).
        """
        y = F.pad(z, (2, 2, 2, 2), mode='replicate')
        y = (y[..., :, :-4] + y[..., :, 4:]) + 4 * (y[..., :, 1:-3] + y[..., :, 3:-1]) + 6 * y[..., :, 2:-2]
        y = (y[..., :-4, :] + y[..., 4:, :]) + 4 * (y[..., 1:-3, :] + y[..., 3:-1, :]) + 6 * y[..., 2:-2, :]
        return y / 256

    def _valid_trunk(self, x):
        assert x.shape[-1] == x.shape[-2] == self.input_size
        lowpass = getattr(self, 'pre_lowpass', False)
        for block in self.blocks:
            residual = block.skip(x) if block.skip is not None else x
            z = block.norm(F.conv2d(x, block.conv.weight, block.conv.bias))
            y = block.act(self._pre_lowpass(z) if lowpass else z)
            x = y + block.residual_scale * residual[..., 1:-1, 1:-1]
        return x.contiguous()

    def _trunk_feature(self, x):
        if getattr(self, 'input_size', None) is not None:
            return self._valid_trunk(x)
        return super()._trunk_feature(x)

    # ============================================================
    # Weighted Global Average Pool
    # ============================================================

    @staticmethod
    def _weighted_gap(feat: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """
        Weighted global average pooling.
        feat: [B,C,H,W]
        w:    [B,1,H,W]
        """
        ws = w.sum(dim=(-2, -1), keepdim=True)
        v = (feat * w).sum(dim=(-2, -1), keepdim=True) / ws
        return v.squeeze(-1).squeeze(-1)  # [B,C]


    def _get_cached_weight(self, hw, sigma, device, dtype):
        key = (hw[0], hw[1], sigma, device.type, str(dtype))
        w = self._w_cache.get(key)
        if w is None:
            w = gaussian_weight_2d(hw, sigma, dtype=dtype, suppress=self.suppress).to(device=device)  # [H,W]
            w = w.unsqueeze(0).unsqueeze(0)  # [1,1,H,W] for cheap broadcast
            self._w_cache[key] = w
        return self._w_cache[key]

    def _pool_components(self, w):
        return (self._weighted_gap(self.f_mean_e1, w),
                self._weighted_gap(self.f_mean_e2, w))

    # ============================================================
    # Forward pass
    # ============================================================

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._crop_input(x)
        # --- Normalization ---
        w = self._get_cached_weight(x.shape[-2:], sigma=self.nl_sigma, device=x.device, dtype=x.dtype)
        m = (x*w).sum(dim=(1, 2, 3), keepdim=True)*100 # mean brightness
        norm = 1 + F.softplus(m - 1, beta=10.0, threshold=20)
        x = x / norm
        #x = asinh_norm_torch(x)

        # --- D4 orbit and inverse aggregation ---
        xs = self._d4_orbit_stack(x)              # [8,B,C,H,W]
        B = x.size(0)
        x8 = xs.view(-1, xs.size(2), xs.size(3), xs.size(4))  # [8B,C,H,W]

        # 2) Single trunk pass
        f8 = self._trunk_feature(x8)              # [8B,C,Hf,Wf]
        self.f8 = f8.view(8, B, f8.size(1), f8.size(2), f8.size(3))  # [8,B,C,Hf,Wf]

        # 3) Bulk inverse back to canonical frame
        self.feats = self._d4_inverse_stack(self.f8)        # [8,B,C,Hf,Wf]

        # --- Sign-weighted mean for e1/e2 --- 
        s1 = self.signs_e1.view(8, 1, 1, 1, 1).to(self.feats) 
        s2 = self.signs_e2.view(8, 1, 1, 1, 1).to(self.feats)
        # --- Sign-weighted mean for e1/e2 ---
        self.f_mean_e1 = (self.feats * s1).mean(dim=0)  # [B,C,Hf,Wf]
        self.f_mean_e2 = (self.feats * s2).mean(dim=0)  # [B,C,Hf,Wf]

        # --- Align weight map to feature map size (center crop) ---
        Hf, Wf = self.f_mean_e1.shape[-2:]
        w_f = self._get_cached_weight((Hf, Wf), sigma=self.gw_sigma, device=self.f_mean_e1.device, dtype=self.f_mean_e1.dtype) # [Hf,Wf]

        # --- Weighted pooling ---
        v1, v2 = self._pool_components(w_f)

        return self._readout(v1, v2)

    def _readout(self, v1, v2):
        """Map pooled spin-2 features [B,C] x2 to [B,2]; subclasses may override."""
        if self.shared_head:
            e1 = self.head_shared(v1)
            e2 = self.head_shared(v2)
        else:
            e1 = self.head_e1(v1)
            e2 = self.head_e2(v2)
        return torch.cat([e1, e2], dim=-1)  # [B,2]
    
class Forward8_poolmix_CNN(Forward8_fixW_CNN):
    """D4 CNN with a learned convex mixture of spatial pooling windows.

    ``pool_sigmas`` are Gaussian widths in feature pixels; ``None`` selects
    uniform global average pooling. Each window is normalized separately.
    Softmax weights start uniform and are shared across channels, objects,
    and both ellipticity components in the default global mode. With
    ``adaptive_pool=True``, a D4-invariant gate predicts weights per object,
    shared across channels and components. Mixing happens before the MLP heads.
    Input/output follow Forward8_fixW_CNN: [B, C, H, H] -> [B, 2].
    """

    def __init__(
        self, in_dim=1, base_channels=32, head_hidden=128, out_dim=2,
        num_layers=5, res_factor=0.1, nl_sigma=16,
        pool_sigmas=(4.0, 8.0, 16.0), shared_head=False,
        suppress=True, r_cut=24.0, suppress_order=8,
        adaptive_pool=False, pool_gate_hidden=32,
    ):
        pool_sigmas = tuple(pool_sigmas)
        if not pool_sigmas or any(
            s is not None and (not float(s) > 0 or not float(s) < float("inf"))
            for s in pool_sigmas
        ):
            raise ValueError("pool_sigmas must contain positive finite widths or None")
        if not 0 < float(nl_sigma) < float("inf"):
            raise ValueError("nl_sigma must be positive and finite")
        super().__init__(
            in_dim=in_dim, base_channels=base_channels, head_hidden=head_hidden,
            out_dim=out_dim, num_layers=num_layers, res_factor=res_factor,
            nl_sigma=nl_sigma, shared_head=shared_head,
        )
        if not 0 < float(r_cut) < float("inf") or suppress_order < 2 or suppress_order % 2:
            raise ValueError("r_cut must be positive and finite; suppress_order must be positive and even")
        self.suppress, self.r_cut, self.suppress_order = bool(suppress), float(r_cut), int(suppress_order)
        self.pool_sigmas = tuple(None if s is None else float(s) for s in pool_sigmas)
        self.adaptive_pool = bool(adaptive_pool)
        if self.adaptive_pool:
            if not isinstance(pool_gate_hidden, int) or pool_gate_hidden <= 0:
                raise ValueError("pool_gate_hidden must be a positive integer")
            self.pool_gate = nn.Sequential(
                nn.Linear(base_channels * len(self.pool_sigmas), pool_gate_hidden),
                nn.GELU(), nn.Linear(pool_gate_hidden, len(self.pool_sigmas)),
            )
            # Begin with uniform mixing; the gate learns object dependence.
            nn.init.zeros_(self.pool_gate[-1].weight)
            nn.init.zeros_(self.pool_gate[-1].bias)
            self._object_pool_weights = None
        else:
            self.pool_logits = nn.Parameter(torch.zeros(len(self.pool_sigmas)))

    @property
    def pool_weights(self):
        """Weights in sigma order: [K] globally, or [B,K] from the last forward.

        Adaptive weights retain their gradient path for response calibration.
        """
        if self.adaptive_pool:
            if self._object_pool_weights is None:
                raise RuntimeError("Adaptive pool weights require a forward pass first")
            return self._object_pool_weights
        return self.pool_logits.softmax(dim=0)

    def _pool_components(self, w):
        if self.adaptive_pool:
            # Unsigned orbit average is a scalar feature field under D4.
            # Radial pooling then makes the gate descriptor D4 invariant.
            invariant_features = self.feats.mean(dim=0)
            descriptor = self._pool_scales(invariant_features).flatten(1)
            self._object_pool_weights = self.pool_gate(descriptor).softmax(dim=-1)
        return super()._pool_components(w)

    def _get_cached_weight(self, hw, sigma, device, dtype):
        # Include the device index so replicas on different GPUs cannot share maps.
        key = (tuple(hw), sigma, torch.device(device), dtype)
        if key not in self._w_cache:
            self._w_cache[key] = gaussian_weight_2d(
                hw, sigma, device=device, dtype=dtype,
                suppress=self.suppress, r_cut=self.r_cut, suppress_order=self.suppress_order,
            )[None, None]
        return self._w_cache[key]

    def _trunk_feature(self, x):
        # These padded blocks preserve both even and odd spatial sizes.
        for block in self.blocks:
            x = block(x)
        return x.contiguous()

    def _pool_scales(self, feat):
        pooled = []
        for sigma in self.pool_sigmas:
            if sigma is None:
                pooled.append(feat.mean(dim=(-2, -1)))
            else:
                window = self._get_cached_weight(
                    feat.shape[-2:], sigma, feat.device, feat.dtype,
                )
                pooled.append(super()._weighted_gap(feat, window))
        return torch.stack(pooled, dim=-1)  # [B, C, K]

    def _weighted_gap(self, feat, w):
        values = self._pool_scales(feat)
        weights = self.pool_weights.to(values)
        if self.adaptive_pool:
            weights = weights.unsqueeze(1)  # [B, 1, K]
        return (values * weights).sum(dim=-1)

    def forward(self, x):
        if x.ndim != 4 or x.shape[-2] != x.shape[-1]:
            raise ValueError("Expected square images with shape [B, C, H, H]")
        return super().forward(x)


class Forward8_simp_CNN(Forward8_base):
    """
    D4-equivariant CNN model with Gaussian smoothing and weighted pooling.
    Input:  [B, in_dim, H, W]
    Output: [B, 2]  (e1, e2)
    """

    def __init__(
        self,
        in_dim=1,
        base_channels=16,
        head_hidden=64,
        out_dim=2,
        num_layers=3,
        res_factor=0,
        nl_sigma=6,
        gw_sigma=16,
    ):
        super().__init__()
        assert out_dim == 2, "This D4-equivariant head assumes two shape components."
        C = base_channels
        self._w_cache = {}

        # --- CNN trunk ---
        self.blocks = nn.ModuleList(
            [ConvGELU_Res(in_dim, C, res_factor)] + [ConvGELU_Res(C, C, res_factor) for _ in range(max(0, num_layers - 1))]
        )
        # --- Heads ---
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.head_e1 = BiasFreeMLP_2l(C, hidden=head_hidden)
        self.head_e2 = BiasFreeMLP_2l(C, hidden=head_hidden)

        self.nl_sigma = nl_sigma
        self.gw_sigma = gw_sigma

    # ============================================================
    # Weighted Global Average Pool
    # ============================================================

    @staticmethod
    def _weighted_gap(feat: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """
        Weighted global average pooling.
        feat: [B,C,H,W]
        w:    [B,1,H,W]
        """
        ws = w.sum(dim=(-2, -1), keepdim=True)
        v = (feat * w).sum(dim=(-2, -1), keepdim=True) / ws
        return v.squeeze(-1).squeeze(-1)  # [B,C]

    def _get_cached_weight(self, hw, sigma, device, dtype):
        key = (hw[0], hw[1], sigma, device.type, str(dtype))
        w = self._w_cache.get(key)
        if w is None:
            w = gaussian_weight_2d(hw, sigma).to(device=device, dtype=dtype)  # [H,W]
            w = w.unsqueeze(0).unsqueeze(0)  # [1,1,H,W] for cheap broadcast
            self._w_cache[key] = w
        return self._w_cache[key]

    # ============================================================
    # Forward pass
    # ============================================================

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # --- Normalization ---
        w = self._get_cached_weight(x.shape[-2:], sigma=self.nl_sigma, device=x.device, dtype=x.dtype)
        m = (x*w).sum(dim=(1, 2, 3), keepdim=True)*100 # mean brightness
        norm = 1 + F.softplus(m - 1, beta=10.0, threshold=20)
        x = x / norm
        #x = asinh_norm_torch(x)

        # --- D4 orbit and inverse aggregation ---
        xs = self._d4_orbit_stack(x)              # [8,B,C,H,W]
        B = x.size(0)
        x8 = xs.view(-1, xs.size(2), xs.size(3), xs.size(4))  # [8B,C,H,W]

        # 2) Single trunk pass
        f8 = self._trunk_feature(x8)              # [8B,C,Hf,Wf]
        f8 = f8.view(8, B, f8.size(1), f8.size(2), f8.size(3))  # [8,B,C,Hf,Wf]

        # 3) Bulk inverse back to canonical frame
        self.feats = self._d4_inverse_stack(f8)        # [8,B,C,Hf,Wf]

        # --- Sign-weighted mean for e1/e2 --- 
        s1 = self.signs_e1.view(8, 1, 1, 1, 1).to(self.feats) 
        s2 = self.signs_e2.view(8, 1, 1, 1, 1).to(self.feats)
        # --- Sign-weighted mean for e1/e2 ---
        self.f_mean_e1 = (self.feats * s1).mean(dim=0)  # [B,C,Hf,Wf]
        self.f_mean_e2 = (self.feats * s2).mean(dim=0)  # [B,C,Hf,Wf]

        # --- Align weight map to feature map size (center crop) ---
        Hf, Wf = self.f_mean_e1.shape[-2:]
        #w_f = self._get_cached_weight((Hf, Wf), sigma=self.gw_sigma, device=self.f_mean_e1.device, dtype=self.f_mean_e1.dtype) # [Hf,Wf]
        w_f = torch.ones((1,1,Hf,Wf), device=self.f_mean_e1.device, dtype=self.f_mean_e1.dtype)

        # --- Weighted pooling ---
        v1 = self._weighted_gap(self.f_mean_e1, w_f)
        v2 = self._weighted_gap(self.f_mean_e2, w_f)

        # --- Heads ---
        e1 = self.head_e1(v1)
        e2 = self.head_e2(v2)
        e = torch.cat([e1, e2], dim=-1)  # [B,2]
        return e



class Forward8_fixW_nores_CNN(Forward8_base):
    """
    D4-equivariant CNN model with Gaussian smoothing and weighted pooling.
    Input:  [B, in_dim, H, W]
    Output: [B, 2]  (e1, e2)
    """

    def __init__(
        self,
        in_dim=1,
        base_channels=32,
        head_hidden=128,
        out_dim=2,
        num_layers=5,
    ):
        super().__init__()
        assert out_dim == 2, "This D4-equivariant head assumes two shape components."
        C = base_channels
        self.inpad = nn.ConstantPad2d(10, 0.0)
        # --- CNN trunk ---
        self.blocks = nn.ModuleList(
            [ConvGELU_layernorm(in_dim, C)] + [ConvGELU_layernorm(C, C) for _ in range(max(0, num_layers - 1))]
        )
        # --- Heads ---
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.head_e1 = BiasFreeMLP(C, hidden=head_hidden)
        self.head_e2 = BiasFreeMLP(C, hidden=head_hidden)

    # ============================================================
    # Weighted Global Average Pool
    # ============================================================

    @staticmethod
    def _weighted_gap(feat: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """
        Weighted global average pooling.
        feat: [B,C,H,W]
        w:    [B,1,H,W]
        """
        ws = w.sum(dim=(-2, -1), keepdim=True)
        v = (feat * w).sum(dim=(-2, -1), keepdim=True) / ws
        return v.squeeze(-1).squeeze(-1)  # [B,C]

    def _trunk_feature(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extracts the feature map from the CNN trunk and crops
        to match the original spatial extent (minus convolutional shrinkage).
        """
        H = x.size(-2) - 6
        W = x.size(-1) - 6

        # Apply zero padding before convolution blocks
        x = self.inpad(x)

        # Sequentially apply Conv blocks
        for blk in self.blocks:
            x = blk(x)

        # Crop back to the central region
        h_, w_ = x.size(-2), x.size(-1)
        x = x[
            :,
            :,
            (h_ // 2 - H // 2):(h_ // 2 + H // 2),
            (w_ // 2 - W // 2):(w_ // 2 + W // 2),
        ]
        return x
    # ==========================================
    # Forward pass
    # ============================================================

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # --- Normalization ---
        w = gaussian_weight_2d((x.size(-2), x.size(-1))).to(x)
        m = (x*w).sum(dim=(1, 2, 3), keepdim=True)*100 # mean brightness
        norm = 0.5 + F.softplus(m - 0.5, beta=10.0, threshold=20)
        x = x / norm
        #x = asinh_norm_torch(x)

        # --- D4 orbit and inverse aggregation ---
        xs = self._d4_augment(x)
        self.feats = []
        for idx, xi in enumerate(xs):
            f = self._trunk_feature(xi)
            f_inv = self._d4_inverse(f, idx)
            self.feats.append(f_inv)
        self.feats = torch.stack(self.feats, dim=0)  # [8,B,C,H,W]

        # --- Sign-weighted mean for e1/e2 ---
        s1 = self.signs_e1.view(8, 1, 1, 1, 1).to(self.feats)
        s2 = self.signs_e2.view(8, 1, 1, 1, 1).to(self.feats)

        self.f_mean_e1 = (self.feats * s1).mean(dim=0)  # [B,C,H,W]
        self.f_mean_e2 = (self.feats * s2).mean(dim=0)  # [B,C,H,W]

        # --- Align weight map to feature map size (center crop) ---
        Hf, Wf = self.f_mean_e1.shape[-2:]
        w_f = gaussian_weight_2d((Hf, Wf), 16).to(self.f_mean_e1)  # [Hf,Wf]

        # --- Weighted pooling ---
        v1 = self._weighted_gap(self.f_mean_e1, w_f)
        v2 = self._weighted_gap(self.f_mean_e2, w_f)

        # --- Heads ---
        e1 = self.head_e1(v1)
        e2 = self.head_e2(v2)
        e = torch.cat([e1, e2], dim=-1)  # [B,2]

        return e
    
class SmoothCNN_GeLU_mocked(Forward8_base):
    """
    D4-equivariant CNN model with Gaussian smoothing and weighted pooling.
    Input:  [B, in_dim, H, W]
    Output: [B, 2]  (e1, e2)
    """

    def __init__(
        self,
        in_dim=1,
        base_channels=32,
        head_hidden=128,
        out_dim=2,
        num_layers=5,
    ):
        super().__init__()
        assert out_dim == 2, "This D4-equivariant head assumes two shape components."
        C = base_channels
        self.inpad = nn.ConstantPad2d(10, 0.0)
        # --- CNN trunk ---
        self.blocks = nn.ModuleList(
            [ConvGELU(in_dim, C)] + [ConvGELU(C, C) for _ in range(max(0, num_layers - 1))]
        )
        self.smooth = T.GaussianBlur(kernel_size=5, sigma=0.7)
        # --- Heads ---
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.head_e1 = BiasFreeMLP(C, hidden=head_hidden)
        self.head_e2 = BiasFreeMLP(C, hidden=head_hidden)

    # ============================================================
    # Weighted Global Average Pool
    # ============================================================

    @staticmethod
    def _weighted_gap(feat: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """
        Weighted global average pooling.
        feat: [B,C,H,W]
        w:    [B,1,H,W]
        """
        ws = w.sum(dim=(-2, -1), keepdim=True)
        v = (feat * w).sum(dim=(-2, -1), keepdim=True) / ws
        return v.squeeze(-1).squeeze(-1)  # [B,C]


    def _trunk_feature(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extracts the feature map from the CNN trunk and crops
        to match the original spatial extent (minus convolutional shrinkage).
        """
        H = x.size(-2) - 6
        W = x.size(-1) - 6

        # Apply zero padding before convolution blocks
        x = self.inpad(x)

        # Sequentially apply Conv blocks
        for blk in self.blocks:
            x = blk(x)

        # Crop back to the central region
        h_, w_ = x.size(-2), x.size(-1)
        x = x[
            :,
            :,
            (h_ // 2 - H // 2):(h_ // 2 + H // 2),
            (w_ // 2 - W // 2):(w_ // 2 + W // 2),
        ]
        return x
    # ============================================================
    # Forward pass
    # ============================================================

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # --- Normalization ---
        self.m   = x.sum(dim=(1, 2, 3), keepdim=True)         # [B,1,1,1]
        self.eps = x.std(dim=(1, 2, 3), keepdim=True)
        x = x / (self.m + self.eps.clamp(min=1e-6))          
        self.x = self.smooth(x)                                    # [B,in_dim,H,W]
        norm = torch.sqrt(self.x.pow(2).sum(dim=(-1, -2), keepdim=True) + 1e-6)  # [B,in_dim,1,1]
        w = self.x / norm
        w = w.mean(dim=1, keepdim=True)                 

        # --- D4 orbit and inverse aggregation ---
        xs = self._d4_augment(x)
        self.feats = []
        for idx, xi in enumerate(xs):
            f = self._trunk_feature(xi)
            f_inv = self._d4_inverse(f, idx)
            self.feats.append(f_inv)
        self.feats = torch.stack(self.feats, dim=0)  # [8,B,C,H,W]



        # --- Sign-weighted mean for e1/e2 ---
        s1 = self.signs_e1.view(8, 1, 1, 1, 1).to(self.feats)
        s2 = self.signs_e2.view(8, 1, 1, 1, 1).to(self.feats)

        self.f_mean_e1 = (self.feats * s1).mean(dim=0)  # [B,C,H,W]
        self.f_mean_e2 = (self.feats * s2).mean(dim=0)  # [B,C,H,W]

        Hf, Wf = self.f_mean_e1.shape[-2], self.f_mean_e1.shape[-1]
        Hw, Ww = w.shape[-2], w.shape[-1]
        if (Hw != Hf) or (Ww != Wf):
            dh = (Hw - Hf) // 2
            dw = (Ww - Wf) // 2
            w = w[:, :, dh:dh + Hf, dw:dw + Wf]   # center crop


        # --- Weighted pooling ---
        v1 = self._weighted_gap(self.f_mean_e1, w)
        v2 = self._weighted_gap(self.f_mean_e2, w)

        # --- Heads ---
        e1 = self.head_e1(v1)
        e2 = self.head_e2(v2)
        e = torch.cat([e1, e2], dim=-1)  # [B,2]

        return e
    
class Forward8_RELU_CNN(Forward8_base):
    """
    D4-equivariant CNN model with ReLU activation (instead of GeLU).
    Input:  [B, in_dim, H, W]
    Output: [B, 2]  (e1, e2)
    """

    def __init__(
        self,
        in_dim=1,
        base_channels=32,
        head_hidden=128,
        out_dim=2,
        num_layers=5,
        res_factor=0.1,
        nl_sigma=16,
        gw_sigma=16,
    ):
        super().__init__()
        assert out_dim == 2, "This D4-equivariant head assumes two shape components."
        C = base_channels
        self._w_cache = {}

        # --- CNN trunk with ReLU ---
        self.blocks = nn.ModuleList(
            [ConvReLU_Res(in_dim, C, res_factor)] + [ConvReLU_Res(C, C, res_factor) for _ in range(max(0, num_layers - 1))]
        )
        # --- Heads ---
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.head_e1 = BiasFreeMLP(C, hidden=head_hidden)
        self.head_e2 = BiasFreeMLP(C, hidden=head_hidden)

        self.nl_sigma = nl_sigma
        self.gw_sigma = gw_sigma

    # ============================================================
    # Weighted Global Average Pool
    # ============================================================

    @staticmethod
    def _weighted_gap(feat: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """
        Weighted global average pooling.
        feat: [B,C,H,W]
        w:    [B,1,H,W]
        """
        ws = w.sum(dim=(-2, -1), keepdim=True)
        v = (feat * w).sum(dim=(-2, -1), keepdim=True) / ws
        return v.squeeze(-1).squeeze(-1)  # [B,C]


    def _get_cached_weight(self, hw, sigma, device, dtype):
        key = (hw[0], hw[1], sigma, device.type, str(dtype))
        w = self._w_cache.get(key)
        if w is None:
            w = gaussian_weight_2d(hw, sigma, dtype=dtype).to(device=device)  # [H,W]
            w = w.unsqueeze(0).unsqueeze(0)  # [1,1,H,W] for cheap broadcast
            self._w_cache[key] = w
        return self._w_cache[key]

    # ============================================================
    # Forward pass
    # ============================================================

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # --- Normalization ---
        w = self._get_cached_weight(x.shape[-2:], sigma=self.nl_sigma, device=x.device, dtype=x.dtype)
        m = (x*w).sum(dim=(1, 2, 3), keepdim=True)*100 # mean brightness
        norm = 1 + F.softplus(m - 1, beta=10.0, threshold=20)
        x = x / norm

        # --- D4 orbit and inverse aggregation ---
        xs = self._d4_orbit_stack(x)              # [8,B,C,H,W]
        B = x.size(0)
        x8 = xs.view(-1, xs.size(2), xs.size(3), xs.size(4))  # [8B,C,H,W]

        # 2) Single trunk pass
        f8 = self._trunk_feature(x8)              # [8B,C,Hf,Wf]
        f8 = f8.view(8, B, f8.size(1), f8.size(2), f8.size(3))  # [8,B,C,Hf,Wf]

        # 3) Bulk inverse back to canonical frame
        self.feats = self._d4_inverse_stack(f8)        # [8,B,C,Hf,Wf]

        # --- Sign-weighted mean for e1/e2 --- 
        s1 = self.signs_e1.view(8, 1, 1, 1, 1).to(self.feats) 
        s2 = self.signs_e2.view(8, 1, 1, 1, 1).to(self.feats)
        # --- Sign-weighted mean for e1/e2 ---
        self.f_mean_e1 = (self.feats * s1).mean(dim=0)  # [B,C,Hf,Wf]
        self.f_mean_e2 = (self.feats * s2).mean(dim=0)  # [B,C,Hf,Wf]

        # --- Align weight map to feature map size (center crop) ---
        Hf, Wf = self.f_mean_e1.shape[-2:]
        w_f = self._get_cached_weight((Hf, Wf), sigma=self.gw_sigma, device=self.f_mean_e1.device, dtype=self.f_mean_e1.dtype) # [Hf,Wf]

        # --- Weighted pooling ---
        v1 = self._weighted_gap(self.f_mean_e1, w_f)
        v2 = self._weighted_gap(self.f_mean_e2, w_f)

        # --- Heads ---
        e1 = self.head_e1(v1)
        e2 = self.head_e2(v2)
        e = torch.cat([e1, e2], dim=-1)  # [B,2]
        return e

class NonEq_CNN(Forward8_base):
    def __init__(self, in_dim=1, base_channels=32, head_hidden=128, num_layers=5,
                 res_factor=0.1, nl_sigma=16, gw_sigma=16):
        super().__init__()
        C = base_channels
        self._w_cache = {}

        self.blocks = nn.ModuleList(
            [ConvGELU_Res(in_dim, C, res_factor)] +
            [ConvGELU_Res(C, C, res_factor) for _ in range(max(0, num_layers - 1))]
        )
        self.head = nn.Sequential(
            nn.Linear(C, head_hidden),
            nn.GELU(),
            nn.Linear(head_hidden, 2),
        )

        self.nl_sigma = nl_sigma
        self.gw_sigma = gw_sigma

    @staticmethod
    def _weighted_gap(feat, w):
        ws = w.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-12)
        v = (feat * w).sum(dim=(-2, -1), keepdim=True) / ws
        return v.squeeze(-1).squeeze(-1)

    def _get_cached_weight(self, hw, sigma, device, dtype):
        key = (hw[0], hw[1], sigma, device.type, str(dtype))
        w = self._w_cache.get(key)
        if w is None:
            w = gaussian_weight_2d(hw, sigma, dtype=dtype).to(device=device)
            w = w.unsqueeze(0).unsqueeze(0)
            self._w_cache[key] = w
        return self._w_cache[key]

    def forward(self, x):
        # same normalization as D4
        #w = self._get_cached_weight(x.shape[-2:], self.nl_sigma, x.device, x.dtype)
        w = self._get_cached_weight(x.shape[-2:], self.nl_sigma, x.device, x.dtype)
        m = (x*w).sum(dim=(1,2,3), keepdim=True)*100 # mean brightness
        norm = 1 + F.softplus(m - 1, beta=10.0, threshold=20)
        x = x / norm

        feat = self._trunk_feature(x)     
        Hf, Wf = feat.shape[-2:]
        w_f = self._get_cached_weight((Hf, Wf), self.gw_sigma, feat.device, feat.dtype)

        v = self._weighted_gap(feat, w_f)
        #v = feat.mean(dim=(-2, -1))  # simple global average pool
        e = self.head(v)
        return e


class _Forward8_AdaptiveW_CNN(Forward8_fixW_CNN):
    """Shared scalar-field pooling machinery; kernels retain their autograd path."""

    def __init__(self, in_dim=1, base_channels=32, head_hidden=128, out_dim=2,
                 num_layers=5, res_factor=0.1, nl_sigma=16, shared_head=False,
                 suppress=True, r_cut=24.0, suppress_order=8):
        if not 0 < float(nl_sigma) < float('inf'):
            raise ValueError('nl_sigma must be positive and finite')
        if not 0 < float(r_cut) < float('inf'):
            raise ValueError('r_cut must be positive and finite')
        if (not isinstance(suppress_order, int) or suppress_order < 2
                or suppress_order % 2):
            raise ValueError('suppress_order must be a positive even integer')
        super().__init__(in_dim, base_channels, head_hidden, out_dim, num_layers,
                         res_factor, nl_sigma, shared_head=shared_head)
        self.suppress = bool(suppress)
        self.r_cut = float(r_cut)
        self.suppress_order = suppress_order
        self._grid_cache = {}
        self._pool_kernel = None

    @property
    def pool_kernel(self):
        """Normalized [B,1,H,W] kernel from the last forward, without detach."""
        if self._pool_kernel is None:
            raise RuntimeError('pool_kernel requires a forward pass first')
        return self._pool_kernel

    def _get_cached_weight(self, hw, sigma, device, dtype):
        key = (tuple(hw), sigma, torch.device(device), dtype,
               self.suppress, self.r_cut, self.suppress_order)
        if key not in self._w_cache:
            self._w_cache[key] = gaussian_weight_2d(
                hw, sigma, device=device, dtype=dtype, suppress=self.suppress,
                r_cut=self.r_cut, suppress_order=self.suppress_order,
            )[None, None]
        return self._w_cache[key]

    def _trunk_feature(self, x):
        if getattr(self, 'input_size', None) is not None:
            return self._valid_trunk(x)
        # Preserve odd as well as even sizes, as in Forward8_poolmix_CNN.
        for block in self.blocks:
            x = block(x)
        return x.contiguous()

    def _coordinates(self, hw, device, dtype):
        key = (tuple(hw), torch.device(device), dtype)
        if key not in self._grid_cache:
            h, w = hw
            y = torch.arange(h, device=device, dtype=dtype) - (h - 1) / 2
            x = torch.arange(w, device=device, dtype=dtype) - (w - 1) / 2
            yy, xx = torch.meshgrid(y, x, indexing='ij')
            self._grid_cache[key] = (xx, yy)
        return self._grid_cache[key]

    def _normalize_logits(self, logits):
        # Keep exponentials and normalization in at least float32 under AMP.
        if logits.dtype in (torch.float16, torch.bfloat16):
            logits = logits.float()
        xx, yy = self._coordinates(logits.shape[-2:], logits.device, logits.dtype)
        if self.suppress:
            logits = logits - ((xx.square() + yy.square()) / self.r_cut**2).pow(
                self.suppress_order // 2)
        return logits.flatten(2).softmax(dim=-1).view_as(logits)

    def _pool_components(self, w):
        scalar_field = self.feats.mean(dim=0)
        self._pool_kernel = self._make_pool_kernel(scalar_field)
        return tuple((feat * self._pool_kernel).sum(dim=(-2, -1)).to(feat.dtype)
                     for feat in (self.f_mean_e1, self.f_mean_e2))

    def forward(self, x):
        if x.ndim != 4 or x.shape[-2] != x.shape[-1] or x.shape[-1] < 1:
            raise ValueError('Expected nonempty square images [B,C,H,H]')
        return super().forward(x)


class Forward8_AdaSigma_CNN(_Forward8_AdaptiveW_CNN):
    """Per-object Gaussian pooling with D4-invariant sigma = 1 + softplus(a).

    Sigma is measured in feature pixels and shared across channels and e1/e2.
    Orbit averaging followed by spatial averaging supplies the invariant gate.
    """

    def __init__(self, in_dim=1, base_channels=32, head_hidden=128, out_dim=2,
                 num_layers=5, res_factor=0.1, nl_sigma=16, shared_head=False,
                 suppress=True, r_cut=24.0, suppress_order=8,
                 sigma_init=8.0, sigma_hidden=32):
        super().__init__(in_dim, base_channels, head_hidden, out_dim, num_layers,
                         res_factor, nl_sigma, shared_head, suppress, r_cut,
                         suppress_order)
        if not 1 < float(sigma_init) < float('inf'):
            raise ValueError('sigma_init must be finite and greater than 1')
        if not isinstance(sigma_hidden, int) or sigma_hidden <= 0:
            raise ValueError('sigma_hidden must be a positive integer')
        self.sigma_gate = nn.Sequential(
            nn.Linear(base_channels, sigma_hidden), nn.GELU(),
            nn.Linear(sigma_hidden, 1))
        nn.init.zeros_(self.sigma_gate[-1].weight)
        # Stable inverse softplus, including large initial widths.
        import math
        delta = float(sigma_init) - 1.0
        nn.init.constant_(self.sigma_gate[-1].bias,
                          delta + math.log(-math.expm1(-delta)))
        self._pool_sigma = None

    @property
    def pool_sigma(self):
        """[B,1] widths from the last forward, retaining the gradient path."""
        if self._pool_sigma is None:
            raise RuntimeError('pool_sigma requires a forward pass first')
        return self._pool_sigma

    def _make_pool_kernel(self, scalar_field):
        raw = self.sigma_gate(scalar_field.mean(dim=(-2, -1)))
        if raw.dtype in (torch.float16, torch.bfloat16):
            raw = raw.float()
        self._pool_sigma = 1 + F.softplus(raw)
        xx, yy = self._coordinates(scalar_field.shape[-2:], raw.device, raw.dtype)
        logits = -0.5 * (xx.square() + yy.square()) / self._pool_sigma[..., None, None].square()
        return self._normalize_logits(logits)


class Forward8_AttnW_CNN(_Forward8_AdaptiveW_CNN):
    """Smooth per-object pooling from low-resolution D4-equivariant attention.

    Position channels (x^2+y^2, x^2*y^2) are D4 invariant. The resulting
    kernel follows the object under D4; it need not itself be radial.
    """

    def __init__(self, in_dim=1, base_channels=32, head_hidden=128, out_dim=2,
                 num_layers=5, res_factor=0.1, nl_sigma=16, shared_head=False,
                 suppress=True, r_cut=24.0, suppress_order=8,
                 attn_grid_size=16, attn_dim=32, attn_heads=4,
                 head_num_hidden_layers=3):
        super().__init__(in_dim, base_channels, head_hidden, out_dim, num_layers,
                         res_factor, nl_sigma, shared_head, suppress, r_cut,
                         suppress_order)
        if any(not isinstance(v, int) or v <= 0
               for v in (attn_grid_size, attn_dim, attn_heads)):
            raise ValueError('Attention sizes must be positive integers')
        if attn_dim % attn_heads:
            raise ValueError('attn_dim must be divisible by attn_heads')
        if isinstance(num_layers, bool) or not isinstance(num_layers, int) or num_layers < 1:
            raise ValueError('num_layers must be a positive integer')
        if not 0 <= float(res_factor) <= 1:
            raise ValueError('res_factor must be finite and in [0, 1]')
        if isinstance(head_num_hidden_layers, bool) or not isinstance(head_num_hidden_layers, int) or head_num_hidden_layers < 1:
            raise ValueError('head_num_hidden_layers must be a positive integer')
        # Default depth preserves historical initialization and state_dict keys.
        if head_num_hidden_layers != 3:
            if self.shared_head:
                self.head_shared = BiasFreeMLP(base_channels, hidden=head_hidden,
                                              num_hidden_layers=head_num_hidden_layers)
            else:
                self.head_e1 = BiasFreeMLP(base_channels, hidden=head_hidden,
                                          num_hidden_layers=head_num_hidden_layers)
                self.head_e2 = BiasFreeMLP(base_channels, hidden=head_hidden,
                                          num_hidden_layers=head_num_hidden_layers)
        self.attn_grid_size, self.attn_heads = attn_grid_size, attn_heads
        self.token_projection = nn.Linear(base_channels + 2, attn_dim)
        self.attn_norm = nn.LayerNorm(attn_dim)
        self.qkv = nn.Linear(attn_dim, 3 * attn_dim)
        self.attn_output = nn.Linear(attn_dim, attn_dim)
        self.ffn_norm = nn.LayerNorm(attn_dim)
        self.ffn = nn.Sequential(nn.Linear(attn_dim, 2 * attn_dim), nn.GELU(),
                                 nn.Linear(2 * attn_dim, attn_dim))
        self.kernel_head = nn.Linear(attn_dim, 1)
        nn.init.zeros_(self.kernel_head.weight)
        nn.init.zeros_(self.kernel_head.bias)
        coord = torch.arange(-2, 3, dtype=torch.float64)
        blur = torch.exp(-0.5 * (coord[:, None].square() + coord[None, :].square()))
        self.register_buffer('kernel_blur', (blur / blur.sum())[None, None])

    def _make_pool_kernel(self, scalar_field):
        b, _, h, w = scalar_field.shape
        size = min(self.attn_grid_size, h)
        field = F.adaptive_avg_pool2d(scalar_field, (size, size))
        xx, yy = self._coordinates((size, size), field.device, field.dtype)
        scale = max((size - 1) / 2, 1)
        xx, yy = xx / scale, yy / scale
        position = torch.stack((xx.square() + yy.square(), xx.square() * yy.square()))
        tokens = torch.cat((field, position[None].expand(b, -1, -1, -1)), dim=1)
        tokens = self.token_projection(tokens.flatten(2).transpose(1, 2))
        n, d = tokens.shape[1:]
        qkv = self.qkv(self.attn_norm(tokens)).reshape(
            b, n, 3, self.attn_heads, d // self.attn_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        # Explicit operations support higher derivatives without fused backends.
        if q.dtype in (torch.float16, torch.bfloat16):
            q, k, v = q.float(), k.float(), v.float()
        scores = (q @ k.transpose(-2, -1)) * ((d // self.attn_heads) ** -0.5)
        context = (scores.softmax(dim=-1) @ v).transpose(1, 2).reshape(b, n, d)
        tokens = tokens + self.attn_output(context.to(tokens.dtype))
        tokens = tokens + self.ffn(self.ffn_norm(tokens))
        logits = self.kernel_head(tokens).transpose(1, 2).reshape(b, 1, size, size)
        if logits.dtype in (torch.float16, torch.bfloat16):
            logits = logits.float()
        logits = F.interpolate(logits, size=(h, w), mode='bilinear', align_corners=False)
        logits = F.conv2d(F.pad(logits, (2, 2, 2, 2), mode='replicate'),
                          self.kernel_blur.to(logits))
        return self._normalize_logits(logits)


class F8_noTanh(Forward8_AttnW_CNN):
    """Forward8_AttnW_CNN trunk/pooling with the D4inv_noTanh scalar-conditioned readout.

    Shapes (default): pooled scalar [B,C], pooled spin features [B,C,2],
    channel coefficients [B,C], output [B,2]. The scalar is the orbit mean of
    the canonical-frame features (D4 invariant), pooled with the same
    attention kernel as the spin-2 sign-weighted means.
    Readout = mean_channels(MLP(scalar / scalar_scale) * spin).

    The MLP is Linear-GELU-Linear with no output activation; its signed
    coefficients are shared by e1 and e2, so the output stays spin-2 under D4.
    scalar_scale is a fixed per-channel training-set RMS buffer (default ones);
    set it once before optimization from training descriptors only.

    fixed_pool_sigma (feature pixels): replace the attention kernel by an exact
    normalized Gaussian centred on the feature grid; attention modules are
    removed. input_size (odd; implemented in Forward8_fixW_CNN): crop the
    central input_size pixels first, so
    the brightness normalization and everything after it see only that
    region; the 3x3 convolutions run unpadded (valid), so every feature pixel
    is computed from real data and the feature grid, input_size -
    2*num_layers, stays odd and centred on the detection pixel. The output
    depends on no pixel outside the crop. D4 stays exact.
    """

    def __init__(self, in_dim=1, base_channels=32, readout_hidden=64,
                 scalar_scale=None, fixed_pool_sigma=None, input_size=None,
                 pool_parabola=None, pool_quartic=None, **kwargs):
        if (isinstance(readout_hidden, bool)
                or not isinstance(readout_hidden, int) or readout_hidden < 1):
            raise ValueError('readout_hidden must be a positive integer')
        super().__init__(in_dim=in_dim, base_channels=base_channels, **kwargs)
        if fixed_pool_sigma is not None and not 0 < float(fixed_pool_sigma) < float('inf'):
            raise ValueError('fixed_pool_sigma must be finite and positive')
        if sum(v is not None for v in (fixed_pool_sigma, pool_parabola, pool_quartic)) > 1:
            raise ValueError('choose at most one of fixed_pool_sigma, pool_parabola, pool_quartic')
        self.fixed_pool_sigma = None if fixed_pool_sigma is None else float(fixed_pool_sigma)
        self.pool_parabola = None
        if pool_parabola is not None:
            r0, blur = (float(v) for v in pool_parabola)
            if not (0 < r0 < float('inf') and 0 <= blur < r0):
                raise ValueError('pool_parabola must be (r0 > 0, 0 <= blur < r0)')
            self.pool_parabola = (r0, blur)
        self.pool_quartic = None
        if pool_quartic is not None:
            if not 0 < float(pool_quartic) < float('inf'):
                raise ValueError('pool_quartic must be finite and positive')
            self.pool_quartic = float(pool_quartic)
        if (self.fixed_pool_sigma is not None or self.pool_parabola is not None
                or self.pool_quartic is not None):
            for name in ('token_projection', 'attn_norm', 'qkv', 'attn_output',
                         'ffn_norm', 'ffn', 'kernel_head', 'kernel_blur'):
                delattr(self, name)
        self._set_input_size(input_size)
        # The scalar MLP replaces the per-component BiasFreeMLP heads.
        for name in ('head_shared', 'head_e1', 'head_e2'):
            if hasattr(self, name):
                delattr(self, name)
        self.scalar_readout = nn.Sequential(
            nn.Linear(base_channels, readout_hidden),
            nn.GELU(),
            nn.Linear(readout_hidden, base_channels),
        )
        nn.init.normal_(self.scalar_readout[-1].weight, std=1e-3)
        nn.init.ones_(self.scalar_readout[-1].bias)
        self.register_buffer('scalar_scale', torch.ones(base_channels))
        if scalar_scale is not None:
            self.set_scalar_scale(scalar_scale)

    @torch.no_grad()
    def set_scalar_scale(self, scale):
        """Install fixed training-set channel RMS; no per-object normalization."""
        scale = torch.as_tensor(scale, device=self.scalar_scale.device,
                                dtype=self.scalar_scale.dtype)
        if (scale.shape != self.scalar_scale.shape
                or not torch.isfinite(scale).all() or not (scale > 0).all()):
            raise ValueError('scalar_scale must be a finite positive [channels] vector')
        self.scalar_scale.copy_(scale)

    @staticmethod
    def smooth_parabola_kernel(hw, r0, blur, oversample=8):
        """Normalized (1 - r^2/r0^2)_+ convolved with a Gaussian of width blur (pixels).

        Pooling term from sub-pixel translation is driven by the spin-2 Hessian
        of the kernel, (K'' - K'/r) cos 2theta for radial K, which vanishes
        exactly for K = a - b r^2. Gaussian blurring keeps a quadratic quadratic,
        so the kernel is translation-neutral for r < r0 - ~3 blur and only the
        smoothed rim contributes. Built once in float64 on a D4-symmetric
        oversampled grid, then symmetrized exactly over D4.
        """
        h, w = hw
        y = torch.arange(h, dtype=torch.float64) - (h - 1) / 2
        x = torch.arange(w, dtype=torch.float64) - (w - 1) / 2
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        if blur == 0:
            kernel = (1 - (xx.square() + yy.square()) / r0**2).clamp(min=0)
        else:
            extent = r0
            n = int(2 * extent * oversample)
            fine = (torch.arange(n, dtype=torch.float64) + 0.5) / oversample - extent
            fy, fx = torch.meshgrid(fine, fine, indexing='ij')
            hard = (1 - (fx.square() + fy.square()) / r0**2).clamp(min=0).flatten()
            keep = hard > 0
            fx, fy, hard = fx.flatten()[keep], fy.flatten()[keep], hard[keep]
            d2 = (xx.flatten()[:, None] - fx[None]).square() + (yy.flatten()[:, None] - fy[None]).square()
            kernel = (torch.exp(-0.5 * d2 / blur**2) @ hard).reshape(h, w)
        kernel = kernel + kernel.flip(0) + kernel.flip(1) + kernel.flip(0).flip(1)
        kernel = kernel + kernel.transpose(0, 1)
        return kernel / kernel.sum()

    @staticmethod
    def quartic_kernel(hw, a):
        """Normalized flat-top window exp(-(r/a)^4) sampled at feature-pixel centres.

        Smooth everywhere (no rim discontinuity). Effective area
        sum(K)/K(0) = pi^1.5 a^2 / 2, so a = 4.5 matches the r0 = 6 paraboloid.
        Built in float64; the centred grid makes it exactly D4-symmetric.
        """
        h, w = hw
        y = torch.arange(h, dtype=torch.float64) - (h - 1) / 2
        x = torch.arange(w, dtype=torch.float64) - (w - 1) / 2
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        kernel = torch.exp(-((xx.square() + yy.square()) / a**2).square())
        return kernel / kernel.sum()

    def _make_pool_kernel(self, scalar_field):
        if getattr(self, 'pool_quartic', None) is not None:
            key = ('quartic', tuple(scalar_field.shape[-2:]), scalar_field.device, scalar_field.dtype)
            if key not in self._w_cache:
                self._w_cache[key] = self.quartic_kernel(
                    scalar_field.shape[-2:], self.pool_quartic).to(
                    device=scalar_field.device, dtype=scalar_field.dtype)[None, None]
            return self._w_cache[key].expand(scalar_field.shape[0], -1, -1, -1)
        if getattr(self, 'pool_parabola', None) is not None:
            key = ('parabola', tuple(scalar_field.shape[-2:]), scalar_field.device, scalar_field.dtype)
            if key not in self._w_cache:
                self._w_cache[key] = self.smooth_parabola_kernel(
                    scalar_field.shape[-2:], *self.pool_parabola).to(
                    device=scalar_field.device, dtype=scalar_field.dtype)[None, None]
            return self._w_cache[key].expand(scalar_field.shape[0], -1, -1, -1)
        if self.fixed_pool_sigma is None:
            return super()._make_pool_kernel(scalar_field)
        # Exact normalized Gaussian in feature pixels; no additional taper.
        xx, yy = self._coordinates(scalar_field.shape[-2:], scalar_field.device, scalar_field.dtype)
        logits = -0.5 * (xx.square() + yy.square()) / self.fixed_pool_sigma**2
        kernel = logits.flatten().softmax(0).reshape(1, 1, *logits.shape)
        return kernel.expand(scalar_field.shape[0], -1, -1, -1)

    def _has_fixed_kernel(self):
        return not (self.pool_quartic is None and self.pool_parabola is None
                    and self.fixed_pool_sigma is None)

    def _fixed_kernel(self, hw, device, dtype):
        """[H,W] D4-symmetric pooling kernel, or None for attention pooling."""
        if not self._has_fixed_kernel():
            return None
        key = ('fixed', tuple(hw), device, dtype)
        if key not in self._w_cache:
            if self.pool_quartic is not None:
                kernel = self.quartic_kernel(hw, self.pool_quartic)
            elif self.pool_parabola is not None:
                kernel = self.smooth_parabola_kernel(hw, *self.pool_parabola)
            else:
                xx, yy = self._coordinates(hw, 'cpu', torch.float64)
                kernel = (-0.5 * (xx.square() + yy.square()) / self.fixed_pool_sigma**2).flatten().softmax(0).view(hw)
            self._w_cache[key] = kernel.to(device=device, dtype=dtype)
        return self._w_cache[key]

    def forward(self, x):
        """With a fixed kernel K, pool each D4 view in its own frame.

        K(g u) = K(u), so sum_u K(u) (g^-1 f_g)(u) = sum_u K(u) f_g(u): the
        inverse stack and the [8,B,C,H,W] orbit means are skipped. Same output
        up to summation order. Set inframe_pool = False to restore the
        full-field path (feats, f_mean_e1/e2), e.g. for primal traces.
        """
        if x.ndim != 4 or x.shape[-2] != x.shape[-1] or x.shape[-1] < 1:
            raise ValueError('Expected nonempty square images [B,C,H,H]')
        if not getattr(self, 'inframe_pool', True) or not self._has_fixed_kernel():
            return super().forward(x)
        x = self._crop_input(x)
        w = self._get_cached_weight(x.shape[-2:], sigma=self.nl_sigma, device=x.device, dtype=x.dtype)
        m = (x * w).sum(dim=(1, 2, 3), keepdim=True) * 100
        x = x / (1 + F.softplus(m - 1, beta=10.0, threshold=20))
        b = x.size(0)
        f8 = self._trunk_feature(self._d4_orbit_stack(x).flatten(0, 1))  # [8B,C,h,w]
        kernel = self._fixed_kernel(f8.shape[-2:], f8.device, f8.dtype)
        # Elementwise product-sum (not matmul): unaffected by TF32 matmul settings.
        pooled = (f8 * kernel).sum(dim=(-2, -1)).view(8, b, -1)          # [8,B,C]
        v1 = (self.signs_e1.to(pooled).view(8, 1, 1) * pooled).mean(0)
        v2 = (self.signs_e2.to(pooled).view(8, 1, 1) * pooled).mean(0)
        self.f8 = self.feats = self.f_mean_e1 = self.f_mean_e2 = None
        self._pool_kernel = kernel[None, None].expand(b, -1, -1, -1)
        self.scalar_pooled = pooled.mean(0)
        self.spin_pooled = torch.stack((v1, v2), dim=-1)
        self.channel_weights = self.scalar_readout(self.scalar_pooled / self.scalar_scale)
        return (self.spin_pooled * self.channel_weights[..., None]).mean(dim=1)

    def _readout(self, v1, v2):
        # pool_kernel already sums to one over pixels.
        scalar = self.feats.mean(dim=0)
        self.scalar_pooled = (scalar * self.pool_kernel).sum(dim=(-2, -1)).to(scalar.dtype)
        self.spin_pooled = torch.stack((v1, v2), dim=-1)
        self.channel_weights = self.scalar_readout(self.scalar_pooled / self.scalar_scale)
        return (self.spin_pooled * self.channel_weights[..., None]).mean(dim=1)


class WindowedChannelNorm2d(nn.Module):
    """Channel norm with one window-weighted scale per image, replacing ChannelLayerNorm2d.

    ChannelLayerNorm2d divides every pixel by its own channel std, so pixels at
    1e-3 of the source peak come out O(1) and the spin density no longer
    follows the light (it peaks on the pooling-window rim, which produces the
    translation term). Here each pixel is still centred over channels (a linear
    map), but all pixels share one scale:

        s^2 = sum_u W(u) var_c(u) / sum_u W(u),   y = (x - mean_c x) / sqrt(s^2 + eps)

    W is fixed, centred on the odd feature grid (the detection pixel) and
    D4-symmetric, so D4 stays exact. Spatial amplitude ratios survive the norm.
    window: ('parabola', r0, blur) uses F8_noTanh.smooth_parabola_kernel, which
    is exactly zero beyond r0 + ~3 blur, so far neighbours do not enter s;
    ('gaussian', sigma); ('global',) is a uniform W (GroupNorm-like scale).
    Parameter names and shapes match ChannelLayerNorm2d.
    """

    def __init__(self, num_channels, window, eps=1e-5, share_orbit=False):
        super().__init__()
        kind = window[0]
        if kind not in ('parabola', 'gaussian', 'global'):
            raise ValueError(f'unknown norm window {window!r}')
        self.window = tuple(window)
        self.eps = eps
        # share_orbit: one scale per object, averaged over the 8 orbit-major D4
        # views ([8B] batch). Per-view scales differ for a non-equivariant
        # trunk and act as a hidden orientation/position-dependent channel.
        self.share_orbit = bool(share_orbit)
        self.weight = nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self._cache = {}

    def _window(self, hw, device, dtype):
        key = (tuple(hw), device, dtype)
        if key not in self._cache:
            kind = self.window[0]
            if kind == 'parabola':
                w = F8_noTanh.smooth_parabola_kernel(hw, *self.window[1:])
            elif kind == 'gaussian':
                h, wd = hw
                y = torch.arange(h, dtype=torch.float64) - (h - 1) / 2
                x = torch.arange(wd, dtype=torch.float64) - (wd - 1) / 2
                yy, xx = torch.meshgrid(y, x, indexing='ij')
                w = torch.exp(-0.5 * (xx.square() + yy.square()) / self.window[1] ** 2)
            else:
                w = torch.ones(*hw, dtype=torch.float64)
            self._cache[key] = (w / w.sum()).to(device=device, dtype=dtype)
        return self._cache[key]

    def forward(self, x):
        centred = x - x.mean(dim=1, keepdim=True)
        variance = centred.square().mean(dim=1)                       # [N,H,W]
        scale = (variance * self._window(x.shape[-2:], x.device, x.dtype)).sum((-2, -1))
        if self.share_orbit:
            if scale.shape[0] % 8:
                raise ValueError('share_orbit expects an orbit-major [8B] batch')
            scale = scale.view(8, -1).mean(0).repeat(8)
        return centred * torch.rsqrt(scale + self.eps)[:, None, None, None] * self.weight + self.bias


class QuadAct(nn.Module):
    """Per-channel quadratic initialized to the GELU Taylor coefficient."""
    def __init__(self, channels, coefficient=0.4):
        super().__init__()
        self.c = nn.Parameter(torch.full((1, channels, 1, 1), float(coefficient)))

    def forward(self, z):
        return z / 2 + self.c * z.square()


# Fixed-Gaussian, cropped F8_noTanh: 5 unpadded layers on the central 43x43
# (feature grid 33x33 = +-4 sigma), centred on the integer detection pixel.
F8_RECIPES = {
    'f8c_sigma4_rdsm': dict(depth=5, target='rdsm', ssl=False, parent=None, forward8=True,
                            fixed_pool_sigma=4., input_size=43),
    'f8c_sigma4_truth': dict(depth=5, target='truth', ssl=False, parent=None, forward8=True,
                             fixed_pool_sigma=4., input_size=43),
    # Same as f8c_sigma4_* except pooling: smoothed paraboloid, r0=6, blur 0.5
    # feature pixels (translation-neutral core r < ~4.5).
    'f8c_parab6_rdsm': dict(depth=5, target='rdsm', ssl=False, parent=None, forward8=True,
                            pool_parabola=(6., .5), input_size=43),
    'f8c_parab6_truth': dict(depth=5, target='truth', ssl=False, parent=None, forward8=True,
                             pool_parabola=(6., .5), input_size=43),
    # Same as f8c_sigma4_* except pooling: exp(-(r/4.5)^4), same effective
    # area as parab6 (~56 feature pixels^2).
    'f8c_quart45_rdsm': dict(depth=5, target='rdsm', ssl=False, parent=None, forward8=True,
                             pool_quartic=4.5, input_size=43),
    'f8c_quart45_truth': dict(depth=5, target='truth', ssl=False, parent=None, forward8=True,
                              pool_quartic=4.5, input_size=43),
}
# Same as f8c_parab6_* except the trunk norm: WindowedChannelNorm2d instead of
# the per-pixel ChannelLayerNorm2d. wnp4: smoothed paraboloid r0=4, blur .5
# (target core only; few noise pixels). wng6: Gaussian sigma=6 (wide, many
# pixels, sees more of the neighbours). Widths in feature pixels.
for _name, _window in (('wnp4', ('parabola', 4., .5)), ('wng6', ('gaussian', 6.))):
    for _target in ('truth', 'rdsm'):
        F8_RECIPES[f'f8c_parab6_{_name}_{_target}'] = dict(
            depth=5, target=_target, ssl=False, parent=None, forward8=True,
            pool_parabola=(6., .5), input_size=43, norm_window=_window)
del _name, _window, _target
# Same as f8c_parab6_wng6_* plus a fixed binomial low-pass on every
# pre-activation (anti-aliasing at ~3% cost). wng6lp keeps per-view norm
# scales; wng6lps also shares the norm scale over the 8 D4 views.
for _name, _share in (('wng6lp', False), ('wng6lps', True)):
    for _target in ('truth', 'rdsm'):
        F8_RECIPES[f'f8c_parab6_{_name}_{_target}'] = dict(
            depth=5, target=_target, ssl=False, parent=None, forward8=True,
            pool_parabola=(6., .5), input_size=43, norm_window=('gaussian', 6.),
            norm_share_orbit=_share, pre_lowpass=True)
del _name, _share, _target


# Shared D4 norm, no low-pass; readout retains GELU.
for _depth in (2, 3, 5):
    _prefix = 'f8c' if _depth == 5 else f'f8c{_depth}'
    F8_RECIPES[f'{_prefix}_parab6_wng6s_truth'] = dict(
        depth=_depth, target='truth', ssl=False, parent=None, forward8=True,
        pool_parabola=(6., .5), input_size=43, norm_window=('gaussian', 6.),
        norm_share_orbit=True, act='gelu')
for _act in ('softplus', 'tanh', 'quad'):
    F8_RECIPES[f'f8c_parab6_wng6s_truth_{_act}'] = dict(
        F8_RECIPES['f8c_parab6_wng6s_truth'], act=_act)
del _depth, _prefix, _act


# Pool-radius ablation: identical trunk, normalization and training target.
F8_RECIPES['f8c_parab10_wng6s_truth'] = dict(
    F8_RECIPES['f8c_parab6_wng6s_truth'], pool_parabola=(10., .5))


# Depth/width controls: all other parab10 settings and the 64-unit readout stay fixed.
for _depth in (1, 2, 3, 4, 5):
    for _channels in (8, 16, 24, 32):
        F8_RECIPES[f'f8c_parab10_d{_depth}c{_channels}_truth'] = dict(
            F8_RECIPES['f8c_parab10_wng6s_truth'], depth=_depth, channels=_channels,
            input_size=33 + 2 * _depth)
del _depth, _channels


def build_f8_model(kind, seed=42, state=None):
    """Random-init (or state-restored) F8 recipe with the common input wrappers:
    InputCrop63 (odd grid on the detection pixel), linear trainable gain, and
    ObservableCenter disabled."""
    from src.architecture.CNN_toolkit import (
        use_explicit_layernorm, InputCrop63, ObservableCenter, TrainableRadial)
    recipe = F8_RECIPES[kind]
    torch.manual_seed(seed)
    base = F8_noTanh(
        num_layers=recipe['depth'], base_channels=recipe.get('channels', 32), res_factor=0.25, head_hidden=128,
        nl_sigma=16.0, suppress=True, r_cut=24.0, suppress_order=8, head_num_hidden_layers=1,
        attn_grid_size=16, attn_dim=32, attn_heads=4,
        fixed_pool_sigma=recipe.get('fixed_pool_sigma'), input_size=recipe.get('input_size'),
        pool_parabola=recipe.get('pool_parabola'), pool_quartic=recipe.get('pool_quartic'))
    use_explicit_layernorm(base)
    if recipe.get('norm_window') is not None:
        for block in base.blocks:
            block.norm = WindowedChannelNorm2d(block.norm.weight.shape[1], recipe['norm_window'],
                                               eps=block.norm.eps,
                                               share_orbit=recipe.get('norm_share_orbit', False))
    activation = recipe.get('act', 'gelu')
    if activation not in ('gelu', 'softplus', 'tanh', 'quad'):
        raise ValueError(f'unknown trunk activation {activation!r}')
    for block in base.blocks:
        if activation == 'softplus':
            block.act = nn.Softplus(beta=1)
        elif activation == 'tanh':
            block.act = nn.Tanh()
        elif activation == 'quad':
            block.act = QuadAct(block.norm.weight.shape[1])
    base.pre_lowpass = bool(recipe.get('pre_lowpass', False))
    model = ObservableCenter(InputCrop63(TrainableRadial(base)), sigma=2.0, enabled=False).float()
    if state is not None:
        model.load_state_dict(state, strict=True)
    return model


def load_f8_checkpoint(path, *, device='cpu', dtype=torch.float64):
    """Strictly restore a checkpoint written by src.train_d4 for an F8_RECIPES kind."""
    saved = torch.load(path, map_location='cpu', weights_only=False)
    if saved.get('kind') not in F8_RECIPES:
        raise ValueError(f"not an F8_RECIPES checkpoint: {saved.get('kind')}")
    model = build_f8_model(saved['kind'], saved['seed'], saved['state_dict'])
    model.model.model.radial_alpha = 0.0
    return model.to(device=device, dtype=dtype).eval()
