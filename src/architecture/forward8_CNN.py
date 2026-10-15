import torch
from torch import nn
import torch.nn.functional as F
import torchvision.transforms as T

from src.architecture.CNN_toolkit import gaussian_weight_2d
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

    # ----- D4 transforms (vectorized) -----

    @staticmethod
    def _rot90_k(x: torch.Tensor, k: int) -> torch.Tensor:
        # NOTE: rot90+flip can create negative strides; we’ll call .contiguous() later once, globally.
        return torch.rot90(x, k % 4, dims=(-2, -1))

    @staticmethod
    def _reflect_x(x: torch.Tensor) -> torch.Tensor:
        return torch.flip(x, dims=[-2])

    def _d4_orbit_stack(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns stacked orbit as [8, B, C, H, W].
        We call .contiguous() once after stacking to avoid hidden copies in convs.
        """
        x0 = x
        x1 = self._rot90_k(x, 1)
        x2 = self._rot90_k(x, 2)
        x3 = self._rot90_k(x, 3)

        # reflections
        xr0 = self._reflect_x(x0)
        xr1 = self._reflect_x(x1)
        xr2 = self._reflect_x(x2)
        xr3 = self._reflect_x(x3)

        xs = torch.stack([x0, x1, x2, x3, xr0, xr1, xr2, xr3], dim=0)  # [8,B,C,H,W]
        return xs.contiguous()  # ensure positive strides for downstream convs

    def _d4_inverse_stack(self, f8: torch.Tensor) -> torch.Tensor:
        """
        Bulk inverse on stacked features.
        Input:  f8 [8, B, C, H, W] in transform order above
        Output: inv_f8 [8, B, C, H, W], each map restored to canonical frame.
        """
        assert f8.dim() == 5 and f8.size(0) == 8
        # Inverses: for r^k -> r^{-k}; for r^k*sx -> r^{-k}*sx
        g0, g1, g2, g3, g4, g5, g6, g7 = f8.unbind(dim=0)

        # rotate back
        g0i = self._rot90_k(g0, 0)
        g1i = self._rot90_k(g1, -1)
        g2i = self._rot90_k(g2, -2)
        g3i = self._rot90_k(g3, -3)

        # reflect then rotate back
        g4i = self._rot90_k(self._reflect_x(g4), 0)
        g5i = self._rot90_k(self._reflect_x(g5), -1)
        g6i = self._rot90_k(self._reflect_x(g6), -2)
        g7i = self._rot90_k(self._reflect_x(g7), -3)

        inv = torch.stack([g0i, g1i, g2i, g3i, g4i, g5i, g6i, g7i], dim=0).contiguous()
        return inv

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

        # --- Heads ---
        if self.shared_head:
            e1 = self.head_shared(v1)
            e2 = self.head_shared(v2)
        else:
            e1 = self.head_e1(v1)
            e2 = self.head_e2(v2)
        e = torch.cat([e1, e2], dim=-1)  # [B,2]
        return e
    
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
