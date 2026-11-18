"""D4-invariant scalar trunk with spin-2 shape output and adaptive pooling.

The current model evaluates symmetrized kernels and matched derivatives once;
there is no eight-image Forward8 pass. The inherited src attention scaffold
preserves historical parameter names and initialization. ``linear_f8`` is a
legacy constructor flag, not eight-view execution in the current configuration.

Use build_single_stage_model() for random-init training and
load_d4_checkpoint() for a selected or single-stage checkpoint. The core class
also accepts historical research configs; the factories select the current one.
"""
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from src.architecture.forward8_CNN import Forward8_AttnW_CNN, F8_noTanh
from src.architecture.CNN_module import frozen_weight_cache
from src.architecture.CNN_toolkit import (
    D4Residual,
    D4IsotropicConv2d,
    matched_spin_basis,
    use_explicit_layernorm,
    ObservableCenter,
    InputCrop63,
    RadialAmplitudeReadout,
    TrainableRadial,
    OddGELU,
    OddGELUDifference,
    CubicEllipticityReadout,
)


class D4InvCNN(Forward8_AttnW_CNN):
    """Current D4 scalar/spin-2 core, before centering/crop/radial wrappers.

    Defaults are the tested two-layer, isotropic, matched-derivative Tanh core.
    build_single_stage_model() adds the complete 64-pixel input pipeline.
    """

    def __init__(
        self,
        in_dim=1,
        d4_layers=2,
        base_channels=32,
        res_factor=0.25,
        shared_head=False,
        linear_f8=True,
        linear_head=True,
        matched_spin_kernel=5,
        spin_gate=True,
        isotropic_front=True,
        **kwargs
    ):
        if isinstance(d4_layers, bool) or not isinstance(d4_layers, int) or d4_layers < 1:
            raise ValueError('d4_layers must be a positive integer')
        if base_channels < 2:
            raise ValueError('Channel LayerNorm requires at least two front-end channels')
        kwargs.setdefault('head_num_hidden_layers', 1)
        super().__init__(
            in_dim=base_channels,
            num_layers=1,
            base_channels=base_channels,
            res_factor=res_factor,
            shared_head=shared_head,
            **kwargs
        )
        self.linear_f8 = bool(linear_f8)
        self.linear_head = bool(linear_head)
        self.matched_spin_kernel = int(matched_spin_kernel)
        if self.matched_spin_kernel not in (0, 3, 5) or (
            self.matched_spin_kernel
            and (not self.linear_f8 or not self.linear_head or shared_head)
        ):
            raise ValueError(
                'Matched spin kernels require linear F8 and independent linear heads; size3 or5'
            )
        if self.linear_head:
            heads = [self.head_shared] if shared_head else [self.head_e1, self.head_e2]
            for head in heads:
                for i, layer in enumerate(head.net):
                    if isinstance(layer, nn.Tanh):
                        head.net[i] = nn.Identity()
        self.spin_gate = bool(spin_gate)
        self.isotropic_front = bool(isotropic_front)
        if self.spin_gate:
            if not self.matched_spin_kernel:
                raise ValueError('Spin gate requires a matched spin derivative kernel')
            self.spin_gate_net = nn.Sequential(
                nn.Linear(base_channels, 64),
                nn.Tanh(),
                nn.Linear(64, base_channels)
            )
            nn.init.zeros_(self.spin_gate_net[-1].weight)
            nn.init.zeros_(self.spin_gate_net[-1].bias)
            self.register_buffer('spin_feature_scale', torch.ones(base_channels))
        self.input_channels = in_dim
        self.d4_layers = d4_layers
        self.d4_blocks = nn.Sequential(*[
            D4Residual(
                in_dim if i == 0 else base_channels,
                base_channels,
                res_factor,
                self.isotropic_front,
            )
            for i in range(d4_layers)
        ])

    def _matched_spin_components(self, inverse_weights):
        """One shared channel field and analytically matched spin-2 derivatives.

        The 5-point formulas remove the unequal O(k^2) relative errors of
        the axial second difference and diagonal mixed difference. All
        parameter dependence stays in autograd; no fitted test correction.
        """
        weight = inverse_weights[0]
        yy, xx = torch.meshgrid(
            torch.arange(-1, 2, dtype=weight.dtype, device=weight.device),
            torch.arange(-1, 2, dtype=weight.dtype, device=weight.device),
            indexing='ij'
        )
        vectors = []
        for axis in (1, 2):
            signs = getattr(self, 'signs_e' + str(axis)).to(weight)
            kernel = (inverse_weights * signs.view(8, 1, 1, 1, 1)).mean(0)
            linear_head = getattr(self, 'head_e' + str(axis))
            channel_basis = torch.eye(kernel.shape[0], dtype=weight.dtype, device=weight.device)
            head = linear_head(channel_basis).flatten()
            effective = torch.einsum('o,oihw->ihw', head, kernel)
            moment = (xx * xx - yy * yy) / 4 if axis == 1 else xx * yy / 2
            vectors.append((effective * moment).sum((-2, -1)))
        common = (vectors[0] + vectors[1]) / 2
        basis = matched_spin_basis(weight, self.matched_spin_kernel)
        return (common, basis)

    def _matched_spin_weights(self, inverse_weights):
        common, basis = self._matched_spin_components(inverse_weights)
        return basis[:, None] * common[None, :, None, None]

    def forward(self, x, *, reference=False):
        if (
            x.ndim != 4
            or x.shape[0] < 1
            or x.shape[1] != self.input_channels
            or x.shape[-1] < 1
            or x.shape[-2] != x.shape[-1]
        ):
            raise ValueError('Expected nonempty square [B,in_dim,H,H] input')
        if reference and self.matched_spin_kernel:
            raise ValueError('Matched derivative readout has its own direct convolution reference')
        # Normalize brightness before constructing scalar and spin-2 features.
        w = self._get_cached_weight(x.shape[-2:], self.nl_sigma, x.device, x.dtype)
        brightness = (x * w).sum((1, 2, 3), keepdim=True) * 100
        x = x / (1 + F.softplus(brightness - 1, beta=10.0, threshold=20))
        b = x.shape[0]
        if (self.linear_f8 or getattr(self, 'nonlinear_f8_kernel', False)) and not reference:
            # Align the eight convolution kernels instead of copying rotated
            # feature images. The nonlinear variant retains all eight maps
            # through normalization and GELU before taking orbit projections.
            stem = self.d4_blocks(x)
            block = self.blocks[0]
            assert len(self.blocks) == 1 and block.skip is None
            weight = block.conv.weight
            # Weight-only kernels are reused across batches once frozen (eval).
            inverse_weights = frozen_weight_cache(
                self, 'inverse_weights',
                lambda: self._d4_inverse_stack(weight[None].expand(8, *weight.shape)))
            if getattr(self, 'nonlinear_f8_kernel', False):
                channels = stem.shape[1]
                bias = None if block.conv.bias is None else block.conv.bias.repeat(8)
                aligned = F.conv2d(stem, inverse_weights.flatten(0, 1), bias, padding=1)
                aligned = aligned.reshape(b, 8, channels, *stem.shape[-2:]).transpose(0, 1).contiguous()
                aligned = block.act(block.norm(aligned.flatten(0, 1))).reshape_as(aligned)
                self.feats = aligned + block.residual_scale * stem[None]
                self.f8 = None
                self.f_mean_e1 = (self.feats * self.signs_e1.to(stem).view(8, 1, 1, 1, 1)).mean(0)
                self.f_mean_e2 = (self.feats * self.signs_e2.to(stem).view(8, 1, 1, 1, 1)).mean(0)
                v1, v2 = self._pool_components(None)
                return self._readout(v1, v2)
            if self.matched_spin_kernel:
                scalar = F.conv2d(stem, frozen_weight_cache(
                    self, 'inverse_mean', lambda: inverse_weights.mean(0)), padding=1)
                if self.spin_gate:
                    if hasattr(self, 'odd_readout') or hasattr(self, 'scalar_readout'):
                        basis = frozen_weight_cache(
                            self, 'spin_basis',
                            lambda: matched_spin_basis(inverse_weights[0], self.matched_spin_kernel))
                    else:
                        self.spin_common, basis = frozen_weight_cache(
                            self, 'spin_components',
                            lambda: self._matched_spin_components(inverse_weights))
                    channels = stem.shape[1]
                    if getattr(self, 'quadratic_spin', False):
                        self.f_mean_e1, self.f_mean_e2 = self._quadratic_spin(stem)
                    else:
                        spin = F.conv2d(
                            stem,
                            frozen_weight_cache(
                                self, 'spin_kernel',
                                lambda: basis[:, None].repeat(channels, 1, 1, 1)),
                            padding=self.matched_spin_kernel // 2,
                            groups=channels
                        )
                        spin = spin.reshape(b, channels, 2, *spin.shape[-2:])
                        self.f_mean_e1, self.f_mean_e2 = (spin[:, :, 0], spin[:, :, 1])
                else:
                    spin = F.conv2d(
                        stem,
                        self._matched_spin_weights(inverse_weights),
                        padding=self.matched_spin_kernel // 2
                    )
                    self.f_mean_e1, self.f_mean_e2 = spin.chunk(2, 1)
            else:
                kernels = torch.cat(
                    (
                        inverse_weights.mean(0),
                        (inverse_weights * self.signs_e1.to(weight).view(8, 1, 1, 1, 1)).mean(0),
                        (inverse_weights * self.signs_e2.to(weight).view(8, 1, 1, 1, 1)).mean(0)
                    ),
                    0
                )
                scalar, self.f_mean_e1, self.f_mean_e2 = F.conv2d(
                    stem,
                    kernels,
                    padding=1
                ).chunk(3, 1)
            scalar = scalar + block.residual_scale * stem
            if block.conv.bias is not None:
                scalar = scalar + block.conv.bias[None, :, None, None]
            # Pooling uses only the invariant orbit mean; storing one field
            # avoids allocating eight identical copies of that mean.
            self.feats = scalar[None]
            self.f8 = None
        else:
            if reference:
                orbit = self._d4_orbit_stack(x)
                x8 = self.d4_blocks(orbit.flatten(0, 1))
            else:
                x8 = self._d4_orbit_stack(self.d4_blocks(x)).flatten(0, 1)
            if self.linear_f8:
                block = self.blocks[0]
                residual = block.skip(x8) if block.skip is not None else x8
                f8 = block.conv(x8) + block.residual_scale * residual
            else:
                f8 = self._trunk_feature(x8)
            self.f8 = f8.reshape(8, b, *f8.shape[1:])
            self.feats = self._d4_inverse_stack(self.f8)
            self.f_mean_e1 = (self.feats * self.signs_e1.to(f8).view(8, 1, 1, 1, 1)).mean(0)
            self.f_mean_e2 = (self.feats * self.signs_e2.to(f8).view(8, 1, 1, 1, 1)).mean(0)
        # Adaptive pooling ignores the fixed window and retains its full autograd path.
        v1, v2 = self._pool_components(None)
        return self._readout(v1, v2)

    def _readout(self, v1, v2):
        if self.spin_gate:
            # Quartic invariants: at a D4-centered round source v=O(g), so
            # magnitude-dependent gating first contributes at order g^5.
            self.spin_pooled = torch.stack((v1, v2), dim=-1)
            if hasattr(self, 'odd_readout'):
                normalized = self.spin_pooled / self.spin_feature_scale[None, :, None]
                return self.odd_readout(normalized.transpose(1, 2)).squeeze(-1)
            radial2 = self.spin_pooled.square().sum(-1) / self.spin_feature_scale.square()
            gate = 1 + 0.5 * torch.tanh(self.spin_gate_net(torch.log1p(radial2.square())))
            return (self.spin_pooled * (self.spin_common[None, :] * gate)[..., None]).sum(1)
        if self.matched_spin_kernel:
            return torch.cat((v1, v2), dim=-1)
        if self.shared_head:
            return torch.cat((self.head_shared(v1), self.head_shared(v2)), dim=-1)
        return torch.cat((self.head_e1(v1), self.head_e2(v2)), dim=-1)



class D4inv_noTanh(D4InvCNN):
    """Scalar-conditioned GELU readout with the unchanged D4/attention trunk.

    Default shapes: scalar descriptor [B,32], pooled spin features [B,32,2],
    channel coefficients [B,32], output [B,2]. Both branches use the same
    normalized spatial weight (attention by default; exact fixed Gaussian
    with ``fixed_pool_sigma`` set, measured in feature pixels); derivatives act BEFORE attention weighting.
    Readout = mean_channels(MLP(scalar / scalar_scale) * spin).

    ``noTanh`` refers only to this readout, not the inherited trunk. The MLP
    has two Linear layers and one GELU, with no output activation, amplitude
    powers/logarithms, odd projection, or radial compression. Its signed
    coefficients are shared by both spin components. D4 symmetry is retained;
    continuous rotation covariance still needs independent validation.

    scalar_scale is a fixed positive per-channel training-set RMS (default
    ones), saved as a buffer. Set it once before optimization, using only
    training descriptors; never update it from validation/test batches.
    Like D4InvCNN, this is the core; centering/cropping wrappers are external.
    """

    def __init__(self, in_dim=1, d4_layers=2, base_channels=32,
                 readout_hidden=64, scalar_scale=None, fixed_pool_sigma=None, **kwargs):
        if (isinstance(readout_hidden, bool)
                or not isinstance(readout_hidden, int) or readout_hidden < 1):
            raise ValueError('readout_hidden must be a positive integer')
        required = dict(shared_head=False, linear_f8=True,
                        linear_head=True, spin_gate=True)
        for name, value in required.items():
            if name in kwargs and kwargs[name] != value:
                raise ValueError(f'D4inv_noTanh requires {name}={value}')
            kwargs[name] = value
        super().__init__(in_dim=in_dim, d4_layers=d4_layers,
                         base_channels=base_channels, **kwargs)
        if fixed_pool_sigma is not None and not 0 < float(fixed_pool_sigma) < float('inf'):
            raise ValueError('fixed_pool_sigma must be finite and positive')
        self.fixed_pool_sigma = None if fixed_pool_sigma is None else float(fixed_pool_sigma)
        if self.fixed_pool_sigma is not None:
            for name in ('token_projection', 'attn_norm', 'qkv', 'attn_output',
                         'ffn_norm', 'ffn', 'kernel_head', 'kernel_blur'):
                delattr(self, name)
        # The scalar MLP replaces BOTH the old gate and common linear heads.
        del self.spin_gate_net
        del self.spin_feature_scale
        del self.head_e1
        del self.head_e2
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

    def _make_pool_kernel(self, scalar_field):
        if self.fixed_pool_sigma is None:
            return super()._make_pool_kernel(scalar_field)
        # Exact normalized Gaussian in feature pixels; no additional taper.
        xx, yy = self._coordinates(scalar_field.shape[-2:],
                                   scalar_field.device, scalar_field.dtype)
        logits = -0.5 * (xx.square() + yy.square()) / self.fixed_pool_sigma**2
        kernel = logits.flatten().softmax(0).reshape(1, 1, *logits.shape)
        return kernel.expand(scalar_field.shape[0], -1, -1, -1)

    @torch.no_grad()
    def set_scalar_scale(self, scale):
        """Install fixed training-set channel RMS; no per-object normalization."""
        scale = torch.as_tensor(scale, device=self.scalar_scale.device,
                                dtype=self.scalar_scale.dtype)
        if (scale.shape != self.scalar_scale.shape
                or not torch.isfinite(scale).all() or not (scale > 0).all()):
            raise ValueError('scalar_scale must be a finite positive [channels] vector')
        self.scalar_scale.copy_(scale)

    def _readout(self, v1, v2):
        # pool_kernel already sums to one: another spatial mean would add 1/HW.
        scalar = self.feats.mean(dim=0)
        self.scalar_pooled = (scalar * self.pool_kernel).sum(dim=(-2, -1))
        self.spin_pooled = torch.stack((v1, v2), dim=-1)
        self.channel_weights = self.scalar_readout(self.scalar_pooled / self.scalar_scale)
        # Keep BOTH branches and attention in autograd for full shear response.
        return (self.spin_pooled * self.channel_weights[..., None]).mean(dim=1)


class D4inv_Quad(D4inv_noTanh):
    """Local quadratic spin-2 density in place of the linear derivative field.

    Why: the linear field D[s] is a total derivative, so sum(K * D[s]) =
    sum(s * D[K]); the shear signal is carried entirely by the curvature of
    the pooling kernel K, i.e. a windowed moment about the kernel centre. K
    must then sit on the source to sub-pixel accuracy, and a stamp-anchored
    16-token attention kernel cannot. Measured on native cutouts: fixed
    sigma-4 pooling gives a position response term of +2.1e-2/+2.5e-2 of R;
    learned attention shrinks K to 1.3 px and cancels most, not all, of it.

    Here each channel u and a channel-mixed partner p = A u (1x1, identity
    initialised) form the gradient bilinear
        T1 = dy(p) dy(u) - dx(p) dx(u),   T2 = -(dx(p) dy(u) + dy(p) dx(u)),
    the by-parts image of p * D[u] with the source itself as the window.
    Its pixel sum is translation invariant for a flat K (exactly so for a
    band-limited field) and the window term falls roughly as sigma_K^-4, so
    K needs neither curvature nor sub-pixel tracking. A pointwise sigmoid
    gate (zero-initialised weights, open at start) lets training suppress
    sky pixels without narrowing K. No image resampling or centroid is used.
    D4 covariance is exact: the 5-point gradient is a D4 vector, the mixing
    and the gate are D4 scalars.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.quadratic_spin = True
        channels = self.scalar_scale.numel()
        self.spin_partner = nn.Conv2d(channels, channels, 1, bias=False)
        with torch.no_grad():
            self.spin_partner.weight.copy_(torch.eye(channels)[:, :, None, None])
        self.spin_local_gate = nn.Conv2d(channels, channels, 1)
        nn.init.zeros_(self.spin_local_gate.weight)
        nn.init.constant_(self.spin_local_gate.bias, 4.0)

    def _quadratic_spin(self, stem):
        channels = stem.shape[1]
        d1 = stem.new_tensor([1 / 12, -2 / 3, 0, 2 / 3, -1 / 12])
        both = torch.cat((stem, self.spin_partner(stem)), 1)
        gx = F.conv2d(both, d1.view(1, 1, 1, 5).repeat(2 * channels, 1, 1, 1),
                      padding=(0, 2), groups=2 * channels)
        gy = F.conv2d(both, d1.view(1, 1, 5, 1).repeat(2 * channels, 1, 1, 1),
                      padding=(2, 0), groups=2 * channels)
        ux, px = gx.chunk(2, 1)
        uy, py = gy.chunk(2, 1)
        gate = torch.sigmoid(self.spin_local_gate(stem))
        return (gate * (py * uy - px * ux), -gate * (px * uy + py * ux))


class D4inv_F8GELU(D4inv_noTanh):
    """One nonlinear Forward8 block in place of fixed spin derivatives.

    The D4 stem, scalar MLP and pooling are retained. Eight aligned convolution
    maps pass through the original channel normalization, GELU and residual
    before scalar/B1/B2 projection. Kernel alignment is exactly equivalent to
    the ordinary eight-view block; ``forward(reference=True)`` evaluates that
    slower reference. There is no linear projection before the nonlinearity.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.linear_f8 = False
        self.matched_spin_kernel = 0
        self.nonlinear_f8_kernel = True


# Original noTanh arms plus fixed-Gaussian controls, all with explicit initialization.
NOTANH_RECIPES = {
    'f8_notanh_rdsm': dict(depth=5, target='rdsm', ssl=False, parent=None, forward8=True),
    'notanh_truth2': dict(depth=2, target='truth', ssl=False, parent=None),
    'notanh_rdsm2': dict(depth=2, target='rdsm', ssl=False, parent=None),
    'notanh_truth2_ssl': dict(depth=2, target='truth', ssl=True, parent='notanh_truth2'),
    'notanh_rdsm2_ssl': dict(depth=2, target='rdsm', ssl=True, parent='notanh_rdsm2'),
    'notanh_truth4': dict(depth=4, target='truth', ssl=False, parent=None),
    'notanh_rdsm4': dict(depth=4, target='rdsm', ssl=False, parent=None),
    'notanh_ssl2': dict(depth=2, target='truth', ssl=True, parent=None),
    'notanh_sigma4_truth2': dict(depth=2, target='truth', ssl=False, parent=None, fixed_pool_sigma=4.),
    'notanh_sigma4_rdsm2': dict(depth=2, target='rdsm', ssl=False, parent=None, fixed_pool_sigma=4.),
    'notanh_sigma4_ssl2': dict(depth=2, target='truth', ssl=True, parent=None, fixed_pool_sigma=4.),
    'notanh_f8gelu_rdsm2': dict(depth=2, target='rdsm', ssl=False, parent=None, f8_gelu=True),
    # F8GELU with AttnW removed: exact fixed Gaussian pooling, sigma 4 feature pixels.
    'notanh_sigma4_f8gelu_truth2': dict(depth=2, target='truth', ssl=False, parent=None,
                                        f8_gelu=True, fixed_pool_sigma=4.),
    # Local quadratic spin density; wide pooling prior (see D4inv_Quad).
    'notanh_quad_truth2': dict(depth=2, target='truth', ssl=False, parent=None, quad=True, pool_prior_sigma=12.),
    'notanh_quad_rdsm2': dict(depth=2, target='rdsm', ssl=False, parent=None, quad=True, pool_prior_sigma=12.),
}


def build_single_stage_model(kind='single_score', seed=42, state=None, center=False):
    """Build the tested single-stage model; no prior weights are read.

    single_score: Tanh gate; single_score_odd: biased GELU inside that gate.
    single_rdsm_odd_readout: whole gated contraction replaced by shared odd GELU.
    Every kind now uses linear final gain, without radial compression.
    Legacy kinds are retained only to read earlier experimental checkpoints.
    Feature/output RMS initialization is performed by the trainer on train data.
    """
    if kind not in (
        'single',
        'single_odd',
        'single_radial',
        'single_odd_difference',
        'single_score',
        'single_score_odd',
        'single_rdsm_linear',
        'single_rdsm_odd_readout'
    ) and kind not in NOTANH_RECIPES:
        raise ValueError(f'Unknown D4 training kind: {kind}')
    torch.manual_seed(seed)
    cfg = dict(
        base_channels=32,
        res_factor=0.25,
        head_hidden=128,
        nl_sigma=16.0,
        suppress=True,
        r_cut=24.0,
        suppress_order=8,
        head_num_hidden_layers=1,
        attn_grid_size=16,
        attn_dim=32,
        attn_heads=4,
        linear_f8=True,
        linear_head=True,
        matched_spin_kernel=5,
        spin_gate=True,
        isotropic_front=True
    )
    recipe = NOTANH_RECIPES.get(kind)
    constructor = (D4inv_F8GELU if recipe and recipe.get('f8_gelu')
                   else D4inv_Quad if recipe and recipe.get('quad') else D4inv_noTanh)
    if recipe and recipe.get('forward8'):
        f8_cfg = {k: v for k, v in cfg.items() if k not in (
            'linear_f8', 'linear_head', 'matched_spin_kernel', 'spin_gate', 'isotropic_front')}
        base = F8_noTanh(num_layers=recipe['depth'], **f8_cfg)
    else:
        base = (constructor(d4_layers=recipe['depth'], fixed_pool_sigma=recipe.get('fixed_pool_sigma'), **cfg) if recipe
                else D4InvCNN(d4_layers=2, **cfg))
    use_explicit_layernorm(base)
    if (kind in ('single_radial', 'single_score', 'single_score_odd', 'single_rdsm_linear', 'single_rdsm_odd_readout') or recipe) and not (recipe and recipe.get('fixed_pool_sigma')):
        # Same function class and trainable parameters; only initialization changes.
        with torch.no_grad():
            base.token_projection.weight[0].zero_()
            base.token_projection.weight[0, 32] = 1.0
            base.token_projection.bias[0] = 0.0
            base.attn_output.weight.zero_()
            base.attn_output.bias.zero_()
            base.ffn[-1].weight.zero_()
            base.ffn[-1].bias.zero_()
            base.kernel_head.weight.zero_()
            prior_sigma = recipe.get('pool_prior_sigma', 4.) if recipe else 4.
            base.kernel_head.weight[0, 0] = -0.5 * (31 / prior_sigma) ** 2
            base.kernel_head.bias.zero_()
    if kind in ('single_odd', 'single_odd_difference', 'single_score_odd'):
        old = base.spin_gate_net
        base.spin_gate_net = (OddGELUDifference if kind in (
            'single_odd_difference',
            'single_score_odd'
        ) else OddGELU)(old[0], nn.GELU(), old[2])
    if kind == 'single_rdsm_odd_readout':
        # Replace the entire amplitude-gated contraction, not its activation.
        # Biased GELU MLP odd part; both spin components share all parameters.
        del base.spin_gate_net
        del base.head_e1
        del base.head_e2
        base.odd_readout = OddGELU(
            nn.Linear(32, 64), nn.GELU(), nn.Linear(64, 1, bias=False)
        )
    readout = TrainableRadial(base)
    if kind == 'single_rdsm_linear':
        readout.radial_alpha = 0.0
    model = ObservableCenter(InputCrop63(readout), sigma=2.0, enabled=center).float()
    if state is not None:
        model.load_state_dict(state, strict=True)
    return model


def load_d4_checkpoint(path, *, device='cpu', dtype=torch.float64, center=False):
    """Strictly restore a trusted D4 state checkpoint and all input wrappers.

    ObservableCenter is disabled by default (center=True restores it; the
    weights are unchanged either way).

    The selected checkpoint uses a fixed gain; single-stage checkpoints use a
    trainable log_gain. These are deliberately distinct parameterizations.
    """
    saved = torch.load(Path(path), map_location='cpu', weights_only=False)
    if 'base_model' in saved:
        config = dict(saved['base_model']['config'])
        config.pop('num_layers', None)
        base = D4InvCNN(d4_layers=saved['base_model']['depth'], **config)
        use_explicit_layernorm(base)
        radial = saved['base_model']['radial_readout']
        base = InputCrop63(RadialAmplitudeReadout(base, radial['alpha'], radial['gain']))
        center_cfg = saved['center']
        model = ObservableCenter(
            base,
            sigma=center_cfg['sigma'],
            iterations=center_cfg['iterations'],
            max_shift=center_cfg['max_shift'],
            pad=center_cfg['padding'],
            enabled=center
        )
        # Load FP64 source weights without an intermediate FP32 rounding.
        model = model.double()
        model.load_state_dict(saved['state_dict'], strict=True)
    elif 'kind' in saved:
        model = build_single_stage_model(saved['kind'], saved['seed'], saved['state_dict'], center=center)
    else:
        raise ValueError('Expected a selected D4 or single-stage checkpoint')
    # All future measurements use a linear final amplitude readout.
    # Legacy state tensors remain loadable; historical catalogs are unchanged.
    model.model.model.radial_alpha = 0.0
    return model.to(device=device, dtype=dtype).eval()



# Historical public name retained for old imports and the matched constructor.
D4Forward8_AttnW_CNN = D4InvCNN


class D4Matched(D4Forward8_AttnW_CNN):
    """Original F8(3)-matched D4(2)+F8(1) constructor, unchanged state keys."""

    def __init__(self, num_layers=3, **kwargs):
        if num_layers != 3:
            raise ValueError('Matched experiment requires total depth 3')
        if kwargs.get('shared_head', False):
            raise ValueError('Reference F8(3) has independent heads')
        super().__init__(d4_layers=2, **kwargs)
