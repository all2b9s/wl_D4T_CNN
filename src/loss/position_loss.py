"""Position (sub-pixel translation) losses on noiseless isolated partners.

PositionAliasLoss (gelu_commutator_cross_v2): GELU commutator energy, the
cross product of two independent noise draws of each noiseless isolated
partner, weighted by the pool-reachable window, ratio per channel. Its hooks
exist only during the loss; inference is unchanged.

PositionNullLoss (signed_position_null_v1): nulls the SIGNED output position
term per size bin. On noiseless isolated partners at their true sub-pixel
offsets, the cross product of two half-sample means of the position response
P over the bin mean R, for e1 and e2. The residual term is one D4 lattice
harmonic whose sign depends on the seed; the v2 energy only scales it.
Offsets enter the training loss only.
"""
import numpy as np
import torch
from src.loss.base import AuxLoss
from src.loss.common import jacobian, position_response, blended_psf as _blended_psf, partner_q

# Independent version: runs with position_weight=0 keep the original objective.
POSITION_LOSS_VERSION = 'gelu_commutator_cross_v2'
POSITION_COUNT = 16
POSITION_VALIDATION_COUNT = 64
# Image noise of the training stamps (MAD of blended - isolated partner 0.61-0.66,
# 1-px differences 0.61) and the AnaCal renoise level used by training_q.
POSITION_NOISE = 0.594
# i-band magnitude bin edges for the projected-alias validation.
POSITION_MAG_BINS = (23.5, 24.5)
# Signed position-term nulling (independent version; weight 0 keeps the objective).
POSITION_NULL_VERSION = 'signed_position_null_v1'
# Size bins: Gaussian(sigma 4 px)-weighted RMS radius of the noiseless partner
# q0, in pixels. Quantile-matched on 1200 noiseless xlens_shift stamps
# (groups 5300-5311, Spearman 0.85 with catalog hlr) to hlr .075/.1/.15/.2/.3
# arcsec. Partner hlr columns are NaN and the CatSim radius formula disagrees
# with the xlens hlr for some sources, so the image measure is used.
POSITION_NULL_SIZE_EDGES = (2.46, 2.50, 2.60, 2.71, 2.95)
POSITION_NULL_PER_BIN = 8           # partners per size bin and step (split in two halves)
POSITION_NULL_SCALE = 1e6           # loss in units of (1e-3)^2 of (<P>/<R>)
POSITION_NULL_VALIDATION_PER_BIN = 256


def _position_trunk(model):
    """(F8_noTanh base, its exact spatial GELUs). Scalar head and norms are not regularized."""
    from src.architecture.forward8_CNN import F8_noTanh
    bases = [module for module in model.modules() if isinstance(module, F8_noTanh)]
    if len(bases) != 1:
        raise ValueError('position loss requires one F8_noTanh trunk')
    base = bases[0]
    acts = tuple(block.act for block in base.blocks)
    if not acts or any(not isinstance(act, torch.nn.GELU) or act.approximate != 'none'
                       for act in acts):
        raise ValueError('position loss requires exact spatial GELUs (no gate/tanh/quad)')
    if getattr(base, 'input_size', None) is None or not base._has_fixed_kernel():
        raise ValueError('position loss requires the valid-crop trunk with a fixed pooling kernel')
    return base, acts


def _position_activations(model):
    return _position_trunk(model)[1]


def _reach_windows(base, device, dtype):
    """Per-layer pool-reachable weight on each pre-activation grid.

    The fixed pooling kernel K lives on the final feature grid. A layer-l
    pre-activation pixel reaches the pool only through the depth-1-l later
    valid 3x3 convolutions (the residual crop lies inside their support), so
    its weight is K convolved with a 3x3 box depth-1-l times, which is exactly
    the layer-l grid. Unit sum; zero where no path to the pool exists. On
    noisy v1 inputs 66-84% of the unwindowed energy came from r > 10 px.
    """
    depth = len(base.blocks)
    size = base.input_size - 2 * depth
    weight = base._fixed_kernel((size, size), 'cpu', torch.float64)[None, None]
    box = torch.ones(1, 1, 3, 3, dtype=torch.float64)
    windows = [None] * depth
    for layer in reversed(range(depth)):
        windows[layer] = (weight / weight.sum())[0, 0].to(device=device, dtype=dtype)
        weight = torch.nn.functional.conv2d(weight, box, padding=2)
    return windows


def isolated_noise_pair(images, rng, noise=POSITION_NOISE):
    """Two independent noisy q0 realizations [2, N, 1, H, W] of noiseless isolated partners.

    Each realization draws fresh image noise and fresh AnaCal renoise maps
    (all sigma=noise, as in training_q; spectral renoising by default, so a
    second map for the filler), so the two share only the source.
    Training stamps are already noisy; renoising them twice would share the
    field noise and leave half of the jitter floor.
    """
    from src.anacal.pixel_response import anacal_pix_r
    psf = _blended_psf()
    images = np.asarray(images, dtype=np.float64)
    center = np.full(2, images.shape[-1] // 2, dtype=np.float64)
    out = np.empty((2, len(images), 1) + images.shape[-2:], dtype=np.float32)
    for realization in range(2):
        for i, image in enumerate(images):
            out[realization, i, 0] = anacal_pix_r(
                image + rng.normal(0, noise, image.shape), psf, center=center,
                scale_arcsec_per_pix=0.2, sigma_arcsec=0.85 / 2.355, freq_lim=10.0,
                noise_map=rng.normal(0, noise, image.shape).astype(np.float32),
                noise2_map=rng.normal(0, noise, image.shape).astype(np.float32))[0]
    return out


def _mirror_derivative(value, dim):
    """Fourier derivative per pixel with an even, endpoint-excluding extension.

    Native features are NOT periodic. Mirroring makes a 2*(n-1) periodic
    sequence without joining unrelated edges or tapering the source. This
    declares a boundary convention for the local training proxy.
    """
    v = value.movedim(dim, -1)
    n = v.shape[-1]
    if n < 3:
        raise ValueError('position loss requires spatial dimensions >=3')
    v = v - v[..., :1]  # remove DC exactly, including constant non-power-of-two grids
    extended = torch.cat((v, v[..., 1:-1].flip(-1)), -1)
    k = 2 * torch.pi * torch.fft.fftfreq(extended.shape[-1], device=v.device, dtype=v.dtype)
    # The real derivative of the even-length Nyquist cosine is zero on-grid.
    k[extended.shape[-1] // 2] = 0
    derivative = torch.fft.ifft(1j * k * torch.fft.fft(extended, dim=-1), dim=-1).real
    return derivative[..., :n].movedim(-1, dim)


def _gelu_defects(z):
    """[(defect, ideal transport)] along x and y: GELU'(z) dz - d GELU(z), d GELU(z)."""
    phi = torch.nn.functional.gelu(z, approximate='none')
    prime = .5 * (1 + torch.erf(z / np.sqrt(2.))) + z * torch.exp(-.5 * z.square()) / np.sqrt(2 * np.pi)
    out = []
    for dim in (-1, -2):
        ideal = _mirror_derivative(phi, dim)
        out.append((prime * _mirror_derivative(z, dim) - ideal, ideal))
    return out


def _position_energy(z, window, orbit=8, *, return_energies=False):
    """Cross-realization, pool-windowed GELU commutator energy, ratio per channel.

    z: [orbit*2N, C, H, W] pre-activations, orbit-major over a batch whose
    first N rows are realization a and last N realization b of the same N
    sources. Numerator per channel: sum over objects, views, directions and
    pixels of W * D_a * D_b. The draws are independent given the source, so
    its expectation is sum W |E_noise D|^2, the noise-averaged (bias-making)
    alias, without the jitter floor that dominated v1 (one batch may be
    negative; the expectation is not). Denominator: the same windowed sum of
    the single-realization transport (d GELU z)^2, averaged over a and b;
    positive and differentiable, so a common channel rescale cancels and
    amplitude shrinking is not rewarded. Ratios are formed per channel, then
    averaged over active channels: one large linear ballast channel can no
    longer dilute the others (v1 pooled channels: layer 0 0.046 -> 0.006).
    """
    if z.ndim != 4 or z.shape[0] % (2 * orbit):
        raise ValueError('expected orbit-major [orbit*2N,C,H,W] activations of realization pairs')
    if z.dtype not in (torch.float32, torch.float64):
        raise ValueError('position loss requires float32 or float64 activations')
    if window.shape != z.shape[-2:]:
        raise ValueError('reach window does not match the activation grid')
    cross = auto = transport = 0
    for defect, ideal in _gelu_defects(z):
        d = defect.unflatten(0, (orbit, 2, -1))   # [view, realization, N, C, H, W]
        t = ideal.unflatten(0, (orbit, 2, -1))
        cross = cross + (window * d[:, 0] * d[:, 1]).sum((-2, -1)).mean(0)        # [N, C]
        auto = auto + (window * d.square()).sum((-2, -1)).mean((0, 1))           # [N, C]
        transport = transport + (window * t.square()).sum((-2, -1)).mean((0, 1))  # [N, C]
    channel_transport = transport.sum(0)
    active = channel_transport > 1e-6 * channel_transport.detach().max()
    safe = torch.where(active, channel_transport, torch.ones_like(channel_transport))
    ratio = torch.where(active, cross.sum(0) / safe, torch.zeros_like(safe))
    loss = ratio.sum() / active.sum().clamp_min(1)
    if not torch.isfinite(loss):
        raise FloatingPointError('nonfinite normalized GELU commutator energy')
    if not return_energies:
        return loss
    with torch.no_grad():
        auto_loss = (torch.where(active, auto.sum(0) / safe, torch.zeros_like(safe)).sum()
                     / active.sum().clamp_min(1))
        per_object = cross.sum(1) / transport.sum(1).clamp_min(torch.finfo(z.dtype).tiny)
    return loss, dict(auto=auto_loss, per_object=per_object, active=int(active.sum()),
                      transport=transport.sum(1).mean())


def position_penalty(model, data, rng, count=POSITION_COUNT, *, layer=None, return_details=False):
    """Cross-realization alias loss on two noise draws of noiseless isolated partners.

    ``data`` is the training batch (dict with 'iso', the noiseless isolated
    partner stamps of its rows, kept on CPU) or an [N, H, W] array of them.
    One uniformly sampled layer gives an unbiased estimate of the layer-mean
    objective; layer='all' evaluates all layers. No model hooks persist.

    Still a LOCAL commutator proxy: the output-level alias is logged by
    position_validation (projected_alias), not trained on. Blending
    neighbours are absent from the partners.
    """
    base, acts = _position_trunk(model)
    images = data['iso'] if isinstance(data, dict) else data
    images = images.detach().cpu().numpy() if torch.is_tensor(images) else np.asarray(images)
    if images.ndim != 3 or len(images) == 0 or count < 1:
        raise ValueError('position loss needs nonempty [N,H,W] isolated partner stamps and positive count')
    if layer is None:
        selected = (int(rng.integers(len(acts))),)
    elif layer == 'all':
        selected = tuple(range(len(acts)))
    elif isinstance(layer, (int, np.integer)) and 0 <= layer < len(acts):
        selected = (int(layer),)
    else:
        raise ValueError('layer must be None, an in-range index, or all')
    ix = rng.choice(len(images), min(count, len(images)), replace=False)
    parameter = next(model.parameters())
    pair = torch.as_tensor(isolated_noise_pair(images[ix], rng), device=parameter.device,
                           dtype=parameter.dtype).flatten(0, 1)   # [2N,1,H,W]: a then b
    windows = _reach_windows(base, parameter.device, parameter.dtype)
    captured = {}
    handles = []
    def capture(index):
        def hook(module, args):
            if index in captured:
                raise RuntimeError('spatial activation reused during one forward')
            captured[index] = args[0]
        return hook
    try:
        for index in selected:
            handles.append(acts[index].register_forward_pre_hook(capture(index)))
        model(pair)
    finally:
        for handle in handles:
            handle.remove()
    if set(captured) != set(selected):
        raise RuntimeError('selected spatial activation was not evaluated')
    results = [_position_energy(captured[index], windows[index], return_energies=True) for index in selected]
    loss = torch.stack([value[0] for value in results]).mean()
    if not return_details:
        return loss
    per_object = torch.stack([value[1]['per_object'] for value in results])
    details = dict(version=POSITION_LOSS_VERSION, layers=list(selected), objects=len(ix),
                   cross_per_layer=[float(value[0].detach()) for value in results],
                   auto_per_layer=[float(value[1]['auto']) for value in results],
                   active_channels=[value[1]['active'] for value in results],
                   transport_per_layer=[float(value[1]['transport']) for value in results],
                   object_p95_per_layer=torch.quantile(per_object, .95, dim=1).tolist())
    return loss, details


def _view_shift_coefficients(base, device, dtype):
    """[8, 2(canonical x, y), 2(view x, y)]: canonical unit shift expressed per D4 view."""
    c = torch.arange(5, device=device, dtype=dtype) - 2
    ramps = torch.stack((c[None, :].expand(5, 5), c[:, None].expand(5, 5)))[:, None]   # [2,1,5,5]
    views = base._d4_orbit_stack(ramps)[:, :, 0]   # [8, 2, 5, 5]
    return torch.stack((views[..., 2, 3] - views[..., 2, 2], views[..., 3, 2] - views[..., 2, 2]), -1)


def projected_alias(model, x):
    """Output-level GELU alias generator A [B, 2(e), 2(shift x, y)] per input pixel.

    A = sum_l sum_views c_view . <de/dy_l, D_l>, with D_l the layer-l GELU
    commutator. For content translated by +delta, de/ddelta = -(pool-window
    term + A); checked against sinc-shift finite differences to 9% on
    parab10_wng6s (remainder: norm scale, brightness normalization, edges).
    """
    base, acts = _position_trunk(model)
    x = x.detach().clone().requires_grad_(True)
    zs, ys, handles = {}, {}, []
    for index, act in enumerate(acts):
        handles.append(act.register_forward_pre_hook(lambda m, args, i=index: zs.__setitem__(i, args[0])))
        handles.append(act.register_forward_hook(lambda m, args, out, i=index: ys.__setitem__(i, out)))
    try:
        with torch.enable_grad():
            e = model(x)
            outputs = [ys[i] for i in range(len(acts))]
            grads = [torch.autograd.grad(e[:, a].sum(), outputs, retain_graph=a == 0) for a in range(2)]
    finally:
        for handle in handles:
            handle.remove()
    batch = len(x)
    coefficients = _view_shift_coefficients(base, x.device, x.dtype)   # [8, k, j]
    alias = x.new_zeros(batch, 2, 2)
    for i in range(len(acts)):
        defects = [defect for defect, _ in _gelu_defects(zs[i].detach())]
        for a in range(2):
            # [8, B, j]: projection of the view-frame defect on de_a/dy_l
            inner = torch.stack([(grads[a][i] * d).sum((1, 2, 3)).view(8, batch) for d in defects], -1)
            alias[:, a] += torch.einsum('vbj,vkj->bk', inner, coefficients)
    return alias


def position_validation(model, fields, count=POSITION_VALIDATION_COUNT):
    """Fixed held-out isolated partners of the validation fields; logged, not selected on.

    (i) Training proxy (cross energy, all layers) and its single-realization
    noise floor (auto). (ii) Noise-free projected alias A on the noiseless
    partners, per i-band magnitude bin: rms|A| / <R>. This includes R and is
    the guard against the proxy's blind spot: a large ballast channel
    inflates the next WindowedChannelNorm scale and drives the proxy to ~0
    while R collapses (frozen parab10: rms|A|/R 4.5e-3 -> 1.45e-2, R 1.06 ->
    0.04). The signed <-A . delta>/<R> over true offsets was ~1e-5 +- 5e-5
    (not the m-setting quantity) and is not logged; m per size bin still
    needs the shear-partner evaluation.
    """
    from src.anacal.pixel_response import anacal_pix_r
    parameter = next(model.parameters())
    psf = _blended_psf()
    proxies, alias, response, mags = [], [], [], []
    for field in fields:
        if 'iso' not in field:
            raise ValueError('position validation needs isolated partners (TrainingData.isolated)')
        field_id = int(field['field'][0])
        rng = np.random.default_rng([19301, field_id])
        ix = rng.choice(len(field['iso']), min(count, len(field['iso'])), replace=False)
        with torch.no_grad():
            _, details = position_penalty(model, field['iso'][ix], rng, count=len(ix), layer='all',
                                          return_details=True)
        proxies.append(details)
        q = np.stack([anacal_pix_r(image.astype(np.float64), psf, center=np.full(2, image.shape[-1] // 2, dtype=np.float64),
                                   scale_arcsec_per_pix=0.2, sigma_arcsec=0.85 / 2.355, freq_lim=10.0)[:3]
                      for image in field['iso'][ix]])
        q = torch.as_tensor(q, device=parameter.device, dtype=parameter.dtype)
        for start in range(0, len(q), 32):
            chunk = q[start:start + 32]
            alias.append(projected_alias(model, chunk[:, :1]).detach().cpu().double())
            with torch.no_grad():
                response.append(jacobian(model, chunk[:, :1], chunk[:, 1:3], False)[1].diagonal(dim1=1, dim2=2).cpu().double())
        mags.append(field['mag'][ix])
    if not proxies:
        raise ValueError('position validation needs nonempty fields')
    alias, response = torch.cat(alias).numpy(), torch.cat(response).numpy()
    mags = np.concatenate(mags)
    weights = np.asarray([p['objects'] for p in proxies])
    bins = {}
    edges = (-np.inf,) + POSITION_MAG_BINS + (np.inf,)
    for low, high in zip(edges[:-1], edges[1:]):
        keep = (mags >= low) & (mags < high)
        n = int(keep.sum())
        if n < 2:
            bins[f'{low}-{high}'] = dict(n=n)
            continue
        r = response[keep].mean(0)
        bins[f'{low}-{high}'] = dict(
            n=n, R=r.tolist(),
            alias_rms_over_R=float(np.sqrt(np.square(alias[keep]).sum((1, 2)).mean()) / r.mean()))
    return dict(version=POSITION_LOSS_VERSION, objects=int(weights.sum()),
                cross_per_layer=np.average([p['cross_per_layer'] for p in proxies], axis=0, weights=weights).tolist(),
                auto_per_layer=np.average([p['auto_per_layer'] for p in proxies], axis=0, weights=weights).tolist(),
                projected_alias_by_mag=bins)


def _phase_position(images):
    """Source position [N, 2] relative to the stamp centre from the lowest Fourier mode."""
    n = images.shape[-1]
    k = 2 * np.pi / n
    phase = np.exp(-1j * k * (np.arange(n) - n // 2))
    return np.stack([-np.angle((images * phase[None, None, :]).sum((1, 2))) / k,
                     -np.angle((images * phase[None, :, None]).sum((1, 2))) / k], 1)


def partner_geometry(images, truth_xy):
    """(delta [N,2] f4, size [N] f4) of noiseless isolated partners.

    Partner stamps are centred on round(truth), so delta = truth -
    floor(truth + .5), the xlens_shift convention (xy - floor(xy + .5)).
    NOT truth - det: on field 0 that has slope -0.26 against the actual
    source position. The convention is asserted per field from the Fourier
    phase position of the brightest partners (exact for centrally symmetric
    noiseless profiles; residual 0.000 px on field 0).
    size: Gaussian(sigma 4 px)-weighted RMS radius of the noiseless q0.
    """
    images = np.asarray(images, dtype=np.float64)
    delta = truth_xy - np.floor(truth_xy + .5)
    bright = np.argsort(-images.sum((1, 2)))[:16]
    error = np.abs(_phase_position(images[bright]) - delta[bright])
    if not np.median(error) < .02:
        raise ValueError(f'partner stamps are not centred on round(truth): median error {np.median(error):.3f} px')
    q0 = partner_q(images)[:, 0].astype(np.float64)
    yy, xx = np.mgrid[:images.shape[-2], :images.shape[-1]]
    r2 = (xx - images.shape[-1] // 2) ** 2 + (yy - images.shape[-2] // 2) ** 2
    window = np.exp(-r2 / (2 * 4. ** 2))
    flux = (window * q0).sum((1, 2))
    size = np.sqrt(np.clip((window * r2 * q0).sum((1, 2)), 0, None) / np.where(flux > 0, flux, np.nan))
    # Nonpositive windowed flux (none in field 0) lands in the largest-size bin.
    size = np.where(np.isfinite(size), size, np.inf)
    return delta.astype(np.float32), size.astype(np.float32)


def _null_bins(size):
    return np.searchsorted(POSITION_NULL_SIZE_EDGES, np.asarray(size), side='right')


def _null_terms(p, r, halves):
    """Per-axis cross estimate <P>_A <P>_B / <R>^2 and the full-sample <P>/<R> of one bin."""
    pd, rd = p.diagonal(dim1=1, dim2=2), r.diagonal(dim1=1, dim2=2)   # [n, 2]: e1 vs g1, e2 vs g2
    a, b = halves
    mean_r = rd.mean(0)
    cross = pd[a].mean(0) * pd[b].mean(0) / mean_r.square()
    return cross, (pd.mean(0) / mean_r).detach()


def position_null_penalty(model, data, rng, per_bin=POSITION_NULL_PER_BIN, *, return_details=False):
    """Signed position-term nulling on noiseless isolated partners (version POSITION_NULL_VERSION).

    The position response P (``position_response`` at the partner's true
    sub-pixel offset) is the shear response the network adds by following
    the moving source across the pixel lattice; AnaCal's R omits it, so the
    bin bias is -<P>/<R>. On d3c16 parab10 it is one D4 lattice harmonic,
    e1(delta) ~ a [cos 2pi dx - cos 2pi dy], with a seed-dependent sign.
    Each size bin draws ``per_bin`` partners from the batch, splits them in
    two random halves A, B and contributes <P>_A <P>_B / <R>^2 per axis: the
    halves are independent, so the expectation is (<P>/<R>)^2 without the
    per-source scatter floor of a single-sample square (one batch may be
    negative). Bins with fewer than 2 partners are skipped. Uniform weight
    over active bins, so compact bins (largest term, ~10% of sources) are not
    swamped. Differentiable R in the denominator: shrinking R is not rewarded.
    Runs in the caller's precision (FP32 outside the TF32 data term).
    """
    images, delta, size = data['iso'], data['iso_delta'], data['iso_size']
    images = images.detach().cpu().numpy() if torch.is_tensor(images) else np.asarray(images)
    bins = _null_bins(size)
    chosen, groups = [], []
    for b in range(len(POSITION_NULL_SIZE_EDGES) + 1):
        members = np.flatnonzero(bins == b)
        if len(members) < 2:
            continue
        pick = rng.choice(members, min(per_bin, len(members)), replace=False)
        start = sum(len(c) for c in chosen)
        half = len(pick) // 2
        groups.append((b, start, len(pick), half))
        chosen.append(pick)
    if not chosen:
        # Tiny final FoF batches can contain only singleton size bins. There
        # is no two-half estimator for this batch; skip just the null term.
        loss = next(model.parameters()).sum() * 0.
        if not return_details:
            return loss
        return loss, dict(version=POSITION_NULL_VERSION, bins={}, skipped_no_pairs=True)
    index = np.concatenate(chosen)
    parameter = next(model.parameters())
    q = torch.as_tensor(partner_q(images[index]), device=parameter.device, dtype=parameter.dtype)
    _, r, p = position_response(model, q, np.asarray(delta)[index], create_graph=True)
    terms, details = [], {}
    for b, start, count, half in groups:
        rows = slice(start, start + count)
        cross, ratio = _null_terms(p[rows], r[rows], (slice(0, half), slice(half, count)))
        terms.append(cross.sum())
        details[str(b)] = dict(n=count, ratio=ratio.tolist())
    loss = POSITION_NULL_SCALE * torch.stack(terms).mean()
    if not return_details:
        return loss
    return loss, dict(version=POSITION_NULL_VERSION, bins=details)


def position_null_validation(model, fields, per_bin=POSITION_NULL_VALIDATION_PER_BIN, batch=64):
    """Held-out <P>/<R> per size bin and axis (x1e3 = -m_pos x1e3), noiseless partners; logged."""
    images = np.concatenate([field['iso'] for field in fields])
    delta = np.concatenate([field['iso_delta'] for field in fields])
    bins = _null_bins(np.concatenate([field['iso_size'] for field in fields]))
    rng = np.random.default_rng(19302)
    out = {}
    edges = (0.,) + POSITION_NULL_SIZE_EDGES + (np.inf,)
    for b in range(len(edges) - 1):
        members = np.flatnonzero(bins == b)
        if len(members) < 2:
            out[f'{edges[b]}-{edges[b + 1]}'] = dict(n=int(len(members)))
            continue
        pick = np.sort(rng.choice(members, min(per_bin, len(members)), replace=False))
        P, R = [], []
        parameter = next(model.parameters())
        for start in range(0, len(pick), batch):
            rows = pick[start:start + batch]
            q = torch.as_tensor(partner_q(images[rows]), device=parameter.device, dtype=parameter.dtype)
            with torch.no_grad():
                _, r, p = position_response(model, q, delta[rows], create_graph=False)
            P.append(p.diagonal(dim1=1, dim2=2).double().cpu())
            R.append(r.diagonal(dim1=1, dim2=2).double().cpu())
        P, R = torch.cat(P).numpy(), torch.cat(R).numpy()
        out[f'{edges[b]}-{edges[b + 1]}'] = dict(
            n=int(len(pick)), R=R.mean(0).tolist(),
            P_over_R_x1e3=(1e3 * P.mean(0) / R.mean(0)).tolist(),
            SE_x1e3=(1e3 * P.std(0, ddof=1) / np.sqrt(len(P)) / np.abs(R.mean(0))).tolist())
    return dict(version=POSITION_NULL_VERSION, size_edges_px=list(POSITION_NULL_SIZE_EDGES), bins=out)


class PositionAliasLoss(AuxLoss):
    """Cross-realization GELU commutator energy (POSITION_LOSS_VERSION) of
    ``count`` isolated partners per step (batch['iso']); one uniformly
    sampled spatial layer per call. Requires one F8_noTanh trunk with exact
    GELUs and a fixed pooling kernel."""
    name = 'position'

    def __init__(self, weight=0., every=2, *, count=POSITION_COUNT,
                 validation_count=POSITION_VALIDATION_COUNT):
        super().__init__(weight, every)
        if count < 1 or validation_count < 1:
            raise ValueError('position counts must be positive')
        self.count, self.validation_count = int(count), int(validation_count)

    def check_model(self, model):
        if self.weight:
            _position_activations(model)

    def __call__(self, model, batch, rng):
        return position_penalty(model, batch, rng, count=self.count, return_details=True)

    def log(self, value, info):
        return dict(auxiliary={self.name + '_details': info, self.name: float(value.detach())})

    def config(self):
        if not self.weight:
            return {}
        return dict(position_loss_version=POSITION_LOSS_VERSION,
                    position_source='isolated_partner_two_noise_draws',
                    position_noise=POSITION_NOISE, position_window='pool_reach',
                    position_count=self.count, position_layers='uniform_one_layer_channel_mean')

    def validate(self, model, fields):
        """Logged, not selected on; see position_validation."""
        return position_validation(model, fields, count=self.validation_count)


class PositionNullLoss(AuxLoss):
    """Signed per-size-bin position-term nulling (POSITION_NULL_VERSION):
    ``per_bin`` partners per size bin and step (batch['iso', 'iso_delta',
    'iso_size']), split into two halves. Applied every step."""
    name = 'position_null'

    def __init__(self, weight=0., *, per_bin=POSITION_NULL_PER_BIN,
                 validation_per_bin=POSITION_NULL_VALIDATION_PER_BIN):
        super().__init__(weight)
        if per_bin < 2 or validation_per_bin < 2:
            raise ValueError('position-null per-bin counts must be >= 2 (two halves)')
        self.per_bin, self.validation_per_bin = int(per_bin), int(validation_per_bin)

    def __call__(self, model, batch, rng):
        return position_null_penalty(model, batch, rng, per_bin=self.per_bin, return_details=True)

    def log(self, value, info):
        return dict(auxiliary={self.name + '_details': info, self.name: float(value.detach())})

    def config(self):
        if not self.weight:
            return {}
        return dict(position_null_weight=self.weight,
                    position_null_version=POSITION_NULL_VERSION,
                    position_null_source='isolated_partner_noiseless_true_offset',
                    position_null_size_edges_px=list(POSITION_NULL_SIZE_EDGES),
                    position_null_per_bin=self.per_bin,
                    position_null_scale=POSITION_NULL_SCALE)

    def validate(self, model, fields):
        """Held-out <P>/<R> per size bin; logged."""
        return position_null_validation(model, fields, per_bin=self.validation_per_bin)
