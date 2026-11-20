"""Single-stage D4 training: target-initialized log-score + high-order loss.

Run from the project root with ``python -m src.train_d4 --help``. No allocation
is requested. Objective (OBJECTIVE_VERSION logscore_v3, 2026-11-12):

  sup(t) * weighted target MSE
  + score(t) * [sum_i log(S_i / R_i^2) + (log S_1/S_2)^2 + (log R_1/R_2)^2]
  + 10 * rotation covariance of e and R (noiseless partners, continuous angles)
  + 100 * [unbiased (population curve residual)^2 + .01 per-source residual^2]
        / endpoint moment

S_i = sum over friends-of-friends groups (GROUP_LINK px) of (sum_g w e_i)^2,
per object; R_i = <w R_ii + dw_i e_i>. sup goes 1->0 and score 0->SCORE_MAX on
a cosine over the whole run (ssl recipes: sup=0, score=1; --supervised-only:
sup=1, score=0). Batches hold whole groups. The score uses
running-moment coefficients (--no-score-ema: current-batch exact gradient).
With EMA, Forward8's 16-object microbatches omit FoF
cross terms spanning microbatches; validation retains whole-field groups. There is no
geometry term by default. Optional --position-weight penalizes the GELU
commutator energy (v2): cross product of two independent noise draws of each
noiseless isolated partner, weighted by the pool-reachable window, ratio per
channel. Its hooks exist only during the loss; inference is unchanged.
Optional --position-null-weight nulls the SIGNED output position term per
size bin (signed_position_null_v1): on noiseless isolated partners at their
true sub-pixel offsets, the cross product of two half-sample means of the
position response P over the bin mean R, for e1 and e2. The residual term is
one D4 lattice harmonic whose sign depends on the seed; the v2 energy only
scales it. Offsets enter the training loss only. LR: linear warmup then per-step cosine. The Polyak (EMA)
weights are validated and saved as final; selection minimizes the worst-axis
validation shape-noise variance among epochs passing the high-order gate.
Speed: the data term runs on TF32 tensor cores (--fp32-data disables it);
the high-order term and validation stay FP32. The next field pool is
preprocessed on CPU while the current pool trains. The FP32 high-order term
(~57% of a TF32 step) is applied every 4th step at 4x weight. GH200
benchmark 2026-11-12, 512 objects: old FP32 1.645 s/step, TF32 + every-step
high-order 0.631 s, TF32 data alone 0.274 s; TF32 changed batch R by 4e-4
relative (batch sampling scatter 31-37%) and gradient cosine 0.9999996.
Results of the earlier S - .2R / geometry objective are not reproduced.
Checkpoints and logs stay in models/d4_f8 and
output/Codex_freestyle/d4_f8_integrated/single_stage.

Each loss term is a class in src/loss (supervised_loss, score_SSL, rot_loss,
nolinear_loss, position_loss); this module keeps data, banks, the step
(train_batch_score), scheduling, selection and checkpoints. The isolated
partner counts per step are --rotation-count / --highorder-count /
--position-count / --position-null-per-bin.
"""


import argparse
import copy
import hashlib
import json
import os
import time
from pathlib import Path
import numpy as np
import torch
from src.architecture.d4inv_CNN import build_single_stage_model, NOTANH_RECIPES
from src.architecture.forward8_CNN import build_f8_model, F8_RECIPES
from src.loss import (SupervisedLoss, ScoreLoss, RotationLoss, NonlinearLoss,
                      PositionAliasLoss, PositionNullLoss, set_response_mode)
# Re-exported for scripts that import them from src.train_d4.
from src.loss.common import (jacobian, shear_response, position_response, GraphedResponse,
                             group_square_sum, normalized_weights, tf32, batch_tensors,
                             blended_psf as _blended_psf)
from src.loss.score_SSL import _gradient_vector
from src.loss.rot_loss import ROTATION_WEIGHT
from src.loss.nolinear_loss import HIGHORDER_GATE
from src.loss.position_loss import partner_geometry

ROOT = Path(__file__).resolve().parents[1]
MD = ROOT / 'models/d4_f8'
STUDY_OUT = ROOT / 'output/Codex_freestyle/d4_f8_integrated/single_stage'
OBJECTIVE_VERSION = 'logscore_v3'
# Friends-of-friends linking length (detection pixels) for the score's noise
# groups. Measured on the logscore_v1 Quad candidate (500 sim_mode80 tracts,
# mag<25.5): per-object sigma .270/.277 grows to .395/.410 once detections
# within 32 px are summed (tract .380/.430), while selected/F8 stay flat to
# 32 px. Neighbours of one blend share errors; per-object variance hides it.
GROUP_LINK = 32.0
# Score weight at the end of the supervised hand-over. Measured gradient norm
# per unit weight at step 651 of logscore_v1: score/MSE ~ 110.
SCORE_MAX = 0.01
ISOLATED_DIR = Path('/work/hdd/bfcn/wenyinli/blended_training_isoIMG')
# --bank-large: extra rotation-only partners (fit, validation) with flux-weighted
# circularized CatSim half-light radius >= LARGE_HLR arcsec (1658 candidates).
# Regular partners have q-image r50 median 2.2 px, 99th percentile 4.5 px;
# parab6 rotation errors exceeded 2% only above ~5 px.
LARGE_HLR = 0.8
LARGE_COUNT = (128, 32)
# CPU preprocessing threads (finite shears, AnaCal); numpy releases the GIL.
CPU_WORKERS = max(1, min(16, len(os.sched_getaffinity(0))))
IDHASH = '3e4c65203e301c4bff922a0a7d0add4a9291e76e9f52e0db97ea3ae976082590'


def record_result(key, value, output_dir=STUDY_OUT):
    import fcntl
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with os.fdopen(
        os.open(output_dir / 'results.json', os.O_RDWR | os.O_CREAT, 432),
        'r+'
    ) as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        stream.seek(0)
        raw = stream.read()
        data = json.loads(raw) if raw else {}
        data[key] = value
        serialized = json.dumps(data, indent=2, allow_nan=False)
        stream.seek(0)
        stream.write(serialized)
        stream.truncate()


def training_q(field, rows=None, renoise=None):
    """Reconstruct the training Q and responses from the raw stamps.

    renoise: RenoiseConfig; None = RenoiseConfig() (spectral, c=0.1), which
    matches blended_training_q_n0594_spectral_c01 (filler noise seeded
    SeedSequence(20020620, sample_id, 1)).  RenoiseConfig(mode='rotate')
    reproduces the original rotate renoise of blended_training_q_n0594.
    """
    from src.anacal.pixel_response import anacal_pix_r, RenoiseConfig
    if renoise is None:
        renoise = RenoiseConfig()
    with np.load(f'/work/hdd/bfcn/wenyinli/blended_training_stamp/blended_{field:05d}.npz') as source:
        raw, ids = (source['blended_stamps'], source['sample_id'])
    if rows is not None:
        raw, ids = (raw[rows], ids[rows])
    psf = _blended_psf()
    noise = np.stack([
        np.random.default_rng(np.random.SeedSequence([20020620, int(sample_id)]))
        .normal(0, 0.594, (64, 64))
        .astype(np.float32)
        for sample_id in ids
    ])
    noise2 = [None] * len(ids)
    if renoise.spectral:
        noise2 = [np.random.default_rng(np.random.SeedSequence([20020620, int(sample_id), 1]))
                  .normal(0, 0.594, (64, 64)).astype(np.float32) for sample_id in ids]
    # The old workers=1 API still forked a process. Calling it from training's
    # threaded prefetch after CUDA initialization can deadlock. Use the same
    # source operation and identical explicit centers without creating a pool.
    center = np.full(2, raw.shape[1] // 2, dtype=np.float64)
    q = np.stack([anacal_pix_r(
        image,
        psf,
        center=center,
        scale_arcsec_per_pix=0.2,
        sigma_arcsec=0.85 / 2.355,
        freq_lim=10.0,
        noise_map=n,
        noise2_map=n2,
        renoise=renoise,
    ) for image, n, n2 in zip(raw, noise, noise2)])
    return (ids, q)


class TrainingData:

    def __init__(self, target="truth"):
        if target not in ("truth", "rdsm"):
            raise ValueError(target)
        self.target = target
        # True: field() also returns the noiseless isolated partners (position loss).
        self.isolated = False
        # True (requires isolated): also return partner sub-pixel offsets and sizes.
        self.position_null = False
        self.ids = np.load(ROOT / 'models/blended_rdsm_runs/renoised_q_n0594_330k_3m/attnw/train_sample_ids.npy')
        assert len(self.ids) == 330000 and hashlib.sha256(self.ids.tobytes()).hexdigest() == IDHASH
        path = ROOT.parent / ('opt_e/datasets/isolated_rdsm_best_m002/rdsm_catalog.npz' if target == 'rdsm' else 'opt_e/datasets/isolated_truth_weighted_m002/truth_catalog.npz')
        with np.load(path) as d:
            order = np.argsort(d['sample_id'])
            self.sid = d['sample_id'][order]
            self.exp = d['exp_id'][order]
            self.det = d['det_id'][order]
            keys = ('rdsm_e1', 'rdsm_e2') if target == 'rdsm' else ('truth_e1_weighted', 'truth_e2_weighted')
            self.truth = np.stack([d[keys[0]], d[keys[1]]], 1)[order]
            self.valid = d['rdsm_valid' if target == 'rdsm' else 'truth_valid'][order] & np.isfinite(self.truth).all(1)
        ix = np.searchsorted(self.sid, self.ids)
        assert np.array_equal(self.sid[ix], self.ids) and self.valid[ix].all()
        self.target_scale = float(np.mean(self.truth[ix] ** 2))
        self.truth_scale = self.target_scale
        if target == 'rdsm':
            # Keep the original objective normalization: only target values change.
            with np.load(ROOT.parent / 'opt_e/datasets/isolated_truth_weighted_m002/truth_catalog.npz') as reference:
                order = np.argsort(reference['sample_id'])
                np.testing.assert_array_equal(self.sid, reference['sample_id'][order])
                original = np.stack([reference['truth_e1_weighted'], reference['truth_e2_weighted']], 1)[order]
                original_valid = reference['truth_valid'][order] & np.isfinite(original).all(1)
                required = np.isin(self.sid, self.ids) | ((self.exp >= 935) & (self.exp < 940) & original_valid)
                assert self.valid[required].all(), 'RDSM lacks labels required by the original split'
                self.valid = original_valid
                self.truth_scale = float(np.mean(original[ix] ** 2))

    def field(self, field, train=True, limit=None):
        with np.load(f'/work/hdd/bfcn/wenyinli/blended_training_stamp/blended_{field:05d}.npz') as d:
            ids = d['sample_id']
            rows = np.flatnonzero(np.isin(
                ids,
                self.ids
            ) if train else self.valid[np.searchsorted(self.sid, ids)])
            if limit is not None:
                rows = rows[:limit]
            ids = ids[rows]
            detid = d['det_id'][rows]
            xy = np.stack([d['det_x'][rows], d['det_y'][rows]], 1)
        ix = np.searchsorted(self.sid, ids)
        assert np.array_equal(self.sid[ix], ids)
        path = (
            Path('/work/hdd/bfmo/wenyinli/measurement/blended_sims/g1_2/blended_det_fpfs_training_s4')
            / f'catalog_{field}.npz'
        )
        with np.load(path) as d:
            idx = np.argsort(d['det_id'])
            loc = idx[np.searchsorted(d['det_id'][idx], detid)]
            assert np.array_equal(d['det_id'][loc], detid)
            np.testing.assert_allclose(np.stack([d['x'][loc], d['y'][loc]], 1), xy, atol=0.001)
            w = d['fpfs_w'][loc].astype('f4')
            dw = np.stack([d['fpfs_dw_dg1'][loc], d['fpfs_dw_dg2'][loc]], 1).astype('f4')
        sid, q = training_q(field, rows)
        assert np.array_equal(sid, ids)
        group = field * 1_000_000 + fof_labels(xy, GROUP_LINK)
        extra = {}
        if self.position_null and not self.isolated:
            raise ValueError('position_null requires isolated partners')
        if self.isolated:
            with np.load(ISOLATED_DIR / f'isolated_{field:05d}.npz') as z:
                order = np.argsort(z['sample_id'])
                loc = order[np.searchsorted(z['sample_id'][order], ids)]
                assert np.array_equal(z['sample_id'][loc], ids)
                extra = dict(iso=z['isolated_stamps'][loc].astype('f4'),
                             mag=z['i_ab'][loc].astype('f4'))
                if self.position_null:
                    truth_xy = np.stack([z['truth_x'][loc], z['truth_y'][loc]], 1).astype(np.float64)
            if self.position_null:
                extra.update(zip(('iso_delta', 'iso_size'), partner_geometry(extra['iso'], truth_xy)))
        return dict(
            group=group,
            q=q[:, :3].astype('f4'),
            truth=self.truth[ix],
            w=w,
            dw=dw,
            ids=ids,
            field=np.full(len(ids), field),
            **extra
        )


def fof_labels(xy, link):
    """Friends-of-friends group label per detection (pixel coordinates)."""
    from scipy.spatial import cKDTree
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    pairs = cKDTree(xy).query_pairs(link, output_type='ndarray')
    if len(pairs) == 0:
        return np.arange(len(xy))
    graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(xy),) * 2)
    return connected_components(graph, directed=False)[1]


class OriginalSourceBank:
    """Exact finite shear curves of existing allowed training partners.

    Used by the population high-order term (curves) and the rotation term
    (q = [image, g1 tangent, g2 tangent], exact zero-shear FFT tangents).
    No new galaxy rendering,
    teacher model or test images. Validation sources are disjoint from the
    fitting sources; both belong to the original supervised training pool.
    curves: [source, axis, step, sign(-,+), (image, endpoint tangent), 64, 64].
    """

    def __init__(self, device='cpu', validation=False, count=None, large=False):
        from concurrent.futures import ThreadPoolExecutor
        from scripts.d4_f8.efficiency_eval import finite_noiseless_shear
        from src.anacal.pixel_response import anacal_pix_r

        def finite_path(images, axis, t, pad):
            if t == 0.:
                # Exact zero-shear polynomial derivative via FFT (FP64).
                size = images.shape[-1]
                n = size + 2 * pad
                z = np.pad(images, ((0, 0), (pad, pad), (pad, pad)))
                spectrum = np.fft.fft2(np.fft.ifftshift(z, axes=(-2, -1)), axes=(-2, -1))
                freq = np.fft.fftfreq(n) * 2 * np.pi
                ky, kx = np.meshgrid(freq, freq, indexing='ij')

                def inverse(coeff):
                    full = np.fft.fftshift(np.fft.ifft2(coeff, axes=(-2, -1)), axes=(-2, -1)).real
                    return full[:, pad:pad + size, pad:pad + size]
                gx, gy = inverse(1j * kx * spectrum), inverse(1j * ky * spectrum)
                X, Y = np.meshgrid(np.arange(size) - size // 2, np.arange(size) - size // 2, indexing='xy')
                sigma2 = (.85 / 2.355 / .2) ** 2
                tangent = (inverse(sigma2 * (kx * kx - ky * ky) * spectrum) - X * gx + Y * gy if axis == 0
                           else inverse(2 * sigma2 * kx * ky * spectrum) - Y * gx - X * gy)
                return images.copy(), tangent
            pieces = [images[i:i + 32] for i in range(0, len(images), 32)]
            with ThreadPoolExecutor(max_workers=CPU_WORKERS) as executor:
                result = list(executor.map(lambda v: finite_noiseless_shear(v, axis, t, pad), pieces))
            return tuple(np.concatenate([v[k] for v in result]) for k in (0, 1))

        bank_start = time.monotonic()
        print('AUXILIARY_START', 'validation' if validation else 'fit', flush=True)
        data = TrainingData(target='rdsm')
        _, sources, fields, chosen, large_chosen = bank_selection(data, validation, count, large)

        def read_partner_field(field, indices, root):
            ids = data.ids[indices]
            with np.load(Path(root) / f'isolated_{field:05d}.npz') as z:
                o = np.argsort(z['sample_id'])
                loc = o[np.searchsorted(z['sample_id'][o], ids)]
                np.testing.assert_array_equal(z['sample_id'][loc], ids)
                np.testing.assert_array_equal(z['catsim_index'][loc], sources[indices])
                psf = z['psf'].astype(float)[:48, :48]
                images = [anacal_pix_r(im.astype(float), psf)[0] for im in z['isolated_stamps'][loc]]
            return images, ids.tolist()

        def read_partners(indices, root):
            # Fixed order (field, then position in `indices`), the original
            # bank order, so that the regular and 45-degree banks align row by row.
            images, identifiers = [], []
            with ThreadPoolExecutor(max_workers=CPU_WORKERS) as executor:
                jobs = [(f, indices[fields[indices] == f]) for f in np.unique(fields[indices])]
                for field_images, field_ids in executor.map(lambda j: read_partner_field(*j, root), jobs):
                    images.extend(field_images)
                    identifiers.extend(field_ids)
            return np.asarray(images), identifiers

        q0, identifiers = read_partners(chosen, ISOLATED_DIR)
        print('AUXILIARY_IMAGES', len(q0), 'seconds', time.monotonic() - bank_start, flush=True)
        self.ids = identifiers
        # Rotation-only sources: the regular partners plus, with large=True,
        # extra large partners. Curves (high-order term) keep the regular set.
        if len(large_chosen):
            q_large, ids_large = read_partners(large_chosen, ISOLATED_DIR)
            q_rotation, self.rotation_ids = np.concatenate([q0, q_large]), identifiers + ids_large
        else:
            q_rotation, self.rotation_ids = q0, list(identifiers)
        self.large_count = len(large_chosen)
        self.steps = (.01, .02)
        curves = []
        for axis in (0, 1):
            amplitudes = []
            for h in self.steps:
                signs = []
                for sign in (-1, 1):
                    im, tangent = finite_path(q0, axis, sign * h, 64)
                    signs.append(np.stack([im, tangent], axis=1))
                amplitudes.append(np.stack(signs, axis=1))
            curves.append(np.stack(amplitudes, axis=1))
            print('AUXILIARY_CURVE_AXIS', axis, 'seconds', time.monotonic() - bank_start, flush=True)
        self.curves = torch.as_tensor(np.stack(curves, axis=1), dtype=torch.float32, device=device)
        tangents = [finite_path(q_rotation, axis, 0., 64)[1] for axis in (0, 1)]
        self.q = torch.as_tensor(np.stack([q_rotation, *tangents], 1), dtype=torch.float32, device=device)
        print('AUXILIARY_READY', len(q0), 'rotation', len(self.rotation_ids),
              'seconds', time.monotonic() - bank_start, flush=True)


def bank_selection(data, validation=False, count=None, large=False):
    """Deterministic auxiliary-bank rows (indices into data.ids).

    Regular partners: 512 fit / next 128 validation unique CatSim sources
    (unchanged). large=True adds LARGE_COUNT (fit, validation) further unique
    sources with catalog radius >= LARGE_HLR arcsec, disjoint from all regular
    partners, for the rotation term only. The training catalogs' hlr column is
    NaN, so the radius is the flux-weighted circularized CatSim half-light
    radius, (f_d sqrt(a_d b_d) + f_b sqrt(a_b b_b)) / (f_d + f_b).
    Returns (radius per data.ids row, sources, fields, chosen, large_chosen).
    """
    with np.load(ROOT.parent / 'opt_e/datasets/isolated_rdsm_best_m002/rdsm_catalog.npz') as z:
        order = np.argsort(z['sample_id'])
        ix = order[np.searchsorted(z['sample_id'][order], data.ids)]
        sources, fields = z['catsim_index'][ix], z['exp_id'][ix]
    hlr = np.full(len(sources), np.nan)
    if large:
        import fitsio
        c = fitsio.read(ROOT.parent / 'data/training_extend/OneDegSq.fits',
                        columns=['a_d', 'b_d', 'a_b', 'b_b', 'fluxnorm_bulge', 'fluxnorm_disk'])[sources]
        fd, fb = c['fluxnorm_disk'].astype(float), c['fluxnorm_bulge'].astype(float)
        hlr = (fd * np.sqrt(c['a_d'] * c['b_d']) + fb * np.sqrt(c['a_b'] * c['b_b'])) / (fd + fb)
    unique, first = np.unique(sources, return_index=True)
    order = np.random.default_rng(20261110).permutation(len(unique))
    chosen = first[order[512:640] if validation else order[:512]]
    if count is not None:
        chosen = chosen[:count]
    large_chosen = np.zeros(0, dtype=chosen.dtype)
    if large:
        regular = set(first[order[:640]].tolist())
        candidates = np.array([i for i in first if hlr[i] >= LARGE_HLR and i not in regular])
        candidates = candidates[np.random.default_rng(20261114).permutation(len(candidates))]
        n_fit, n_validation = LARGE_COUNT
        if len(candidates) < n_fit + n_validation:
            raise ValueError(f'only {len(candidates)} large partners with hlr >= {LARGE_HLR}')
        large_chosen = candidates[n_fit:n_fit + n_validation] if validation else candidates[:n_fit]
    return hlr, sources, fields, chosen, large_chosen


def _gradient_cosine(a, b):
    denominator = float(a.norm() * b.norm())
    return float(a @ b) / denominator if denominator > 1e-30 else None


def _diagnostic_json(value):
    """Strict JSON: disabled response/score measurements are null, never NaN."""
    if isinstance(value, dict):
        return {k: _diagnostic_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_diagnostic_json(v) for v in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


def auxiliary_loss_config(position=None, position_null=None):
    """Checkpoint fragment of the position losses; versions only enabled ones."""
    position = position or PositionAliasLoss()
    position_null = position_null or PositionNullLoss()
    config = dict(position_weight=position.weight, position_every=position.every)
    config.update(position_null.config())
    config.update(position.config())
    return config


def train_batch_score(model, batch, optimizer, *, supervised, score, aux, step, rng,
                      supervised_weight=1.0, score_weight=0.1, data_microbatch=None,
                      measure_gradients=False, tf32_data=False, data_response=None,
                      diagnostic=False, diagnostic_reference=None):
    """One optimizer step of  sup * SupervisedLoss + score * ScoreLoss + sum_aux weight_at(step) * aux.

    Q has shape [batch, 3, height, width]: image, g1 tangent, g2 tangent.
    ``score`` keeps its own running moments (ScoreLoss.ema). ``aux`` is the
    ordered list of AuxLoss terms; their order is the RNG draw order.
    tf32_data runs only the data term (batch-averaged S, R, MSE; forward and
    backward) on TF32 tensor cores. The auxiliary terms always run in FP32.
    """
    params = [p for p in model.parameters() if p.requires_grad]
    if diagnostic and not measure_gradients:
        raise ValueError('diagnostic requires component gradients')
    parameter_before = (torch.cat([p.detach().flatten() for p in params])
                        if diagnostic else None)
    score_noise_grad = torch.zeros_like(parameter_before) if diagnostic else None

    def flat_grad():
        return torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).flatten() for p in params])

    use_score = bool(score_weight)
    if use_score:
        score.prepare(model, batch, data_microbatch)
    optimizer.zero_grad(set_to_none=True)

    weights, weight_response = normalized_weights(batch)
    count = len(weights)
    chunk = data_microbatch or count
    S = torch.zeros(2, device=weights.device)
    R = torch.zeros(2, device=weights.device)
    truth_value = 0.0
    grads = dict(truth=0., score=0., rotation=0., highorder=0.)
    data_precision = tf32(tf32_data)
    data_precision.__enter__()
    for begin in range(0, count, chunk):
        sl = slice(begin, min(begin + chunk, count))
        w, dw, group = weights[sl], weight_response[sl], batch['group'][sl]
        # Groups straddling a microbatch (forward8 only) lose their cross terms
        # in the EMA score (the EMA-free microbatched score keeps them).
        if use_score:
            if data_response is None:
                shape, response = shear_response(model, batch['q'][sl], True)
            else:
                shape, response = data_response(batch['q'][sl], eager=measure_gradients)
            surrogate_c, S_c, R_c = score(shape, response, w, dw, group, count)
        else:  # pure supervision needs no response: one forward/backward
            shape = model(batch['q'][sl, 0:1])
            response = None
            S_c, R_c = ScoreLoss.moments(shape, None, w, dw, group, count)
            surrogate_c = shape.new_zeros(())
        truth_c = supervised(shape, batch['truth'][sl], w, count)
        if measure_gradients:
            before = flat_grad()
            if diagnostic and use_score:
                score_noise_grad += _gradient_vector(
                    score_weight * score.noise_surrogate(shape, response, w, dw, group, count), params,
                    retain_graph=True)
            if supervised_weight:
                (supervised_weight * truth_c).backward(retain_graph=True)
            middle = flat_grad()
            if use_score:
                (score_weight * surrogate_c).backward()
            grads['truth'] = grads['truth'] + (middle - before)
            grads['score'] = grads['score'] + (flat_grad() - middle)
        else:
            (supervised_weight * truth_c + score_weight * surrogate_c).backward()
        S += S_c.detach()
        R += R_c.detach()
        truth_value += float(truth_c.detach())
        del shape, response, S_c, R_c, truth_c, surrogate_c
    data_precision.__exit__(None, None, None)

    # Auxiliary terms in fixed order (RNG stream: rotation, high-order, position, position null).
    logs, auxiliary, step_weights = {}, {}, {}
    for loss in aux:
        weight = step_weights[loss.name] = loss.weight_at(step)
        if not weight:
            entries = loss.empty_log()
        else:
            before = flat_grad() if measure_gradients else None
            value, info = loss(model, batch, rng)
            if not torch.isfinite(value):
                raise FloatingPointError(f'{loss.name} penalty is not finite')
            ((weight * loss.scale) * value).backward()
            entries = loss.log(value, info)
            if measure_gradients:
                grads[loss.name] = flat_grad() - before
        auxiliary.update(entries.pop('auxiliary', {}))
        logs.update(entries)
    highorder_value = logs.get('highorder', 0.0)
    rotation_values = logs.get('rotation', [0.0, 0.0])

    batch_score = score.value(S, R) if use_score else 0.0
    if not np.isfinite([truth_value, batch_score, highorder_value] + rotation_values).all():
        raise FloatingPointError([truth_value, batch_score, highorder_value] + rotation_values)
    diagnostic_values = {}
    if diagnostic:
        vectors = {k: v.detach() for k, v in grads.items() if torch.is_tensor(v)}
        if use_score:
            vectors['score_noise'] = score_noise_grad
            vectors['score_isotropy'] = vectors['score'] - score_noise_grad
            if score.ema['S'] is not None:
                diagnostic_values['moment_reference'] = {k: v.tolist() for k, v in score.ema.items()}
                diagnostic_values['moment_batch_to_ema'] = dict(
                    S=(S / score.ema['S']).tolist(), R=(R / score.ema['R']).tolist())
        diagnostic_values['gradient_norms'] = {k: float(v.norm()) for k, v in vectors.items()}
        parameter_ranges = []
        cursor = 0
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                parameter_ranges.append((name, slice(cursor, cursor + parameter.numel())))
                cursor += parameter.numel()
        diagnostic_values['component_parameter_norms'] = {
            key: {name: float(vector[sl].norm()) for name, sl in parameter_ranges}
            for key, vector in vectors.items()}
        diagnostic_values['gradient_cosines'] = {
            a + ':' + b: _gradient_cosine(va, vb)
            for i, (a, va) in enumerate(vectors.items())
            for b, vb in list(vectors.items())[i + 1:]}
        reference_gradients = {}
        if diagnostic_reference is not None:
            train_stats, train_gradients = score.gradient_probe(
                model, batch, chunk=data_microbatch or 16, tf32_data=tf32_data, moments=score.ema)
            val_stats, val_gradients = score.gradient_probe(model, diagnostic_reference)
            diagnostic_values.update(train_probe=train_stats, validation_probe=val_stats)
            reference_gradients = {'train_' + k: v for k, v in train_gradients.items()}
            reference_gradients.update({'validation_' + k: v for k, v in val_gradients.items()})
            diagnostic_values['reference_gradient_norms'] = {
                k: float(v.norm()) for k, v in reference_gradients.items()}
            diagnostic_values['reference_parameter_norms'] = {
                key: {name: float(vector[sl].norm()) for name, sl in parameter_ranges}
                for key, vector in reference_gradients.items()}
            diagnostic_values['reference_cosines'] = {
                a + ':' + b: _gradient_cosine(va, vb)
                for a, va in vectors.items() for b, vb in reference_gradients.items()}
            diagnostic_values['probe_gradient_cosines'] = {
                a + ':' + b: _gradient_cosine(va, vb)
                for i, (a, va) in enumerate(reference_gradients.items())
                for b, vb in list(reference_gradients.items())[i + 1:]}
    total_norm = float(torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True))
    optimizer.step()
    if diagnostic:
        update = torch.cat([p.detach().flatten() for p in params]) - parameter_before
        diagnostic_values.update(
            clipping_factor=min(1., 1. / (total_norm + 1e-6)),
            learning_rate=optimizer.param_groups[0]['lr'], update_norm=float(update.norm()),
            update_dot_gradient={k: float(update @ v) for k, v in {**vectors, **reference_gradients}.items()},
            update_descent_cosine={k: _gradient_cosine(-update, v)
                                   for k, v in {**vectors, **reference_gradients}.items()})
        diagnostic_values['update_parameter_norms'] = {
            name: float(update[sl].norm()) for name, sl in parameter_ranges}
        if diagnostic_reference is not None:
            after, _ = score.gradient_probe(model, diagnostic_reference, with_gradients=False)
            diagnostic_values['validation_probe_after'] = after
            diagnostic_values['actual_validation_log_noise_change'] = (
                after['log_noise'] - diagnostic_values['validation_probe']['log_noise'])
    if use_score:
        score.update(S, R)
    metrics = dict(
        truth=truth_value, score=batch_score, highorder=highorder_value,
        S=S.tolist(), R=R.tolist(), sigma2=(S / R.square()).tolist(),
        auxiliary=auxiliary, rotation=rotation_values, curve_m=logs.get('curve_m'), grad_norm_preclip=total_norm,
        weights=dict(supervised=supervised_weight, score=score_weight,
                     rotation=step_weights.get('rotation', 0.), highorder=step_weights.get('highorder', 0.)),
    )
    if measure_gradients:
        metrics['grad_norm'] = {k: float(v.norm()) if torch.is_tensor(v) else 0. for k, v in grads.items()}
    if diagnostic:
        metrics['diagnostic'] = diagnostic_values
        metrics['weights'].update(position=step_weights.get('position', 0.),
                                  position_null=step_weights.get('position_null', 0.))
    return metrics


def epoch_batches(data, seed, epoch):
    """Optimizer steps in one epoch, reproducing the loop's field pools exactly."""
    counts = np.bincount(data.exp[np.searchsorted(data.sid, data.ids)], minlength=191)
    assert len(counts) == 191 and counts.sum() == len(data.ids)
    fields = np.random.default_rng(seed + epoch).permutation(191)
    return sum(int(np.ceil(counts[fields[i:i + 8]].sum() / 512)) for i in range(0, 191, 8))


def train(
    kind='single_score',
    seconds=1500,
    seed=42,
    max_epochs=2,
    *,
    device='cuda',
    threads=2,
    model_dir=MD,
    output_dir=STUDY_OUT,
    record=None,
    learning_rate=1e-4,
    warmup_fraction=0.05,
    weight_ema=0.995,
    tf32_data=True,
    highorder_every=4,
    highorder_weight=0.01,
    supervised_only=False,
    ssl_only=False,
    isotropy_weight=1.,
    noise_only_selection=False,
    tag=None,
    rotation_weight=ROTATION_WEIGHT,
    bank_large=False,
    position_weight=0.,
    position_every=2,
    position_null_weight=0.,
    rotation_count=16,
    highorder_count=32,
    position_count=16,
    position_null_per_bin=8,
    score_ema_decay=.9,
    cuda_graph=True,
    diagnostic_every=0,
    diagnostic_validation_every=200,
):
    from concurrent.futures import ThreadPoolExecutor
    if kind not in ('single_score', 'single_score_odd', 'single_rdsm_linear', 'single_rdsm_odd_readout') and kind not in NOTANH_RECIPES and kind not in F8_RECIPES:
        raise ValueError('The src trainer supports the current fixed-score objective only')
    if max_epochs < 1 or seconds <= 0 or threads < 1:
        raise ValueError('epochs, seconds and threads must be positive')
    if diagnostic_every < 0 or diagnostic_validation_every < 1:
        raise ValueError('invalid diagnostic cadence')
    # Loss terms (each validates its own weight/cadence/counts). Banks are attached once built.
    score = ScoreLoss(isotropy_weight, ema_decay=score_ema_decay)
    rotation = RotationLoss(None, None, rotation_weight, count=rotation_count)
    nonlinear = NonlinearLoss(None, highorder_weight, highorder_every, count=highorder_count)
    position = PositionAliasLoss(position_weight, position_every, count=position_count)
    position_null = PositionNullLoss(position_null_weight, per_bin=position_null_per_bin)
    aux = [rotation, nonlinear, position, position_null]  # order = RNG draw order; do not change
    auxiliary_config = auxiliary_loss_config(position, position_null)
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    record = record or (lambda key, result: record_result(key, result, output_dir))
    torch.set_num_threads(threads)
    # FP32 by default (validation, high-order term); see tf32_data for the data term.
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = True  # fixed 64x64 shapes: autotune conv algorithms once
    start_time = time.monotonic()
    recipe = NOTANH_RECIPES.get(kind) or F8_RECIPES.get(kind)
    ssl = bool(ssl_only or (recipe and recipe['ssl']))
    if supervised_only and ssl:
        raise ValueError('supervised_only conflicts with an ssl recipe')
    optimization_config = dict(ssl=ssl, supervised_only=supervised_only,
                               highorder_weight=highorder_weight, rotation_weight=rotation_weight,
                               isotropy_weight=isotropy_weight, noise_only_selection=noise_only_selection)
    # Non-default counts / EMA only: unchanged configs keep old checkpoints resumable.
    optimization_config.update(rotation.config(), **nonlinear.config())
    if score_ema_decay != .9:
        optimization_config['score_ema_decay'] = score_ema_decay
    # Eight-view models need microbatches only for the score's response pass (16 also fixes which FoF cross terms enter S).
    microbatch = 16 if recipe and recipe.get('forward8') and not supervised_only else None
    data = TrainingData(target=recipe['target'] if recipe else
                        ("rdsm" if kind.startswith("single_rdsm") else "truth"))
    data.isolated = bool(position_weight or position_null_weight)
    data.position_null = bool(position_null_weight)
    supervised = SupervisedLoss(data.truth_scale)
    rotation.scale_e = data.truth_scale
    name = kind + (f'_{tag}' if tag else '') + f'_seed{seed}'
    resume = model_dir / f'{name}_resume.pth'
    model = (build_f8_model(kind, seed) if kind in F8_RECIPES else build_single_stage_model(kind, seed)).to(device)
    for loss in aux:
        loss.check_model(model)  # fail before training for unsupported activations
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1.3e-06)
    total_steps = sum(epoch_batches(data, seed, epoch) for epoch in range(max_epochs))
    warmup = max(1, round(warmup_fraction * total_steps))

    def lr_factor(step):
        if step < warmup:
            return (step + 1) / warmup
        progress = min(1.0, (step - warmup) / max(1, total_steps - warmup))
        return 0.01 + 0.99 * 0.5 * (1 + np.cos(np.pi * progress))

    def objective_weights(step):
        # Target acts as initialization: cosine hand-over from MSE to the score.
        if supervised_only:
            return dict(supervised_weight=1.0, score_weight=0.0)
        if ssl:
            return dict(supervised_weight=0.0, score_weight=1.0)
        supervised = 0.5 * (1 + np.cos(np.pi * min(1.0, step / total_steps)))
        return dict(supervised_weight=float(supervised), score_weight=float(SCORE_MAX * (1 - supervised)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    state = dict(epoch=0, pool=0, offset=0, steps=0, best=None, history=[],
                 seed=seed, kind=kind, total_steps=total_steps, score_ema=None)

    parent_path = (model_dir / (recipe['parent'] + f'_seed{seed}_final.pth')
                   if recipe and recipe['parent'] else None)
    parent_sha = hashlib.sha256(parent_path.read_bytes()).hexdigest() if parent_path else None
    ema_state = None
    if resume.exists():
        saved = torch.load(resume, map_location=device, weights_only=False)
        if (saved['kind'], saved['seed'], saved['id_sha256']) != (kind, seed, IDHASH):
            raise ValueError('Resume checkpoint kind, seed or training IDs do not match')
        if saved.get('objective_version') != OBJECTIVE_VERSION:
            raise ValueError('Resume checkpoint was written by a different objective; remove it to restart')
        if (saved.get('rotation_mode', 'continuous'), saved.get('bank_large', False)) != ('continuous', bank_large):
            raise ValueError('Resume checkpoint used a different rotation mode or bank')
        if saved['progress']['total_steps'] != total_steps:
            raise ValueError('max_epochs changed during continuation; schedules would not match')
        if saved.get('auxiliary_config', auxiliary_loss_config()) != auxiliary_config:
            raise ValueError('Resume checkpoint used different auxiliary losses or position-loss version; '
                             'start a new run/tag rather than mixing objectives')
        legacy_defaults = not (ssl_only or highorder_weight != .01 or isotropy_weight != 1.
                               or noise_only_selection or len(optimization_config) != 6)
        if saved.get('optimization_config', optimization_config if legacy_defaults else None) != optimization_config:
            raise ValueError('Resume objective or selection configuration changed; use a new tag')
        model.load_state_dict(saved['state_dict'])
        optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler'])
        state = saved['progress']
        ema_state = saved['ema_state_dict']
        if recipe and saved.get('parent_sha256') != parent_sha:
            raise ValueError('Warm-start parent changed during continuation')
    elif parent_path:
        parent = torch.load(parent_path, map_location=device, weights_only=False)
        if parent['kind'] != recipe['parent'] or parent['id_sha256'] != IDHASH:
            raise ValueError('Wrong warm-start parent')
        model.load_state_dict(parent['state_dict'], strict=True)
        record(name + '_initialization', dict(parent=str(parent_path), parent_sha256=parent_sha,
               optimizer_reset=True, scalar_scale_retained=True, radial_alpha=0))
    else:
        # Calibrate feature scale and output RMS using training inputs only.
        fields = [data.field(0, limit=1768), data.field(1, limit=280)]
        q = np.concatenate([f['q'] for f in fields])[:2048]
        assert len(q) == 2048
        base = model.model.model.model
        total = torch.zeros_like(base.scalar_scale if recipe else base.spin_feature_scale, device=device)
        with torch.no_grad():
            for start in range(0, len(q), 32):
                model(torch.from_numpy(q[start:start + 32, 0:1]).to(device))
                total += (base.scalar_pooled.square().sum(0) if recipe
                          else base.spin_pooled.square().sum((0, 2)))
            feature_scale = (total / len(q)).sqrt().clamp_min(1e-06)
            if recipe:
                base.set_scalar_scale(feature_scale)
            else:
                base.spin_feature_scale.copy_(feature_scale)
            output_square = 0.0
            for start in range(0, len(q), 32):
                calibrated = model(torch.from_numpy(q[start:start + 32, 0:1]).to(device))
                output_square += float(calibrated.square().sum())
            initial_rms = max((output_square / (2 * len(q))) ** 0.5, 1e-08)
            model.model.model.log_gain.fill_(np.log(0.05 / initial_rms))
        record(name + '_initialization', dict(
            pool_prior_sigma_pixels=(recipe or {}).get('pool_prior_sigma', 4.0),
            gain_initial=0.05 / initial_rms, raw_output_RMS=initial_rms,
            initialization_data_N=len(q), truth_used_for_initialization=False,
            objective_version=OBJECTIVE_VERSION))
        del fields, q

    # Polyak average of weights; validated, selected and saved as final.
    ema_model = copy.deepcopy(model)
    data_response = (GraphedResponse(model, microbatch)
                     if cuda_graph and microbatch and str(device).startswith('cuda') else None)
    if ema_state is not None:
        ema_model.load_state_dict(ema_state)
    for parameter in ema_model.parameters():
        parameter.requires_grad_(False)

    @torch.no_grad()
    def update_ema():
        decay = min(weight_ema, (1 + state['steps']) / (10 + state['steps']))
        for (key, average), current in zip(ema_model.state_dict().items(), model.state_dict().values()):
            if average.dtype.is_floating_point:
                average.mul_(decay).add_(current, alpha=1 - decay)
            else:
                average.copy_(current)

    def save(final=False):
        checkpoint = dict(
            kind=kind, seed=seed, id_sha256=IDHASH, target=data.target,
            objective_version=OBJECTIVE_VERSION, objective_scale=data.truth_scale,
            target_second_moment=data.target_scale, recipe=recipe, parent_sha256=parent_sha,
            progress=state, radial_alpha=model.model.model.radial_alpha,
            bank_large=bank_large, auxiliary_config=auxiliary_config, optimization_config=optimization_config,
        )
        if final:
            checkpoint.update(state_dict={k: v.detach().cpu() for k, v in ema_model.state_dict().items()},
                              weights='ema', selected_epoch=state['best']['epoch'],
                              selected_validation=state['best'])
        else:
            checkpoint.update(state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()},
                              ema_state_dict={k: v.detach().cpu() for k, v in ema_model.state_dict().items()},
                              optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict())
        target = model_dir / f"{name}_{('final' if final else 'resume')}.pth"
        temporary = target.with_suffix('.partial.pth')
        torch.save(checkpoint, temporary)
        os.replace(temporary, target)

    validation_start = time.monotonic()
    with ThreadPoolExecutor(max_workers=CPU_WORKERS) as workers:
        validation = list(workers.map(lambda i: data.field(i, False), range(935, 940)))
    print('VALIDATION_PREPARED', 'seconds', time.monotonic() - validation_start, flush=True)
    bank = OriginalSourceBank(device, large=bank_large) if rotation_weight or highorder_weight else None
    validation_bank = OriginalSourceBank(device, validation=True, large=bank_large)
    rotation.bank = nonlinear.bank = bank
    record(name + '_objective', dict(
        objective_version=OBJECTIVE_VERSION, total_steps=total_steps, warmup_steps=warmup,
        learning_rate=learning_rate, weight_ema=weight_ema, isotropy_weight=isotropy_weight, tf32_data=tf32_data, rotation_weight=rotation_weight,
        optimization_config=optimization_config,
        bank_large=bank_large, rotation_ids=bank.rotation_ids if bank is not None else [],
        validation_rotation_ids=validation_bank.rotation_ids,
        auxiliary_config=auxiliary_config, highorder_every=highorder_every,
        train_curve_ids=bank.ids if bank is not None else [], validation_curve_ids=validation_bank.ids,
        loss=('sup(t) target MSE + score(t) [sum_i log S_i/R_i^2 + isotropy], S over FoF groups '
              f'+ {rotation_weight:g} rotation covariance (e, R) on noiseless partners '
              f'(bicubic random angles{", + large partners" if bank_large else ""}) + {highorder_weight * 1e4:g} [unbiased (population curve residual)^2 + .01 per-source residual^2]; isotropy coefficient and auxiliary losses recorded separately'),
        schedule=('supervised only: sup=1, score=0' if supervised_only else 'ssl: sup=0, score=1' if ssl
                  else f'sup cosine 1->0, score 0->{SCORE_MAX}'),
        group_link_pixels=GROUP_LINK,
        selection=('EMA weights; min worst-axis validation sigma2 WITHOUT bias gate' if noise_only_selection else
                   f'EMA weights; min worst-axis validation sigma2 subject to |curve m(h=.02)|<={HIGHORDER_GATE}'),
    ))

    score.load_state(state['score_ema'], device)
    diagnostic_reference = None
    diagnostic_training = None
    if diagnostic_every:
        diagnostic_reference = batch_tensors(validation[0], np.arange(len(validation[0]['q'])), device)
        # Same fixed training and held-out fields throughout; auxiliary sample RNG is untouched.
        diagnostic_training = [data.field(0), data.field(1)]
        record(name + '_diagnostic_protocol', dict(
            every=diagnostic_every, validation_every=diagnostic_validation_every,
            training_fields=[0, 1], validation_fields=list(range(935, 940)),
            gradient_reference_field=935, optimizer_unchanged=True,
            gradient_norms='Weighted, before clipping; score_noise/isotropy partition score, not extra losses',
            update_dot_gradient='First-order loss change under actual AdamW update; negative predicts improvement',
            full_group_probe='Two-pass exact first derivative at current full-group moments; EMA-full probe isolates microbatch truncation',
            scope='One seed retraining with observational diagnostics; not a randomized loss ablation'))

    def diagnostic_validation():
        if diagnostic_every:
            record(name + f'_diagnostic_validation_{state["steps"]:05d}', dict(
                steps=state['steps'], epoch=state['epoch'],
                weights=objective_weights(state['steps']), learning_rate=optimizer.param_groups[0]['lr'],
                raw_training=score.validate(model, diagnostic_training, data.truth_scale, batch_size=32),
                raw_validation=score.validate(model, validation, data.truth_scale, batch_size=32),
                ema_training=score.validate(ema_model, diagnostic_training, data.truth_scale, batch_size=32),
                ema_validation=score.validate(ema_model, validation, data.truth_scale, batch_size=32)))

    diagnostic_validation()
    def load_pool(pool_fields):
        with ThreadPoolExecutor(max_workers=CPU_WORKERS) as workers:
            parts = list(workers.map(lambda f: data.field(int(f)), pool_fields))
        return {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}

    # The next pool's AnaCal preprocessing (CPU) overlaps the current pool's GPU steps.
    loader = ThreadPoolExecutor(max_workers=1)
    while state['epoch'] < max_epochs:
        epoch = state['epoch']
        fields = np.random.default_rng(seed + epoch).permutation(191)
        pools = [fields[i:i + 8] for i in range(0, len(fields), 8)]
        pending = None
        for pool_index in range(state['pool'], len(pools)):
            pool = pending.result() if pending is not None else load_pool(pools[pool_index])
            pending = (loader.submit(load_pool, pools[pool_index + 1])
                       if pool_index + 1 < len(pools) else None)
            # Permute whole FoF groups so each group's rows share a batch (score needs their covariance).
            _, inverse = np.unique(pool['group'], return_inverse=True)
            rank = np.random.default_rng(seed + epoch * 1000 + pool_index).permutation(inverse.max() + 1)
            order = np.argsort(rank[inverse], kind='stable')
            start_offset = state['offset'] if pool_index == state['pool'] else 0
            for offset in range(start_offset, len(order), 512):
                batch = batch_tensors(pool, order[offset:offset + 512], device)
                probe_step = bool(diagnostic_every and (state['steps'] % diagnostic_validation_every == 0
                                                       or state['steps'] == total_steps - 1))
                diagnostic_step = bool(diagnostic_every and (state['steps'] % diagnostic_every == 0 or probe_step))
                metrics = train_batch_score(
                    model, batch, optimizer, supervised=supervised, score=score, aux=aux,
                    step=state['steps'], rng=np.random.default_rng(seed + state['steps']),
                    data_microbatch=microbatch, measure_gradients=state['steps'] % 50 == 0 or diagnostic_step,
                    data_response=data_response,
                    diagnostic=diagnostic_step,
                    diagnostic_reference=diagnostic_reference if probe_step else None,
                    tf32_data=tf32_data,
                    **objective_weights(state['steps']),
                )
                scheduler.step()
                state.update(pool=pool_index, offset=offset + 512, steps=state['steps'] + 1,
                             score_ema=score.state())
                update_ema()
                if diagnostic_step:
                    record(name + f'_diagnostic_step_{state["steps"]:05d}', _diagnostic_json(dict(
                        step=state['steps'], epoch=epoch, pool=pool_index, metrics=metrics)))
                if diagnostic_every and (state['steps'] % diagnostic_validation_every == 0
                                         or state['steps'] == total_steps):
                    diagnostic_validation()
                if state['steps'] % 100 == 0:
                    save()
                if state['steps'] % 10 == 1:
                    print('TRAIN', name, epoch, pool_index, offset, state['steps'],
                          'lr', optimizer.param_groups[0]['lr'], metrics,
                          'seconds', time.monotonic() - start_time, flush=True)
                if time.monotonic() - start_time > seconds:
                    save()
                    record(name + '_progress', dict(**state, seconds_this_segment=time.monotonic() - start_time,
                                                    job=os.environ.get('SLURM_JOB_ID')))
                    loader.shutdown(wait=False, cancel_futures=True)
                    return
            state.update(pool=pool_index + 1, offset=0)
            del pool

        # Physical selection on EMA weights: worst-axis noise within the bias gate.
        result = score.validate(ema_model, validation, data.truth_scale)
        result['curve_m'] = nonlinear.validate(ema_model, validation_bank)
        result['rotation'] = rotation.validate(ema_model, validation_bank)
        if position_weight:
            result['position_alias'] = position.validate(ema_model, validation)
        if position_null_weight:
            result['position_null'] = position_null.validate(ema_model, validation)
        result['highorder_pass'] = nonlinear.passes(result['curve_m'])
        result.update(epoch=epoch + 1, learning_rate=optimizer.param_groups[0]['lr'])
        key = (not result['highorder_pass'] and not noise_only_selection, result['worst_sigma2'])
        best = state['best']
        if best is None or key < (not best['highorder_pass'] and not noise_only_selection, best['worst_sigma2']):
            state['best'] = result
            save(final=True)
        state['history'].append(result)
        state.update(epoch=epoch + 1, pool=0, offset=0)
        save()
        record(name + '_progress', state)
        diagnostic_validation()
        print('EPOCH', name, epoch + 1, result, flush=True)
    loader.shutdown()
    record(name + '_complete', state)
    resume.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--kind',
        choices=['single_score', 'single_score_odd', 'single_rdsm_linear', 'single_rdsm_odd_readout'] + list(NOTANH_RECIPES) + list(F8_RECIPES),
        default='single_score'
    )
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max-epochs', type=int, default=2)
    parser.add_argument('--seconds', type=float, default=1500)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--model-dir', type=Path, default=MD)
    parser.add_argument('--output-dir', type=Path, default=STUDY_OUT)
    parser.add_argument('--fp32-data', dest='tf32_data', action='store_false',
                        help='run the data term in full FP32 instead of TF32')
    parser.add_argument('--supervised-only', action='store_true',
                        help='target MSE only (score off) plus the high-order term')
    parser.add_argument('--ssl-only', action='store_true',
                        help='constant score weight1 and supervision0; auxiliary losses have separate switches')
    parser.add_argument('--isotropy-weight', type=float, default=1.,
                        help='isotropy coefficient inside score; 0 gives pure log noise variance')
    parser.add_argument('--highorder-weight', type=float, default=.01,
                        help='high-order coefficient before internal1e4 normalization; 0 disables')
    parser.add_argument('--noise-only-selection', action='store_true',
                        help='select by noise without the nonlinear-bias gate; diagnostic experiments only')
    parser.add_argument('--rotation-weight', type=float, default=ROTATION_WEIGHT,
                        help='weight of the continuous-rotation consistency term (0 disables)')
    parser.add_argument('--bank-large', action='store_true',
                        help=f'add {LARGE_COUNT} rotation-only partners with CatSim radius >= {LARGE_HLR} arcsec')
    parser.add_argument('--position-weight', type=float, default=0.,
                        help='fixed dimensionless weight of the cross-realization GELU alias loss (0 disables); '
                             'v1 coefficients do not transfer; grad_norm.position is logged every 50 steps')
    parser.add_argument('--position-every', type=int, default=2,
                        help='apply alias energy every k steps at k times its weight; sample one spatial layer')
    parser.add_argument('--position-null-weight', type=float, default=0.,
                        help='weight of the signed per-size-bin position-term nulling on noiseless partners '
                             '(0 disables); loss is in (1e-3)^2 units of <P>/<R>, applied every step')
    parser.add_argument('--rotation-count', type=int, default=16,
                        help='isolated partners per step of the rotation term (split over 4 angles)')
    parser.add_argument('--highorder-count', type=int, default=32,
                        help='isolated partners per step of the finite-curve high-order term')
    parser.add_argument('--position-count', type=int, default=16,
                        help='isolated partners per step of the position alias term')
    parser.add_argument('--position-null-per-bin', type=int, default=8,
                        help='isolated partners per size bin and step of the position-null term')
    parser.add_argument('--no-score-ema', dest='score_ema_decay', action='store_const', const=None, default=.9,
                        help='score from the current batch only (exact whole-batch gradient) instead of EMA moments')
    parser.add_argument('--response-mode', choices=['jvp', 'reverse'], default='jvp',
                        help='shear response by forward-mode JVP (default) or the original double backward')
    parser.add_argument('--no-cuda-graph', dest='cuda_graph', action='store_false',
                        help='do not replay the data-term microbatch (jvp mode) as a CUDA graph')
    parser.add_argument('--tag', default=None, help='appended to the run name, e.g. sup')
    parser.add_argument('--highorder-every', type=int, default=4,
                        help='apply the FP32 high-order term every k steps with k times its weight')
    parser.add_argument('--diagnostic-every', type=int, default=0,
                        help='record weighted loss gradients, conflicts and Adam updates every k steps; 0 disables')
    parser.add_argument('--diagnostic-validation-every', type=int, default=200,
                        help='fixed raw/EMA validation and full-group gradient probes cadence when diagnostics enabled')
    args = vars(parser.parse_args())
    set_response_mode(args.pop('response_mode'))
    train(**args)


if __name__ == '__main__':
    main()
