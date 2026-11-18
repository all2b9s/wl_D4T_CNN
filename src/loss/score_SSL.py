"""Self-supervised log-score of the shape-noise variance.

    L = sum_i [log S_i - 2 log R_i] + lambda [(log S_1/S_2)^2 + (log R_1/R_2)^2]

S_i = sum over friends-of-friends groups of (sum_g w e_i)^2 / n, R_i = <w R_ii
+ dw_i e_i>. ScoreLoss.__call__ returns a tensor whose gradient is the score
gradient of the step:
  * EMA on (default): linear surrogate c_S . S_c + c_R . R_c with coefficients
    at running (EMA) moments; microbatches add up, FoF cross terms spanning
    microbatches are dropped (logscore_v3 behaviour).
  * EMA off, one chunk: the batch log-score itself.
  * EMA off, microbatched: exact gradient of the whole-batch log-score from a
    detached first pass (current-batch coefficients and full group sums),
    including cross-microbatch group terms; costs one extra no-grad pass.
"""
import numpy as np
import torch
from src.loss.common import (shear_response, group_square_sum, normalized_weights,
                             tf32, batch_tensors)


@torch.no_grad()
def batch_moments(model, batch, data_microbatch=None):
    """Detached batch S_i = sum_groups (sum_g w e_i)^2 / n and R_i = mean(w R_ii + dw_i e_i)."""
    weights, weight_response = normalized_weights(batch)
    count = len(weights)
    chunk = data_microbatch or count
    S = torch.zeros(2, device=weights.device)
    R = torch.zeros(2, device=weights.device)
    for begin in range(0, count, chunk):
        sl = slice(begin, min(begin + chunk, count))
        with torch.enable_grad():
            shape, response = shear_response(model, batch['q'][sl], False)
        shape, response = shape.detach(), response.detach()
        S += group_square_sum(weights[sl, None] * shape, batch['group'][sl]) / count
        R += (weights[sl, None] * response.diagonal(dim1=1, dim2=2) + weight_response[sl] * shape).sum(0) / count
    return S, R


def score_coefficients(S_bar, R_bar, isotropy_weight):
    """Linear-surrogate coefficients whose gradient equals that of

        L = sum_i [log S_i - 2 log R_i]
            + lambda [(log S_1 - log S_2)^2 + (log R_1 - log R_2)^2]

    evaluated at running (EMA) moments. sum_i log(S_i/R_i^2) is the summed
    log shape-noise variance: scale invariant, and diverges if either axis'
    response collapses, unlike the previous S - .2R form whose per-axis
    optimum -.01 R^2/S is merely zero for a dead axis. The isotropy term
    encodes the population 45-degree symmetry D4 does not impose. The
    surrogate is linear in the batch moments, so microbatches accumulate.
    """
    R_bar = torch.maximum(R_bar, 1e-3 * S_bar.clamp_min(1e-30).sqrt())
    log_S, log_R = S_bar.log(), R_bar.log()
    sign = R_bar.new_tensor([1., -1.])
    c_S = (1 + 2 * isotropy_weight * sign * (log_S[0] - log_S[1])) / S_bar
    c_R = (-2 + 2 * isotropy_weight * sign * (log_R[0] - log_R[1])) / R_bar
    return c_S.detach(), c_R.detach()


def score_tensor(S, R, isotropy_weight):
    """Differentiable log-score (response floored as in score_coefficients)."""
    R = torch.maximum(R, 1e-3 * S.clamp_min(1e-30).sqrt())
    log_S, log_R = S.log(), R.log()
    return ((log_S - 2 * log_R).sum()
            + isotropy_weight * ((log_S[0] - log_S[1]) ** 2 + (log_R[0] - log_R[1]) ** 2))


def score_value(S, R, isotropy_weight):
    R = torch.maximum(R, 1e-3 * S.clamp_min(1e-30).sqrt())
    log_S, log_R = S.log(), R.log()
    return float((log_S - 2 * log_R).sum()
                 + isotropy_weight * ((log_S[0] - log_S[1]) ** 2 + (log_R[0] - log_R[1]) ** 2))


def _gradient_vector(value, params, retain_graph=False):
    parts = torch.autograd.grad(value, params, retain_graph=retain_graph, allow_unused=True)
    return torch.cat([(torch.zeros_like(p) if g is None else g).detach().flatten()
                      for p, g in zip(params, parts)])


class ScoreLoss:
    """Log-score of the shape-noise variance; see the module docstring.

    Per step: prepare(model, batch, microbatch) before zero_grad, then
    loss, S_c, R_c = self(shape, response, w, dw, group, count) per
    (micro)batch slice (w, dw from common.normalized_weights of the whole
    batch, count = whole-batch size) and loss.backward(); after the
    optimizer step, update(S, R) with the summed detached S_c, R_c.
    """

    def __init__(self, isotropy_weight=1., ema_decay=.9):
        if not 0 <= isotropy_weight < np.inf:
            raise ValueError('invalid isotropy weight')
        if ema_decay is not None and not 0 <= ema_decay < 1:
            raise ValueError('ema_decay must be in [0, 1) or None')
        self.isotropy_weight, self.ema_decay = isotropy_weight, ema_decay
        self.ema = dict(S=None, R=None)
        self._mode = None

    # ---- per-step protocol -------------------------------------------------
    @staticmethod
    def moments(shape, response, w, dw, group, count):
        """(S_c, R_c) of one slice; R_c is NaN without a response (pure supervision)."""
        if response is None:
            R_c = shape.new_full((2,), float('nan'))
        else:
            R_c = (w[:, None] * response.diagonal(dim1=1, dim2=2) + dw * shape).sum(0) / count
        S_c = group_square_sum(w[:, None] * shape, group) / count
        return S_c, R_c

    def prepare(self, model, batch, microbatch=None):
        """Fix this step's coefficients (call once per step, before zero_grad)."""
        if self.ema_decay is not None:
            if self.ema['S'] is None:
                self.ema['S'], self.ema['R'] = batch_moments(model, batch, microbatch)
            self._mode = 'ema'
            self._coefficients = score_coefficients(self.ema['S'], self.ema['R'], self.isotropy_weight)
            self._noise = None
            return
        n = len(batch['w'])
        if not microbatch or microbatch >= n:
            self._mode = 'direct'
            return
        # Exact whole-batch gradient through a detached first pass.
        self._mode = 'exact'
        w, dw = normalized_weights(batch)
        signal, response = [], []
        with torch.no_grad():
            for start in range(0, n, microbatch):
                sl = slice(start, start + microbatch)
                with torch.enable_grad():
                    e, r = shear_response(model, batch['q'][sl], False)
                e, r = e.detach(), r.detach()
                signal.append(w[sl, None] * e)
                response.append(w[sl, None] * r.diagonal(dim1=1, dim2=2) + dw[sl] * e)
            signal = torch.cat(signal)
            self._groups, inverse = torch.unique(batch['group'], return_inverse=True)
            self._sums = signal.new_zeros(len(self._groups), 2).index_add(0, inverse, signal)
            S = self._sums.square().sum(0) / n
            R = torch.cat(response).mean(0)
        self._coefficients = score_coefficients(S, R, self.isotropy_weight)
        self._noise = score_coefficients(S, R, 0.)

    def _terms(self, shape, response, w, dw, group, count):
        S_c, R_c = self.moments(shape, response, w, dw, group, count)
        if self._mode == 'exact':
            sums = self._sums[torch.searchsorted(self._groups, group)]
            S_term = (2 * sums * w[:, None] * shape).sum(0) / count
        else:
            S_term = S_c
        return S_term, S_c, R_c

    def _linear(self, coefficients, S_term, R_c):
        c_S, c_R = coefficients
        return (c_S * S_term).sum() + (c_R * R_c).sum()

    def __call__(self, shape, response, w, dw, group, count):
        """(loss, S_c, R_c) of one slice; loss.backward() gives the score gradient."""
        if self._mode is None:
            raise RuntimeError('ScoreLoss.prepare must run before each step')
        if response is None:
            raise ValueError('the score needs the shear response')
        S_term, S_c, R_c = self._terms(shape, response, w, dw, group, count)
        if self._mode == 'direct':
            if len(shape) != count:
                raise ValueError('EMA-free score without prepare(microbatch) needs the whole batch')
            return score_tensor(S_c, R_c, self.isotropy_weight), S_c, R_c
        return self._linear(self._coefficients, S_term, R_c), S_c, R_c

    def noise_surrogate(self, shape, response, w, dw, group, count):
        """Tensor whose gradient is the isotropy-free (log noise) part of the score gradient."""
        S_term, S_c, R_c = self._terms(shape, response, w, dw, group, count)
        if self._mode == 'direct':
            return score_tensor(S_c, R_c, 0.)
        if self._mode == 'ema':
            self._noise = self._noise or score_coefficients(self.ema['S'], self.ema['R'], 0.)
        return self._linear(self._noise, S_term, R_c)

    def update(self, S, R):
        """EMA update after the optimizer step (no-op without EMA)."""
        self._mode = None
        if self.ema_decay is None:
            return
        self.ema['S'] = self.ema_decay * self.ema['S'] + (1 - self.ema_decay) * S
        self.ema['R'] = self.ema_decay * self.ema['R'] + (1 - self.ema_decay) * R

    def value(self, S, R):
        return score_value(S, R, self.isotropy_weight)

    # ---- checkpoint state --------------------------------------------------
    def state(self):
        return {k: (None if v is None else v.tolist()) for k, v in self.ema.items()}

    def load_state(self, state, device):
        self.ema = {k: (None if state is None or state[k] is None
                        else torch.as_tensor(state[k], device=device)) for k in ('S', 'R')}

    # ---- diagnostics and validation ----------------------------------------
    def gradient_probe(self, model, batch, *, chunk=16, tf32_data=False, moments=None, with_gradients=True):
        """Read-only two-pass gradients of full-group log noise and isotropy.

        Groups are summed across all forward microbatches. Detached group sums
        supply the exact first derivative in the second pass; no optimizer or
        random-number state is changed. Optional EMA coefficients isolate the
        missing cross-group terms from the separate effect of stale moments.
        At the response floor, coefficients follow the trainer's surrogate;
        response_floor_active flags points that are not exact log-score gradients.
        """
        isotropy_weight = self.isotropy_weight
        params = [p for p in model.parameters() if p.requires_grad]
        n = len(batch['w'])
        w = batch['w'] / batch['w'].mean().clamp_min(1e-8)
        dw = batch['dw'] / batch['w'].mean().clamp_min(1e-8)
        signals, responses = [], []
        truncated = w.new_zeros(2)
        with tf32(tf32_data):
            for start in range(0, n, chunk):
                sl = slice(start, start + chunk)
                e, r = shear_response(model, batch['q'][sl], False)
                signal = (w[sl, None] * e).detach()
                signals.append(signal)
                responses.append((w[sl, None] * r.diagonal(dim1=1, dim2=2)
                                  + dw[sl] * e).detach())
                truncated += group_square_sum(signal, batch['group'][sl]) / n
                del e, r
            signal = torch.cat(signals)
            _, inverse = torch.unique(batch['group'], return_inverse=True)
            sums = signal.new_zeros(int(inverse.max()) + 1, 2).index_add(0, inverse, signal)
            S = sums.square().sum(0) / n
            R = torch.cat(responses).mean(0)
            cs, cr = score_coefficients(S, R, isotropy_weight)
            ns, nr = score_coefficients(S, R, 0.)
            coefficients = dict(noise=(ns, nr), isotropy=(cs - ns, cr - nr))
            if moments is not None and moments['S'] is not None:
                coefficients['ema_full'] = score_coefficients(moments['S'], moments['R'], isotropy_weight)
            gradients = {key: torch.zeros(sum(p.numel() for p in params), device=w.device, dtype=params[0].dtype)
                         for key in coefficients}
            for start in (range(0, n, chunk) if with_gradients else ()):
                sl = slice(start, start + chunk)
                e, r = shear_response(model, batch['q'][sl], True)
                # The surrogate's VALUE is immaterial; its gradient is exact.
                ds = (2 * sums[inverse[sl]].detach() * w[sl, None] * e).sum(0) / n
                dr = (w[sl, None] * r.diagonal(dim1=1, dim2=2) + dw[sl] * e).sum(0) / n
                for j, (key, (a, b)) in enumerate(coefficients.items()):
                    gradients[key] += _gradient_vector((a * ds + b * dr).sum(), params,
                                                       retain_graph=j + 1 < len(coefficients))
                del e, r, ds, dr
        safe_R = torch.maximum(R, 1e-3 * S.clamp_min(1e-30).sqrt())
        stats = dict(N=n, S_full=S.tolist(), S_microbatch=truncated.tolist(), R=R.tolist(),
                     sigma2_full=(S / R.square()).tolist(),
                     sigma2_microbatch=(truncated / R.square()).tolist(),
                     missing_cross_fraction=((S - truncated) / S.clamp_min(1e-30)).tolist(),
                     log_noise=float((S.log() - 2 * safe_R.log()).sum()),
                     isotropy=float((S[0].log() - S[1].log()).square()
                                    + (safe_R[0].log() - safe_R[1].log()).square()),
                     response_floor_active=(R < 1e-3 * S.sqrt()).tolist())
        return stats, gradients

    @staticmethod
    def validate(model, fields, scale, batch_size=512):
        """Whole-validation-set moments, sigma2_i = N S_i / (sum w R_ii + dw_i e_i)^2.

        sigma2 uses group S = sum_groups (sum_g w e_i)^2 (FoF groups of the
        fields), the selection criterion; sigma2_object uses per-object
        (w e_i)^2. truth_mse (target MSE / scale) comes from the same pass.
        """
        device = next(model.parameters()).device
        S = torch.zeros(2, dtype=torch.float64, device=device)
        S_group = torch.zeros(2, dtype=torch.float64, device=device)
        R = torch.zeros(2, dtype=torch.float64, device=device)
        truth = 0.0
        weight_sum = 0.0
        count = 0
        for field in fields:
            signal, groups = [], []
            for start in range(0, len(field['q']), batch_size):
                batch = batch_tensors(field, np.arange(start, min(start + batch_size, len(field['q']))), device)
                shape, response = shear_response(model, batch['q'], False)
                shape, response = shape.detach().double(), response.detach().double()
                w, dw = batch['w'].double(), batch['dw'].double()
                S += (w[:, None] * shape).square().sum(0)
                signal.append(w[:, None] * shape)
                groups.append(batch['group'])
                R += (w[:, None] * response.diagonal(dim1=1, dim2=2) + dw * shape).sum(0)
                truth += float((w * (shape - batch['truth'].double()).square().mean(1)).sum())
                weight_sum += float(w.sum())
                count += len(w)
            S_group += group_square_sum(torch.cat(signal), torch.cat(groups))
        sigma2 = [float(count * s / r ** 2) if r > 0 else 1e30 for s, r in zip(S_group.tolist(), R.tolist())]
        sigma2_object = [float(count * s / r ** 2) if r > 0 else 1e30 for s, r in zip(S.tolist(), R.tolist())]
        return dict(sigma2=sigma2, worst_sigma2=max(sigma2), sigma2_object=sigma2_object,
                    R=(R / weight_sum).tolist(), truth_mse=truth / weight_sum / scale, N=count)
