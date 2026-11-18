"""Finite-shear (high-order / nonlinear) response bias on noiseless isolated partners.

For exact finite shear curves +-h of bank partners, the residual is the
secant e(+h) - e(-h) minus the endpoint-response prediction tanh(h) (R(-h) +
R(+h)). Training penalty: unbiased (population mean residual)^2 + .01
per-source residual^2, both over the endpoint second moment.
"""
import numpy as np
import torch
from src.loss.base import AuxLoss
from src.loss.common import jacobian

# Max |population finite-curve m| at h=.02 on validation partners for selection.
HIGHORDER_GATE = 1e-3
HIGHORDER_COUNT = 32


def curve_residuals(model, bank, ix, axis, hi, create_graph):
    """Per-source secant minus endpoint-response prediction at shear +-h."""
    paths = bank.curves[ix, axis, hi]
    x = paths[:, :, 0].reshape(-1, 1, 64, 64)
    e, r = jacobian(model, x, paths[:, :, 1].reshape_as(x), create_graph)
    r = r[:, axis, 0].reshape(-1, 2).double()
    e = e[:, axis].reshape(-1, 2).double()
    denominator = np.tanh(bank.steps[hi]) * r.sum(1)
    return e[:, 1] - e[:, 0] - denominator, denominator


class NonlinearLoss(AuxLoss):
    """Population + per-source finite-curve residual of ``count`` bank partners,
    one random axis and step per call.

    The per-source term is needed: with the mean alone (logscore_v1) the
    training-partner population m was ~6e-5 while analytic round-source
    C3/C1 reached -1.17/+2.11 (HLR .8, axes 1/2), and isolated m on
    xlens_shift was -2.2e-3/+4.5e-3. Cancellation over one population does
    not transfer to another. Internal scale 1e4 (fixed coefficient retained
    from the previous objective).
    """
    name = 'highorder'
    scale = 1e4

    def __init__(self, bank, weight=.01, every=4, *, count=HIGHORDER_COUNT):
        super().__init__(weight, every)
        if count < 2:
            raise ValueError('highorder count must be >= 2 (two halves)')
        self.bank, self.count = bank, int(count)

    def __call__(self, model, batch, rng):
        bank = self.bank
        ix = torch.as_tensor(rng.choice(len(bank.ids), min(self.count, len(bank.ids)), replace=False),
                             device=bank.curves.device)
        axis = int(rng.integers(2))
        hi = int(rng.integers(2))
        residual, denominator = curve_residuals(model, bank, ix, axis, hi, True)
        norm = denominator.square().mean().clamp_min(1e-18)
        half = len(residual) // 2
        penalty = (residual[:half].mean() * residual[half:].mean() + .01 * residual.square().mean()) / norm
        return penalty.float(), dict(axis=axis, step=bank.steps[hi],
                                     m=float(residual.mean() / denominator.mean()))

    def log(self, value, info):
        return dict(highorder=float(value.detach()) * 1e4, curve_m=info.get('m'))

    def empty_log(self):
        return dict(highorder=0.0, curve_m=None)

    def config(self):
        return {} if self.count == HIGHORDER_COUNT else dict(highorder_count=self.count)

    @staticmethod
    def validate(model, bank, chunk=32):
        """Population m = mean residual / mean endpoint prediction over all sources,
        plus rms_* = per-source rms of residual / endpoint prediction."""
        out = {}
        for axis in (0, 1):
            for hi, step in enumerate(bank.steps):
                numerator = denominator = 0.0
                relative = []
                for start in range(0, len(bank.ids), chunk):
                    ix = torch.arange(start, min(start + chunk, len(bank.ids)), device=bank.curves.device)
                    residual, prediction = curve_residuals(model, bank, ix, axis, hi, False)
                    numerator += float(residual.sum())
                    denominator += float(prediction.sum())
                    relative.append((residual / prediction).detach().cpu().numpy())
                out[f'g{axis + 1}_h{step}'] = numerator / denominator
                out[f'rms_g{axis + 1}_h{step}'] = float(np.sqrt(np.mean(np.square(np.concatenate(relative)))))
        return out

    @staticmethod
    def passes(curve_m, gate=HIGHORDER_GATE):
        """High-order selection gate: max |m| at h=.02 over both axes."""
        return max(abs(v) for k, v in curve_m.items() if k.startswith('g') and k.endswith('_h0.02')) <= gate
