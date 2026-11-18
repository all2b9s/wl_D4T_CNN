"""Continuous-rotation consistency of e and R on noiseless isolated partners.

Partners (q = [image, g1 tangent, g2 tangent]) are rotated with the fixed
bicubic map geometry_inputs; shear tangents rotate as spin-2. Without it, D4
alone left 40-50% rotation error (supervised Quad / sigma-4 F8GELU runs);
with the old geometry term it was <0.3%. The validation rotation numbers
(ROTATION_ANGLES) are logged only, not used for selection: bicubic
resampling alone gives ~2-3%.
"""
import numpy as np
import torch
from src.architecture.CNN_toolkit import geometry_inputs
from src.loss.base import AuxLoss
from src.loss.common import shear_response

ROTATION_WEIGHT = 10.0
ROTATION_ANGLES = tuple(np.linspace(0.13, 1.45, 6))
ROTATION_COUNT = 16
ROTATION_ANGLES_PER_STEP = 4


def rotation_residuals(model, q, angles, create_graph):
    """Rotation covariance residuals of e and R.

    q [n, 3, 64, 64] is split evenly over `angles`. Returns (e_rot - spin e,
    R_rot - spin R spin^T, e, R) for the unrotated sources' e and R.
    """
    shape, response = shear_response(model, q, create_graph)
    groups = torch.arange(len(q), device=q.device).chunk(len(angles))
    shape_residual, response_residual = [], []
    for angle, ix in zip(angles, groups):
        rotated, spin = geometry_inputs(q[ix], float(angle))
        rotated_shape, rotated_response = shear_response(model, rotated, create_graph)
        shape_residual.append(rotated_shape - shape[ix] @ spin.T)
        response_residual.append(rotated_response - spin @ response[ix] @ spin.T)
    return torch.cat(shape_residual), torch.cat(response_residual), shape, response


class RotationLoss(AuxLoss):
    """shape + response rotation residual; per step ``count`` partners over
    ``angles_per_step`` uniform angles in [0, pi/2).

    bank: training OriginalSourceBank (uses bank.q); scale: target second
    moment normalizing the shape residual.
    """
    name = 'rotation'

    def __init__(self, bank, scale, weight=ROTATION_WEIGHT, *, count=ROTATION_COUNT,
                 angles_per_step=ROTATION_ANGLES_PER_STEP):
        super().__init__(weight)
        if count < angles_per_step or angles_per_step < 1:
            raise ValueError('rotation count must be >= angles_per_step >= 1')
        self.bank, self.scale_e = bank, scale
        self.count, self.angles_per_step = int(count), int(angles_per_step)

    def __call__(self, model, batch, rng):
        bank = self.bank
        ix = torch.as_tensor(rng.choice(len(bank.q), min(self.count, len(bank.q)), replace=False),
                             device=bank.q.device)
        angles = rng.uniform(0, np.pi / 2, self.angles_per_step)
        shape_residual, response_residual, _, response = rotation_residuals(model, bank.q[ix], angles, True)
        shape_loss = shape_residual.square().mean() / self.scale_e
        response_loss = response_residual.square().mean() / (response.square().mean().detach() + 0.01)
        return shape_loss + response_loss, [float(shape_loss.detach()), float(response_loss.detach())]

    def log(self, value, info):
        return dict(rotation=info)

    def empty_log(self):
        return dict(rotation=[0.0, 0.0])

    def config(self):
        changed = dict(rotation_count=self.count, rotation_angles_per_step=self.angles_per_step)
        defaults = dict(rotation_count=ROTATION_COUNT, rotation_angles_per_step=ROTATION_ANGLES_PER_STEP)
        return {k: v for k, v in changed.items() if v != defaults[k]}

    @staticmethod
    def validate(model, bank, angles=ROTATION_ANGLES):
        """Deterministic rotation errors on held-out sources, relative to rms e (R) of the sources."""
        shape_residual, response_residual, shape, response = rotation_residuals(model, bank.q, angles, False)
        shape_residual, response_residual = shape_residual.detach(), response_residual.detach()
        e_rms, r_rms = float(shape.detach().square().mean().sqrt()), float(response.detach().square().mean().sqrt())
        return dict(e_rel=float(shape_residual.square().mean().sqrt()) / e_rms,
                    R_rel=float(response_residual.square().mean().sqrt()) / r_rms,
                    e_max_rel=float(shape_residual.abs().max()) / e_rms,
                    R_max_rel=float(response_residual.abs().max()) / r_rms)
