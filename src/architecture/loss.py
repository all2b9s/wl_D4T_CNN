"""Ellipticity objectives for individual galaxies and grouped mean errors."""

import math

import torch


def ellipticity_mixing_loss(pred, target, tract_ids=None, mean_weight=1.0,
                            detection_weight=None, return_components=False):
    """Detection-weighted individual MSE plus grouped mean-residual MSE.

    For residual r = pred - target and nonnegative detection weights w:
      individual = sum_i(w_i * mean_c(r_ic**2)) / sum_i(w_i)
      mean_error = mean_g,c((sum_i_in_g(w_i*r_ic)/sum_i_in_g(w_i))**2)
      total = individual + mean_weight * mean_error

    ``pred`` and ``target`` are [B, 2]. ``detection_weight`` is [B]; None
    uses unit weights. Integer ``tract_ids`` [B] select groups; None treats
    the batch as one group. Positive-weight groups contribute equally to
    mean_error; zero-weight groups are omitted. An all-zero-weight batch
    returns differentiable zeros. Detection weights are fixed data.

    Group means only include samples in this call, with no cross-batch or
    distributed gathering. Undo each object's augmentation before calling
    this function so grouped residuals share a physical coordinate frame.
    This penalizes ellipticity error, not calibrated shear error.

    ``return_components=True`` returns (total, individual, mean_error), where
    mean_error is the raw term before multiplication by mean_weight.
    """
    if not math.isfinite(mean_weight) or mean_weight < 0:
        raise ValueError("mean_weight must be nonnegative and finite")
    if pred.ndim != 2 or pred.shape[1] != 2 or pred.shape[0] == 0:
        raise ValueError("pred must have nonempty shape [B, 2]")
    if target.shape != pred.shape or target.device != pred.device:
        raise ValueError("target must have the same shape and device as pred")
    if not pred.is_floating_point() or not target.is_floating_point():
        raise ValueError("pred and target must be floating tensors")
    # Upcast before subtraction for mixed precision.
    if pred.dtype in (torch.float16, torch.bfloat16):
        pred = pred.float()
    if target.dtype in (torch.float16, torch.bfloat16):
        target = target.float()
    residual = pred - target
    if detection_weight is None:
        weight = residual.new_ones(len(pred))
    else:
        if detection_weight.shape != (len(pred),):
            raise ValueError("detection_weight must have shape [B]")
        weight = detection_weight.detach().to(residual)
        if not torch.isfinite(weight).all() or (weight < 0).any():
            raise ValueError("detection_weight must be finite and nonnegative")
    if tract_ids is None:
        inverse = torch.zeros(len(pred), dtype=torch.long, device=pred.device)
        n_groups = 1
    else:
        if tract_ids.ndim != 1 or tract_ids.shape[0] != len(pred):
            raise ValueError("tract_ids must have shape [B]")
        if tract_ids.dtype not in (
            torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64,
        ):
            raise ValueError("tract_ids must contain integer labels")
        groups, inverse = torch.unique(tract_ids.to(pred.device), return_inverse=True)
        n_groups = groups.numel()
    weight_sum = weight.sum()
    safe_sum = torch.where(weight_sum > 0, weight_sum, torch.ones_like(weight_sum))
    individual = (weight * residual.square().mean(dim=1)).sum() / safe_sum
    sums = residual.new_zeros((n_groups, 2)).index_add(0, inverse, weight[:, None] * residual)
    group_weights = weight.new_zeros(n_groups).index_add(0, inverse, weight)
    active = group_weights > 0
    safe_weights = torch.where(active, group_weights, torch.ones_like(group_weights))
    means = sums / safe_weights[:, None]
    mean_error = (means.square().mean(dim=1) * active).sum() / active.sum().clamp_min(1)
    total = individual + mean_weight * mean_error
    return (total, individual, mean_error) if return_components else total
