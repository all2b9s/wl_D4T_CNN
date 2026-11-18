"""Weighted target MSE (supervised data term)."""


class SupervisedLoss:
    """sum_i w_i mean_a (e_ia - t_ia)^2 / (count * scale) on one (micro)batch slice.

    count is the size of the whole batch, so microbatch values add up to the
    batch mean; scale is the target second moment (TrainingData.truth_scale).
    """

    def __init__(self, scale):
        self.scale = scale

    def __call__(self, shape, truth, w, count):
        return (w * (shape - truth).square().mean(1)).sum() / (count * self.scale)
