"""Common interface of the auxiliary losses that run their own forward pass."""


class AuxLoss:
    """Auxiliary loss term.

    The trainer applies ``((weight_at(step) * scale) * value).backward()`` with
    ``value, info = loss(model, batch, rng)``; ``log`` / ``empty_log`` give the
    metric entries of an applied / skipped step, ``config`` the checkpoint
    fragment, ``validate`` held-out diagnostics.
    """
    name = 'aux'
    scale = 1.0

    def __init__(self, weight=0., every=1):
        if not 0 <= weight < float('inf') or int(every) < 1:
            raise ValueError(f'invalid {self.name} weight or cadence')
        self.weight, self.every = weight, int(every)

    def weight_at(self, step):
        """Every k-th step at k times the weight: same expected gradient, 1/k the cost."""
        return self.weight * self.every if step % self.every == 0 else 0.

    def check_model(self, model):
        """Raise before training if the model is unsupported."""

    def __call__(self, model, batch, rng):
        raise NotImplementedError

    def log(self, value, info):
        return {}

    def empty_log(self):
        return {}

    def config(self):
        return {}

    def validate(self, model, held_out):
        raise NotImplementedError
