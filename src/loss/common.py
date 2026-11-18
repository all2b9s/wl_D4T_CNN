"""Shared differentiable utilities of the D4/F8 training losses (no loss semantics).

Shear response (forward-mode JVP or reverse autograd), its CUDA-graph replay,
FoF group sums, the TF32 switch, batch tensor conversion and the AnaCal
preprocessing of noiseless isolated partners.
"""
import contextlib
import hashlib
from pathlib import Path
import numpy as np
import torch
from src.architecture.CNN_toolkit import jacobian_jvp

ROOT = Path(__file__).resolve().parents[2]

# 'jvp': forward-mode response (first-order backward only); 'reverse': the
# original reverse-over-reverse autograd. Same quantities to rounding.
RESPONSE_MODE = 'jvp'


def set_response_mode(mode):
    global RESPONSE_MODE
    if mode not in ('jvp', 'reverse'):
        raise ValueError(mode)
    RESPONSE_MODE = mode


def jacobian(model, x, tangents, create_graph=True):
    """e [B,2] and J[:, a, k] = sum_pixels d e_a/d x * tangents[:, k]."""
    if RESPONSE_MODE == 'jvp':
        with torch.set_grad_enabled(create_graph and torch.is_grad_enabled()):
            return jacobian_jvp(model, x, tangents)
    x = x.detach().requires_grad_(True)
    e = model(x)
    rows = [torch.autograd.grad(e[:, a].sum(), x, create_graph=create_graph,
                                retain_graph=True)[0] for a in range(e.shape[1])]
    return e, torch.stack([(j * tangents).sum((-2, -1)) for j in rows], 1)


def shear_response(model, q, create_graph=True):
    """(e, R) as in CNN_toolkit.response_torch."""
    return jacobian(model, q[:, 0:1], q[:, 1:3], create_graph)


def position_response(model, q, delta, create_graph=True):
    """Return e, native R, and the full position response P."""
    e, jac = jacobian(model, q[:, :1], q[:, 1:], create_graph)
    r, j = jac[:, :, :2], jac[:, :, 2:]
    delta = torch.as_tensor(delta, device=q.device, dtype=q.dtype)
    dx, dy = delta[:, 0, None], delta[:, 1, None]
    p = .2 * torch.stack((dx * j[:, :, 0] - dy * j[:, :, 1],
                          dy * j[:, :, 0] + dx * j[:, :, 1]), -1)
    return e, r, p


class _ResponseModule(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, q):
        return shear_response(self.model, q, True)


class GraphedResponse:
    """Data-term (e, R) of fixed-size CUDA microbatches replayed as a CUDA graph.

    Forward (primal + JVP tangents) and its first-order backward are captured
    once, on the first full microbatch, under the TF32 setting active then;
    parameters are updated in place by the optimizer, so the graph stays valid.
    Other sizes, CPU tensors, reverse mode and retain_graph (measure_gradients)
    use the eager shear_response.
    """

    def __init__(self, model, size):
        self.model, self.size, self.graphed, self.tf32 = model, int(size), None, None

    def __call__(self, q, eager=False):
        tf32 = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
        if (eager or RESPONSE_MODE != 'jvp' or not q.is_cuda or q.shape[0] != self.size
                or (self.graphed is not None and tf32 != self.tf32)):
            return shear_response(self.model, q, True)
        if self.graphed is None:
            self.tf32 = tf32
            self.graphed = torch.cuda.make_graphed_callables(
                _ResponseModule(self.model), (q.detach().clone(),), allow_unused_input=True)
        # The graphed backward returns its static buffers; with p.grad None,
        # AccumulateGrad would adopt the buffer as p.grad and the next replay
        # would overwrite it. Existing grads make every accumulation in place.
        for p in self.model.parameters():
            if p.requires_grad and p.grad is None:
                p.grad = torch.zeros_like(p)
        return self.graphed(q.detach())


def group_square_sum(values, group):
    """sum over groups of (sum of rows in the group)^2; values [n, 2], group [n]."""
    _, inverse = torch.unique(group, return_inverse=True)
    sums = values.new_zeros(int(inverse.max()) + 1, values.shape[1]).index_add(0, inverse, values)
    return sums.square().sum(0)


def normalized_weights(batch):
    """(w, dw) normalized by the batch mean weight."""
    weights = batch['w'] / batch['w'].mean().clamp_min(1e-8)
    weight_response = batch['dw'] / batch['w'].mean().clamp_min(1e-8)
    return weights, weight_response


@contextlib.contextmanager
def tf32(enabled):
    """Temporarily allow TF32 tensor cores for CUDA matmul/convolution."""
    old = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = bool(enabled)
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = old


def batch_tensors(data, indices, device):
    batch = {key: torch.as_tensor(
        np.ascontiguousarray(data[key][indices]),
        device=device
    ) for key in ['q', 'truth', 'w', 'dw', 'group']}
    for key in ('iso', 'iso_delta', 'iso_size'):
        if key in data:
            batch[key] = data[key][indices]  # stays on CPU: the AnaCal preprocessing runs there
    return batch


def blended_psf():
    psf = np.load(ROOT / 'logs/blended_test_psf.npy').astype(np.float64)[:48, :48]
    assert hashlib.sha256(psf.tobytes()).hexdigest() == (
        '2432023954b9270531650770ec6ab6f8525baa8b58c8d4c361b9b8d2551811c1'
    )
    return psf


def partner_q(images):
    """Noiseless AnaCal q [N, 5, H, W] (image, g1/g2, x/y translation tangents) of isolated partners."""
    from src.anacal.pixel_response import anacal_pix_r
    psf = blended_psf()
    images = np.asarray(images, dtype=np.float64)
    center = np.full(2, images.shape[-1] // 2, dtype=np.float64)
    return np.stack([anacal_pix_r(image, psf, center=center, scale_arcsec_per_pix=0.2,
                                  sigma_arcsec=0.85 / 2.355, freq_lim=10.0)[:5]
                     for image in images]).astype(np.float32)
