"""CPU inference for D4InvCNN checkpoints (src/architecture/d4inv_CNN.py, models/d4_f8).

Exact forward-mode (JVP) evaluation of e and R[i, j] = de_i/dt_j for q stamps
[N, 1+T, 64, 64] = [v, t_1..t_T] (AnaCal order v, g1, g2, x1, x2). Same
quantity as src.architecture.CNN_toolkit.response_torch / Calibrator.

Usage (from single_galaxies/):
    python -m cpu_src.d4inv build
    python -m cpu_src.d4inv validate models/d4_f8/single_score_seed42_final.pth
    python -m cpu_src.d4inv bench models/d4_f8/*.pth
    python -m cpu_src.d4inv export CKPT OUT.npz      # torch-free deployment

    from cpu_src.d4inv import D4InvCPU
    model = D4InvCPU.load("models/d4_f8/observable_center_selected_final.pth")
    e, R = model.measure(q, n_tangent=2, nthreads=1)   # e [N,2], R [N,2,2]

Algebraic reductions used by native/d4inv.cpp are listed at the top of that file.
Torch is needed only to export tables from a .pth checkpoint.
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


# ----------------------------------------------------------------------------- export
def export_tables(checkpoint):
    """Derive every constant table from a trusted checkpoint in float64."""
    import torch
    from torch.nn import functional as F
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from src.architecture.d4inv_CNN import load_d4_checkpoint
    from src.architecture.CNN_toolkit import (TrainableRadial, RadialAmplitudeReadout, OddGELU,
                                              OddGELUDifference, matched_spin_basis)
    model = load_d4_checkpoint(checkpoint, dtype=torch.float64)
    crop, readout = model.model, model.model.model
    base = readout.model
    if hasattr(base, "odd_readout") or hasattr(base, "scalar_readout"):
        raise ValueError("Nonlinear learned readout requires the PyTorch backend; native export is unsupported")
    assert type(crop).__name__ == 'InputCrop63'
    if isinstance(readout, TrainableRadial):
        gain = float(readout.log_gain.exp())
    elif isinstance(readout, RadialAmplitudeReadout):
        gain = float(readout.radial_gain)
    else:
        raise ValueError('Unsupported readout')
    assert base.linear_f8 and base.matched_spin_kernel == 5 and base.spin_gate and base.d4_layers == 2
    assert base.isotropic_front and base.input_channels == 1 and len(base.blocks) == 1
    assert base.attn_grid_size == 16 and base.attn_heads == 4 and base.qkv.in_features == 32
    t = {}
    d = lambda x: x.detach().to(torch.float64).cpu().numpy().copy()

    def taps(W):
        """Return center, edge, corner [out,in] and check exact D4 symmetry."""
        c, e, k = W[..., 1, 1], W[..., 0, 1], W[..., 0, 0]
        rebuilt = torch.stack((k, e, k, e, c, e, k, e, k), -1).reshape_as(W)
        assert torch.allclose(rebuilt, W, rtol=0, atol=1e-14 * float(W.abs().max())), 'kernel not D4 symmetric'
        return c, e, k

    with torch.no_grad():
        t.update(center_sigma=model.sigma, center_iterations=model.iterations, center_max_shift=model.max_shift,
                 center_pad=model.pad, gain=gain, alpha=float(readout.radial_alpha), nl_beta=10.0, nl_threshold=20.0)
        t['wb'] = d(base._get_cached_weight((63, 63), base.nl_sigma, torch.device('cpu'), torch.float64)[0, 0])
        b0, b1 = base.d4_blocks
        c, e, k = taps(b0.block._D4_avg(b0.block.P))
        t['l1'] = np.stack([d(c[:, 0]), d(e[:, 0]), d(k[:, 0]), d(b0.block.bias), d(b0.block.norm.weight),
                            d(b0.block.norm.bias), b0.res_factor * d(b0.skip.weight[:, 0, 0, 0])])
        t['eps1'] = b0.block.norm.eps
        assert isinstance(b1.skip, torch.nn.Identity)
        c, e, k = taps(b1.block._D4_avg(b1.block.P))
        # D4IsotropicConv2d fixes edge = 4 * corner: two taps, [x, 4*cross + diag].
        assert torch.equal(e, 4 * k), 'front kernel is not fourth-order isotropic'
        t['W2'] = np.concatenate([d(c.T), d(k.T)])
        t.update(b2=d(b1.block.bias), g2=d(b1.block.norm.weight), beta2=d(b1.block.norm.bias),
                 eps2=b1.block.norm.eps, res2=float(b1.res_factor))
        block = base.blocks[0]
        weight = block.conv.weight
        inv = base._d4_inverse_stack(weight[None].expand(8, *weight.shape))
        c, e, k = taps(inv.mean(0))
        t['W3'] = np.concatenate([d(c.T) + block.residual_scale * np.eye(32), d(e.T), d(k.T)])
        t['b3'] = d(block.conv.bias)
        # Attention: position channels folded into a per-token bias.
        Wt = base.token_projection.weight
        size = 16
        xx, yy = base._coordinates((size, size), torch.device('cpu'), torch.float64)
        xx, yy = xx / ((size - 1) / 2), yy / ((size - 1) / 2)
        pos = torch.stack((xx.square() + yy.square(), xx.square() * yy.square())).flatten(1)  # [2,256]
        t['Wtok'] = d(Wt[:, :32].T)
        t['tokbias'] = d((Wt[:, 32:] @ pos).T + base.token_projection.bias)
        lin = lambda m: (d(m.weight.T), d(m.bias))
        t['an_g'], t['an_b'], t['eps_a'] = d(base.attn_norm.weight), d(base.attn_norm.bias), base.attn_norm.eps
        t['Wqkv'], t['bqkv'] = lin(base.qkv)
        t['Wo'], t['bo'] = lin(base.attn_output)
        t['fn_g'], t['fn_b'], t['eps_f'] = d(base.ffn_norm.weight), d(base.ffn_norm.bias), base.ffn_norm.eps
        t['Wf1'], t['bf1'] = lin(base.ffn[0])
        t['Wf2'], t['bf2'] = lin(base.ffn[2])
        assert isinstance(base.ffn[1], torch.nn.GELU) and base.ffn[1].approximate == 'none'
        t['wk'], t['bk'] = d(base.kernel_head.weight[0]), float(base.kernel_head.bias[0])
        # Upsample + replicate pad + blur as a separable operator M [63,16].
        eye = torch.eye(256, dtype=torch.float64).reshape(256, 1, 16, 16)
        up = F.interpolate(eye, size=(63, 63), mode='bilinear', align_corners=False)
        phi = F.conv2d(F.pad(up, (2, 2, 2, 2), mode='replicate'), base.kernel_blur.to(up)).reshape(16, 16, 63, 63)
        s = torch.sqrt(torch.stack([phi[a, a].sum() for a in range(16)]))
        M = torch.stack([phi[a, 0].sum(1) / s[0] for a in range(16)], 1)  # M[i,a]
        assert torch.allclose(torch.einsum('ia,jb->abij', M, M), phi, atol=1e-13), 'blur not separable'
        t['M'] = d(M)
        gx, gy = base._coordinates((63, 63), torch.device('cpu'), torch.float64)
        t['sup'] = d(((gx.square() + gy.square()) / base.r_cut ** 2).pow(base.suppress_order // 2)) \
            if base.suppress else np.zeros((63, 63))
        t['B'] = d(matched_spin_basis(weight, 5))
        t['common'] = d(base._matched_spin_components(inv)[0])
        t['fscale'] = d(base.spin_feature_scale)
        net = base.spin_gate_net
        if isinstance(net, OddGELU):
            assert isinstance(net[1], torch.nn.GELU) and net[1].approximate == 'none'
            t['gate_mode'] = 2 if isinstance(net, OddGELUDifference) else 1
            assert type(net) in (OddGELU, OddGELUDifference)
        else:
            assert isinstance(net[1], torch.nn.Tanh)
            t['gate_mode'] = 0
        t['gW1'], t['gb1'] = d(net[0].weight), d(net[0].bias)
        t['gW2'], t['gb2'] = d(net[2].weight), d(net[2].bias)
    return {k: np.asarray(v, dtype=np.float64) for k, v in t.items()}


# ------------------------------------------------------------------------------ build
def build():
    import shlex, subprocess, sysconfig, os, platform
    import pybind11
    out = HERE / ('_d4inv' + sysconfig.get_config_var('EXT_SUFFIX'))
    arch = '-mcpu=native' if platform.machine() in ('aarch64', 'arm64') else '-march=native'
    cmd = (shlex.split(os.environ.get('CPU_CXX', 'g++')) +
           ['-std=c++17', '-O3', arch] + shlex.split(os.environ.get('D4INV_FLAGS', '')) + [ '-ffp-contract=fast', '-fno-math-errno', '-fPIC', '-shared',
            '-fvisibility=hidden', '-pthread', '-I' + pybind11.get_include(), '-I' + sysconfig.get_paths()['include'],
            str(HERE / 'native' / 'd4inv.cpp'), '-o', str(out) + '.tmp'])
    subprocess.run(cmd, check=True)
    os.replace(str(out) + '.tmp', out)  # never truncate a mapped extension
    print('Built', out)


# -------------------------------------------------------------------------------- API
class D4InvCPU:
    def __init__(self, tables, precision='float32'):
        from . import _d4inv
        self.tables = dict(tables, alpha=np.asarray(0.0))
        self.precision = precision
        self._model = _d4inv.create({k: np.asarray(v) for k, v in self.tables.items()}, precision)

    @classmethod
    def load(cls, path, precision='float32'):
        path = Path(path)
        if path.suffix == '.npz':
            with np.load(path, allow_pickle=False) as z:
                tables = {k: z[k] for k in z.files}
        else:
            tables = export_tables(path)
        return cls(tables, precision)

    def measure(self, q, n_tangent=2, nthreads=1):
        """q [N,C,64,64] float32/float64. Returns e [N,2], R [N,2,n_tangent] (model precision)."""
        q = np.asarray(q)
        if q.dtype != np.float32:
            q = np.ascontiguousarray(q, dtype=np.float64)
        else:
            q = np.ascontiguousarray(q)
        e, R = self._model.measure(q, n_tangent, nthreads)
        return e.astype(self.precision), R.astype(self.precision)


# ---------------------------------------------------------------------------- testing
def sample_q(n=64):
    """Real 5-channel training q stamps (field 0), reconstructed with training noise."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from src.train_d4 import training_q
    return training_q(0, rows=np.arange(n))[1].astype(np.float32)


def validate(checkpoints, n=64):
    """Compare e and R (all 4 AnaCal tangents) with float64 torch autograd."""
    import torch
    from src.architecture.d4inv_CNN import load_d4_checkpoint
    q = sample_q(n)
    torch.set_num_threads(4)
    rel = lambda a, b: float(np.abs(a - b).max() / np.abs(b).max())

    def reference(model, dtype):
        x = torch.tensor(q[:, :1], dtype=dtype, requires_grad=True)
        e = model(x)
        grads = [torch.autograd.grad(e[:, i].sum(), x, retain_graph=True)[0] for i in range(2)]
        tangents = torch.tensor(q[:, 1:5], dtype=dtype)
        R = torch.stack([(g * tangents).sum((-2, -1)) for g in grads], 1)  # [N, 2, 4]
        return e.detach().numpy(), R.detach().numpy()

    for ck in checkpoints:
        e_ref, R_ref = reference(load_d4_checkpoint(ck, dtype=torch.float64), torch.float64)
        e32, R32 = reference(load_d4_checkpoint(ck, dtype=torch.float32), torch.float32)
        tables = export_tables(ck)
        print(Path(ck).name, '(max-abs error / max-abs value vs torch float64)')
        print(f'  torch float32      : e {rel(e32, e_ref):.2e}  R {rel(R32, R_ref):.2e}')
        for precision in ('float64', 'float32'):
            model = D4InvCPU(tables, precision)
            e, R = model.measure(q, 4)
            e2, R2 = model.measure(q, 2, nthreads=3)
            assert np.array_equal(e, e2) and np.array_equal(R[..., :2], R2), 'T/threads changed results'
            print(f'  cpu {precision:7s}T=4  : e {rel(e, e_ref):.2e}  R {rel(R, R_ref):.2e}')


def bench(checkpoints, n=256, repeat=3):
    """Single-core throughput, gal/(CPU s). Tables and stamps are prepared in a
    subprocess so torch's thread pools never run in the timed process."""
    import os, subprocess, tempfile, time
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run([sys.executable, '-m', 'cpu_src.d4inv', '_prepare', tmp, str(n)] + list(checkpoints),
                       check=True, cwd=ROOT)
        q = np.load(Path(tmp) / 'q.npy')
        models = [D4InvCPU.load(Path(tmp) / f'{i}.npz') for i in range(len(checkpoints))]
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    for ck, model in zip(checkpoints, models):
        for T in (2, 0):
            model.measure(q[:8], T)
            best = min(_timed(model, q, T, time) for _ in range(repeat))
            print(f'{Path(ck).name:42s} T={T}: {len(q) / best:7.1f} gal/(CPU s)  ({1e3 * best / len(q):.3f} ms/gal)')


def _timed(model, q, T, time):
    t0 = time.perf_counter()
    model.measure(q, T, nthreads=1)
    return time.perf_counter() - t0


def _prepare(out, n, checkpoints):
    q = sample_q(min(n, 64))
    np.save(Path(out) / 'q.npy', np.concatenate([q] * -(-n // len(q)))[:n])
    for i, ck in enumerate(checkpoints):
        np.savez(Path(out) / f'{i}.npz', **export_tables(ck))


if __name__ == '__main__':
    cmd, args = sys.argv[1], sys.argv[2:]
    if cmd == 'build':
        build()
    elif cmd == 'export':
        np.savez(args[1], **export_tables(args[0]))
    elif cmd == 'validate':
        validate(args)
    elif cmd == 'bench':
        bench(args)
    elif cmd == '_prepare':
        _prepare(args[0], int(args[1]), args[2:])
    else:
        raise SystemExit(__doc__)
