"""CPU inference for cropped F8_noTanh checkpoints (16 channels, parabola pooling).

Covers src.architecture.forward8_CNN F8_RECIPES 'f8c_parab10_d{D}c16_truth', e.g. the
selected legacy model models/legacy/f8_3c16_s43.pth (depth 3, 16 channels, input 39,
WindowedChannelNorm gaussian 6 shared over the D4 orbit, GELU, parab10 pool).

Exact forward-mode evaluation of e and R[i, j] = de_i/dt_j for q stamps
[N, 1+T, 64, 64] = [v, t_1..t_T] (AnaCal order v, g1, g2, x1, x2), the same quantity
as CNN_toolkit.response_torch / response_jvp.

Usage (from single_galaxies/):
    python -m cpu_src.f8notanh build
    python -m cpu_src.f8notanh validate models/legacy/f8_3c16_s43.pth
    python -m cpu_src.f8notanh bench models/legacy/f8_3c16_s43.pth
    python -m cpu_src.f8notanh export CKPT OUT.npz      # torch-free deployment

    from cpu_src.f8notanh import F8NoTanhCPU
    model = F8NoTanhCPU.load("models/legacy/f8_3c16_s43.pth")   # or exported .npz
    e, R = model.measure(q, n_tangent=2, nthreads=1)   # e [N,2], R [N,2,2]

Reductions and memory layout are documented at the top of native/f8notanh.cpp.
Torch is needed only to export tables from a .pth checkpoint and for validation.
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
K_CUT = 1e-14  # pool-kernel support: K > K_CUT * max K


def _load_torch(checkpoint, dtype):
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from src.architecture.forward8_CNN import load_f8_checkpoint
    return load_f8_checkpoint(checkpoint, dtype=dtype)


# ----------------------------------------------------------------------------- export
def export_tables(checkpoint):
    """Derive every constant table from a trusted checkpoint in float64."""
    import torch
    from src.architecture.forward8_CNN import F8_noTanh, WindowedChannelNorm2d
    from src.architecture.CNN_toolkit import TrainableRadial, d4_orbit_stack
    model = _load_torch(checkpoint, torch.float64)
    crop = model.model
    radial = crop.model
    base = radial.model
    assert not model.enabled, 'ObservableCenter must be disabled'
    assert type(crop).__name__ == 'InputCrop63' and isinstance(radial, TrainableRadial)
    assert float(radial.radial_alpha) == 0.0
    assert type(base) is F8_noTanh and base.pool_parabola is not None
    assert base.input_size is not None and not getattr(base, 'pre_lowpass', False)
    assert getattr(base, 'inframe_pool', True)
    n, depth = base.input_size, len(base.blocks)
    C = base.blocks[0].conv.out_channels
    if C != 16:
        raise ValueError('native F8_noTanh kernel is specialized to 16 channels')
    d = lambda x: x.detach().to(torch.float64).cpu().numpy().copy()
    t = dict(channels=C, depth=depth, input_size=n, gain=float(radial.log_gain.exp()),
             nl_beta=10.0, nl_threshold=20.0, k_cut=K_CUT)
    with torch.no_grad():
        t['wb'] = 100 * d(base._get_cached_weight((n, n), base.nl_sigma, torch.device('cpu'), torch.float64)[0, 0])
        idx = torch.arange(n * n, dtype=torch.float64).reshape(1, 1, n, n)
        t['perm'] = d(d4_orbit_stack(idx)[:, 0, 0].reshape(8, -1))
        res = {blk.residual_scale for blk in base.blocks}
        assert len(res) == 1
        t['res'] = float(res.pop())
        for l, blk in enumerate(base.blocks):
            norm = blk.norm
            assert isinstance(norm, WindowedChannelNorm2d) and norm.share_orbit
            assert isinstance(blk.act, torch.nn.GELU) and blk.act.approximate == 'none'
            assert blk.conv.kernel_size == (3, 3) and (blk.skip is not None) == (l == 0)
            W, b = blk.conv.weight, blk.conv.bias
            # Channel centring of the norm folded into the conv: W - mean_out W.
            t[f'W{l}'] = d((W - W.mean(0, keepdim=True)).permute(2, 3, 1, 0))
            t[f'b{l}'] = d(b - b.mean())
            t[f'g{l}'], t[f'be{l}'], t[f'eps{l}'] = d(norm.weight.flatten()), d(norm.bias.flatten()), norm.eps
            ho = n - 2 * (l + 1)
            t[f'wn{l}'] = d(norm._window((ho, ho), torch.device('cpu'), torch.float64))
        t['skip'] = d(base.blocks[0].skip.weight[:, 0, 0, 0])
        nf = n - 2 * depth
        t['K'] = d(base._fixed_kernel((nf, nf), torch.device('cpu'), torch.float64))
        t['signs_e1'], t['signs_e2'] = d(base.signs_e1), d(base.signs_e2)
        ro = base.scalar_readout
        assert isinstance(ro[1], torch.nn.GELU) and ro[1].approximate == 'none' and len(ro) == 3
        t['readout_hidden'] = ro[0].out_features
        t['R1W'], t['R1b'] = d(ro[0].weight), d(ro[0].bias)
        t['R2W'], t['R2b'] = d(ro[2].weight), d(ro[2].bias)
        t['scalar_scale'] = d(base.scalar_scale)
    return {k: np.asarray(v, dtype=np.float64) for k, v in t.items()}


# ------------------------------------------------------------------------------ build
def build():
    import os, platform, shlex, subprocess, sysconfig
    import pybind11
    out = HERE / ('_f8notanh' + sysconfig.get_config_var('EXT_SUFFIX'))
    arch = '-mcpu=native' if platform.machine() in ('aarch64', 'arm64') else '-march=native'
    cmd = (shlex.split(os.environ.get('CPU_CXX', 'g++')) +
           ['-std=c++17', '-O3', arch, '-ffp-contract=fast', '-fno-math-errno', '-fPIC', '-shared',
            '-fvisibility=hidden', '-pthread', '-I' + pybind11.get_include(), '-I' + sysconfig.get_paths()['include']] +
           shlex.split(os.environ.get('F8_FLAGS', '')) +
           [str(HERE / 'native' / 'f8notanh.cpp'), '-o', str(out) + '.tmp'])
    subprocess.run(cmd, check=True)
    os.replace(str(out) + '.tmp', out)  # never truncate a mapped extension
    print('Built', out)


# -------------------------------------------------------------------------------- API
class F8NoTanhCPU:
    def __init__(self, tables, precision='float32', conv='winograd'):
        """conv: 'winograd' (F(2x2,3x3) for the 16->16 layers, default) or 'direct'."""
        from . import _f8notanh
        self.tables = dict(tables)
        self.precision, self.conv = precision, conv
        self._model = _f8notanh.create({k: np.asarray(v) for k, v in self.tables.items()}, precision, conv)

    @classmethod
    def load(cls, path, precision='float32', conv='winograd'):
        path = Path(path)
        if path.suffix == '.npz':
            with np.load(path, allow_pickle=False) as z:
                tables = {k: z[k] for k in z.files}
        else:
            tables = export_tables(path)
        return cls(tables, precision, conv)

    def measure(self, q, n_tangent=2, nthreads=1):
        """q [N,C,64,64] float32/float64. Returns e [N,2], R [N,2,n_tangent] (model precision)."""
        q = np.asarray(q)
        q = np.ascontiguousarray(q) if q.dtype == np.float32 else np.ascontiguousarray(q, dtype=np.float64)
        e, R = self._model.measure(q, n_tangent, nthreads)
        return e.astype(self.precision), R.astype(self.precision)


# ---------------------------------------------------------------------------- testing
def sample_q(n=64):
    """Real 5-channel blended training q stamps (field 0), training renoise."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from src.train_d4 import training_q
    return training_q(0, rows=np.arange(n))[1].astype(np.float32)


def _torch_reference(model, q, dtype, T=4):
    """Reverse-mode autograd e and R contracted with T tangents."""
    import torch
    x = torch.tensor(q[:, :1], dtype=dtype, requires_grad=True)
    e = model(x)
    grads = [torch.autograd.grad(e[:, i].sum(), x, retain_graph=True)[0] for i in range(2)]
    tangents = torch.tensor(q[:, 1:1 + T], dtype=dtype)
    R = torch.stack([(g * tangents).sum((-2, -1)) for g in grads], 1)
    return e.detach().numpy(), R.detach().numpy()


def validate(checkpoints, n=64):
    """Compare e and R (all 4 AnaCal tangents) with float64 torch autograd."""
    import torch
    q = sample_q(n)
    torch.set_num_threads(4)
    rel = lambda a, b: float(np.abs(a - b).max() / np.abs(b).max())
    mean = lambda a, b: float(np.abs((a.astype(np.float64) - b).reshape(len(b), -1).mean(0)).max())
    for ck in checkpoints:
        e_ref, R_ref = _torch_reference(_load_torch(ck, torch.float64), q, torch.float64)
        e32, R32 = _torch_reference(_load_torch(ck, torch.float32), q, torch.float32)
        tables = export_tables(ck)
        print(Path(ck).name, '(max-abs error / max-abs value vs torch float64; max |component mean bias|)')
        print(f'  torch float32      : e {rel(e32, e_ref):.2e}  R {rel(R32, R_ref):.2e}'
              f'   mean e {mean(e32, e_ref):.1e} R {mean(R32, R_ref):.1e}')
        for precision, conv in (('float64', 'winograd'), ('float32', 'winograd'), ('float32', 'direct')):
            model = F8NoTanhCPU(tables, precision, conv)
            e, R = model.measure(q, 4)
            e2, R2 = model.measure(q, 2, nthreads=3)
            e0, _ = model.measure(q, 0)
            assert np.array_equal(e, e2) and np.array_equal(R[..., :2], R2), 'T/threads changed results'
            # T=0 uses the plain erf GELU; T>0 shares exp between Phi and phi (float32 only).
            assert rel(e0, e) < 1e-5, 'e-only path disagrees'
            print(f'  cpu {precision:7s} {conv:8s} T=4: e {rel(e, e_ref):.2e}  R {rel(R, R_ref):.2e}'
                  f'   mean e {mean(e, e_ref):.1e} R {mean(R, R_ref):.1e}')


def bench(checkpoints, n=256, repeat=5, torch_n=64):
    """Single-core throughput in gal/(CPU s) for the native kernel and the PyTorch
    float32 CPU baseline (1 intra-op thread, batched autograd, as cpu_src.reference)."""
    import os, time
    q = sample_q(min(n, 64))
    q = np.concatenate([q] * -(-n // len(q)))[:n]
    allowed = os.sched_getaffinity(0)
    core = min(allowed)
    for ck in checkpoints:
        tables = export_tables(ck)
        os.sched_setaffinity(0, {core})
        for conv in ('winograd', 'direct'):
            model = F8NoTanhCPU(tables, 'float32', conv)
            for T in (2, 4, 0):
                model.measure(q[:8], T)
                best = min(_timed(lambda: model.measure(q, T, nthreads=1), time) for _ in range(repeat))
                print(f'  native float32 {conv:8s} T={T}: {n / best:8.1f} gal/(CPU s)  ({1e3 * best / n:.3f} ms/gal)')
        import torch
        torch.set_num_threads(1)
        tm = _load_torch(ck, torch.float32)
        qt = q[:torch_n]
        for label, fn in (('autograd e+R(T=2)', lambda: _torch_reference(tm, qt, torch.float32, 2)),
                          ('forward e only', lambda: _torch_e(tm, qt))):
            fn()
            best = min(_timed(fn, time) for _ in range(3))
            print(f'  torch  float32 {label:20s}: {len(qt) / best:8.1f} gal/(CPU s)  ({1e3 * best / len(qt):.3f} ms/gal)')
        os.sched_setaffinity(0, allowed)


def _torch_e(model, q):
    import torch
    with torch.no_grad():
        return model(torch.tensor(q[:, :1]))


def _timed(fn, time):
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0


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
    else:
        raise SystemExit(__doc__)
