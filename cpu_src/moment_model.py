"""Fixed-moment shear estimator measured directly in AnaCal Fourier space.

NumPy port of score_research_20261029/model.py (CrossScaleScore, 1,080 frozen
weights). The 30 Gaussian moments (5 scales x F,Q1,Q2,dx,dy,T) are built from
AnaCal shapelets and applied to the PSF-deconvolved stamp spectrum, so no
q-image is formed. Shear/position tangents are the moments of AnaCal's
g1,g2,x1,x2 channels, obtained from the adjoint of ImageQ's operators; the
head propagates them in forward mode. Version-2 exports default to a fourth-order
inverse pixel-average correction of the final nonlinear outputs and responses,
assuming uniform independent subpixel phases. This needs no true centroid but
returns a population response, not an individual corrected-output derivative.
Runtime never imports torch.

Kernels stay centred on pixel 31.5 (half a pixel from AnaCal's centre 32),
exactly as the weights were fitted. Corrected measurement aligns the raw cutout's
mean centre 32 to 31.5 before applying the offset quadrature. Legacy v1 archives
retain the original convention unless explicitly overridden. Use float64.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from .api import Measurement

FORMAT = "moment_cross_scale"
FORMAT_VERSION = 2
STAMP = 64
KERNEL_CENTER = (STAMP - 1) / 2
SCALES = (2.0, 3.0, 4.0, 6.0, 8.0)
FLOORS = (30.0, 100.0, 300.0)
RADIAL_WIDTHS = (None, 0.2, 0.4, 0.8)
# Pair scales 2->3, 3->4, 4->6, 6->8, 8->6 pixels.
REFERENCE_SCALES = (1, 2, 3, 4, 3)
BASE_SHAPE = (5, 3, 4, 2, 2, 3)
CROSS_SHAPE = (5, 3, 4, 2, 3)
PREPROCESSING = {"pixel_scale": 0.2, "sigma_arcsec": 0.85 / 2.355, "flim": 10.0, "klim": 50.0}
CHANNELS = ("v", "g1", "g2", "x1", "x2")
MIN_DECONV_RATIO = 1e-5  # anacal/image.h min_deconv_ratio


def fixed_filters(scales=SCALES, n=STAMP):
    """[scale, moment, y, x] pixel kernels; moments F, Q1, Q2, dx, dy, T."""
    y, x = np.mgrid[:n, :n].astype(float)
    x -= (n - 1) / 2
    y -= (n - 1) / 2
    kernels = []
    for s in scales:
        g = np.exp(-(x * x + y * y) / (2 * s * s))
        kernels.append([g, g * (x * x - y * y) / (2 * s * s), g * x * y / (s * s),
                        g * x / s, g * y / s, g * (x * x + y * y) / (2 * s * s)])
    return np.array(kernels)


def _frequencies(n=STAMP):
    """AnaCal half-plane grid in rad/pixel: ky row (Nyquist negative), kx column."""
    ky = 2 * np.pi * np.fft.fftfreq(n)[:, None]
    kx = 2 * np.pi * np.fft.rfftfreq(n)[None, :]
    return ky, kx


def shapelet_spectra(scales=SCALES, n=STAMP):
    """Kernel spectra from AnaCal shapelets; returns (spectra, truncation residual).

    With A=2*pi*s^2*n^2 and the 31.5 phase, F=A chi00, Q1+iQ2=A chi22/sqrt2,
    dx=iA Im chi11, dy=-iA Re chi11, T=A(chi00+chi20)/2. These are the continuous
    transforms; the residual (rfft2 of the pixel kernel minus the shapelet term)
    restores the 64x64 truncation and aliasing exactly. It is ~1e-13 for s=3,4
    and reaches ~3e-3 of the kernel for s=8.
    """
    from anacal.fpfs import shapelets2d_func
    ky, kx = _frequencies(n)
    phase = np.exp(-1j * (kx + ky) * (n - 1) / 2)
    spectra = []
    for s in scales:
        chi = shapelets2d_func(n, 2, 1.0 / s, 1e3) * (2 * np.pi * s * s * n * n)
        c00, c20, c11, c22 = chi[0], chi[6], chi[4], chi[8]  # index n*3+m
        spectra.append(np.stack([c00, c22.real / np.sqrt(2), c22.imag / np.sqrt(2),
                                 1j * c11.imag, -1j * c11.real, (c00 + c20) / 2]) * phase)
    spectra = np.array(spectra)
    residual = np.fft.rfft2(fixed_filters(scales, n)) - spectra
    return spectra + residual, residual


def tangent_filters(pixel_scale, sigma_arcsec, n=STAMP):
    """Real [2*n*(n/2+1), 5*30] matrix: moments of AnaCal's v,g1,g2,x1,x2 channels.

    AnaCal: g1 = IFFT[(kx^2-ky^2) s0^2 c] + X x1 - Y x2, g2 = IFFT[2kxky s0^2 c]
    + Y x1 + X x2, x_a = IFFT[-i k_a c], X = i - n/2. A pixel moment sum(K q) of
    c2r output equals sum_k w Re(conj(rfft2 K) Qhat)/n^2, w=1 on kx=0,Nyquist.
    """
    ky, kx = _frequencies(n)
    s0 = sigma_arcsec / pixel_scale
    kernels = fixed_filters(SCALES, n).reshape(-1, n, n)
    spectra, residual = shapelet_spectra(SCALES, n)
    k = np.conj(spectra.reshape(-1, n, n // 2 + 1))
    X = np.arange(n)[None, :] - n // 2
    Y = np.arange(n)[:, None] - n // 2
    kX = np.conj(np.fft.rfft2(kernels * X))
    kY = np.conj(np.fft.rfft2(kernels * Y))
    dx, dy = -1j * kx, -1j * ky
    filters = [
        k + 0j * kx,
        k * (kx * kx - ky * ky) * s0 * s0 + kX * dx - kY * dy,
        k * 2 * kx * ky * s0 * s0 + kY * dx + kX * dy,
        k * dx / pixel_scale,  # AnaCal x1/x2 use k in 1/arcsec
        k * dy / pixel_scale,
    ]
    weight = np.full(n // 2 + 1, 2.0)
    weight[[0, -1]] = 1.0
    stack = np.stack(filters) * weight / (n * n)  # [5, 30, ky, kx]
    stack = stack.reshape(5 * len(kernels), -1).T
    return np.concatenate([stack.real, -stack.imag]), residual


def _place(array, xcen, ycen, n, ishift):
    """Image::set_r: n x n window at (xcen,ycen), zero fill; ishift moves it to the origin."""
    out = np.zeros(array.shape[:-2] + (n, n))
    ny, nx = array.shape[-2:]
    y0, x0 = ycen - n // 2, xcen - n // 2
    ys, xs = max(y0, 0), max(x0, 0)
    ye, xe = min(y0 + n, ny), min(x0 + n, nx)
    out[..., ys - y0:ye - y0, xs - x0:xe - x0] = array[..., ys:ye, xs:xe]
    return np.roll(out, (n // 2, n // 2), (-2, -1)) if ishift else out


def _rotate90(f):
    """Image::_rotate90_f(flip=1) on [..., n, n/2+1] half-plane spectra."""
    n = f.shape[-2]
    h = n // 2
    out = np.empty_like(f)
    r = np.arange(h)
    i = np.arange(h + 1)
    out[..., r[:, None], i[None, :]] = f[..., (-i[None, :]) % n, r[:, None]]
    r = np.arange(h, n)
    i = np.arange(h)
    out[..., r[:, None], i[None, :]] = np.conj(f[..., i[None, :], (n - r)[:, None]])
    out[..., r, h] = np.conj(f[..., h, n - r])
    return out


def lens_kernel(psf, pixel_scale, sigma_arcsec, klim, n=STAMP):
    """ImageQ::get_lens_kernel_vec: Gaussian / PSF with AnaCal's clamp, [..., n, n/2+1]."""
    psf = np.asarray(psf, dtype=np.float64)
    p = np.fft.rfft2(_place(psf, psf.shape[-1] // 2, psf.shape[-2] // 2, n, True))
    ky, kx = _frequencies(n)
    k2 = kx * kx + ky * ky
    s0 = sigma_arcsec / pixel_scale
    gauss = np.exp(-k2 * s0 * s0 / 2) * (k2 <= (klim * pixel_scale) ** 2)
    floor = MIN_DECONV_RATIO * p[..., :1, :1].real
    return gauss / np.where(np.abs(p) < floor, floor, p)


def _jet_head(m, dm, base, cross):
    """Vectorized CrossScaleScore head with forward-mode tangents.

    m: [N,S,6] moments, dm: [N,T,S,6] tangents. Returns y [N,2], dy [N,T,2].
    Axes: s scale, l floor, w radial window, p gate power, j size power, k norm.
    """
    c2 = np.square(FLOORS)
    lift = lambda a: a[:, None]
    F, q1, q2, x, y, T = (m[..., i, None] for i in range(6))
    dF, dq1, dq2, dx, dy, dT = (dm[..., i, None] for i in range(6))
    fn2 = F * F + c2
    fn = np.sqrt(fn2)
    dfn = lift(F / fn) * dF
    cf = F / fn2
    dcf = lift((c2 - F * F) / (fn2 * fn2)) * dF
    A, dA = x * x - y * y, 2 * (lift(x) * dx - lift(y) * dy)
    B, dB = x * y, dx * lift(y) + lift(x) * dy
    C, dC = x * x + y * y, 2 * (lift(x) * dx + lift(y) * dy)
    Q = np.stack([q1 - 0.5 * A * cf, q2 - B * cf], -1)
    dQ = np.stack([dq1 - 0.5 * (dA * lift(cf) + lift(A) * dcf),
                   dq2 - (dB * lift(cf) + lift(B) * dcf)], -1)
    Tc = T - 0.5 * C * cf
    dTc = dT - 0.5 * (dC * lift(cf) + lift(C) * dcf)
    tn = np.sqrt(Tc * Tc + c2)
    mn = np.sqrt(F * F + Tc * Tc + c2)
    dtn = lift(Tc / tn) * dTc
    dmn = (lift(F) * dF + lift(Tc) * dTc) / lift(mn)
    g, dg = F / fn, dF / lift(fn) - lift(F / fn2) * dfn
    s, ds = Tc / mn, dTc / lift(mn) - lift(Tc / (mn * mn)) * dmn
    norm = np.stack([fn, tn, mn], -1)  # [N,S,L,K]
    dnorm = np.stack([dfn, dtn, dmn], -1)
    e = Q[..., None, :] / norm[..., None]  # [N,S,L,K,2]
    de = dQ[..., None, :] / lift(norm)[..., None] - lift(e / norm[..., None]) * dnorm[..., None]
    r2 = np.square(e).sum(-1)
    dr2 = 2 * (lift(e) * de).sum(-1)
    rate = np.array([0.0] + [1 / (2 * w * w) for w in RADIAL_WIDTHS[1:]])[:, None]
    win = np.exp(-r2[..., None, :] * rate)  # [N,S,L,W,K]
    dwin = -lift(win * rate) * dr2[..., None, :]
    G = np.stack([np.ones_like(g), g], -1)  # [N,S,L,P]
    dG = np.stack([np.zeros_like(dg), dg], -1)
    basis = np.stack([G, G * s[..., None]], -1)  # [N,S,L,P,J]
    dbasis = np.stack([dG, dG * lift(s)[..., None] + lift(G) * ds[..., None]], -1)
    refs = list(REFERENCE_SCALES)
    area = np.square(np.array(SCALES) / np.array(SCALES)[refs])
    Fs, dFs = m[..., 0], dm[..., 0]
    Fr, dFr = Fs[:, refs] * area, dFs[..., refs] * area
    D = np.sqrt((Fs * Fs + Fr * Fr)[..., None] + c2)  # [N,S,L]
    conc = (Fs - Fr)[..., None] / D
    dconc = ((dFs - dFr)[..., None] - lift(conc / D) * (lift(Fs) * dFs + lift(Fr) * dFr)[..., None]) / lift(D)
    gated = np.einsum("nslp,slwpk->nslwk", G, cross)
    amp = np.einsum("nslpj,slwpjk->nslwk", basis, base) + conc[..., None, None] * gated
    damp = (np.einsum("ntslpj,slwpjk->ntslwk", dbasis, base) + dconc[..., None, None] * lift(gated)
            + lift(conc)[..., None, None] * np.einsum("ntslp,slwpk->ntslwk", dG, cross))
    out = np.einsum("nslwk,nslkc->nc", amp * win, e)
    dout = (np.einsum("ntslwk,nslkc->ntc", damp * lift(win) + lift(amp) * dwin, e)
            + np.einsum("nslwk,ntslkc->ntc", amp * win, de))
    return out, dout


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def pixel_average_stencil(order=4):
    """Inverse uniform-pixel average through the requested even Taylor order.

    For a uniform displacement u in [-1/2, 1/2], averaging is the operator
    sinh(D/2)/(D/2). Its inverse starts 1 - D^2/24 + 7 D^4/5760.
    Tensor-product shifts act on the final nonlinear estimator AND response;
    this is a population correction, not an estimate of an object's centroid.
    """
    if order == 0:
        return [(0., 0., 1.)]
    if order not in (4, 6):
        raise ValueError("pixel_average_order must be None (legacy), 0 (aligned), 4 or 6")
    nodes = np.arange(-order // 2, order // 2 + 1, dtype=float) / 2
    moments = np.array([1., 0., -1/12, 0., 7/240, 0., -31/1344])[:order + 1]
    weights = np.linalg.solve(np.vander(nodes, order + 1, increasing=True).T, moments)
    return [(x, y, wx * wy) for x, wx in zip(nodes, weights)
            for y, wy in zip(nodes, weights)]


class MomentModel:
    """Same measure_* interface as api.Model, returning api.Measurement."""

    precision = "float64"
    isa = "numpy"

    def __init__(self, metadata, base_weights, cross_weights):
        if metadata.get("format") != FORMAT or metadata.get("format_version") not in (1, FORMAT_VERSION):
            raise ValueError("Not a moment_cross_scale model file")
        if metadata.get("stamp_size") != STAMP or metadata.get("kernel_center") != KERNEL_CENTER:
            raise ValueError("Unsupported stamp size or kernel centre")
        self.base = np.ascontiguousarray(base_weights, dtype=np.float64)
        self.cross = np.ascontiguousarray(cross_weights, dtype=np.float64)
        if self.base.shape != BASE_SHAPE or self.cross.shape != CROSS_SHAPE:
            raise ValueError("Unexpected weight shapes")
        self.metadata = json.loads(json.dumps(metadata))
        p = self.metadata["preprocessing"]
        self._filters, self.shapelet_residual = tangent_filters(p["pixel_scale"], p["sigma_arcsec"])
        self._pixel_filters = fixed_filters().reshape(30, -1).T
        self.pixel_average_order = metadata.get("pixel_average_order")
        self._stencil = (None if self.pixel_average_order is None
                         else pixel_average_stencil(self.pixel_average_order))
        # Raw cutouts are centred at 32; align their mean centre to the fixed
        # head's 31.5, and use 31.5 for the shear generator as well.
        self._aligned_filters = self._filters.copy()
        self._aligned_filters[:, 30:60] += .5 * p["pixel_scale"] * (
            self._filters[:, 90:120] - self._filters[:, 120:150])
        self._aligned_filters[:, 60:90] += .5 * p["pixel_scale"] * (
            self._filters[:, 90:120] + self._filters[:, 120:150])

    @classmethod
    def load(cls, path, *, pixel_average_order="metadata"):
        with np.load(path, allow_pickle=False) as archive:
            meta = json.loads(archive["metadata"].tobytes().decode("utf-8"))
            if pixel_average_order != "metadata":
                meta["pixel_average_order"] = pixel_average_order
            return cls(meta, archive["base_weights"], archive["cross_weights"])

    def _metadata(self, n_tangent):
        return {"model": FORMAT, "precision": self.precision, "isa": self.isa, "n_tangent": n_tangent,
                "reference": "rounded_stamp_center", "position_derivative_units": "arcsec",
                "pixel_average_order": self.pixel_average_order,
                "offset_prior": (None if self._stencil is None else
                                 "fixed at rounded centre" if self.pixel_average_order == 0 else
                                 "independent uniform [-0.5,0.5] pixels"),
                "response_kind": "population_pixel_average" if self.pixel_average_order in (4, 6) else "per_image",
                "checkpoint_sha256": self.metadata.get("checkpoint_sha256")}

    def measure_jets(self, jets, n_tangent=2):
        """jets [N,1+T,5,6] moments of v and tangent channels -> Measurement."""
        jets = np.asarray(jets, dtype=np.float64)
        y, dy = _jet_head(jets[:, 0], jets[:, 1:1 + n_tangent], self.base, self.cross)
        return Measurement(y, dy[:, :2].transpose(0, 2, 1).copy(),
                           dy[:, 2:4].transpose(0, 2, 1).copy() if n_tangent == 4 else None,
                           self._metadata(n_tangent))

    def fourier_jets(self, stamps, psf, noise=None, n_tangent=2, nthreads=1):
        """Moments of AnaCal's q channels straight from [N,64,64] stamps (source at 32).

        psf is one [h,w] stamp or a per-object [N,h,w] stack, centred at h//2, w//2.
        """
        import scipy.fft
        p = self.metadata["preprocessing"]
        stamps = np.asarray(stamps, dtype=np.float64)
        kernel = lens_kernel(psf, p["pixel_scale"], p["sigma_arcsec"], p["klim"])
        c = scipy.fft.rfft2(stamps, workers=nthreads) * kernel
        ncol = 30 * (1 + n_tangent)
        flat = lambda z: np.concatenate([z.real.reshape(len(z), -1), z.imag.reshape(len(z), -1)], 1)
        if noise is None:
            jets = flat(c) @ self._filters[:, :ncol]
        else:
            cn = scipy.fft.rfft2(np.asarray(noise, dtype=np.float64), workers=nthreads) * _rotate90(kernel)
            jets = np.concatenate([flat(c + cn) @ self._filters[:, :30],
                                   flat(c - cn) @ self._filters[:, 30:ncol]], 1)
        return jets.reshape(len(stamps), 1 + n_tangent, 5, 6)

    def qstamp_jets(self, q):
        """Pixel-space moments of existing q-stamps [N,C,64,64] (C>=3)."""
        q = np.asarray(q, dtype=np.float64)
        return (q.reshape(q.shape[0], q.shape[1], -1) @ self._pixel_filters).reshape(len(q), -1, 5, 6)

    @staticmethod
    def _check(n_tangent, nthreads, batch_size):
        if n_tangent not in (2, 4) or type(nthreads) is not int or nthreads < 1:
            raise ValueError("n_tangent must be 2 or 4; nthreads must be a positive integer")
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")

    def _batched(self, n, batch_size, n_tangent, jets_for):
        parts = [self.measure_jets(jets_for(slice(i, i + batch_size)), n_tangent)
                 for i in range(0, n, batch_size)]
        if not parts:
            parts = [self.measure_jets(np.empty((0, 1 + n_tangent, 5, 6)), n_tangent)]
        return Measurement(np.concatenate([r.e for r in parts]), np.concatenate([r.R for r in parts]),
                           np.concatenate([r.dE_dx for r in parts]) if n_tangent == 4 else None,
                           self._metadata(n_tangent))

    def measure_stamps(self, stamps, psf, noise=None, n_tangent=2, nthreads=1, batch_size=256):
        """Isolated [N,64,64] cutouts with the source at pixel 32, as anacal_pix_r.

        Stamps and noise are rounded to float32 like AnaCal's pixel_t; the PSF stays float64.
        With pixel_average_order=4/6, the mean input centre is 32, internally
        aligned to kernel centre 31.5. R is a population response under an
        independent uniform [-0.5,0.5] pixel offset in each axis, not the
        derivative of this corrected output for an individual arbitrary offset.
        No truth position is estimated or accepted by this method.
        """
        self._check(n_tangent, nthreads, batch_size)
        stamps = np.asarray(stamps, dtype=np.float32)
        psf = np.asarray(psf, dtype=np.float64)
        if stamps.ndim != 3 or stamps.shape[1:] != (STAMP, STAMP) or not np.isfinite(stamps).all():
            raise ValueError(f"stamps must be finite [N,{STAMP},{STAMP}]")
        if psf.ndim not in (2, 3) or (psf.ndim == 3 and len(psf) != len(stamps)) or not np.isfinite(psf).all():
            raise ValueError("psf must be finite [h,w] or [N,h,w]")
        if noise is not None:
            noise = np.asarray(noise, dtype=np.float32)
            if noise.shape != stamps.shape or not np.isfinite(noise).all():
                raise ValueError("noise must be finite and have the stamps' shape")
        if self._stencil is not None:
            parts = [self._pixel_average_stamps(
                stamps[i:i + batch_size], psf[i:i + batch_size] if psf.ndim == 3 else psf,
                None if noise is None else noise[i:i + batch_size], n_tangent, nthreads)
                for i in range(0, len(stamps), batch_size)]
            if not parts:
                return self.measure_jets(np.empty((0, 1 + n_tangent, 5, 6)), n_tangent)
            return Measurement(np.concatenate([r.e for r in parts]), np.concatenate([r.R for r in parts]),
                               np.concatenate([r.dE_dx for r in parts]) if n_tangent == 4 else None,
                               self._metadata(n_tangent))
        return self._batched(len(stamps), batch_size, n_tangent, lambda b: self.fourier_jets(
            stamps[b], psf[b] if psf.ndim == 3 else psf,
            None if noise is None else noise[b], n_tangent, nthreads))

    def _pixel_average_stamps(self, stamps, psf, noise, n_tangent, nthreads):
        """Correct the output and response jointly for a uniform pixel phase."""
        import scipy.fft
        p = self.metadata["preprocessing"]
        kernel = lens_kernel(psf, p["pixel_scale"], p["sigma_arcsec"], p["klim"])
        signal = scipy.fft.rfft2(np.asarray(stamps, dtype=np.float64), workers=nthreads) * kernel
        extra = (0 if noise is None else scipy.fft.rfft2(np.asarray(noise, dtype=np.float64), workers=nthreads) * _rotate90(kernel))
        plus, minus = signal + extra, signal - extra
        return self._pixel_average_spectra(plus, minus, n_tangent)

    def _pixel_average_spectra(self, plus, minus, n_tangent):
        """Shared nonlinear quadrature for raw-image and AnaCal-q entry points."""
        count = len(plus)
        ky, kx = _frequencies(STAMP)
        filters = self._aligned_filters
        flatten = lambda z: np.concatenate([z.real.reshape(len(z), -1), z.imag.reshape(len(z), -1)], axis=1)
        jet_blocks = []
        weights = []
        for x, y, weight in self._stencil:
            phase = np.exp(-1j * ((x - .5) * kx + (y - .5) * ky))
            moments = flatten(plus * phase) @ filters[:, :30]
            tangents = flatten(minus * phase) @ filters[:, 30:30 * (1 + n_tangent)]
            jet_blocks.append(np.concatenate([moments, tangents], axis=1).reshape(count, 1 + n_tangent, 5, 6))
            weights.append(weight)
        result = self.measure_jets(np.concatenate(jet_blocks), n_tangent)
        weighted = lambda values: np.einsum("s,sn...->n...", weights,
            values.reshape(len(weights), count, *values.shape[1:]))
        return Measurement(weighted(result.e), weighted(result.R),
                           weighted(result.dE_dx) if n_tangent == 4 else None,
                           self._metadata(n_tangent))

    def measure_sources(self, image, psf, positions_xy, noise=None, n_tangent=2, nthreads=1, batch_size=256):
        """Stamps cut at floor(position+0.5); float32 at the AnaCal boundary like api.Model."""
        self._check(n_tangent, nthreads, batch_size)
        image = np.asarray(image, dtype=np.float32)
        pos = np.asarray(positions_xy, dtype=np.float64)
        if image.ndim != 2 or not np.isfinite(image).all():
            raise ValueError("image must be a finite 2D array")
        if pos.ndim != 2 or pos.shape[1] != 2 or not np.isfinite(pos).all():
            raise ValueError("positions_xy must be finite [N,2]")
        if noise is not None:
            noise = np.asarray(noise, dtype=np.float32)
            if noise.shape != image.shape or not np.isfinite(noise).all():
                raise ValueError("noise must be finite and have image shape")
        centers = np.floor(pos + 0.5).astype(np.int64)
        h = STAMP // 2
        inside = ((centers >= h) & (centers + h <= np.array(image.shape[::-1]))).all(1)
        if not inside.all():
            raise ValueError("Sources whose stamp crosses the image edge are not supported")
        cut = lambda a, b: np.stack([a[y - h:y + h, x - h:x + h] for x, y in centers[b]]).reshape(-1, STAMP, STAMP)
        if self._stencil is not None:
            # Reuse the stamp API so catalogue and isolated measurements apply
            # exactly the same population correction.
            result = self.measure_stamps(cut(image, slice(None)), psf,
                None if noise is None else cut(noise, slice(None)),
                n_tangent, nthreads, batch_size)
        else:
            result = self._batched(len(pos), batch_size, n_tangent, lambda b: self.fourier_jets(
                cut(image, b), psf, None if noise is None else cut(noise, b), n_tangent, nthreads))
        return Measurement(result.e, result.R, result.dE_dx, result.metadata, centers)

    def measure_qstamps(self, q, n_tangent=2, nthreads=1, batch_size=256):
        """Measure native-centred AnaCal q images, before any half-pixel shift.

        Corrected modes require all five channels to retain signal-minus-noise
        position information. Legacy mode also accepts three channels with T=2.
        """
        self._check(n_tangent, nthreads, batch_size)
        q = np.asarray(q)
        if q.ndim != 4 or q.shape[1] < 1 + n_tangent or q.shape[2:] != (STAMP, STAMP):
            raise ValueError(f"q must have shape [N,{1 + n_tangent}+,{STAMP},{STAMP}]")
        if self._stencil is not None:
            if q.shape[1] < 5:
                raise ValueError("Pixel-average correction requires all five AnaCal q channels, including x1/x2")
            import scipy.fft
            ky, kx = _frequencies(STAMP)
            scale = self.metadata["preprocessing"]["pixel_scale"]
            # Recover the signal-minus-renoise spectrum from position channels.
            # DC is unconstrained, but contributes zero shear/position response.
            # Nyquist-axis derivatives are zero for a real sampled image.
            dx = np.broadcast_to(-1j * kx / scale, (STAMP, STAMP // 2 + 1)).copy()
            dy = np.broadcast_to(-1j * ky / scale, dx.shape).copy()
            dx[:, -1] = 0
            dy[STAMP // 2, :] = 0
            norm = np.abs(dx)**2 + np.abs(dy)**2
            parts = []
            for first in range(0, len(q), batch_size):
                fq = scipy.fft.rfft2(np.asarray(q[first:first + batch_size, [0, 3, 4]], dtype=np.float64), workers=nthreads)
                minus = np.divide(np.conj(dx) * fq[:, 1] + np.conj(dy) * fq[:, 2], norm,
                                  out=np.zeros_like(fq[:, 0]), where=norm > 0)
                parts.append(self._pixel_average_spectra(fq[:, 0], minus, n_tangent))
            if not parts:
                return self.measure_jets(np.empty((0, 1 + n_tangent, 5, 6)), n_tangent)
            return Measurement(np.concatenate([r.e for r in parts]), np.concatenate([r.R for r in parts]),
                               np.concatenate([r.dE_dx for r in parts]) if n_tangent == 4 else None,
                               self._metadata(n_tangent))
        return self._batched(len(q), batch_size, n_tangent,
                             lambda b: self.qstamp_jets(q[b, :1 + n_tangent]))


def export_checkpoint(checkpoint, output, *, pixel_average_order=4):
    """Export frozen weights with an explicit measurement convention (needs torch).

    The default assumes independent uniform subpixel phases in each axis.
    Use order=0 for a fixed, known pixel phase at the rounded cutout centre;
    use None only to reproduce the historical centre mismatch.
    """
    import torch
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    weights = {k: v.numpy().astype(np.float64) for k, v in state["state_dict"].items()}
    if np.abs(weights["base.filters"] - fixed_filters()).max() > 1e-12:
        raise ValueError("Checkpoint filters differ from the fixed moment kernels")
    meta = {"format": FORMAT, "format_version": FORMAT_VERSION, "stamp_size": STAMP,
            "kernel_center": KERNEL_CENTER, "preprocessing": PREPROCESSING,
            "pixel_average_order": pixel_average_order,
            "checkpoint": str(Path(checkpoint).resolve()), "checkpoint_sha256": _sha256(checkpoint),
            "noise_level": state.get("noise_level"), "frozen_utc": state.get("frozen_utc"),
            "source_metadata": json.loads(json.dumps(state.get("metadata", {}), default=str))}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    encoded = np.frombuffer(json.dumps(meta, sort_keys=True).encode("utf-8"), dtype=np.uint8)
    np.savez(output, metadata=encoded, base_weights=weights["base.weights"].reshape(BASE_SHAPE),
             cross_weights=weights["cross_weights"].reshape(CROSS_SHAPE))
    return MomentModel.load(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    export = sub.add_parser("export", help="model.pt -> pickle-free npz")
    export.add_argument("--checkpoint", required=True)
    export.add_argument("--output", required=True)
    mode = export.add_mutually_exclusive_group()
    mode.add_argument("--pixel-average-order", type=int, choices=(0, 4, 6), default=4,
                      help="0: aligned fixed-phase input; 4/6: uniform-pixel population correction (default 4)")
    mode.add_argument("--legacy-center", action="store_true", help="Reproduce the original checkpoint measurement convention")
    args = parser.parse_args()
    model = export_checkpoint(args.checkpoint, args.output,
                              pixel_average_order=None if args.legacy_center else args.pixel_average_order)
    print(json.dumps({"output": args.output, "max_shapelet_residual_by_scale":
                      np.abs(model.shapelet_residual).max((1, 2, 3)).tolist()}, indent=2))


if __name__ == "__main__":
    main()
