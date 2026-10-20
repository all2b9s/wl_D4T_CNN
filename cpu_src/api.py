"""Public CPU API. Precision applies to every network operation, including heads."""
from dataclasses import dataclass, replace
import numpy as np
from .model_io import load_model, validate

@dataclass(frozen=True)
class Measurement:
    e: np.ndarray
    R: np.ndarray
    dE_dx: np.ndarray | None
    metadata: dict
    centers_xy: np.ndarray | None = None

    def shift_reference(self, delta_xy_arcsec):
        """Apply AnaCal's signed shift; delta is [dx1,dx2] in arcsec."""
        if self.dE_dx is None:
            raise ValueError("Reference shifts require T=4 position derivatives")
        delta = np.asarray(delta_xy_arcsec, dtype=self.R.dtype)
        if delta.shape not in ((2,), (len(self.e), 2)) or not np.isfinite(delta).all():
            raise ValueError("delta_xy_arcsec must be finite [2] or [N,2]")
        delta = np.broadcast_to(delta, (len(self.e), 2))
        x, y = self.dE_dx[..., 0], self.dE_dx[..., 1]
        r = self.R.copy()
        r[..., 0] += delta[:, 0, None] * x - delta[:, 1, None] * y
        r[..., 1] += delta[:, 0, None] * y + delta[:, 1, None] * x
        return replace(self, R=r, metadata={**self.metadata, "reference": "explicitly_shifted"})

class Model:
    def __init__(self, metadata, weights, precision="float32", isa="auto"):
        from . import _native
        if getattr(_native, "api_version", 0) != 3:
            raise RuntimeError("CPU native extension is out of date; rebuild cpu_src with build_local.sh using this Python interpreter")
        validate(metadata, weights)
        if precision not in ("float32", "float64"):
            raise ValueError("precision must be float32 or float64")
        # Own a metadata copy; callers cannot mutate the native configuration.
        import copy
        self.metadata = copy.deepcopy(metadata)
        self.precision = precision
        self.dtype = np.dtype(precision)
        self.isa = "scalar" if precision == "float64" else _native.resolve_isa(isa)
        if precision == "float64" and isa not in ("auto", "scalar"):
            raise ValueError("float64 validation currently uses the scalar kernel")
        native_type = _native.FloatModel if precision == "float32" else _native.DoubleModel
        self._model = native_type(metadata, weights, self.isa)

    @classmethod
    def load(cls, path, precision="float32", isa="auto"):
        return cls(*load_model(path), precision=precision, isa=isa)

    def _metadata(self, n_tangent):
        return {"precision": self.precision, "isa": self.isa, "n_tangent": n_tangent,
                "convolution_algorithm": self._model.convolution_algorithm,
                "reference": "rounded_stamp_center", "position_derivative_units": "arcsec",
                "checkpoint_sha256": self.metadata.get("checkpoint_sha256")}

    def measure_qstamps(self, q, n_tangent=2, nthreads=1):
        if n_tangent not in (2, 4) or type(nthreads) is not int or nthreads < 1:
            raise ValueError("n_tangent must be 2 or 4; nthreads must be a positive integer")
        q = np.asarray(q)
        h = self.metadata["stamp_size"]
        if q.ndim != 4 or q.shape[1:] != (5, h, h):
            raise ValueError(f"q must have shape [N,5,{h},{h}]")
        q = np.ascontiguousarray(q, dtype=self.dtype)
        if not np.isfinite(q).all():
            raise ValueError("q contains non-finite values")
        values = self._model.measure(q, n_tangent, nthreads)
        return Measurement(values[:, 0].copy(), values[:, 1:3].transpose(0, 2, 1).copy(),
                           values[:, 3:5].transpose(0, 2, 1).copy() if n_tangent == 4 else None,
                           self._metadata(n_tangent))

    def prepare_qstamps(self, image, psf, positions_xy, noise=None, nthreads=1):
        from . import _native
        if not _native.has_anacal:
            raise RuntimeError("Rebuild with -DANACAL_SOURCE_DIR=/path/to/AnaCal for source measurement")
        image = np.ascontiguousarray(image, dtype=np.float32)
        psf = np.ascontiguousarray(psf, dtype=np.float64)
        pos = np.ascontiguousarray(positions_xy, dtype=np.float64)
        if image.ndim != 2 or psf.ndim != 2 or not np.isfinite(image).all() or not np.isfinite(psf).all():
            raise ValueError("image and psf must be finite 2D arrays")
        if pos.ndim != 2 or pos.shape[1] != 2 or not np.isfinite(pos).all():
            raise ValueError("positions_xy must be finite [N,2]")
        if type(nthreads) is not int or nthreads < 1:
            raise ValueError("nthreads must be positive")
        if noise is not None:
            noise = np.ascontiguousarray(noise, dtype=np.float32)
            if noise.shape != image.shape or not np.isfinite(noise).all():
                raise ValueError("noise must be finite and have image shape")
        p = self.metadata["preprocessing"]
        return _native.prepare_qstamps(image, psf, pos, noise, self.metadata["stamp_size"],
                                      p["pixel_scale"], p["sigma_arcsec"], p["klim"], nthreads)

    def measure_sources(self, image, psf, positions_xy, noise=None, n_tangent=2, nthreads=1, batch_size=None):
        if type(nthreads) is not int or nthreads < 1:
            raise ValueError("nthreads must be a positive integer")
        if batch_size is None:
            batch_size = max(64, 8 * nthreads)
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be positive")
        pos = np.asarray(positions_xy, dtype=np.float64)
        if pos.ndim != 2 or pos.shape[1] != 2 or not np.isfinite(pos).all():
            raise ValueError("positions_xy must be finite [N,2]")
        # Prepare bounded batches instead of materializing an exposure's q-images.
        from . import _native
        if not _native.has_anacal:
            raise RuntimeError("Rebuild with ANACAL_SOURCE_DIR for source measurement")
        if n_tangent not in (2,4) or type(nthreads) is not int or nthreads < 1:
            raise ValueError("Invalid tangent count or nthreads")
        image = np.ascontiguousarray(image, dtype=np.float32)
        psf = np.ascontiguousarray(psf, dtype=np.float64)
        if image.ndim != 2 or psf.ndim != 2 or not np.isfinite(image).all() or not np.isfinite(psf).all():
            raise ValueError("image and psf must be finite 2D arrays")
        if noise is not None:
            noise = np.ascontiguousarray(noise,dtype=np.float32)
            if noise.shape != image.shape or not np.isfinite(noise).all():
                raise ValueError("noise must be finite and have image shape")
        pre=self.metadata["preprocessing"]
        results = []
        for start in range(0, len(pos), batch_size):
            q = _native.prepare_qstamps(image,psf,pos[start:start+batch_size],noise,self.metadata["stamp_size"],
                                       pre["pixel_scale"],pre["sigma_arcsec"],pre["klim"],nthreads)
            results.append(self.measure_qstamps(q, n_tangent, nthreads))
        if not results:
            q = self.prepare_qstamps(image, psf, pos, noise, nthreads)
            return replace(self.measure_qstamps(q, n_tangent, nthreads), centers_xy=np.empty((0, 2), dtype=np.int64))
        return Measurement(np.concatenate([r.e for r in results]), np.concatenate([r.R for r in results]),
                           np.concatenate([r.dE_dx for r in results]) if n_tangent == 4 else None,
                           self._metadata(n_tangent), np.floor(pos+.5).astype(np.int64))

    def trace(self, q, n_tangent=2):
        q = np.ascontiguousarray(q, dtype=self.dtype)
        if not np.isfinite(q).all():
            raise ValueError("q contains non-finite values")
        return dict(self._model.trace(q, n_tangent))
