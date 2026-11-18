"""Run: python -m unittest cpu_src.tests.test_moment_model -v (from single_galaxies/)."""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
from cpu_src import moment_model as mm

RESEARCH = Path(__file__).resolve().parents[2] / "output" / "Codex_freestyle" / "score_research_20261029"


def synthetic_model(seed=0):
    rng = np.random.default_rng(seed)
    meta = {"format": mm.FORMAT, "format_version": mm.FORMAT_VERSION, "stamp_size": mm.STAMP,
            "kernel_center": mm.KERNEL_CENTER, "preprocessing": mm.PREPROCESSING}
    return mm.MomentModel(meta, rng.normal(size=mm.BASE_SHAPE), rng.normal(size=mm.CROSS_SHAPE))


def synthetic_data(n=6, seed=1):
    """Elliptical Gaussian galaxies (source near pixel 32) and an anisotropic Moffat PSF."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[:64, :64].astype(float)
    stamps = []
    for _ in range(n):
        cx, cy = 32 + rng.uniform(-.5, .5, 2)
        a, e1, e2 = rng.uniform(1.5, 4), *rng.normal(0, .2, 2)
        dx, dy = x - cx, y - cy
        r2 = ((1 - e1) * dx * dx + (1 + e1) * dy * dy - 2 * e2 * dx * dy) / (a * a)
        stamps.append(rng.uniform(5, 300) * np.exp(-r2 / 2) + rng.normal(0, .6, x.shape))
    r2 = (1.03 * (x - 32) ** 2 + .97 * (y - 32) ** 2 + .02 * (x - 32) * (y - 32)) / 2.0 ** 2
    psf = (1 + r2) ** -2.5
    noise = rng.normal(0, .6, (n, 64, 64))
    as32 = lambda a: np.asarray(a, dtype=np.float32).astype(np.float64)
    return as32(stamps), psf / psf.sum(), as32(noise)


def imageq(stamps, psf, noise=None):
    import anacal
    p = mm.PREPROCESSING
    iq = anacal.image.ImageQ(64, 64, p["pixel_scale"], p["sigma_arcsec"], p["klim"], True)
    if noise is None:
        return np.array([iq.prepare_qnumber_image(s, psf, 32, 32) for s in stamps])
    return np.array([iq.prepare_qnumber_image(s, psf, 32, 32, n) for s, n in zip(stamps, noise)])


class MomentModelTests(unittest.TestCase):
    def test_shapelet_spectra_match_pixel_kernels(self):
        spectra, residual = mm.shapelet_spectra()
        exact = np.fft.rfft2(mm.fixed_filters())
        scale = np.abs(exact).max((1, 2, 3))
        self.assertLess(np.abs(spectra - exact).max(), 1e-11 * scale.max())
        # AnaCal's continuous shapelets alone suffice where the 64x64 box does not truncate.
        relative = np.abs(residual).max((1, 2, 3)) / scale
        self.assertLess(relative[1:3].max(), 1e-12)
        self.assertGreater(relative[4], 1e-4)

    def test_fourier_jets_match_imageq_projection(self):
        model = synthetic_model()
        stamps, psf, noise = synthetic_data()
        for n in (None, noise):
            expected = model.qstamp_jets(imageq(stamps, psf, n))
            got = model.fourier_jets(stamps, psf, n, n_tangent=4)
            for channel in range(5):
                err = np.abs(got[:, channel] - expected[:, channel]).max()
                self.assertLess(err, 1e-12 * np.abs(expected[:, channel]).max(), mm.CHANNELS[channel])

    def test_per_object_psf_matches_shared_psf(self):
        model = synthetic_model()
        stamps, psf, noise = synthetic_data(3)
        shared = model.fourier_jets(stamps, psf, noise, 4)
        stacked = model.fourier_jets(stamps, np.repeat(psf[None], 3, 0), noise, 4)
        np.testing.assert_allclose(stacked, shared, rtol=0, atol=1e-12 * np.abs(shared).max())

    def test_head_tangents_match_finite_differences(self):
        model = synthetic_model()
        stamps, psf, noise = synthetic_data(4)
        jets = model.fourier_jets(stamps, psf, noise, 4)
        m, dm = jets[:, 0], jets[:, 1:]
        _, dy = mm._jet_head(m, dm, model.base, model.cross)
        for t in range(4):
            h = 1e-6
            plus, _ = mm._jet_head(m + h * dm[:, t], dm[:, :0], model.base, model.cross)
            minus, _ = mm._jet_head(m - h * dm[:, t], dm[:, :0], model.base, model.cross)
            np.testing.assert_allclose(dy[:, t], (plus - minus) / (2 * h), rtol=1e-6,
                                       atol=1e-7 * np.abs(dy[:, t]).max())

    def test_entry_points_agree(self):
        model = synthetic_model()
        stamps, psf, noise = synthetic_data(5)
        direct = model.measure_stamps(stamps, psf, noise, n_tangent=4, batch_size=2)
        q = model.measure_qstamps(imageq(stamps, psf, noise), n_tangent=4)
        for a, b in ((direct.e, q.e), (direct.R, q.R), (direct.dE_dx, q.dE_dx)):
            np.testing.assert_allclose(a, b, rtol=0, atol=1e-11 * np.abs(b).max())
        canvas, canvas_noise = np.zeros((80, 400)), np.zeros((80, 400))
        positions = []
        for i, (s, n) in enumerate(zip(stamps, noise)):
            canvas[8:72, 8 + 76 * i:72 + 76 * i] = s
            canvas_noise[8:72, 8 + 76 * i:72 + 76 * i] = n
            positions.append((40 + 76 * i + .3, 40 - .4))
        sources = model.measure_sources(canvas, psf, positions, canvas_noise, n_tangent=4, batch_size=3)
        np.testing.assert_array_equal(sources.centers_xy, np.floor(np.array(positions) + .5))
        np.testing.assert_allclose(sources.R, direct.R, rtol=0, atol=1e-13 * np.abs(direct.R).max())
        with self.assertRaises(ValueError):
            model.measure_sources(canvas, psf, [(10., 40.)])

    def test_reference_research_model(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch not installed")
        if not (RESEARCH / "model.py").exists():
            self.skipTest("research directory not available")
        sys.path.insert(0, str(RESEARCH))
        try:
            import model as research
        finally:
            sys.path.remove(str(RESEARCH))
        stamps, psf, noise = synthetic_data(8)
        q = imageq(stamps, psf, noise)
        with tempfile.TemporaryDirectory() as tmp:
            for name in ("model.pt", "model_k0.pt"):
                ours = mm.export_checkpoint(RESEARCH / name, Path(tmp) / "m.npz", pixel_average_order=None)
                ref = research.CrossScaleScore.from_checkpoint(RESEARCH / name)
                y, R = research.measure_response(ref, q[:, :3])
                got = ours.measure_stamps(stamps, psf, noise)
                np.testing.assert_allclose(got.e, y.numpy(), rtol=0, atol=1e-12)
                np.testing.assert_allclose(got.R, R.numpy(), rtol=0, atol=1e-12)

    def corrected_model(self):
        checkpoint = RESEARCH / "model.pt"
        if not checkpoint.exists():
            self.skipTest("research checkpoint not available")
        with tempfile.TemporaryDirectory() as tmp:
            return mm.export_checkpoint(checkpoint, Path(tmp) / "corrected.npz")

    def test_corrected_entry_points_and_noise(self):
        model = self.corrected_model()
        stamps, psf, noise = synthetic_data(3)
        for n in (None, noise):
            direct = model.measure_stamps(stamps, psf, n, n_tangent=4, batch_size=2)
            q = imageq(stamps, psf, n)
            projected = model.measure_qstamps(q, n_tangent=4, batch_size=1)
            np.testing.assert_allclose(direct.e, projected.e, rtol=0, atol=1e-10)
            np.testing.assert_allclose(direct.R, projected.R, rtol=1e-7, atol=1e-7)
            np.testing.assert_allclose(direct.dE_dx, projected.dE_dx, rtol=1e-7, atol=1e-7)
            with self.assertRaisesRegex(ValueError, "five AnaCal"):
                model.measure_qstamps(q[:, :3])
            canvas = np.concatenate(list(stamps), axis=1)
            ncanvas = None if n is None else np.concatenate(list(n), axis=1)
            catalogued = model.measure_sources(canvas, psf, [[32+64*i,32] for i in range(3)],
                                               ncanvas, n_tangent=4, batch_size=2)
            np.testing.assert_allclose(catalogued.e, direct.e, rtol=0, atol=1e-12)
            np.testing.assert_allclose(catalogued.R, direct.R, rtol=0, atol=1e-12)

    def test_corrected_d4_about_input_center(self):
        model = self.corrected_model()
        stamps, psf, noise = synthetic_data(3)
        baseline = model.measure_stamps(stamps, psf, noise)
        # 64x64 raw cutouts are centred at pixel 32. The model itself moves
        # that mean centre to 31.5; rotating raw arrays at 31.5 would be wrong.
        def rotate(a):
            return np.roll(np.rot90(a, axes=(-2,-1)), 1, axis=-2).copy()
        changed = model.measure_stamps(rotate(stamps), rotate(psf), rotate(noise))
        np.testing.assert_allclose(changed.e, -baseline.e, rtol=0, atol=2e-7)
        np.testing.assert_allclose(changed.R, baseline.R, rtol=0, atol=2e-7)
        flip = lambda a: np.roll(np.flip(a, axis=-1), 1, axis=-1).copy()
        changed = model.measure_stamps(flip(stamps), flip(psf), flip(noise))
        signs = np.array([1., -1.])
        np.testing.assert_allclose(changed.e, baseline.e*signs, rtol=0, atol=2e-7)
        np.testing.assert_allclose(changed.R, baseline.R*signs[:,None]*signs[None,:], rtol=0, atol=2e-7)

    def test_uniform_offset_population_against_physical_shear(self):
        import galsim
        model = self.corrected_model()
        legacy = mm.MomentModel({**model.metadata, "pixel_average_order": None}, model.base, model.cross)
        psf = galsim.Convolve([galsim.Moffat(beta=2.5,fwhm=.8),galsim.Pixel(.2)])
        pim = psf.drawImage(nx=64,ny=64,scale=.2,method="no_pixel",
                           offset=galsim.PositionD(.5,.5),dtype=np.float64).array
        gal = galsim.Gaussian(half_light_radius=.6,flux=100).shear(e1=.23,e2=.19)
        nodes, weights = np.polynomial.legendre.leggauss(6)
        nodes /= 2; weights /= 2
        points = [(x,y) for y in nodes for x in nodes]
        probability = (weights[:,None]*weights[None,:]).ravel()
        def draw(g1,g2,offsets):
            obj = galsim.Convolve([gal.lens(g1=g1,g2=g2,mu=1/(1-g1*g1-g2*g2)),psf])
            return np.array([obj.drawImage(nx=64,ny=64,scale=.2,method="no_pixel",
                             offset=galsim.PositionD(x,y),dtype=np.float64).array.copy() for x,y in offsets])
        centered = legacy.measure_stamps(draw(0,0,[(0,0)]),pim).e[0]
        offsets = [(x+.5,y+.5) for x,y in points]
        mean = model.measure_stamps(draw(0,0,offsets),pim)
        np.testing.assert_allclose(probability @ mean.e, centered, rtol=0, atol=3e-5)
        for axis in (0,1):
            h = .001
            plus = model.measure_stamps(draw(h if axis==0 else 0,h if axis==1 else 0,offsets),pim)
            minus = model.measure_stamps(draw(-h if axis==0 else 0,-h if axis==1 else 0,offsets),pim)
            finite = probability @ (plus.e-minus.e)/(2*h)
            analytic = probability @ mean.R[:,:,axis]
            np.testing.assert_allclose(analytic, finite, rtol=5e-4, atol=2e-5)

    def test_import_and_measure_without_torch(self):
        code = '''import importlib.abc,sys
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split(".")[0] in ("torch","torchvision"):raise RuntimeError("torch import attempted")
sys.meta_path.insert(0,Block())
from cpu_src.tests.test_moment_model import synthetic_model,synthetic_data
s,p,n=synthetic_data(2);r=synthetic_model().measure_stamps(s,p,n)
assert r.R.shape==(2,2,2) and "torch" not in sys.modules
'''
        subprocess.run([sys.executable, "-B", "-c", code], check=True, cwd=Path(__file__).resolve().parents[2])


if __name__ == "__main__":
    unittest.main()
