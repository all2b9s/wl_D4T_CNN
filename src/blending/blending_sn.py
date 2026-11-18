"""Full-blending shape noise from the scatter of per-tract (exposure) means.

Estimator (same convention as
``scripts/training_size_test/global_response_tract_noise.py`` and
``ml_wl/steps/blended.py::effective_noise``)::

    m_t      = sum_t(signal) / N_t                       per tract t
    R_global = sum_t sum_t(response) / sum_t N_t         one response per method/component
    sigma    = sqrt(mean_t N_t) * std_t(m_t, ddof=1) / |R_global|

Each exposure is one tract, so blending-induced correlations between
neighbouring detections are included in the scatter (unlike the per-object
``std(e)/|mean(R)|`` of ``test_func.shape_noise_from_stats``).

Per-tract sums come from ``test_func.accumulate_shape_noise``; ``estimate``
below reimplements the tract-scatter estimator.  The bootstrap resamples
tracts with ``sqrt(mean N_t)`` held at its original-sample value.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.blending.test_func import (
    DEFAULT_METHODS,
    accumulate_shape_noise,
    catalog_paths,
    shape_noise_from_stats,
)


def estimate(signal, response, count, scale):
    """Arrays ``[tract, method, component]`` -> ``(noise, tract_std, R_global, tract_shear)``."""
    means = signal / count
    r_global = response.sum(axis=0) / count.sum(axis=0)
    if not np.isfinite(r_global).all() or np.any(np.abs(r_global) < 1e-12):
        raise ValueError("Invalid global response")
    std = means.std(axis=0, ddof=1) / np.abs(r_global)
    return std * scale, std, r_global, means / r_global


def tract_shape_noise_from_stats(
    stats: np.ndarray,
    n_boot: int = 2000,
    seed: int = 42,
    scale: Optional[float] = None,
) -> dict:
    """Tract-scatter shape noise from per-tract sums.

    ``stats`` has shape ``(n_tract, n_method, 4, 2)`` with rows
    ``[N, sum(signal), sum(signal^2), sum(response)]`` (the layout of
    ``accumulate_shape_noise``).  ``scale`` defaults to ``sqrt(mean N_t)``.

    Returns a dict with ``(n_method, 2)`` arrays ``shape_noise``, ``tract_std``,
    ``global_response`` and ``bootstrap`` draws ``(n_boot, n_method, 2)``, the
    per-tract shear ``(n_tract, n_method, 2)``, and ``scale``.
    """
    stats = np.asarray(stats, dtype=np.float64)
    if stats.ndim != 4 or stats.shape[2:] != (4, 2):
        raise ValueError(f"stats must have shape (n_tract, n_method, 4, 2), got {stats.shape}")
    if len(stats) < 2:
        raise ValueError("At least two tracts are required")
    count = stats[:, :, 0]
    if np.any(count <= 0):
        raise ValueError("Every tract needs at least one selected object")
    signal, response = stats[:, :, 1], stats[:, :, 3]
    if scale is None:
        scale = float(np.sqrt(count[:, 0, 0].mean()))

    noise, tract_std, r_global, shear = estimate(signal, response, count, scale)

    draws = np.empty((n_boot,) + noise.shape)
    rng = np.random.default_rng(seed)
    for b in range(n_boot):
        idx = rng.integers(len(stats), size=len(stats))
        draws[b] = estimate(signal[idx], response[idx], count[idx], scale)[0]

    return dict(shape_noise=noise, tract_std=tract_std, global_response=r_global,
                tract_shear=shear, bootstrap=draws, scale=scale)


def blending_shape_noise(
    paths: Sequence[Path | str] | Path | str,
    methods: Sequence[str] = DEFAULT_METHODS,
    detection_weighted: bool = True,
    mag_cut: Optional[Mapping] = None,
    index_range: Optional[Tuple[int, int]] = None,
    n_boot: int = 2000,
    seed: int = 42,
    reference: Optional[str] = "fpfs",
    verbose: bool = True,
) -> Tuple[pd.DataFrame, dict]:
    """Full-blending shape noise from per-exposure ``catalog_*.npz`` files.

    ``paths`` is a list of catalogs or a directory (filtered by
    ``index_range``).  ``methods``/``detection_weighted``/``mag_cut`` are passed
    to ``accumulate_shape_noise``; all methods share one finite mask per tract,
    so the bootstrap is paired across methods.  If ``reference`` is one of the
    methods, each row also carries the ratio to it with a paired bootstrap 95%
    interval.  The per-object ``std(e)/|mean(R)|`` is reported alongside as
    ``object_shape_noise`` for comparison.

    Returns ``(table, raw)``: one row per (method, component), and the raw
    arrays (``stats``, ``manifest``, plus the output of
    ``tract_shape_noise_from_stats``).
    """
    if isinstance(paths, (str, Path)):
        paths = catalog_paths(paths, index_range=index_range)
    methods = list(methods)
    stats, counts, manifest = accumulate_shape_noise(
        paths, methods=methods, detection_weighted=detection_weighted,
        mag_cut=mag_cut, verbose=verbose)
    keep = stats[:, 0, 0, 0] > 0
    if verbose and not keep.all():
        print(f"[blending_sn] dropping {int((~keep).sum())} tracts with no selected objects")
    stats, manifest = stats[keep], manifest[keep].reset_index(drop=True)

    result = tract_shape_noise_from_stats(stats, n_boot=n_boot, seed=seed)
    object_noise = shape_noise_from_stats(stats.sum(axis=0))[2]
    noise, draws = result["shape_noise"], result["bootstrap"]
    ref = methods.index(reference) if reference in methods else None

    rows = []
    for m, method in enumerate(methods):
        for c in range(2):
            row = dict(method=method, component=f"e{c + 1}", n_tract=len(stats),
                       n_object=int(stats[:, m, 0, c].sum()), scale=result["scale"],
                       global_response=result["global_response"][m, c],
                       tract_std=result["tract_std"][m, c],
                       shape_noise=noise[m, c],
                       shape_noise_err=draws[:, m, c].std(ddof=1),
                       object_shape_noise=object_noise[m, c])
            if ref is not None and ref != m:
                ratio = draws[:, m, c] / draws[:, ref, c]
                row.update({f"ratio_to_{reference}": noise[m, c] / noise[ref, c],
                            "ratio_ci_low": np.quantile(ratio, 0.025),
                            "ratio_ci_high": np.quantile(ratio, 0.975)})
            rows.append(row)

    raw = dict(result, stats=stats, counts=counts, manifest=manifest, methods=methods)
    return pd.DataFrame(rows), raw


def aperture_shape_covariance(paths, *, radius_arcsec=300., square_side_arcsec=300.,
                              pixel_scale=.2, field_bounds_px=(32., 1468.),
                              edges_arcsec=(0., .4, .8, 1.6, 3.2, 6.4, 12.8, 25.6, 51.2, 102.4),
                              n_boot=500, seed=42, method="ml", short_range_approx=False,
                              return_field_statistics=False):
    """Connected, response-normalized pair noise and aperture covariance.

    Uses all finite detections, including same-truth detections, to match the
    actual shear estimator. Signal s=w*e; response d=w*R+dw*e. One population
    response per component, never a division by noisy per-object responses.
    Each unordered pair contributes both orientations, including symmetric 12.
    ``method`` selects the saved ML or FPFS measurements.
    ``short_range_approx`` replaces aperture overlap by aperture area,
    giving the correlation-area / aperture-area approximation.
    Translation edge correction divides by the rectangular field overlap.
    Aperture covariance uses the exact circular/square overlap at each pair
    displacement; no bin-centre approximation. Results at successive upper
    separation cutoffs expose tail sensitivity. Contributions outside the
    largest cutoff are unmeasured, not asserted to be zero.

    Field bootstrap resamples response and all pair/self numerators together.
    The global connected-mean correction is fitted once (O(1/Nfield) bias).
    No images or pair lists are retained. Assumes a homogeneous source/noise
    process and a common rectangular detection footprint.
    """
    from scipy.spatial import cKDTree
    from src.blending.test_func import load_exposure_arrays
    edges = np.asarray(edges_arcsec, dtype=float)
    if len(paths) < 2 or np.any(np.diff(edges) <= 0) or edges[0] != 0:
        raise ValueError('Need >=2 fields and increasing separation edges from zero')
    side = (field_bounds_px[1] - field_bounds_px[0]) * pixel_scale
    area = side**2
    aperture_areas = np.array([np.pi*radius_arcsec**2, square_side_arcsec**2])
    records = []
    sum_s = np.zeros(2); sum_w = 0.
    for path in paths:
        cat = load_exposure_arrays(path, methods=[method])
        mask = cat['mask']; e = cat['methods'][method]['e'][mask].astype(float)
        w = cat['w'][mask].astype(float)
        with np.load(path) as z:
            dw = np.stack([z['fpfs_dw_dg1'][mask], z['fpfs_dw_dg2'][mask]], 1)
        response = w[:,None]*cat['methods'][method]['R'][mask] + dw*e
        if not np.isfinite(response).all():
            raise ValueError('Nonfinite full detection response')
        xy = np.column_stack([cat['x'][mask], cat['y'][mask]])
        if np.any(xy < field_bounds_px[0]) or np.any(xy > field_bounds_px[1]):
            raise ValueError('Catalog exceeds declared detection footprint')
        signal = w[:,None]*e
        records.append((xy*pixel_scale, signal, w, response))
        sum_s += signal.sum(0); sum_w += w.sum()
    mean_e = sum_s / sum_w
    pairs_raw=[]; pair_counts=[]; pair_cov=[]; self_cov=[]; responses=[]; counts=[]; totals=[]
    for xy, signal, w, response in records:
        signal = signal - w[:,None]*mean_e
        counts.append(len(signal)); responses.append(response.sum(0)); totals.append(signal.sum(0))
        self_cov.append(signal.T@signal)
        ij = cKDTree(xy).query_pairs(edges[-1], output_type='ndarray')
        delta = xy[ij[:,0]]-xy[ij[:,1]]; distance = np.linalg.norm(delta,axis=1)
        index = np.minimum(np.searchsorted(edges,distance,side='right')-1,len(edges)-2)
        a,b=signal[ij[:,0]],signal[ij[:,1]]
        prod=(a[:,:,None]*b[:,None,:]+b[:,:,None]*a[:,None,:])/2
        overlap_field = (side-np.abs(delta[:,0]))*(side-np.abs(delta[:,1]))
        u = np.clip(distance/(2*radius_arcsec),0,1)
        overlap_circle = 2*radius_arcsec**2*(np.arccos(u)-u*np.sqrt(1-u*u))
        overlap_square = np.prod(np.maximum(square_side_arcsec-np.abs(delta),0),axis=1)
        if short_range_approx:
            overlap_circle = np.full_like(distance, aperture_areas[0])
            overlap_square = np.full_like(distance, aperture_areas[1])
        pair_counts.append(np.bincount(index,minlength=len(edges)-1))
        raw=np.zeros((len(edges)-1,2,2)); cov=np.zeros((2,len(edges)-1,2,2))
        for i in range(2):
            for j in range(2):
                raw[:,i,j]=np.bincount(index,weights=prod[:,i,j],minlength=len(edges)-1)
                for window, overlap in enumerate((overlap_circle,overlap_square)):
                    cov[window,:,i,j]=np.bincount(index,weights=2*prod[:,i,j]*overlap/overlap_field/aperture_areas[window]**2,minlength=len(edges)-1)
        pairs_raw.append(raw);pair_cov.append(cov)
    responses=np.asarray(responses); counts=np.asarray(counts); totals=np.asarray(totals)
    self_cov=np.asarray(self_cov); pair_cov=np.asarray(pair_cov)
    pairs_raw=np.asarray(pairs_raw); pair_counts=np.asarray(pair_counts)
    def reduce(which):
        n=counts[which].sum(); r=responses[which].sum(0)/n
        density=responses[which].mean(0)/area
        norm=density[:,None]*density[None,:]
        shot=self_cov[which].mean(0)[None]/area/aperture_areas[:,None,None]/norm
        corr=pair_cov[which].mean(0).cumsum(axis=1)/norm
        pc=pair_counts[which].sum(0)
        xi=np.divide(pairs_raw[which].sum(0),pc[:,None,None],out=np.full_like(pairs_raw[0],np.nan),where=pc[:,None,None]>0)/(r[:,None]*r[None,:])
        return r,shot,corr,xi
    which=np.arange(len(paths)); r,shot,corr,xi=reduce(which)
    rng=np.random.default_rng(seed)
    draws=[reduce(rng.integers(0,len(paths),len(paths)))[1:] for _ in range(n_boot)]
    def interval(k):
        values=np.asarray([d[k] for d in draws])
        flat=values.reshape(len(values),-1); valid=np.isfinite(flat).any(axis=0)
        se=np.full(flat.shape[1],np.nan);ci=np.full((2,flat.shape[1]),np.nan)
        se[valid]=np.nanstd(flat[:,valid],axis=0,ddof=1)
        ci[:,valid]=np.nanquantile(flat[:,valid],[.025,.975],axis=0)
        return dict(SE=se.reshape(values.shape[1:]).tolist(),CI95=ci.reshape((2,)+values.shape[1:]).tolist())
    # Direct field-scatter covariance, with the same fixed population response.
    g=totals/counts[:,None]/r
    direct=np.cov(g.T,ddof=1)
    result = dict(method=method, short_range_approx=short_range_approx,
        n_fields=len(paths),n_detections=int(counts.sum()),global_response=r.tolist(),
        mean_e_removed=mean_e.tolist(),field_side_arcsec=side,edges_arcsec=edges.tolist(),
        pair_counts=pair_counts.sum(0).tolist(),xi_R_normalized=xi.tolist(),xi_uncertainty=interval(2),
        windows=['circle_radius_'+str(radius_arcsec),'square_side_'+str(square_side_arcsec)],
        aperture_areas_arcsec2=aperture_areas.tolist(),self_covariance=shot.tolist(),self_uncertainty=interval(0),
        correlated_covariance_cumulative=corr.tolist(),correlated_uncertainty=interval(1),
        total_covariance_cumulative=(shot[:,None]+corr).tolist(),
        direct_field_covariance=direct.tolist(),
        direct_area_scaled_covariance=(direct[None]*area/aperture_areas[:,None,None]).tolist(),
        scope='Connected measurement covariance at zero shear, including blends, shared image noise and duplicate detections. Not cosmic-shear covariance. Finite-separation truncation and homogeneous-footprint extrapolation; direct area scaling assumes short-range correlations.',
        bootstrap_fields=n_boot,seed=seed)

    def finite_json(value):
        if isinstance(value, dict):
            return {k:finite_json(v) for k,v in value.items()}
        if isinstance(value, list):
            return [finite_json(v) for v in value]
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value
    result = finite_json(result)
    if return_field_statistics:
        # Compact sufficient statistics for paired resampling of frozen models.
        # No pair list or image is retained; the default public result is unchanged.
        return result, dict(counts=counts, responses=responses, self_covariance=self_cov,
                            pair_covariance=pair_cov, field_area=area,
                            aperture_areas=aperture_areas)
    return result
