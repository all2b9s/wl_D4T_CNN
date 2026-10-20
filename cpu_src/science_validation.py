"""Paired exposure bootstrap, preserving the existing g1 calibration definition."""
import json
from pathlib import Path
import numpy as np
from .validation import paired_catalogs

def compare_runs(left_manifest,right_manifest,repeats=200,seed=42):
    from src.anacal.cal_toolkit import selection_response
    left=json.loads(Path(left_manifest).read_text())
    right=json.loads(Path(right_manifest).read_text())
    a,b=left["config"],right["config"]
    for key in ("sim_modes","exp_range","mag_cut","include_detection_response","noise_level","skip_detection","center_on_truth"):
        if a[key]!=b[key]:
            raise ValueError(f"Science settings differ: {key}")
    if left["model_metadata"]["checkpoint_sha256"]!=right["model_metadata"]["checkpoint_sha256"]:
        raise ValueError("Science comparison requires the same checkpoint")
    if not {"sim_mode0","sim_mode40"}.issubset(a["sim_modes"]):
        raise ValueError("Paired g1 science comparison needs positive and negative shear exposures")
    if repeats<2:
        raise ValueError("At least two bootstrap repeats are required")
    indices=list(range(*a["exp_range"]))
    data=[{},{}]
    catalog_reports=[]
    # Order is positive shear, negative shear, matching _direct_calibrate.
    for mode in (1,0):
        for index in indices:
            paths=[Path(c["output_root"])/f"g1_{mode}"/c["fname"]/f"catalog_{index}.npz" for c in (a,b)]
            catalog_reports.append({"mode":mode,"index":index,"errors":paired_catalogs(*paths)})
            for branch,path in enumerate(paths):
                with np.load(path,allow_pickle=False) as archive:
                    data[branch][mode,index]={k:np.array(archive[k]) for k in archive.files}
            for key in ("fpfs_w","fpfs_dw_dg1","fpfs_dw_dg2","fpfs_m00","fpfs_dm00_dg1","fpfs_dm00_dg2"):
                if not np.array_equal(data[0][mode,index][key],data[1][mode,index][key],equal_nan=True):
                    raise ValueError(f"Detection/selection inputs differ: {key}")

    def estimate(branch,sample):
        signals,responses,fluxes,flux_derivs=[],[],[],[]
        for mode in (1,0):
            arrays=[data[branch][mode,index] for index in sample]
            def cat(key):
                return np.concatenate([x[key] for x in arrays])
            e=np.column_stack((cat("ml_e1"),cat("ml_e2")))
            response=np.empty((len(e),2,2))
            for i in range(2):
                for j in range(2):
                    response[:,i,j]=cat(f"ml_R{i+1}{j+1}")
            if a["include_detection_response"]:
                w=cat("fpfs_w")
                dw=np.column_stack((cat("fpfs_dw_dg1"),cat("fpfs_dw_dg2")))
                response=w[:,None,None]*response+e[:,:,None]*dw[:,None,:]
                e=w[:,None]*e
            signals.append(e)
            responses.append(response)
            fluxes.append(cat("fpfs_m00")*2*np.pi*.45**2)
            flux_derivs.append(np.column_stack((cat("fpfs_dm00_dg1"),cat("fpfs_dm00_dg2")))*2*np.pi*.45**2)
        masks=[np.ones(len(x),dtype=bool) for x in signals]
        if a["mag_cut"] is not None:
            selection=[]
            for e,flux,dflux in zip(signals,fluxes,flux_derivs):
                if len(e):
                    r=selection_response(flux=np.maximum(flux,1e-30),R_flux=dflux,shapes=e,zero_point=30.,mag_cut=a["mag_cut"])
                    if np.isfinite(r).all():
                        selection.append(r)
            correction=np.mean(selection,axis=0) if selection else np.zeros(2)
            for i in range(2):
                responses[i][:,0,0]+=correction[0]
                responses[i][:,1,1]+=correction[1]
                masks[i]=30-2.5*np.log10(np.maximum(fluxes[i],1e-30))<a["mag_cut"]
        if any(not mask.any() for mask in masks):
            return np.array([np.nan,np.nan])
        means=[e[m,0].mean() for e,m in zip(signals,masks)]
        denom=sum(r[m,0,0].mean() for r,m in zip(responses,masks))
        if not np.isfinite(denom) or denom==0:
            return np.array([np.nan,np.nan])
        return np.array([(means[0]-means[1])/denom/.02-1,(means[0]+means[1])/denom])
    original=[estimate(branch,indices) for branch in (0,1)]
    delta=original[1]-original[0]
    rng=np.random.default_rng(seed)
    samples=[]
    for _ in range(repeats):
        chosen=rng.choice(indices,size=len(indices),replace=True)
        samples.append(estimate(1,chosen)-estimate(0,chosen))
    samples=np.asarray(samples)
    finite=np.isfinite(samples).all(1)
    if not np.isfinite(delta).all() or not finite.any():
        raise ValueError("Calibration undefined: empty selection or invalid response")
    interval=np.quantile(samples[finite],[.025,.975],axis=0)
    return {"left_mc":original[0].tolist(),"right_mc":original[1].tolist(),"delta_m":float(delta[0]),
            "delta_c":float(delta[1]),"delta_m_95_interval":interval[:,0].tolist(),
            "delta_c_95_interval":interval[:,1].tolist(),"bootstrap_unit":"paired exposure",
            "bootstrap_seed":seed,"valid_repeats":int(finite.sum()),"exposure_pairs":len(indices),
            "science_evidence":"insufficient: fewer than 20 exposure pairs" if len(indices)<20 else "paired sample; inspect interval",
            "target_abs_delta_m":1e-4,"target_met":bool(abs(delta[0])<1e-4),"catalogs":catalog_reports}
