import json
import platform
import time
from pathlib import Path
import numpy as np
from cpu_src import Model
from cpu_src.model_io import sha256
from .measurement import MeasurementBackend
from .catalog import save_catalog

SIM_MODES={"sim_mode0":("g1",0),"sim_mode40":("g1",1),"sim_mode80":("g1",2)}

def run_measurement(config):
    import anacal
    import fitsio
    from .exposure import load_blended_exposure
    from .detection import _build_truth_detection_catalog, _recenter_on_truth
    from src.anacal.detection_ml_pipeline import run_detection_and_fpfs, _build_cell_list, _build_task_from_fpfs_config
    config.validate()
    model=Model.load(config.model,config.precision,config.isa)
    h=model.metadata["stamp_size"]
    if h%2:
        raise ValueError("The source pipeline requires even stamp size")
    pre=model.metadata["preprocessing"]
    psf=np.asarray(np.load(config.psf_path,allow_pickle=False),dtype=np.float64)
    if psf.ndim==3:
        psf=psf[0]
    if psf.ndim!=2:
        raise ValueError("PSF must be 2D or a stack of 2D images")
    # Preserve the existing pipeline's odd PSF trimming convention.
    if psf.shape[0]%2 or psf.shape[1]%2:
        psf=psf[:-1,:-1]
    psf=np.ascontiguousarray(psf)
    cat_ref=fitsio.read(config.cat_ref_path)
    backend=MeasurementBackend(model,config)
    root=Path(config.output_root)
    root.mkdir(parents=True,exist_ok=True)
    manifest={"config":config.asdict(),"model_sha256":sha256(config.model),"model_metadata":model.metadata,
              "anacal_version":getattr(anacal,"__version__","unknown"),"machine":platform.machine(),
              "isa":model.isa,"exposures":[]}
    start=time.perf_counter()
    for sim_mode in config.sim_modes:
        comp,mode=SIM_MODES[sim_mode]
        for index in range(*config.exp_range):
            row={"sim_mode":sim_mode,"index":index}
            tick=time.perf_counter()
            image,sigma,truth,truth_positions=load_blended_exposure(
                str(Path(config.blended_dir)/sim_mode/f"exp-i-{index:05d}.fits"),
                str(Path(config.blended_dir)/sim_mode/f"truth-{index:05d}.fits"),cat_ref,pre["pixel_scale"],
                "from_variance" if config.noise_level is None else "fixed",config.noise_level)
            row["load_seconds"]=time.perf_counter()-tick
            tick=time.perf_counter()
            noise=np.random.default_rng(19491001+index).normal(0,sigma,image.shape).astype(np.float32)
            if config.skip_detection:
                det=_build_truth_detection_catalog(truth,config.mag_cut)
            else:
                task=_build_task_from_fpfs_config(mag_zero=config.mag_zero,pixel_scale=pre["pixel_scale"],snr_peak_min=config.snr_cut)
                cells=_build_cell_list(*image.shape,pixel_scale=pre["pixel_scale"])
                fpfs=anacal.fpfs.FpfsConfig(sigma_shapelets1=.45,sigma_shapelets2=.55)
                det=run_detection_and_fpfs(image,psf,fpfs,config.mag_zero,pre["pixel_scale"],
                                           sigma**2 if sigma>0 else 1e-4,noise,task,cells)
                det=det[det["fpfs_w"]>=config.weight_threshold]
                if config.center_on_truth:
                    det=_recenter_on_truth(det,truth_positions,config.match_threshold)
            # Match existing 32-pixel edge margin for 64-pixel stamps, while also
            # checking the actual rounded centers for configurable even sizes.
            cx=np.floor(det["x"]+.5).astype(np.int64);cy=np.floor(det["y"]+.5).astype(np.int64)
            margin=max(32,h//2)
            valid=(det["x"]>=margin)&(det["x"]<image.shape[1]-margin)&(det["y"]>=margin)&(det["y"]<image.shape[0]-margin)
            valid&=(cx-h//2>=0)&(cx+h//2<=image.shape[1])&(cy-h//2>=0)&(cy+h//2<=image.shape[0])
            det=det[valid]
            row["detection_seconds"]=time.perf_counter()-tick
            positions=np.column_stack((det["x"],det["y"]))
            tick=time.perf_counter()
            result=backend.measure(image,psf,positions,noise)
            row["measurement_seconds"]=time.perf_counter()-tick
            tick=time.perf_counter()
            save_catalog(det,result,truth_positions,truth,root/f"{comp}_{mode}"/config.fname,index,config)
            row["save_seconds"]=time.perf_counter()-tick
            row["sources"]=len(det)
            manifest["exposures"].append(row)
            print(json.dumps(row),flush=True)
    manifest["wall_seconds"]=time.perf_counter()-start
    manifest["sources"]=sum(r["sources"] for r in manifest["exposures"])
    manifest["gal_per_sec"]=manifest["sources"]/manifest["wall_seconds"]
    (root/f"{config.fname}_manifest.json").write_text(json.dumps(manifest,indent=2))
    return manifest
