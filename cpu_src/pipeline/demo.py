"""Generate a tiny reproducible FITS example; setup and execution need no torch."""
import json
from pathlib import Path
import numpy as np
from cpu_src.model_io import load_model
from .config import PipelineConfig

def create_demo(model_path, directory):
    from astropy.io import fits
    from astropy.table import Table
    model_path=Path(model_path).resolve()
    metadata,_=load_model(model_path)
    stamp=metadata["stamp_size"]
    if stamp%2:
        raise ValueError("The source demo requires an even stamp size")
    root=Path(directory).resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"Choose an empty demo directory: {root}")
    root.mkdir(parents=True,exist_ok=True)
    side=max(128,2*stamp)
    psf=np.zeros((48,48),dtype=np.float64)
    psf[24,24]=1.
    np.save(root/"psf.npy",psf)
    Table({"i_ab":[22.5],"hlr":[.5]}).write(root/"reference.fits")
    y,x=np.mgrid[:side,:side].astype(float)
    center_x,center_y=side/2+.13,side/2+.27
    x-=center_x
    y-=center_y
    # Shared image noise across the two shear conditions, independently of
    # the renoise canvas generated later by the pipeline.
    image_noise=np.random.default_rng(42).normal(0,.1,x.shape)
    scale=metadata["preprocessing"]["pixel_scale"]
    for mode,shear in (("sim_mode0",-.02),("sim_mode40",.02)):
        folder=root/mode
        folder.mkdir()
        xx=(1-shear)*x
        yy=(1+shear)*y
        signal=12*np.exp(-(xx**2/20+yy**2/28))+3*np.exp(-((xx-3)**2/6+(yy+1)**2/9))
        image=(signal+image_noise).astype(np.float32)
        hdus=[fits.PrimaryHDU(),fits.ImageHDU(image),fits.ImageHDU(np.zeros_like(image)),
              fits.ImageHDU(np.full_like(image,.01))]
        fits.HDUList(hdus).writeto(folder/"exp-i-00000.fits")
        Table({"dx":[(center_x-side/2-1500)*scale],"dy":[(center_y-side/2-1500)*scale],
               "gamma1":[shear],"gamma2":[0.],"indices":[0]}).write(folder/"truth-00000.fits")
    config=PipelineConfig(model=str(model_path),blended_dir=str(root),psf_path=str(root/"psf.npy"),
                          cat_ref_path=str(root/"reference.fits"),output_root=str(root/"catalogs"),
                          fname="demo_cpu",nthreads=1,n_tangent=4,bs_times=10)
    (root/"config.json").write_text(json.dumps(config.asdict(),indent=2))
    (root/"README.txt").write_text("Synthetic two-exposure functional demo only. This sample cannot establish scientific m/c accuracy.\n")
    return config
