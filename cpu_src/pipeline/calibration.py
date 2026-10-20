import json
from pathlib import Path
import numpy as np

def calibrate(config):
    from src.blending.det_shape_calibration import get_det_calibration_biases
    if not {"sim_mode0","sim_mode40"}.issubset(config.sim_modes):
        raise ValueError("g1 calibration requires both sim_mode0 and sim_mode40")
    for mode in (0,1):
        for index in range(*config.exp_range):
            path=Path(config.output_root)/f"g1_{mode}"/config.fname/f"catalog_{index}.npz"
            if not path.is_file():
                raise FileNotFoundError(path)
    out=Path(config.output_root)/"calibration"/config.fname
    out.mkdir(parents=True,exist_ok=True)
    result=get_det_calibration_biases(noise_level=config.noise_level or 0.,fname=config.fname,
             index_range=tuple(config.exp_range),shape_mode="ml",mag_cut=config.mag_cut,
             include_detection_response=config.include_detection_response,bs_times=config.bs_times,
             is_twin=False,workers=config.calibration_workers,base_dir=config.output_root,
             output_dir=str(out),log_csv=str(out/"calibration.csv"),shear_comps=["g1"])
    # g2 has no paired exposures; represent unavailable numbers as JSON null.
    summary={key:[float(x) if np.isfinite(x) else None for x in value]
             for key,value in zip(("m","c","m_err","c_err"),result)}
    summary.update({"shear_comps":["g1"],"n_tangent":config.n_tangent,"reference":"rounded_stamp_center"})
    (out/"result.json").write_text(json.dumps(summary,indent=2))
    return summary
