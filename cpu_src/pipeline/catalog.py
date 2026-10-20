import json
from pathlib import Path
import numpy as np
from src.anacal.detection_ml_pipeline import merge_and_save_catalog

def save_catalog(det, result, truth_positions, truth, root, exp_index, config):
    root=Path(root)
    if len(det)!=len(result.e):
        raise ValueError("Detection and measurement counts differ")
    path=merge_and_save_catalog(det_cat=det,ml_shapes=result.e,ml_R=result.R[:,None],positions=truth_positions,
                               output_dir=str(root),truth_cat=truth,fname=f"catalog_{exp_index}",
                               match_threshold=config.match_threshold,save_npz=True,save_csv=config.save_csv,verbose=False)
    sidecar={"e":result.e,"R":result.R}
    if result.dE_dx is not None:
        sidecar["dE_dx"]=result.dE_dx
    if result.centers_xy is not None:
        sidecar["centers_xy"]=result.centers_xy
    sidecar["metadata"]=np.frombuffer(json.dumps(result.metadata).encode(),dtype=np.uint8)
    np.savez(root/f"catalog_{exp_index}_response.npz",**sidecar)
    return path
