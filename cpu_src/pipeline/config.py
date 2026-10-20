from dataclasses import dataclass, field, asdict
import json
import math
from pathlib import Path

@dataclass
class PipelineConfig:
    model: str
    blended_dir: str
    psf_path: str
    cat_ref_path: str
    output_root: str
    fname: str = "det_ml_cpu"
    sim_modes: list[str] = field(default_factory=lambda: ["sim_mode0", "sim_mode40"])
    exp_range: list[int] = field(default_factory=lambda: [0,1])
    precision: str = "float32"
    isa: str = "auto"
    n_tangent: int = 2
    nthreads: int = 1
    batch_size: int | None = None
    noise_level: float | None = None
    mag_zero: float = 30.
    snr_cut: float = 5.
    match_threshold: float = 3.
    weight_threshold: float = 1e-6
    skip_detection: bool = False
    center_on_truth: bool = False
    mag_cut: float | None = None
    save_csv: bool = False
    include_detection_response: bool = True
    bs_times: int = 100
    calibration_workers: int = 1
    backend: str = "cpp"
    reference_device: str = "cpu"

    def validate(self):
        if self.precision not in ("float32","float64") or self.backend not in ("cpp","torch"):
            raise ValueError("Invalid precision or backend")
        if self.n_tangent not in (2,4):
            raise ValueError("n_tangent must be 2 or 4")
        for key in ("nthreads","bs_times","calibration_workers"):
            if type(getattr(self,key)) is not int or getattr(self,key)<1:
                raise ValueError(f"{key} must be a positive integer")
        if self.batch_size is None:
            self.batch_size = max(64, 8 * self.nthreads)
        if type(self.batch_size) is not int or self.batch_size < 1:
            raise ValueError("batch_size must be a positive integer or null (automatic)")
        if len(self.exp_range)!=2 or any(type(x) is not int for x in self.exp_range) or not 0 <= self.exp_range[0]<self.exp_range[1]:
            raise ValueError("exp_range must be [start,end) with 0<=start<end")
        if not self.sim_modes or len(set(self.sim_modes))!=len(self.sim_modes) or any(x not in ("sim_mode0","sim_mode40","sim_mode80") for x in self.sim_modes):
            raise ValueError("Invalid sim_modes")
        if self.noise_level is not None and (not math.isfinite(self.noise_level) or self.noise_level<0):
            raise ValueError("noise_level must be nonnegative and finite")
        if not self.fname or Path(self.fname).name != self.fname or self.fname in (".",".."):
            raise ValueError("fname must be a single directory name")
        for key in ("snr_cut","match_threshold"):
            if not math.isfinite(getattr(self,key)) or getattr(self,key)<=0:
                raise ValueError(f"{key} must be positive and finite")
        return self

    @classmethod
    def load(cls,path):
        path=Path(path).resolve()
        data=json.loads(path.read_text())
        # Relative paths are resolved against the configuration file.
        for key in ("model","blended_dir","psf_path","cat_ref_path","output_root"):
            if key in data:
                p=Path(data[key]).expanduser()
                data[key]=str(p if p.is_absolute() else path.parent/p)
        return cls(**data).validate()

    def asdict(self):
        return asdict(self)
