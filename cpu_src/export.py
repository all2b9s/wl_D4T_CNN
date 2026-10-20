"""Checkpoint export. Torch is imported only when export/reference is requested."""
import inspect
import json
import hashlib
from pathlib import Path
import numpy as np
from .model_io import ARCHITECTURES, FORMAT_VERSION, save_model, sha256

def export_checkpoint(checkpoint, output, config=None, architecture=None, model_overrides=None,
                      stamp_size=64, pixel_scale=None, sigma_arcsec=None, flim=None):
    import torch
    from src.architecture import forward8_CNN
    checkpoint = Path(checkpoint)
    config_path = Path(config) if config else checkpoint.with_name("config.json")
    recorded = json.loads(config_path.read_text()) if config_path.is_file() else {}
    name = architecture or recorded.get("model_class")
    if name not in ARCHITECTURES:
        raise ValueError("Specify a supported architecture explicitly or in config.json")
    if recorded.get("model_class", name) != name:
        raise ValueError("Architecture conflicts with config.json")
    kwargs = dict(recorded.get("model", {}))
    for key, value in (model_overrides or {}).items():
        if key in kwargs and kwargs[key] != value:
            raise ValueError(f"Explicit {key} conflicts with config: {kwargs[key]} != {value}")
        kwargs[key] = value
    for key in (["nl_sigma", "gw_sigma"] if name == "Forward8_fixW_CNN" else ["nl_sigma"]):
        if key not in kwargs:
            raise ValueError(f"{key} is not in checkpoints; provide config or --model-kwargs")
    cls = getattr(forward8_CNN, name)
    signature = inspect.signature(cls)
    bound = signature.bind(**kwargs)
    bound.apply_defaults()
    kwargs = dict(bound.arguments)
    pre_recorded = recorded.get("image_preprocessing", {})
    pre = {}
    for key, explicit in (("pixel_scale", pixel_scale), ("sigma_arcsec", sigma_arcsec), ("flim", flim)):
        value = pre_recorded.get(key)
        if explicit is not None and value is not None and not np.isclose(explicit, value, rtol=1e-12, atol=0):
            raise ValueError(f"Explicit {key} conflicts with preprocessing config")
        pre[key] = explicit if explicit is not None else value
        if pre[key] is None:
            raise ValueError(f"Missing preprocessing {key}; specify explicitly")
    pre["klim"] = pre["flim"] / pre["pixel_scale"]
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if "state_dict" in state:
        state = state["state_dict"]
    elif "model_state_dict" in state:
        state = state["model_state_dict"]
    model = cls(**kwargs)
    model.load_state_dict(state, strict=True)
    weights = {key: value.detach().cpu().numpy() for key, value in state.items()}
    source_hash = hashlib.sha256(Path(forward8_CNN.__file__).read_bytes()).hexdigest()
    meta = {"format_version": FORMAT_VERSION, "architecture": name, "model": kwargs,
            "stamp_size": stamp_size, "preprocessing": pre, "checkpoint_sha256": sha256(checkpoint),
            "checkpoint": str(checkpoint.resolve()), "config": recorded,
            "model_source_sha256": source_hash, "torch_version": torch.__version__,
            "numerics": {"layernorm_eps": 1e-5, "softplus_beta": 10, "softplus_threshold": 20,
                         "gelu": "exact_erf", "reference": "rounded_stamp_center"}}
    save_model(output, meta, weights)
    return meta
