"""Versioned, pickle-free model interchange. This module never imports torch."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np

FORMAT_VERSION = 1
ARCHITECTURES = ("Forward8_fixW_CNN", "Forward8_AttnW_CNN")

def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()

def expected_shapes(meta):
    k = meta["model"]
    c, hidden = k["base_channels"], k["head_hidden"]
    shapes = {"signs_e1": (8,), "signs_e2": (8,)}
    for layer in range(k["num_layers"]):
        ci = 1 if layer == 0 else c
        root = f"blocks.{layer}"
        shapes.update({root + ".conv.weight": (c, ci, 3, 3), root + ".conv.bias": (c,),
                       root + ".norm.weight": (1, c, 1, 1), root + ".norm.bias": (1, c, 1, 1)})
        if ci != c:
            shapes[root + ".skip.weight"] = (c, ci, 1, 1)
    depth = k.get("head_num_hidden_layers", 3)
    for root in (["head_shared"] if k["shared_head"] else ["head_e1", "head_e2"]):
        for layer in range(depth):
            shapes[f"{root}.net.{1+2*layer}.weight"] = (hidden, c if layer == 0 else hidden)
        shapes[f"{root}.net.{1+2*depth}.weight"] = (1, hidden)
    if meta["architecture"] == "Forward8_AttnW_CNN":
        d = k["attn_dim"]
        for key, ci, co in (("token_projection", c+2, d), ("qkv", d, 3*d),
                            ("attn_output", d, d), ("ffn.0", d, 2*d),
                            ("ffn.2", 2*d, d), ("kernel_head", d, 1)):
            shapes[key + ".weight"] = (co, ci)
            shapes[key + ".bias"] = (co,)
        for key in ("attn_norm", "ffn_norm"):
            shapes[key + ".weight"] = (d,)
            shapes[key + ".bias"] = (d,)
        shapes["kernel_blur"] = (1, 1, 5, 5)
    return shapes

def validate(meta, weights):
    if meta.get("format_version") != FORMAT_VERSION:
        raise ValueError("Unsupported model format_version")
    if meta.get("architecture") not in ARCHITECTURES:
        raise ValueError("Only Forward8_fixW_CNN and Forward8_AttnW_CNN are supported")
    numerics = meta.get("numerics", {})
    expected_numerics = {"layernorm_eps": 1e-5, "softplus_beta": 10,
                         "softplus_threshold": 20, "gelu": "exact_erf"}
    if any(numerics.get(key) != value for key,value in expected_numerics.items()):
        raise ValueError("Unsupported numeric operator semantics")
    k = meta["model"]
    ints = ["in_dim", "out_dim", "base_channels", "head_hidden", "num_layers"]
    if meta["architecture"] == "Forward8_AttnW_CNN":
        ints += ["attn_grid_size", "attn_dim", "attn_heads", "head_num_hidden_layers", "suppress_order"]
    for key in ints:
        if type(k.get(key)) is not int or k[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if k["in_dim"] != 1 or k["out_dim"] != 2:
        raise ValueError("Only one input channel and two shape outputs are supported")
    size = meta.get("stamp_size")
    if type(size) is not int or not 1 <= size <= 4096:
        raise ValueError("stamp_size must be an integer in [1,4096]")
    if meta["architecture"] == "Forward8_fixW_CNN" and size % 2:
        raise ValueError("Legacy fixW odd-size cropping is not supported")
    positive = ["nl_sigma"]
    if meta["architecture"] == "Forward8_fixW_CNN":
        positive += ["gw_sigma"]
    else:
        positive += ["r_cut"]
        if k["attn_dim"] % k["attn_heads"] or k["suppress_order"] % 2:
            raise ValueError("Invalid attention head count or taper order")
    for key in positive:
        if not np.isfinite(k[key]) or k[key] <= 0:
            raise ValueError(f"{key} must be positive and finite")
    if not np.isfinite(k["res_factor"]) or not 0 <= k["res_factor"] <= 1:
        raise ValueError("res_factor must be in [0,1]")
    for key in ("shared_head", "suppress"):
        if type(k.get(key)) is not bool:
            raise ValueError(f"{key} must be boolean")
    pre = meta["preprocessing"]
    for key in ("pixel_scale", "sigma_arcsec", "flim", "klim"):
        if not np.isfinite(pre[key]) or pre[key] <= 0:
            raise ValueError(f"preprocessing.{key} must be positive and finite")
    if not np.isclose(pre["klim"], pre["flim"] / pre["pixel_scale"], rtol=1e-12):
        raise ValueError("klim must equal flim / pixel_scale")
    shapes = expected_shapes(meta)
    if set(shapes) != set(weights):
        raise ValueError(f"Weight keys mismatch: missing={set(shapes)-set(weights)}, extra={set(weights)-set(shapes)}")
    for key, shape in shapes.items():
        a = np.asarray(weights[key])
        if a.shape != shape or a.dtype not in (np.dtype("float32"), np.dtype("float64")) or not np.isfinite(a).all():
            raise ValueError(f"Invalid weight {key}: expected finite float array {shape}, got {a.shape} {a.dtype}")
    for key, expected in (("signs_e1", [1,-1,1,-1,1,-1,1,-1]), ("signs_e2", [1,-1,1,-1,-1,1,-1,1])):
        if not np.array_equal(weights[key], expected):
            raise ValueError(f"Unsupported D4 convention in {key}")

def save_model(path, meta, weights):
    validate(meta, weights)
    path = Path(path)
    if path.suffix != ".npz":
        raise ValueError("Model filename must end in .npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = np.frombuffer(json.dumps(meta, sort_keys=True).encode("utf-8"), dtype=np.uint8)
    np.savez(path, metadata=encoded, **weights)

def load_model(path):
    with np.load(path, allow_pickle=False) as archive:
        meta = json.loads(archive["metadata"].tobytes().decode("utf-8"))
        weights = {k: np.ascontiguousarray(archive[k]) for k in archive.files if k != "metadata"}
    validate(meta, weights)
    return meta, weights
