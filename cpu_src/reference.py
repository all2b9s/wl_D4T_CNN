"""Independent PyTorch autograd reference; never imported by CPU runtime."""
import numpy as np
from .model_io import load_model

def torch_model(path, precision="float32", device="cpu"):
    import torch
    from src.architecture import forward8_CNN
    meta, weights = load_model(path)
    dtype = getattr(torch, precision)
    model = getattr(forward8_CNN, meta["architecture"])(**meta["model"]).to(device=device, dtype=dtype)
    model.load_state_dict({k: torch.from_numpy(v.copy()).to(device=device, dtype=dtype) for k, v in weights.items()}, strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model

def measure(path, q, precision="float32", n_tangent=4, device="cpu", batch_size=8, _model=None):
    import torch
    from .api import Measurement
    if n_tangent not in (2,4) or batch_size < 1:
        raise ValueError("Invalid tangent count or batch size")
    model = torch_model(path, precision, device) if _model is None else _model
    dtype = getattr(torch, precision)
    q = np.asarray(q)
    es, derivatives = [], []
    for start in range(0, len(q), batch_size):
        tensor = torch.as_tensor(np.ascontiguousarray(q[start:start+batch_size]), device=device, dtype=dtype)
        x = tensor[:, :1].detach().requires_grad_(True)
        with torch.enable_grad():
            e = model(x)
            jac = []
            for component in range(2):
                gradient, = torch.autograd.grad(e[:, component].sum(), x, retain_graph=component==0)
                jac.append((gradient * tensor[:, 1:1+n_tangent]).sum(dim=(-1,-2)))
        es.append(e.detach().cpu().numpy())
        derivatives.append(torch.stack(jac, dim=1).detach().cpu().numpy())
    if es:
        e, derivative = np.concatenate(es), np.concatenate(derivatives)
    else:
        e, derivative = np.empty((0,2),dtype=precision), np.empty((0,2,n_tangent),dtype=precision)
    return Measurement(e, derivative[..., :2], derivative[..., 2:] if n_tangent==4 else None,
                       {"backend": "torch_autograd", "precision": precision, "device": device})

def synthetic_q(n=4, size=64, seed=42):
    """Asymmetric low-flux stamps with independent, nonzero test directions."""
    rng = np.random.default_rng(seed)
    y,x = np.mgrid[:size,:size].astype(float)
    x -= (size-1)/2
    y -= (size-1)/2
    q = rng.normal(0, .002, (n,5,size,size))
    for i in range(n):
        q[i,0] += .025*np.exp(-((x-.7)**2/15+(y+1.1)**2/27))
        q[i,0] += .008*np.exp(-((x+2.3)**2/5+(y-.8)**2/9))
        iy,ix = np.gradient(q[i,0])
        q[i,1] += -x*ix+y*iy
        q[i,2] += -y*ix-x*iy
        q[i,3] += ix/.2
        q[i,4] += iy/.2
    return q

def primal_trace(path, q, precision="float64"):
    import torch
    model = torch_model(path, precision)
    result = {}
    handles = []
    def hook(name):
        def record(module, inputs, out):
            out = out.detach().cpu()
            if out.ndim == 4:
                result[name] = out[0].permute(1,2,0).reshape(-1,out.shape[1]).numpy()
        return record
    for i, block in enumerate(model.blocks):
        for child, label in ((block.conv, "conv"), (block.norm, "norm"), (block.act, "gelu"), (block, "out")):
            handles.append(child.register_forward_hook(hook(f"blocks.{i}.{label}")))
    tensor = torch.as_tensor(q[None,:1],dtype=getattr(torch,precision))
    with torch.no_grad():
        weight = model._get_cached_weight(tensor.shape[-2:],model.nl_sigma,tensor.device,tensor.dtype)
        m = (tensor*weight).sum((1,2,3),keepdim=True)*100
        norm = 1+torch.nn.functional.softplus(m-1,beta=10,threshold=20)
        result["normalized"] = (tensor/norm)[0].permute(1,2,0).reshape(-1,1).numpy()
        model(tensor)
        fields = {"scalar_field": model.feats.mean(0), "f_mean_e1": model.f_mean_e1, "f_mean_e2": model.f_mean_e2}
        for name, out in fields.items():
            result[name] = out[0].permute(1,2,0).reshape(-1,out.shape[1]).numpy()
        if hasattr(model, "pool_kernel"):
            result["kernel"] = model.pool_kernel[0].permute(1,2,0).reshape(-1,1).numpy()
    for handle in handles:
        handle.remove()
    return result
