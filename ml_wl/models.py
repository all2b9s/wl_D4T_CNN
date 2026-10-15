"""Build a CNN from a yaml {arch, kwargs} block (src is imported lazily: CNN-side jobs only)."""


def build_model(arch, kwargs):
    from src.architecture import forward8_CNN
    cls = getattr(forward8_CNN, arch, None)
    if cls is None:
        raise ValueError(f"unknown architecture {arch!r} in src.architecture.forward8_CNN")
    return cls(**kwargs)
