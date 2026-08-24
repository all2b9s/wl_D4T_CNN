import csv
import torch
import torch.nn as nn
from typing import Literal, Optional, Tuple
from tqdm.auto import tqdm
from src.datasets.single_gal_dataset import make_loaders


@torch.no_grad()
def _compute_label_stats(train_loader, device, max_batches: int = 50) -> Tuple[torch.Tensor, torch.Tensor]:
    ys = []
    for i, (_, y) in enumerate(train_loader):
        ys.append(y)
        if (i + 1) >= max_batches:
            break
    y_all = torch.cat(ys, dim=0).to(device)
    mu = y_all.mean(dim=0)
    std = y_all.std(dim=0).clamp_min(1e-6)
    return mu, std

def _set_requires_grad(module: torch.nn.Module, flag: bool):
    for p in module.parameters():
        p.requires_grad_(flag)

def train_model_noneq(
    model,
    images_path="images.npy",
    csv_path="gt_info.csv",
    target: Literal["e","g"]="e",
    epochs: int = 100,
    batch_size: int = 256,
    num_workers: int = 8,
    huber_delta: float = 1.0,
    lr: float = 1e-3,
    wd: float = 0.0,
    device: Optional[str] = None,
    pt_path: Optional[str] = None,
    print_every: int = 1,

    # --- whitening ---
    use_whitening: bool = True,
    label_stats_max_batches: int = 50,

    # --- zero-point regularization ---
    use_zero_point_penalty: bool = True,
    zero_point_lambda: float = 0.05,

    # --- freezing ---
    freeze_backbone: bool = True,
    freeze_epochs: int = 10,  # first N epochs freeze backbone

    # --- NEW: progress bar, logging, AMP ---
    csv_log_path: Optional[str] = None,
    use_amp: bool = True,

):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # --- AMP setup ---
    use_bf16 = (use_amp and device == "cuda" and torch.cuda.is_bf16_supported())
    amp_dtype = torch.bfloat16 if use_bf16 else (torch.float16 if use_amp and device=="cuda" else None)
    scaler = torch.amp.GradScaler(device, enabled=(amp_dtype is torch.float16))

    train_loader, val_loader, test_loader = make_loaders(
        images_path, csv_path, batch_size, num_workers, target=target, augment=True
    )

    # --- CSV logger initialization ---
    if csv_log_path is not None:
        with open(csv_log_path, "w", newline="") as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(["epoch", "split", "loss", "lr"])

    def log_row(ep, split, loss_val, lr_val):
        if csv_log_path is not None:
            with open(csv_log_path, "a", newline="") as f:
                csv.writer(f).writerow([ep, split, loss_val, lr_val])

    # --- label stats (for whitening the LOSS only) ---
    mu = std = None
    if use_whitening:
        mu, std = _compute_label_stats(train_loader, device=device, max_batches=label_stats_max_batches)
        # mu is not needed for the residual (it cancels), but kept for logging / optional use
        print("[whitening] mu:", mu.detach().cpu(), "std:", std.detach().cpu())

    # --- loss ---
    huber = nn.HuberLoss(delta=huber_delta)

    def loss_fn_phys(pred, y):
        """
        pred, y are in physical units (e1,e2).
        We whiten ONLY the residual for stable optimization.
        """
        if use_whitening:
            r = (pred - y) / std
            # optional: also remove mean bias if you want (usually unnecessary):
            # r = (pred - y) / std  (mu cancels anyway)
            return huber(r, torch.zeros_like(r))
        else:
            return huber(pred, y)

    def zero_point_penalty(pred, y):
        """
        Penalize batch mean offset to suppress zero-point drift:
            E[pred] - E[y] -> 0
        """
        bias = pred.mean(dim=0) - y.mean(dim=0)
        if use_whitening:
            bias = bias / std
        return (bias * bias).mean()

    # --- optimizer & scheduler ---
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.316, patience=10, threshold=0.01, threshold_mode="rel"
    )

    # --- freezing setup ---
    # We try to freeze common attribute names. Adjust if your backbone name differs.
    backbone = None
    for name in ["blocks", "backbone", "trunk", "encoder"]:
        if hasattr(model, name):
            backbone = getattr(model, name)
            break

    if freeze_backbone and backbone is not None and freeze_epochs > 0:
        _set_requires_grad(backbone, False)
        print(f"[freeze] backbone '{name}' frozen for first {freeze_epochs} epochs")
    else:
        freeze_backbone = False  # nothing to freeze

    def run_epoch(loader, train: bool):
        model.train(train)
        total, n = 0.0, 0
        pbar = tqdm(loader, desc=f"{'Train' if train else 'Val'}", leave=False)
        with torch.set_grad_enabled(train):
            for imgs, y in pbar:
                imgs = imgs.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                with torch.amp.autocast(device, enabled=use_amp):
                    pred = model(imgs)  # <-- stays in physical e1,e2 units
                    task_loss = loss_fn_phys(pred, y)
                    if use_zero_point_penalty and zero_point_lambda > 0.0:
                        zp = zero_point_penalty(pred, y)
                        loss = task_loss + zero_point_lambda * zp
                    else:
                        loss = task_loss

                if train:
                    opt.zero_grad(set_to_none=True)
                    scaler.scale(loss).backward()
                    scaler.step(opt)
                    scaler.update()

                bs = imgs.size(0)
                total += float(loss) * bs
                n += bs
                
                # Update progress bar
                pbar.set_postfix({"loss": f"{float(loss):.5f}"})
        return total / max(n, 1)

    best_val = float("inf")
    for ep in range(1, epochs + 1):
        # unfreeze after freeze_epochs
        if freeze_backbone and (ep == freeze_epochs + 1) and (backbone is not None):
            _set_requires_grad(backbone, True)
            print(f"[freeze] backbone unfrozen at epoch {ep}")

        tr = run_epoch(train_loader, train=True)
        va = run_epoch(val_loader, train=False)

        scheduler.step(va)
        current_lr = opt.param_groups[0]["lr"]

        # Log to CSV
        log_row(ep, "train", tr, current_lr)
        log_row(ep, "val", va, current_lr)

        if ep % print_every == 0 or ep == 1:
            print(f"Epoch {ep:03d} | train {tr:.6f} | val {va:.6f} | lr {current_lr:.6g}")

        if current_lr < 1e-5:
            print("Early stopping: learning rate fell below 1e-5.")
            break

        if va < best_val:
            best_val = va
            if pt_path is not None:
                torch.save(model.state_dict(), pt_path)

    test_loss = run_epoch(test_loader, train=False)
    log_row(epochs, "test", test_loss, current_lr)
    print(f"Test loss: {test_loss:.6f}")

    return model
