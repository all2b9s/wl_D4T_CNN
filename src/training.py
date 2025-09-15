import argparse, os, random
import torch
from torch import nn
import torch.nn.functional as F
import random
from typing import Callable, Literal, Optional
import optuna
from src.datasets.single_gal_dataset import make_loaders
from src.architecture.single_gal_CNN import D4T_CNN_GeLU

def train_model(
    model,
    images_path="images.npy",
    csv_path="gt_info.csv",
    target: Literal["e","g"]="e",
    epochs: int = 100,
    batch_size: int = 256,
    num_workers: int = 8,
    lr: float = 1e-3,
    wd: float = 1e-4,
    device: str = None,
    pt_path: str = None,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    train_loader, val_loader, test_loader = make_loaders(
        images_path, csv_path, batch_size, num_workers, target=target, augment=True
    )

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    # Halve LR if val loss doesn't improve by >1% for 20 epochs
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt,
        mode="min",
        factor=1/3.162,
        patience=20,
        threshold=0.01,          # 1% relative improvement required
        threshold_mode="rel",
    )

    loss_fn = nn.MSELoss()

    def run_epoch(loader, train=True):
        model.train(train)
        total_loss, n = 0.0, 0
        with torch.set_grad_enabled(train):
            for imgs, y in loader:
                imgs = imgs.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                pred = model(imgs)
                loss = loss_fn(pred, y)

                if train:
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()

                bs = imgs.size(0)
                total_loss += loss.item() * bs
                n += bs
        return total_loss / max(n, 1)

    best_val = float("inf")
    for ep in range(1, epochs + 1):
        tr = run_epoch(train_loader, train=True)
        va = run_epoch(val_loader, train=False)

        # Step the plateau scheduler on validation loss
        scheduler.step(va)

        # Early stop if LR falls below 1e-5
        current_lr = opt.param_groups[0]["lr"]
        print(f"Epoch {ep:03d} | train {tr:.6f} | val {va:.6f} | lr {current_lr:.6g}")
        if current_lr < 1e-5:
            print("Early stopping: learning rate fell below 1e-5.")
            break

        if va < best_val:
            best_val = va
            if pt_path is not None:
                torch.save(model.state_dict(), pt_path)

    # quick test evaluation
    test_loss = run_epoch(test_loader, train=False)
    print(f"Test loss: {test_loss:.6f}")

    return model

class OptunaTrainer:
    def __init__(
        self,
        images_path: str = "images.npy",
        csv_path: str = "gt_info.csv",
        target: Literal["e", "g"] = "e",
        epochs: int = 100,
        num_workers: int = 8,
        device: Optional[str] = None,
        seed: int = 42,
    ):
        """
        Optuna hyperparameter tuner for PyTorch models.

        Args:
            model_builder: function num_layers -> nn.Module
            images_path: path to npy image data
            csv_path: path to labels CSV
            target: 'e' or 'g' (task target)
            epochs: max epochs per trial
            num_workers: dataloader workers
            device: 'cuda', 'cpu', or None (auto-detect)
            seed: random seed
        """
        self.images_path = images_path
        self.csv_path = csv_path
        self.target = target
        self.epochs = epochs
        self.num_workers = num_workers
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.startswith("cuda") and torch.cuda.is_available():
            # Ensure this process uses the intended GPU
            dev_index = 0 if self.device == "cuda" else int(self.device.split(":")[1])
            torch.cuda.set_device(dev_index)

        self.seed = seed

        self._set_seed(seed)

    def _set_seed(self, s: int):
        random.seed(s)
        torch.manual_seed(s)
        torch.cuda.manual_seed_all(s)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def _objective(self, trial: optuna.trial.Trial) -> float:
        # Search space
        lr = trial.suggest_float("lr", 2e-5, 5e-3, log=True)
        wd = trial.suggest_float("wd", 1e-5, 1e-3, log=True)
        bs = 512 #trial.suggest_categorical("batch_size", [64, 128, 256, 512])

        # model params
        num_layers = trial.suggest_int("num_layers", 2, 8)
        num_filters = trial.suggest_categorical("CNN_filters", [16, 32, 64, 128])
        num_MLPsize = trial.suggest_categorical("MLPsize", [64, 128, 256, 512])


        # Data loaders
        train_loader, val_loader, _ = make_loaders(
            self.images_path,
            self.csv_path,
            bs,
            self.num_workers,
            target=self.target,
            augment=True,
        )

        # Model / optimizer / scheduler
        model = D4T_CNN_GeLU(base_channels=num_filters, 
                             head_hidden= num_MLPsize, 
                             num_layers=num_layers).to(self.device)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode="min",
            factor=0.5,
            patience=20,
            threshold=0.01,  # 1% relative improvement needed
            threshold_mode="rel",
        )
        loss_fn = nn.MSELoss()

        def run_epoch(loader, train: bool):
            model.train(train)
            total_loss, n = 0.0, 0
            with torch.set_grad_enabled(train):
                for imgs, y in loader:
                    imgs = imgs.to(self.device, non_blocking=True)
                    y = y.to(self.device, non_blocking=True)

                    pred = model(imgs)
                    loss = loss_fn(pred, y)

                    if train:
                        opt.zero_grad(set_to_none=True)
                        loss.backward()
                        opt.step()

                    bs_local = imgs.size(0)
                    total_loss += loss.item() * bs_local
                    n += bs_local
            return total_loss / max(n, 1)

        best_val = float("inf")

        for ep in range(1, self.epochs + 1):
            tr = run_epoch(train_loader, train=True)
            va = run_epoch(val_loader, train=False)
            if ep%10 == 0 or ep == 1:
                print(f"Trial {trial.number} Epoch {ep:03d} | train {tr:.6f} | val {va:.6f} | lr {opt.param_groups[0]['lr']:.6g}")
            # Report for pruning
            trial.report(va, step=ep)
            if trial.should_prune():
                raise optuna.TrialPruned(f"Pruned at epoch {ep}")

            scheduler.step(va)

            if va < best_val:
                best_val = va

            # Early stop if LR drops too low
            if opt.param_groups[0]["lr"] < 5e-6:
                break

        return best_val

    def run(
        self,
        n_trials: int = 15,
        study_name: Optional[str] = None,
        storage: str = "sqlite:///D4T_cnn.db",
        show_progress_bar: bool = False,
        warmup_trials: int = 5,
    ):
        """Run Optuna hyperparameter optimization."""
        existing_trials = 0
        try:
            existing = optuna.load_study(study_name=study_name, storage=storage)
            # Count all recorded trials (completed, pruned, failed, etc.)
            existing_trials = len(existing.trials)
        except Exception:
            existing_trials = 0  # no DB/study yet

        # Warm-up remaining after accounting for existing trials
        warmup_remaining = max(0, warmup_trials - existing_trials)

        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=warmup_remaining ,  # don't prune first N trials
            n_warmup_steps=30,               # allow a few epochs before pruning inside later trials
        )
        sampler = optuna.samplers.TPESampler(
            seed=self.seed,
            n_startup_trials=warmup_remaining ,  # random sampling for first N trials
        )
        try:
            study = optuna.create_study(
                direction="minimize",
                sampler=sampler,
                pruner=pruner,
                study_name=study_name,
                storage="sqlite:///D4T_cnn.db",
                load_if_exists=True, 
            )
            print('Created new study.')
        except Exception:
            study = optuna.load_study(study_name=study_name, storage=storage)
            print('Loaded existing study.')

        study.optimize(
            self._objective, n_trials=n_trials, show_progress_bar=show_progress_bar
        )

        print("Best trial:")
        print(f"  value (best val loss): {study.best_trial.value:.6f}")
        print("  params:")
        for k, v in study.best_trial.params.items():
            print(f"    {k}: {v}")

        return study

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--images", type=str, required=True)
    p.add_argument("--csv", type=str, required=True)
    p.add_argument("--target", type=str, default="e", choices=["e", "g"])
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--study-name", type=str, default="D4T_cnn_tuning")
    p.add_argument("--storage", type=str, default="sqlite:///D4T_cnn.db")
    p.add_argument("--n-trials", type=int, default=25)
    p.add_argument("--device", type=str, default=None)  # e.g. "cuda:0"
    p.add_argument("--seed", type=int, default=520)
    args = p.parse_args()

    # If you restrict visibility outside (CUDA_VISIBLE_DEVICES=<one gpu>),
    # then using "cuda" is enough (it will refer to that single visible GPU).
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    tuner = OptunaTrainer(
        images_path=args.images,
        csv_path=args.csv,
        target=args.target,
        epochs=args.epochs,
        num_workers=args.num_workers,
        device=device,
        seed=args.seed,
    )
    tuner.run(
        n_trials=args.n_trials,
        study_name=args.study_name,
        storage=args.storage,
        show_progress_bar=True,
        warmup_trials=15,
    )

if __name__ == "__main__":
    main()