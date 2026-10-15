#!/usr/bin/env python3
"""Train the blended-stamp CNN against isolated RDSM labels.

This is the in-tree copy of the RDSM student trainer, extended with the two
``Forward8_fixW_CNN`` Gaussian weight-map sigmas so a sigma sweep can be
launched from ``scripts/sigma_test/train_rdsm_sigma.sh``.

    python -u src/train_blended_rdsm.py --run-dir models/blended_rdsm_runs/blended_rdsm_s8 \
        --nl-sigma 8 --gw-sigma 8

The chosen sigmas (and the rest of the architecture) are written into the run's
``config.json``, which is the source of truth used later by
``src/blending/test_func.py::build_cnn_from_run_dir`` to rebuild the *same*
architecture when measuring.  Everything else -- labels, exposure split,
augmentation, optimizer, scheduler and metrics -- is unchanged.

Run from the repository root (``codes/single_galaxies``) so that ``src`` is
importable, matching the other launchers.
"""
import argparse
from pathlib import Path

# ``src/train_blended_rdsm.py`` -> CODE_DIR = codes/single_galaxies
CODE_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = CODE_DIR.parents[1]


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def positive_float(value):
    import math
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return value


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--num-samples', '--num_samples', type=positive_int, default=100_000,
                        help='Exact number of usable training objects (default: 100000).')
    parser.add_argument('--num-epoch', '--num_epoch', '--epochs', type=positive_int, default=10)
    parser.add_argument('--save-epochs', type=positive_int, nargs='*', default=[],
                        help='Also save epoch_NNN.pth after these epochs; does not change training.')
    parser.add_argument('--batch-size', '--batch_size', type=positive_int, default=750)
    parser.add_argument('--val-exposures', type=positive_int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--device', default='auto', help='auto, cpu, cuda, or cuda:N')
    parser.add_argument('--blended-dir', type=Path,
                        default=Path('/work/hdd/bfcn/wenyinli/blended_training_q_n0594'))
    parser.add_argument('--label-path', type=Path,
                        default=PROJECT_ROOT / 'codes/opt_e/datasets/isolated_rdsm_best_m002/rdsm_catalog.npz')
    parser.add_argument('--e1-column', default='rdsm_e1',
                        help='Label catalog column holding the e1 target (default: %(default)s).')
    parser.add_argument('--e2-column', default='rdsm_e2',
                        help='Label catalog column holding the e2 target (default: %(default)s).')
    parser.add_argument('--valid-column', default='rdsm_valid',
                        help='Label catalog column holding the usable mask (default: %(default)s).')
    parser.add_argument('--run-dir', type=Path,
                        default=CODE_DIR / 'models/blended_rdsm_runs/blended_rdsm_s16')
    parser.add_argument('--nl-sigma', '--nl_sigma', type=positive_float, default=16.0,
                        help='Gaussian sigma of the brightness-normalization weight map '
                             '(Forward8_fixW_CNN.nl_sigma, default: 16).')
    parser.add_argument('--gw-sigma', '--gw_sigma', type=positive_float, default=16.0,
                        help='Gaussian sigma of the weighted-pooling weight map at feature '
                             'resolution (Forward8_fixW_CNN.gw_sigma, default: 16).')
    args = parser.parse_args()
    import math
    if not math.isfinite(args.lr) or args.lr <= 0:
        parser.error('--lr must be finite and positive')
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        parser.error('--weight-decay must be finite and nonnegative')
    if any(epoch > args.num_epoch for epoch in args.save_epochs):
        parser.error('--save-epochs cannot exceed --num-epoch')
    return args


def main():
    args = parse_args()
    if args.run_dir.exists() and any(args.run_dir.iterdir()):
        raise FileExistsError(f'Choose an empty --run-dir to preserve existing output: {args.run_dir}')
    from pathlib import Path
    import sys
    import json
    import random
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import torch
    from torch.utils.data import IterableDataset, DataLoader

    sys.path.insert(0, str(CODE_DIR))
    from src.architecture.forward8_CNN import Forward8_fixW_CNN
    from src.datasets.single_gal_dataset import rotate_image_90, rotate_spin2, flip_spin2

    BLENDED_DIR = args.blended_dir.resolve()
    from scripts.prepare_blended_q import validate_dataset, read_q_images
    preprocessing = validate_dataset(BLENDED_DIR)
    LABEL_PATH = args.label_path.resolve()
    RUN_DIR = args.run_dir.resolve()
    SEED, EPOCHS, BATCH_SIZE = args.seed, args.num_epoch, args.batch_size
    LR, WEIGHT_DECAY = args.lr, args.weight_decay
    NUM_WORKERS = 0
    DEVICE = torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device)
    MODEL_KWARGS = dict(num_layers=5, base_channels=32, res_factor=0.1,
                        nl_sigma=args.nl_sigma, gw_sigma=args.gw_sigma)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    print('device:', DEVICE)
    print('model kwargs:', MODEL_KWARGS)
    print('run dir:', RUN_DIR)


    label_columns = ('sample_id', 'exp_id', 'det_id', 'catsim_index',
                     args.valid_column, args.e1_column, args.e2_column)
    with np.load(LABEL_PATH) as saved:
        missing = [k for k in label_columns if k not in saved.files]
        if missing:
            raise KeyError(f'{LABEL_PATH} is missing label columns {missing}; '
                           f'available={sorted(saved.files)}')
        labels = pd.DataFrame({k: saved[k] for k in label_columns})
        # Teacher provenance is bookkeeping only: recorded in config.json,
        # never used to build targets.  A truth-label catalog has none.
        teacher_checkpoint = (str(saved['checkpoint'].item())
                              if 'checkpoint' in saved.files else 'n/a')
        teacher_norm = (float(saved['rdsm_norm'])
                        if 'rdsm_norm' in saved.files else float('nan'))
    labels = labels.rename(columns={args.valid_column: 'valid',
                                    args.e1_column: 'e1',
                                    args.e2_column: 'e2'})
    if labels['sample_id'].duplicated().any():
        raise ValueError('Duplicate sample_id in the label catalog')
    labels['usable'] = labels['valid'] & np.isfinite(labels[['e1', 'e2']]).all(axis=1)
    print(f'label catalog : {LABEL_PATH}')
    print(f'label columns : e1={args.e1_column} e2={args.e2_column} '
          f'valid={args.valid_column}  usable={int(labels["usable"].sum()):,}/{len(labels):,}')
    label_groups = {int(exp): group.set_index('sample_id') for exp, group in labels.groupby('exp_id')}
    shard_paths = {int(p.stem.split('_')[-1]): p for p in BLENDED_DIR.glob('blended_*.npz')}
    if set(shard_paths) != set(label_groups):
        raise ValueError('Blended and RDSM exposure sets differ')
    # Fully deterministic split (no randomness): process exposures in ascending id
    # order. Accumulate whole exposures into the training partition until it holds
    # at least TRAIN_OBJECTS (~100k) usable objects, then take the next
    # VAL_EXPOSURES exposures as validation; every remaining exposure is test.
    TRAIN_OBJECTS = args.num_samples
    VAL_EXPOSURES = args.val_exposures
    usable_per_exposure = {int(exp): int(group['usable'].sum()) for exp, group in labels.groupby('exp_id')}
    exp_order = sorted(shard_paths)  # natural exposure order 0, 1, 2, ...
    train_ids, n_train_objects = [], 0
    for exposure in exp_order:
        train_ids.append(int(exposure))
        n_train_objects += usable_per_exposure[int(exposure)]
        if n_train_objects >= TRAIN_OBJECTS:
            break
    consumed = set(train_ids)
    remainder = [int(e) for e in exp_order if int(e) not in consumed]
    val_ids = remainder[:VAL_EXPOSURES]
    test_ids = remainder[VAL_EXPOSURES:]
    split_exposures = dict(
        train=np.asarray(train_ids, dtype=int),
        val=np.asarray(val_ids, dtype=int),
        test=np.asarray(test_ids, dtype=int),
    )
    print(f'Training partition: {len(train_ids)} exposures, {n_train_objects} usable objects (target {TRAIN_OBJECTS}).')
    print(pd.DataFrame([
        dict(split=s, exposures=len(ids), objects=sum(int(label_groups[int(e)]['usable'].sum()) for e in ids))
        for s, ids in split_exposures.items()
    ]))

    if n_train_objects < TRAIN_OBJECTS:
        raise ValueError(f'Requested {TRAIN_OBJECTS} training objects, but only {n_train_objects} are usable')
    if len(val_ids) != VAL_EXPOSURES or not test_ids:
        raise ValueError('Not enough exposures: reduce --num-samples or --val-exposures to leave validation and test data')
    for name in ('val', 'test'):
        if not sum(usable_per_exposure[int(e)] for e in split_exposures[name]):
            raise ValueError(f'Empty usable {name} partition')
    # Select a fixed set before per-epoch shuffling; hold out entire exposures.
    remaining = TRAIN_OBJECTS
    selected_parts = []
    for exposure in train_ids:
        group = label_groups[exposure]
        selected = group.index[group['usable']].to_numpy()[:remaining]
        group['selected_train'] = group.index.isin(selected)
        selected_parts.append(selected)
        remaining -= len(selected)
    train_sample_ids = np.concatenate(selected_parts)
    print(f'Using exactly {len(train_sample_ids):,} training samples per epoch; batch size {BATCH_SIZE}, epochs {EPOCHS}.')


    class BlendedRDSMStream(IterableDataset):
        def __init__(self, exposure_ids, augment=False):
            self.exposure_ids = np.asarray(exposure_ids)
            self.augment = augment
            self.epoch = 0

        def __iter__(self):
            rng = np.random.default_rng(SEED + self.epoch)
            ids = self.exposure_ids.copy()
            if self.augment:
                rng.shuffle(ids)
            for exposure in ids:
                with np.load(shard_paths[int(exposure)]) as saved:
                    sid = saved['sample_id']
                    if len(np.unique(sid)) != len(sid):
                        raise ValueError(f'Duplicate blended IDs in exposure {exposure}')
                    group = label_groups[int(exposure)]
                    if len(group) != len(sid) or not np.isin(sid, group.index).all():
                        raise ValueError(f'Unmatched RDSM IDs in exposure {exposure}')
                    aligned = group.loc[sid]
                    for key in ('exp_id', 'det_id', 'catsim_index'):
                        if not np.array_equal(saved[key], aligned[key].to_numpy()):
                            raise ValueError(f'{exposure}: {key} mismatch')
                    images = read_q_images(saved, preprocessing)
                if images.shape != (len(sid), 64, 64) or not np.isfinite(images).all():
                    raise ValueError(f'{exposure}: invalid blended images')
                targets = aligned[['e1', 'e2']].to_numpy(dtype=np.float32)
                mask = aligned['selected_train'] if self.augment else aligned['usable']
                rows = np.flatnonzero(mask.to_numpy())
                if self.augment:
                    rng.shuffle(rows)
                for row in rows:
                    image = torch.from_numpy(images[row].copy()).unsqueeze(0)
                    target = torch.from_numpy(targets[row].copy())
                    if self.augment:
                        k = int(rng.integers(4))
                        image, target = rotate_image_90(image, k), rotate_spin2(target, k)
                        if rng.random() < 0.5:
                            image, target = torch.flip(image, dims=(-1,)), flip_spin2(target)
                    yield image, target

    # Stream one decompressed shard at a time.

    streams = {s: BlendedRDSMStream(ids, augment=(s == 'train')) for s, ids in split_exposures.items()}
    loaders = {s: DataLoader(ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
                            pin_memory=(DEVICE.type == 'cuda'), drop_last=False)
               for s, ds in streams.items()}


    model = Forward8_fixW_CNN(**MODEL_KWARGS).to(DEVICE)


    from tqdm.auto import tqdm

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    BEST_PATH, LAST_PATH = RUN_DIR / 'best.pth', RUN_DIR / 'last.pth'
    if BEST_PATH.exists() or LAST_PATH.exists():
        raise FileExistsError('Choose a new RUN_DIR to keep existing checkpoints')
    config = dict(num_samples=TRAIN_OBJECTS, val_exposures=VAL_EXPOSURES, device=str(DEVICE), seed=SEED, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR,
                  weight_decay=WEIGHT_DECAY, model=MODEL_KWARGS, labels=str(LABEL_PATH),
                  teacher_checkpoint=teacher_checkpoint, teacher_norm=teacher_norm,
                  target_columns=[args.e1_column, args.e2_column], image_normalization='model internal',
                  blended_dir=str(BLENDED_DIR), image_preprocessing=preprocessing)
    config['save_epochs'] = sorted(set(args.save_epochs))
    (RUN_DIR / 'config.json').write_text(json.dumps(config, indent=2))
    np.savez(RUN_DIR / 'split_exposures.npz', **split_exposures)
    np.save(RUN_DIR / 'train_sample_ids.npy', train_sample_ids)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=1/3.162, patience=20, threshold=0.01)

    def run_regression_epoch(loader, training=False):
        model.train(training)
        total_squared_error, count = 0.0, 0
        iterator = tqdm(loader, desc='train' if training else 'val  ',
                        unit='batch', leave=False)
        with torch.set_grad_enabled(training):
            for images, target in iterator:
                images, target = images.to(DEVICE, non_blocking=True), target.to(DEVICE, non_blocking=True)
                prediction = model(images)
                loss = torch.nn.functional.mse_loss(prediction, target)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite regression loss')
                if training:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                total_squared_error += loss.item() * target.numel()
                count += target.numel()
                iterator.set_postfix(loss=f'{loss.item():.4g}')
        iterator.close()
        if count == 0:
            raise ValueError('Empty data partition')
        return total_squared_error / count

    history, best_val = [], float('inf')
    epoch_bar = tqdm(range(1, EPOCHS + 1), desc='epoch', unit='epoch')
    for epoch in epoch_bar:
        streams['train'].epoch = epoch
        train_mse = run_regression_epoch(loaders['train'], training=True)
        val_mse = run_regression_epoch(loaders['val'])
        scheduler.step(val_mse)
        lr = optimizer.param_groups[0]['lr']
        history.append(dict(epoch=epoch, train_mse=train_mse, val_mse=val_mse, lr=lr))
        if val_mse < best_val:
            best_val = val_mse
            torch.save(model.state_dict(), BEST_PATH)
        torch.save(model.state_dict(), LAST_PATH)
        if epoch in args.save_epochs:
            torch.save(model.state_dict(), RUN_DIR / f'epoch_{epoch:03d}.pth')
        pd.DataFrame(history).to_csv(RUN_DIR / 'history.csv', index=False)
        epoch_bar.set_postfix(train=f'{train_mse:.4g}', val=f'{val_mse:.4g}', lr=f'{lr:.3g}')
    epoch_bar.close()
    print('Best validation MSE:', best_val, '| weights:', BEST_PATH)


    # Evaluate the best validation checkpoint, not the last epoch.
    model.load_state_dict(torch.load(RUN_DIR / 'best.pth', map_location=DEVICE, weights_only=True))
    model.eval()
    squared_error, residual_sum, n_test = np.zeros(2), np.zeros(2), 0
    with torch.no_grad():
        for images, target in loaders['test']:
            prediction = model(images.to(DEVICE)).cpu().numpy()
            residual = prediction.astype(np.float64) - target.numpy().astype(np.float64)
            if not np.isfinite(residual).all():
                raise FloatingPointError('Nonfinite test residual')
            squared_error += np.square(residual.astype(np.float64)).sum(axis=0)
            residual_sum += residual.sum(axis=0, dtype=np.float64)
            n_test += len(target)
    if n_test == 0:
        raise ValueError('Empty test set')
    metrics = dict(n_test=n_test, rmse=(squared_error/n_test)**0.5,
                   mean_residual=residual_sum/n_test)
    print(metrics)
    (RUN_DIR / 'test_metrics.json').write_text(json.dumps(
        {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in metrics.items()}, indent=2))
    history_df = pd.read_csv(RUN_DIR / 'history.csv')
    history_df.plot(x='epoch', y=['train_mse', 'val_mse'], logy=True)
    plt.ylabel('RDSM label MSE')
    plt.savefig(RUN_DIR / 'loss_curve.png', bbox_inches='tight')
    plt.close('all')



if __name__ == '__main__':
    main()
