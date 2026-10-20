"""Compare raw e with detection-weighted w*e on the same catalog objects."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalog', required=True)
    parser.add_argument('--out-dir', required=True)
    args = parser.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with np.load(args.catalog) as z:
        raw = z['raw_e'].astype(np.float64)
        weighted = z['shapes'].astype(np.float64)
    valid = np.isfinite(raw).all(1) & np.isfinite(weighted).all(1)
    raw, weighted = raw[valid], weighted[valid]
    values = [(np.linalg.norm(raw, axis=1), np.linalg.norm(weighted, axis=1)),
              (raw[:, 0], weighted[:, 0]), (raw[:, 1], weighted[:, 1])]
    titles = [r'Ellipticity modulus $|e|$', r'Component $e_1$', r'Component $e_2$']
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    summary = {'catalog': args.catalog, 'n_objects': len(raw), 'excluded_nonfinite': int((~valid).sum()),
               'definition': 'raw_e versus shapes = detection weight * raw_e; equal count per object; no training normalization',
               'distributions': {}}
    for col, ((a, b), title, key) in enumerate(zip(values, titles, ['modulus', 'e1', 'e2'])):
        limit = max(np.abs(a).max(), np.abs(b).max()) * 1.001
        bins = np.linspace(0 if col == 0 else -limit, limit, 201)
        summary['distributions'][key] = {}
        for data, label, color, name in [(a, 'Unweighted: e', '#2864b4', 'unweighted'),
                                         (b, 'Detection-weighted: w e', '#df7427', 'weighted')]:
            counts, _ = np.histogram(data, bins=bins)
            density = counts / (len(data) * np.diff(bins))
            for row in range(2):
                axes[row, col].stairs(density, bins, label=label, color=color, linewidth=1.7)
            summary['distributions'][key][name] = dict(mean=float(data.mean()), std=float(data.std()),
                quantiles=dict(zip(['p01', 'p50', 'p99'], np.quantile(data, [.01, .5, .99]).tolist())))
        axes[0, col].set_title(title)
        for row in range(2):
            axes[row, col].set_xlim(bins[0], bins[-1])
            axes[row, col].set_xlabel(['$|e|$ or $|w e|$', '$e_1$ or $w e_1$', '$e_2$ or $w e_2$'][col])
            axes[row, col].grid(alpha=.2)
        axes[1, col].set_yscale('log')
    axes[0, 0].set_ylabel('Probability density')
    axes[1, 0].set_ylabel('Probability density (log scale)')
    axes[0, 0].legend(frameon=False)
    fig.suptitle(f'AttnW: before / after detection weighting | Same {len(raw):,} objects\n'
                 'All train + validation + test objects; physical ellipticity, before training normalization', fontsize=13)
    for extension in ['png', 'pdf']:
        fig.savefig(out / f'ellipticity_weight_comparison.{extension}', dpi=180)
    (out / 'ellipticity_weight_comparison.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
