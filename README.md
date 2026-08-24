# D₄CNN × AnaCal: Physics-Informed Machine Learning for Accurate and Precise Weak Lensing Shear Estimation

Official implementation of the paper
[*D₄CNN × AnaCal: Physics-Informed Machine Learning for Accurate and Precise Weak Lensing Shear Estimation*](https://arxiv.org/abs/2603.19046)
(Shurui Lin, Xiangchong Li, Ji Li, Shengcao Cao, Xin Liu, Yu-Xiong Wang).

This repository provides a **D₄-equivariant deep neural network** for galaxy shape measurement
whose architecture enforces symmetry under 90° rotations and mirror transformations, calibrated with
the **Analytical Calibration (AnaCal)** framework using the model's backpropagated gradients.

In LSST-like single-band simulations of isolated galaxies, this approach achieves
~10% lower shape noise than the traditional moment-based Fourier Power Function Shapelets (FPFS)
estimator in the high-noise regime (≈20% gain in effective galaxy number density), with multiplicative
biases $|m| < 10^{-3}$ (within the 0.2% LSST requirement) across a wide range of noise levels,
PSF sizes, ellipticities and magnitude selection cuts.

**Calibration framework:** [AnaCal — Analytic Calibration for Perturbation Estimation from Galaxy Images](https://github.com/mr-superonion/AnaCal/)
(Xiangchong Li et al.). AnaCal measures the shear response of shape estimators via the
[pixel shear response](https://ui.adsabs.harvard.edu/abs/2023MNRAS.521.4904L/abstract) — the
derivatives of pixel values w.r.t. applied shear distortions — propagated through the estimator
using quintuple numbers.

## Quick Start

```bash
git clone https://github.com/all2b9s/wl_D4T_CNN.git
cd wl_D4T_CNN
pip install torch numpy matplotlib galsim anacal pandas
```

Then open `example.ipynb` and follow the cells:

1. **Train** a `Forward8_fixW_CNN` on your dataset (or use the provided pretrained weights in `models/`)
2. **Predict** the shape of a simulated galaxy
3. Compute the **pixel response** and **shear response matrix** for calibration

For large-scale shape measurement, see `shape_measurement.py` and `cal_biases.py`.

## Repository Structure

```
.
├── shape_measurement.py   # Batch shear measurement with AnaCal-style calibration
├── cal_biases.py          # Multiplicative/additive bias estimation
├── example.ipynb          # End-to-end tutorial (training + inference + response)
├── models/                # Pretrained weights (.pth)
├── src/
│   ├── architecture/      # CNN model definitions (D4-equivariant, non-eq, ...)
│   ├── anacal/            # AnaCal pixel response & calibration utilities
│   ├── datasets/          # Data loaders & galaxy image simulators
│   └── training.py        # Training loop with data augmentation
└── scripts/               # SLURM job submission helpers
```

## Model Architecture

`Forward8_fixW_CNN` is a D4-equivariant CNN with:
- **8-way D4 feature maps** ($r_0, r_{90}, r_{180}, r_{270}$ and their mirrors)
- **Fixed-channel convolution** blocks (GeLU activation + residual connections)
- **Weighted global average pooling** with a Gaussian weight map
- **Bias-free MLP heads** for $e_1$ and $e_2$

The equivariance ensures that a 90° rotation of the input image rotates the predicted ellipticity
by the correct spin-2 transformation $(e_1 + i e_2) \to -i(e_1 + i e_2)$.

## Calibration with AnaCal

The shape estimator $f$ is pixel-wise differentiable. AnaCal computes the **pixel shear response**
$\partial I / \partial g$ (derivatives of pixel values w.r.t. applied shear) and propagates it through
the estimator. Combined with the model's backpropagated gradient, we obtain the shear response matrix

$$R_{ij} = \sum_{\mathrm{pixels}} \frac{\partial e_i}{\partial I} \cdot \frac{\partial I}{\partial g_j}$$

The response $R$ enters the calibrated shear estimator $\hat{g} = R^{-1} (e - c)$.
See [Li & Mandelbaum (2023)](https://ui.adsabs.harvard.edu/abs/2023MNRAS.521.4904L/abstract) for the
pixel shear response formalism and the [AnaCal repository](https://github.com/mr-superonion/AnaCal/)
for the full implementation.

## Citation

If you use this code, please cite the paper:

```bibtex
@article{lin2026d4cnn,
  title   = {D$_4$CNN$\times$AnaCal: Physics-Informed Machine Learning for Accurate and Precise Weak Lensing Shear Estimation},
  author  = {Lin, Shurui and Li, Xiangchong and Li, Ji and Cao, Shengcao and Liu, Xin and Wang, Yu-Xiong},
  journal = {APJ},
  year    = {2026}
}
```

## License

MIT (code); please refer to [AnaCal](https://github.com/mr-superonion/AnaCal/) for its license.

