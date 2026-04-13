# ebsbi

Simulation-based inference of eclipsing binary star parameters using neural posterior estimation. The package couples [PHOEBE](http://phoebe-project.org/) forward models with normalizing-flow–based neural Bayesian inference ([NBI](https://github.com/kmzzhang/nbi)) to produce amortized posterior distributions over physical parameters from multi-survey light curves, broadband SEDs, and Gaia parallaxes.

## Installation

Create the conda environment and install the package:

```bash
conda env create -f requirements.yml
conda activate ebsbi
pip install -e .
```

## Project Structure

```
ebsbi/
├── src/ebsbi/          # Core library
│   ├── model.py        # PHOEBE-based forward model (light curves + SEDs)
│   ├── priors.py       # Prior distributions with isochrone-based empirical priors
│   ├── observational.py# Observational noise process (cadence, SED, Gaia parallax)
│   ├── engine.py       # NBI engine construction (flow + multi-channel featurizer)
│   ├── shards.py       # Memory-mapped shard dataset for large training sets
│   ├── phoebe_wrapper.py # PHOEBE bundle management and SED computation
│   ├── stellar_utils.py# Isochrone interpolation and stellar parameter utilities
│   ├── constants.py    # Physical constants and PHOEBE twig mappings
│   ├── config.py       # YAML configuration loader
│   └── utils.py        # Phase/noise bank I/O
├── scripts/
│   ├── train_nbi.py    # Training script (config-driven, W&B support)
│   ├── run_inference.py# Posterior sampling from a trained checkpoint
│   ├── generate_training_data.py # Forward-model simulation to shards
│   └── convert_npz_to_npy.py    # Convert .npz shards to mmap-friendly .npy dirs
├── models/             # Trained model checkpoints (Git LFS)
├── examples/           # Example data (Git LFS)
└── docs/               # Configuration files and design documents
```

## Quickstart

### Training

Generate training data shards with the forward model, then train the normalizing flow:

```bash
python scripts/generate_training_data.py --config docs/base.yml
python scripts/train_nbi.py --config docs/base.yml
```

### Inference

Run posterior inference on an observation using a trained checkpoint:

```bash
python scripts/run_inference.py \
    --config docs/base.yml \
    --checkpoint models/best_model.pth \
    --observation examples/training_shard_000000.npz \
    --index 0 \
    --n-posterior 10000
```

This loads the observation from a shard, applies the observational noise process (cadence noise, SED noise, Gaia parallax sampling), then draws posterior samples from the trained flow. Output is saved as a compressed `.npz` with samples and a summary table:

```
Parameter             Mean        Std       2.5%      97.5%       True
----------------------------------------------------------------------
msum               2.3079     0.3036     1.7129     2.8959     2.1953
q                  0.7421     0.1822     0.3255     0.9786     0.9443
log_age            8.6467     0.7247     7.0283     9.6601     9.0459
...
```

## Model Architecture

The network uses a multi-channel featurizer with a masked normalizing flow:

- **4 light-curve channels** (ASAS-SN g, ASAS-SN V, ZTF g, ZTF r): each processed by an independent ResNet, input shape `(4, L)` = [phase, flux, sigma, mask]
- **1 SED channel** (23 broadband filters from GALEX to WISE): ResNet featurizer, input shape `(3, 23)` = [flux, sigma, mask]
- **1 metadata channel** (period, parallax, parallax error, Gaia coverage flag): GRU featurizer, input shape `(4, 1)`

The featurizer outputs are concatenated and passed to a Mixture-of-Gaussians normalizing flow that models the 16-dimensional posterior over:

| Parameter | Description |
|-----------|-------------|
| `msum` | Total system mass (M_sun) |
| `q` | Mass ratio |
| `log_age` | Log stellar age |
| `metallicity` | [Fe/H] |
| `ebv` | E(B-V) extinction |
| `cosi` | Cosine of inclination |
| `rsumfrac` | Sum of fractional radii |
| `ecc` | Eccentricity |
| `per0` | Argument of periastron (deg) |
| `distance` | Distance (pc) |
| `teff1`, `teff2` | Effective temperatures (K) |
| `r1`, `r2` | Radii (R_sun) |
| `log_lum1`, `log_lum2` | Log luminosities (L_sun) |

## License

MIT
