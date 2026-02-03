#!/usr/bin/env python
"""Generate training samples for testing NBI integration."""
import sys
import os
import numpy as np
import h5py
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from ebsbi.model import EBModel

def load_phase_bank(filename):
    """Load full phase bank from HDF5 file."""
    bank = {}
    with h5py.File(filename, "r") as f:
        for survey in f.keys():
            grp = f[survey]
            dset = grp["phases"]
            phase_list = []
            for i in range(len(dset)):
                phase_list.append(np.array(dset[i], dtype=np.float64))
            bank[survey] = phase_list
    return bank

def load_noise_bank(filename):
    """Load full noise bank from HDF5 file."""
    bank = {}
    with h5py.File(filename, "r") as f:
        for survey in f.keys():
            grp = f[survey]
            dset = grp["noise"]
            noise_list = []
            for i in range(len(dset)):
                noise_list.append(np.array(dset[i], dtype=np.float64))
            bank[survey] = noise_list
    return bank

print("="*70)
print("GENERATING TRAINING SAMPLES")
print("="*70)

# Create output directory
output_dir = Path("training_samples_test")
output_dir.mkdir(exist_ok=True)
print(f"\nOutput directory: {output_dir.absolute()}")

# Parameter ranges for sampling
param_ranges = {
    'teff1': (5000, 8000),      # K
    'teff2': (4000, 7000),      # K
    'r1': (0.5, 2.0),           # Rsun
    'r2': (0.3, 1.5),           # Rsun
    'msum': (1.5, 3.0),         # Msun
    'q': (0.5, 1.0),            # mass ratio
    'period': (1.0, 10.0),      # days
    'incl': (75.0, 90.0),       # degrees
    'distance': (50.0, 500.0),  # pc
    'ebv': (0.0, 0.3),          # mag
    'metallicity': (-0.5, 0.5), # [Fe/H]
}

params_dict = {k: None for k in param_ranges.keys()}

# Load real phase/noise banks from HDF5 files
phase_bank_path = "/global/cfs/cdirs/m2218/jrblaum/research/eclipsing_binaries_lfi/data/phase_bank.h5py"
noise_bank_path = "/global/cfs/cdirs/m2218/jrblaum/research/eclipsing_binaries_lfi/data/noise_bank.h5py"

print("\nLoading phase and noise banks...")
phase_bank = load_phase_bank(phase_bank_path)
noise_bank = load_noise_bank(noise_bank_path)

print("Phase bank loaded:")
for survey in phase_bank.keys():
    n_systems = len(phase_bank[survey])
    lengths = [len(arr) for arr in phase_bank[survey]]
    print(f"  {survey}: {n_systems} systems, {min(lengths)}-{max(lengths)} obs (mean {np.mean(lengths):.0f})")

print("\nInitializing EBModel...")
# Create model with binning plot saving enabled
# Plots will be saved to <output_dir>/binning_plots/
model = EBModel(
    eb_path=None,
    params_dict=params_dict,
    phase_bank=phase_bank,
    noise_bank=noise_bank,
    rng=np.random.default_rng(101),  # New seed for fresh samples
    save_binning_plots=True,
    plot_output_dir=output_dir,
)
print("✓ Initialized")

# Generate samples
n_samples = 3
print(f"\nGenerating {n_samples} training samples...")
print("-" * 70)

successful = 0
failed = 0
theta_samples = []

# Use a new random seed for parameter sampling
param_rng = np.random.default_rng(101)

for i in range(n_samples):
    # Sample random parameters
    theta = np.array([
        param_rng.uniform(*param_ranges['teff1']),
        param_rng.uniform(*param_ranges['teff2']),
        param_rng.uniform(*param_ranges['r1']),
        param_rng.uniform(*param_ranges['r2']),
        param_rng.uniform(*param_ranges['msum']),
        param_rng.uniform(*param_ranges['q']),
        param_rng.uniform(*param_ranges['period']),
        param_rng.uniform(*param_ranges['incl']),
        param_rng.uniform(*param_ranges['distance']),
        param_rng.uniform(*param_ranges['ebv']),
        param_rng.uniform(*param_ranges['metallicity']),
    ])

    # Generate training data
    output_path = output_dir / f"sample_{i:03d}.npy"

    try:
        success = model.nbi_simulator(theta, str(output_path))

        if success:
            successful += 1
            theta_samples.append(theta)
            print(f"[{i:2d}] ✓ Generated: {output_path.name}")
            print(f"     Distance={theta[8]:.1f} pc, Period={theta[6]:.2f} d, Incl={theta[7]:.1f}°")
        else:
            failed += 1
            print(f"[{i:2d}] ✗ Failed: {output_path.name}")

    except Exception as e:
        failed += 1
        print(f"[{i:2d}] ✗ Exception: {e}")

print("-" * 70)
print(f"\nGeneration complete:")
print(f"  Successful: {successful}/{n_samples}")
print(f"  Failed:     {failed}/{n_samples}")

if successful > 0:
    # Save parameter samples
    theta_array = np.array(theta_samples)
    params_file = output_dir / "theta_samples.npy"
    np.save(params_file, theta_array)
    print(f"\n✓ Saved parameters: {params_file}")
    print(f"  Shape: {theta_array.shape}")

print("\n" + "="*70)
print("VERIFYING SAMPLE STRUCTURE")
print("="*70)

if successful > 0:
    # Load and inspect first successful sample
    sample_path = output_dir / "sample_000.npy"
    data = np.load(sample_path, allow_pickle=True).item()

    print(f"\nLoaded: {sample_path}")
    print(f"\nData structure:")
    print(f"  Keys: {list(data.keys())}")

    print(f"\nLight curve data:")
    print(f"  lc_phase: {len(data['lc_phase'])} passbands")
    for i, ph in enumerate(data['lc_phase']):
        print(f"    Passband {i}: {len(ph)} points")

    print(f"\nSED data:")
    print(f"  sed: {len(data['sed'])} filters")
    print(f"  sed_err: {len(data['sed_err'])} filters")

    print(f"\nMetadata:")
    print(f"  Shape: {data['meta'].shape}")
    print(f"  Values: {data['meta']}")

    n_passbands = len(data['lc_phase'])
    if len(data['meta']) >= 5 + n_passbands:
        print(f"\nMetadata breakdown:")
        print(f"  [0] Period:           {data['meta'][0]:.4f} days")
        print(f"  [1] Distance (Gaia):  {data['meta'][1]:.2f} pc")
        print(f"  [2] Distance error:   {data['meta'][2]:.2f} pc")
        print(f"  [3] Parallax (Gaia):  {data['meta'][3]:.3f} mas")
        print(f"  [4] Parallax error:   {data['meta'][4]:.3f} mas")
        print(f"  [5:{5+n_passbands}] Obs counts:     {[int(x) for x in data['meta'][5:5+n_passbands]]}")

    print(f"\nMasks:")
    print(f"  mask_lc: {[m.shape for m in data['mask_lc']]}")
    print(f"  mask_sed: {data['mask_sed'].shape}")
    print(f"  mask_meta: {data['mask_meta'].shape}")

    print(f"\nSystem IDs:")
    print(f"  {data['lc_system_ids']}")

    # Check for NaN/Inf
    has_issues = False
    for i, flux in enumerate(data['lc']):
        if np.any(~np.isfinite(flux)):
            print(f"\n⚠ Warning: Non-finite values in lc passband {i}")
            has_issues = True

    if np.any(~np.isfinite(data['sed'])):
        print(f"\n⚠ Warning: Non-finite values in sed")
        has_issues = True

    # Note: metadata can have NaN (for missing Gaia data), that's expected

    if not has_issues:
        print(f"\n✓ All data values are finite (except Gaia metadata, which can be NaN)")

    print("\n" + "="*70)
    print("✓ SAMPLE GENERATION AND VERIFICATION COMPLETE")
    print("="*70)
    print(f"\nGenerated samples are in: {output_dir.absolute()}")
    print(f"Ready for NBI training!")

else:
    print("\n✗ No successful samples generated")
    print("Check PHOEBE configuration and parameter ranges")
