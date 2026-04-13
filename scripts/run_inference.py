#!/usr/bin/env python
"""
Run posterior inference using a trained NBI checkpoint.

Loads a training shard, applies the observational noise process, then draws
posterior samples from the trained normalizing flow.

Usage
-----
    python scripts/run_inference.py \
        --config configs/train.yml \
        --checkpoint models/best_model.pth \
        --observation examples/training_shard_000000.npz \
        --index 0 \
        --n-posterior 10000
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from nbi import NBI

# Reuse config + observational process builders from training script
from train_nbi import (
    load_config,
    get_required,
    build_observational_process,
)
from ebsbi.utils import load_phase_bank, load_noise_bank
from ebsbi.shards import EBSBIShardDatasetMultiChannel
from ebsbi.config import Config as EBSBIConfig


# -------------------------
# Load checkpoint
# -------------------------

def load_engine(checkpoint_path, device="cpu"):
    """
    Reconstruct an NBI engine from a saved checkpoint.

    The checkpoint stores the flow config, featurizer modules, masked channel
    specs, network weights, and scaling statistics — everything needed for
    inference without rebuilding the training architecture.
    """
    engine = NBI(state_dict=str(checkpoint_path), device=device)
    engine.set_params(str(checkpoint_path))
    return engine


# -------------------------
# Posterior sampling
# -------------------------

def sample_posterior(engine, x_list, n_posterior):
    """
    Draw posterior samples for a single observation.

    Parameters
    ----------
    engine : NBI
        Engine with loaded checkpoint.
    x_list : list of Tensor
        Observation channels after noise process, each (C, L).
    n_posterior : int
        Number of posterior samples.

    Returns
    -------
    theta_samples : np.ndarray, shape (n_posterior, D)
    """
    engine.network.eval()

    # Add batch dimension: (C, L) -> (1, C, L)
    x_list = [x.unsqueeze(0) if x.ndim == 2 else x for x in x_list]

    x_scaled = engine.scale_x(x_list)
    x_scaled = [xj.to(engine.device, dtype=torch.float32) for xj in x_scaled]

    with torch.no_grad():
        samples = engine.get_network()(x_scaled, n=n_posterior, sample=True)

    if torch.is_tensor(samples):
        samples = samples.detach().cpu().numpy()
    else:
        samples = np.asarray(samples)

    if samples.ndim == 3:
        theta_samples = engine.scale_y(samples, back=True)[0]
    elif samples.ndim == 2:
        theta_samples = engine.scale_y(samples[None, ...], back=True)[0]
    else:
        raise ValueError(f"Unexpected sample shape from flow: {samples.shape}")

    return np.asarray(theta_samples, dtype=np.float32)


def _as_numpy(t):
    if torch.is_tensor(t):
        return t.detach().cpu().numpy()
    return np.asarray(t)


# -------------------------
# Main
# -------------------------

LABELS = [
    "msum", "q", "log_age", "metallicity", "ebv", "cosi",
    "rsumfrac", "ecc", "per0", "distance", "teff1", "teff2",
    "r1", "r2", "log_lum1", "log_lum2",
]


def main():
    ap = argparse.ArgumentParser(
        description="Run NBI posterior inference on an observation."
    )
    ap.add_argument("--config", required=True,
                    help="Path to training YAML config (for noise process).")
    ap.add_argument("--checkpoint", default="models/best_model.pth",
                    help="Path to trained model checkpoint.")
    ap.add_argument("--observation", required=True,
                    help="Path to observation shard (.npz or npy directory).")
    ap.add_argument("--index", type=int, default=0,
                    help="Sample index within the shard (default: 0).")
    ap.add_argument("--n-posterior", type=int, default=10000,
                    help="Number of posterior samples (default: 10000).")
    ap.add_argument("--output", default=None,
                    help="Output .npz path. Default: <observation>_posterior.npz")
    ap.add_argument("--device", default="cpu",
                    help="Device (default: cpu).")
    ap.add_argument("--seed", type=int, default=42,
                    help="RNG seed for noise process (default: 42).")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    # --- Load training config (for noise process) ---
    cfg = load_config(args.config)

    # --- Build observational noise process ---
    pb_path = get_required(cfg, "banks.phase_bank_path")
    nb_path = get_required(cfg, "banks.noise_bank_path")
    pb = load_phase_bank(pb_path)
    nb = load_noise_bank(nb_path)
    process_fn = build_observational_process(cfg, pb, nb)

    # --- Load shard dataset ---
    conf = EBSBIConfig(get_required(cfg, "ebsbi_conf_path"))
    data_cfg = cfg["data"]
    num_lcs = int(data_cfg.get("num_lcs", 4))

    ds = EBSBIShardDatasetMultiChannel(
        [args.observation],
        conf.labels_dict,
        num_lcs=num_lcs,
        mmap_mode="r",
    )
    print(f"Shard loaded: {len(ds)} valid samples")

    if args.index >= len(ds):
        raise IndexError(f"Requested index {args.index} but shard has {len(ds)} valid samples")

    # --- Get raw sample and apply noise ---
    x_raw, theta_true = ds[args.index]
    theta_true = np.asarray(_as_numpy(theta_true), dtype=np.float32).ravel()

    out = process_fn(x_raw, y=theta_true, rng=rng)
    if isinstance(out, (tuple, list)) and len(out) == 2:
        x_ready, _ = out
    else:
        x_ready = out

    if not isinstance(x_ready, (list, tuple)):
        raise TypeError(f"Expected x_ready list/tuple, got {type(x_ready)}")

    print(f"Channels: {len(x_ready)}, shapes: {[tuple(x.shape) for x in x_ready]}")

    # --- Load model ---
    print(f"Loading checkpoint: {args.checkpoint}")
    engine = load_engine(args.checkpoint, device=args.device)

    # --- Sample posterior ---
    print(f"Drawing {args.n_posterior} posterior samples...")
    theta_samples = sample_posterior(engine, x_ready, args.n_posterior)
    print(f"Posterior shape: {theta_samples.shape}")

    # --- Summary table ---
    labels = LABELS[:theta_samples.shape[1]]
    header = f"{'Parameter':<15} {'Mean':>10} {'Std':>10} {'2.5%':>10} {'97.5%':>10} {'True':>10}"
    print(f"\n{header}")
    print("-" * len(header))
    for i, label in enumerate(labels):
        col = theta_samples[:, i]
        print(f"{label:<15} {col.mean():>10.4f} {col.std():>10.4f} "
              f"{np.percentile(col, 2.5):>10.4f} {np.percentile(col, 97.5):>10.4f} "
              f"{theta_true[i]:>10.4f}")

    # --- Save ---
    out_path = args.output
    if out_path is None:
        stem = Path(args.observation).stem
        out_path = f"{stem}_posterior.npz"

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        out_path,
        theta_samples=theta_samples,
        theta_true=theta_true,
        labels=np.array(labels, dtype=object),
        n_posterior=np.array(args.n_posterior),
    )
    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()
