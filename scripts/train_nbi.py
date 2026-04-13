#!/usr/bin/env python
import argparse
from pathlib import Path
import os
import sys
import warnings
import time

import numpy as np
import torch

# Third-party
import nbi
from isochrones import get_ichrone
from nbi.data import DatasetContainer

# Your package
from ebsbi.config import Config as EBSBIConfig
from ebsbi.engine import create_engine
from ebsbi.model import EBModel
from ebsbi.priors import EBPriors
from ebsbi.shards import EBSBIShardDatasetMultiChannel
from ebsbi.utils import load_phase_bank, load_noise_bank
from ebsbi.observational import (
    CadenceNoiseSampler,
    instrumental_noise,
    load_sed_noise_banks,
    load_gaia_parallax_bank,
    SEDNoiseModel,
    GaiaParallaxSampler,
)

warnings.filterwarnings("ignore", category=FutureWarning, module="isochrones")

try:
    import yaml
except ImportError as e:
    raise ImportError(
        "Missing dependency: pyyaml. Install with `pip install pyyaml`."
    ) from e


# -------------------------
# config helpers
# -------------------------
def deep_update(base: dict, upd: dict) -> dict:
    """Recursively update dict base with dict upd (mutates base)."""
    for k, v in (upd or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base


def load_yaml(path: str | Path) -> dict:
    path = Path(path).expanduser()
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def load_config(main_cfg_path: str | Path) -> dict:
    """
    Supports:
      - a single YAML with everything, OR
      - a YAML that references other YAMLs via `includes: [path1, path2, ...]`
        later files override earlier ones, and main overrides all includes.
    """
    main_cfg_path = Path(main_cfg_path).expanduser()
    main = load_yaml(main_cfg_path)

    cfg = {}
    for inc in main.get("includes", []) or []:
        inc_path = (main_cfg_path.parent / inc).expanduser()
        deep_update(cfg, load_yaml(inc_path))

    deep_update(cfg, {k: v for k, v in main.items() if k != "includes"})
    return cfg


def get_required(cfg: dict, keypath: str):
    cur = cfg
    for k in keypath.split("."):
        if not isinstance(cur, dict) or k not in cur:
            raise KeyError(f"Missing required config key: {keypath}")
        cur = cur[k]
    return cur


def set_nested(cfg: dict, dotted_key: str, value):
    """Set a value in a nested dict using a dotted key like 'train.lr'."""
    keys = dotted_key.split(".")
    d = cfg
    for k in keys[:-1]:
        if k not in d or not isinstance(d[k], dict):
            d[k] = {}
        d = d[k]
    d[keys[-1]] = value


def parse_value(s: str):
    """Parse a CLI override value string into an appropriate Python type."""
    # bool
    if s.lower() in ("true", "false"):
        return s.lower() == "true"
    # int
    try:
        return int(s)
    except ValueError:
        pass
    # float
    try:
        return float(s)
    except ValueError:
        pass
    # string
    return s


def flatten_dict(d: dict, prefix: str = "") -> dict:
    """Flatten nested dict: {'train': {'lr': 0.001}} -> {'train.lr': 0.001}."""
    flat = {}
    for k, v in d.items():
        key = f"{prefix}{k}" if not prefix else f"{prefix}.{k}"
        if isinstance(v, dict):
            flat.update(flatten_dict(v, key))
        elif isinstance(v, (set, frozenset)):
            flat[key] = sorted(v) if v else []
        else:
            flat[key] = v
    return flat


# -------------------------
# wandb helpers
# -------------------------
def init_wandb(args, cfg):
    """Initialize a W&B run and return True, or return False if wandb disabled."""
    if not args.wandb:
        return False

    import wandb

    wandb_kwargs = {
        "project": args.wandb_project,
        "config": flatten_dict(cfg),
    }
    if args.wandb_entity:
        wandb_kwargs["entity"] = args.wandb_entity
    if args.wandb_name:
        wandb_kwargs["name"] = args.wandb_name
    if args.wandb_tags:
        wandb_kwargs["tags"] = args.wandb_tags

    wandb.init(**wandb_kwargs)
    return True


def apply_wandb_sweep_overrides(cfg):
    """If running inside a W&B sweep agent, apply sweep params as config overrides."""
    import wandb
    if wandb.run is None:
        return

    sweep_config = dict(wandb.config)
    for key, value in sweep_config.items():
        if "." in key:
            set_nested(cfg, key, value)
            print(f"[wandb sweep] override: {key} = {value}")

    # Re-sync wandb.config with the fully merged config
    wandb.config.update(flatten_dict(cfg), allow_val_change=True)


def log_training_results(engine):
    """Log post-hoc training metrics to wandb after engine.fit() completes."""
    import wandb
    if wandb.run is None:
        return

    try:
        # Per-epoch losses for round 0
        train_losses = engine.train_losses[-1] if engine.train_losses else []
        val_losses = engine.val_losses[-1] if engine.val_losses else []

        for epoch, (tl, vl) in enumerate(zip(train_losses, val_losses)):
            wandb.log({
                "epoch": epoch,
                "train_loss": float(tl),
                "val_loss": float(vl),
            }, step=epoch)

        # Summary metrics
        if val_losses:
            best_epoch = int(np.argmin(val_losses))
            wandb.summary["best_epoch"] = best_epoch
            wandb.summary["best_val_loss"] = float(val_losses[best_epoch])
            wandb.summary["best_train_loss"] = float(train_losses[best_epoch])
            wandb.summary["total_epochs"] = len(val_losses)

        if engine.best_params:
            wandb.summary["best_checkpoint"] = str(engine.best_params)

    except Exception as e:
        print(f"[wandb] Warning: failed to log training results: {e}")


def log_checkpoint_artifact(engine):
    """Optionally log the best checkpoint as a W&B artifact."""
    import wandb
    if wandb.run is None or not engine.best_params:
        return

    ckpt_path = str(engine.best_params)
    if not os.path.isfile(ckpt_path):
        print(f"[wandb] Checkpoint not found at {ckpt_path}, skipping artifact.")
        return

    try:
        artifact = wandb.Artifact(
            name=f"model-{wandb.run.id}",
            type="model",
        )
        artifact.add_file(ckpt_path)
        wandb.log_artifact(artifact)
        print(f"[wandb] Logged checkpoint artifact: {ckpt_path}")
    except Exception as e:
        print(f"[wandb] Warning: failed to log artifact: {e}")


# -------------------------
# build pieces
# -------------------------
def build_priors(cfg: dict):
    ichrone_name = get_required(cfg, "priors.iso_ichrone")
    ichrone_tracks = bool(cfg.get("priors", {}).get("tracks", True))
    lookup_dir = Path(get_required(cfg, "priors.lookup_table.dir")).expanduser()
    parts = cfg["priors"]["lookup_table"]["parts"]  # e.g. [1..8]
    pattern = cfg["priors"]["lookup_table"].get("pattern", "lookup_table_part{part}_30k.npy")

    arrays = [np.load(lookup_dir / pattern.format(part=p)) for p in parts]
    lookup_table = np.concatenate(arrays, axis=0)
    lookup_table = lookup_table[:, :-2] # Get rid of log_lum1 and log_lum2

    tracks = get_ichrone(ichrone_name, tracks=ichrone_tracks)
    rng = np.random.default_rng(int(get_required(cfg, "seeds.priors_rng")))
    conf = EBSBIConfig(get_required(cfg, "ebsbi_conf_path"))
    priors = EBPriors(conf.params_dict, conf.labels_dict, tracks, rng=rng)

    empirical_prior = nbi.empirical_prior.EmpiricalPrior(lookup_table, priors)
    return conf, priors, empirical_prior


def build_observational_process(cfg: dict, pb, nb):
    # survey mapping
    survey_id_map = cfg["surveys"]["survey_id_map"]
    id_to_survey = {v: k for k, v in survey_id_map.items()}

    # LC cadence/noise sampler
    cadence_cfg = cfg["noise"]["cadence"]
    cadence_noise_sampler = CadenceNoiseSampler(
        phase_bank=pb,
        noise_bank=nb,
        log_noise=bool(cadence_cfg.get("log_noise", True)),
        seed=int(cadence_cfg.get("seed", 42)),
        system_weights=cadence_cfg.get("system_weights", "length"),
    )

    # SED noise model
    sed_cfg = cfg["noise"]["sed"]
    sed_noise_banks = load_sed_noise_banks(
        sed_cfg["banks_path"],
        file_ext=sed_cfg.get("file_ext", ".pkl"),
    )
    sed_noise_model = SEDNoiseModel(
        sed_noise_banks=sed_noise_banks,
        jitter_mode=sed_cfg.get("jitter_mode", "log"),
        jitter_log_s=float(sed_cfg.get("jitter_log_s", 0.005)),
        jitter_log_s_by_key=sed_cfg.get("jitter_log_s_by_key", {}) or {},
        rel_clip=tuple(sed_cfg.get("rel_clip", (1e-6, 10.0))),
        default_rel=float(sed_cfg.get("default_rel", 0.05)),
        floors=sed_cfg.get("floors", None),
    )

    # Gaia parallax sampler
    gaia_cfg = cfg["noise"]["gaia"]
    gaia_parallax_bank = load_gaia_parallax_bank(gaia_cfg["parallax_bank_path"])
    gaia_parallax_sampler = GaiaParallaxSampler(gaia_parallax_bank)

    proc_cfg = cfg["process"]
    process_fn = instrumental_noise(
        cadence_noise_sampler=cadence_noise_sampler,
        sed_noise_model=sed_noise_model,
        gaia_parallax_sampler=gaia_parallax_sampler,
        id_to_survey=id_to_survey,
        L_fixed=int(proc_cfg.get("L_fixed", 2000)),
        rng=np.random.default_rng(int(get_required(cfg, "seeds.process_rng"))),
        gaia_coverage=float(proc_cfg.get("gaia_coverage", 1.0)),
        normalize_lc=bool(proc_cfg.get("normalize_lc", True)),
        ablate=proc_cfg.get("ablate", None),
    )
    return process_fn


def build_datasets(cfg: dict):
    data_cfg = cfg["data"]
    base = Path(data_cfg["shards_base"]).expanduser()
    n_shards = int(data_cfg["n_shards"])
    shard_pattern = data_cfg.get("shard_pattern", "training_shard_{i:06d}.npz")
    shards = np.array([base / shard_pattern.format(i=i) for i in range(n_shards)], dtype=object)

    rng = np.random.default_rng(int(get_required(cfg, "seeds.split_rng")))
    perm = rng.permutation(len(shards))

    test_frac = float(data_cfg.get("test_frac", 0.05))
    n_test = int(round(test_frac * len(shards)))

    test_shards = shards[perm[:n_test]].tolist()
    train_shards = shards[perm[n_test:]].tolist()

    num_lcs = int(data_cfg.get("num_lcs", 4))
    mmap_mode = data_cfg.get("mmap_mode", "r")

    conf = EBSBIConfig(get_required(cfg, "ebsbi_conf_path"))

    ds_train = EBSBIShardDatasetMultiChannel(train_shards, conf.labels_dict, num_lcs=num_lcs, mmap_mode=mmap_mode)
    ds_test  = EBSBIShardDatasetMultiChannel(test_shards,  conf.labels_dict, num_lcs=num_lcs, mmap_mode=mmap_mode)

    print("train shards:", len(train_shards), "train sims:", len(ds_train))
    print("test  shards:", len(test_shards),  "test sims:",  len(ds_test))
    return ds_train, ds_test

def benchmark_dataloader(loader, n_batches=100, warmup=10):
    it = iter(loader)

    for _ in range(warmup):
        next(it)

    times = []
    sizes = []
    t_prev = time.perf_counter()

    for _ in range(n_batches):
        x, y = next(it)
        t_now = time.perf_counter()
        times.append(t_now - t_prev)

        bs = x[0].shape[0] if isinstance(x, (list, tuple)) else x.shape[0]
        sizes.append(bs)
        t_prev = time.perf_counter()

    times = np.array(times)
    sizes = np.array(sizes)

    print(f"warmup batches:    {warmup}")
    print(f"batches measured:  {len(times)}")
    print(f"mean batch time:   {times.mean():.4f} s")
    print(f"median batch time: {np.median(times):.4f} s")
    print(f"p90 batch time:    {np.percentile(times, 90):.4f} s")
    print(f"examples/sec:      {sizes.sum() / times.sum():.1f}")

def benchmark_dataset_getitem(dataset, n_items=1000):
    idxs = np.random.default_rng(0).integers(0, len(dataset), size=n_items)
    times = []

    for idx in idxs:
        t0 = time.perf_counter()
        _ = dataset[idx]
        t1 = time.perf_counter()
        times.append(t1 - t0)

    times = np.array(times)
    print(f"items measured:     {len(times)}")
    print(f"mean item time:     {times.mean():.4f} s")
    print(f"median item time:   {np.median(times):.4f} s")
    print(f"p90 item time:      {np.percentile(times, 90):.4f} s")
    print(f"items/sec:          {len(times) / times.sum():.1f}")

import time
import numpy as np
import torch

def profile_train_loop(engine, n_batches=10):
    engine.network.train()

    fetch_times = []
    prep_times = []
    forward_times = []
    backward_times = []
    total_times = []

    loader_iter = iter(engine.train_loader)

    for _ in range(n_batches):
        t0 = time.perf_counter()
        data = next(loader_iter)
        t1 = time.perf_counter()

        x, y = data

        x = engine.scale_x(x)
        if isinstance(x, (list, tuple)):
            x = [xj.to(engine.device, dtype=torch.float32, non_blocking=True) for xj in x]
        else:
            x = x.to(engine.device, dtype=torch.float32, non_blocking=True)

        y = engine.scale_y(y).to(engine.device, dtype=torch.float32, non_blocking=True)

        if "cuda" in str(engine.device):
            torch.cuda.synchronize()
        t2 = time.perf_counter()

        engine.optimizer.zero_grad(set_to_none=True)

        loss = engine.network(x, y)
        loss = loss.mean()

        if "cuda" in str(engine.device):
            torch.cuda.synchronize()
        t3 = time.perf_counter()

        loss.backward()
        engine.optimizer.step()

        if "cuda" in str(engine.device):
            torch.cuda.synchronize()
        t4 = time.perf_counter()

        fetch_times.append(t1 - t0)
        prep_times.append(t2 - t1)
        forward_times.append(t3 - t2)
        backward_times.append(t4 - t3)
        total_times.append(t4 - t0)

    def summarize(name, arr):
        arr = np.array(arr)
        print(f"{name:>12}: mean={arr.mean():.4f}s median={np.median(arr):.4f}s p90={np.percentile(arr, 90):.4f}s")

    summarize("fetch", fetch_times)
    summarize("prep", prep_times)
    summarize("forward", forward_times)
    summarize("backward", backward_times)
    summarize("total", total_times)

    total = np.mean(total_times)
    print("\nFraction of step time:")
    print(f"fetch    {np.mean(fetch_times)/total:.1%}")
    print(f"prep     {np.mean(prep_times)/total:.1%}")
    print(f"forward  {np.mean(forward_times)/total:.1%}")
    print(f"backward {np.mean(backward_times)/total:.1%}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to main YAML config.")
    ap.add_argument("--benchmark-dataloader", action="store_true")
    ap.add_argument("--n-batches", type=int, default=100)
    ap.add_argument("--benchmark-dataset", action="store_true")
    ap.add_argument("--n-items", type=int, default=1000)
    ap.add_argument("--profile-train-step", action="store_true")
    ap.add_argument("--profile-batches", type=int, default=10)

    # W&B flags
    ap.add_argument("--wandb", action="store_true", help="Enable W&B logging.")
    ap.add_argument("--wandb-project", default="ebsbi-hparam", help="W&B project name.")
    ap.add_argument("--wandb-entity", default=None, help="W&B entity (team or username).")
    ap.add_argument("--wandb-name", default=None, help="W&B run name.")
    ap.add_argument("--wandb-tags", nargs="*", default=None, help="W&B run tags.")
    ap.add_argument("--log-artifact", action="store_true",
                     help="Log best checkpoint as W&B artifact.")

    # Config overrides: --set train.lr=0.0005 --set train.batch_size=256
    ap.add_argument("--set", dest="overrides", action="append", default=[],
                     metavar="KEY=VALUE",
                     help="Override config values using dotted keys, e.g. --set train.lr=0.0005")

    args = ap.parse_args()

    # --- load config ---
    cfg = load_config(args.config)

    # --- apply CLI overrides ---
    for ov in args.overrides:
        if "=" not in ov:
            ap.error(f"--set requires KEY=VALUE format, got: {ov}")
        key, val = ov.split("=", 1)
        set_nested(cfg, key, parse_value(val))
        print(f"[override] {key} = {parse_value(val)}")

    # --- wandb init ---
    use_wandb = init_wandb(args, cfg)

    # --- apply sweep overrides (if running inside a wandb agent) ---
    if use_wandb:
        apply_wandb_sweep_overrides(cfg)

    # --- banks
    pb_path = get_required(cfg, "banks.phase_bank_path")
    nb_path = get_required(cfg, "banks.noise_bank_path")
    pb = load_phase_bank(pb_path)
    nb = load_noise_bank(nb_path)

    # --- priors / lookup / ebsbi config
    conf, priors, empirical_prior = build_priors(cfg)

    # --- simulator (physics-only)
    sim_cfg = cfg["simulator"]
    model = EBModel(
        eb_path=conf.eb_path,
        params_dict=conf.labels_dict,
        phase_bank=pb,
        noise_bank=nb,
        amortized=bool(sim_cfg.get("amortized", True)),
        rng=int(sim_cfg.get("rng_seed", 42)),
    )

    # --- process function (observational noise)
    process_fn = build_observational_process(cfg, pb, nb)

    # --- engine (isolate checkpoint dir per wandb run to avoid collisions)
    ckpt_path = get_required(cfg, "ckpt_path")
    if use_wandb:
        import wandb
        if wandb.run is not None:
            ckpt_path = os.path.join(ckpt_path, wandb.run.id)
            print(f"[wandb] checkpoint dir: {ckpt_path}")
    engine = create_engine(conf, model, empirical_prior, ckpt_path)

    # --- dataset split
    ds_train, ds_test = build_datasets(cfg)

    # --- train
    train_cfg = cfg["train"]

    if args.benchmark_dataset:
        benchmark_dataset_getitem(ds_train, n_items=args.n_items)
        return

    if args.benchmark_dataloader:

        data_container = DatasetContainer(
            ds_train,
            f_test=0.0,
            f_val=float(train_cfg.get("f_val", 0.1)),
            seed=0,
            process=process_fn,
        )

        engine._init_loader(
            data_container,
            batch_size=int(train_cfg.get("batch_size", 128)),
            workers=int(train_cfg.get("workers", 8)),
            pin_memory=bool(train_cfg.get("pin_memory", True)),
            prefetch_factor=int(train_cfg.get("prefetch_factor", 2)),
        )

        benchmark_dataloader(engine.train_loader, n_batches=args.n_batches)
        return

    if args.profile_train_step:
        data_container = DatasetContainer(
            ds_train,
            f_test=0.0,
            f_val=float(train_cfg.get("f_val", 0.1)),
            seed=0,
            process=process_fn,
        )

        engine._init_train(lr=float(train_cfg.get("lr", 1e-3)))
        engine._init_loader(
            data_container,
            batch_size=int(train_cfg.get("batch_size", 128)),
            workers=int(train_cfg.get("workers", 8)),
            pin_memory=bool(train_cfg.get("pin_memory", True)),
            prefetch_factor=int(train_cfg.get("prefetch_factor", 2)),
        )
        profile_train_loop(engine, n_batches=10)
        return

    engine.fit(
        x=ds_train,
        y=None,
        noise=process_fn,
        n_sims=len(ds_train),
        n_rounds=int(train_cfg.get("n_rounds", 1)),
        n_epochs=int(train_cfg.get("n_epochs", 200)),
        batch_size=int(train_cfg.get("batch_size", 128)),
        f_val=float(train_cfg.get("f_val", 0.1)),
        lr=float(train_cfg.get("lr", 1e-3)),
        min_lr=float(train_cfg.get("min_lr", 1e-5)),
        early_stop_patience=int(train_cfg.get("early_stop_patience", 10)),
        workers=int(train_cfg.get("workers", 8)),
        pin_memory=bool(train_cfg.get("pin_memory", True)),
        prefetch_factor=int(train_cfg.get("prefetch_factor", 2)),
    )

    print("Best model checkpoint is saved to:", engine.best_params)

    # --- post-hoc wandb logging ---
    if use_wandb:
        log_training_results(engine)
        if args.log_artifact:
            log_checkpoint_artifact(engine)

        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
