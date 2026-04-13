# generate_training_data.py
import argparse
import os
import time
import traceback
from pathlib import Path
import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

# Do this before ANY phoebe import
os.environ["PHOEBE_ENABLE_MPI"] = "FALSE"
os.environ["PHOEBE_MULTIPROC_NPROCS"] = "0"   # don't let phoebe multiproc inside workers

# also cap BLAS threads so one worker doesn't hog the node
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"



from pathlib import Path
import numpy as np

from generate_shards import generate_shards
from ebsbi.model import EBModel  # adjust import to your project
from ebsbi.config import Config
import multiprocessing as mp



import h5py
import numpy as np

_WORKER = {}

def _unset_mpi_env():
    import os
    for k in [
        "OMPI_COMM_WORLD_SIZE", "OMPI_COMM_WORLD_RANK", "OMPI_COMM_WORLD_LOCAL_RANK",
        "PMI_SIZE", "PMI_RANK",
        "PMIX_RANK", "PMIX_SIZE",
    ]:
        os.environ.pop(k, None)

def _init_worker(worker_id: int = 0):
    global _WORKER
    _unset_mpi_env()
    seed = 12345 + (os.getpid() % 10_000_000)
    model, passbands = build_model(rng_seed=seed)
    _WORKER["model"] = model
    _WORKER["passbands"] = passbands


def _do_chunk(job):
    """
    One job == one shard (or one shard remainder at the end).
    job = (params_npy_path, outdir_str, shard_size, prefix, shard_start, shard_count)
    """

    params_npy, outdir, shard_size, prefix, shard_start, shard_count = job
    model = _WORKER["model"]

    # --- optional time budget controls (set these in your SLURM script) ---
    # MAX_SECONDS: total wallclock seconds for this job allocation
    # SAFETY_SECONDS: stop this many seconds before the limit
    max_seconds = int(os.environ.get("MAX_SECONDS", "0"))          # 0 => disabled
    safety = int(os.environ.get("SAFETY_SECONDS", "900"))          # default 15 min buffer
    t_global0 = float(os.environ.get("JOB_T0", "0"))               # set once by main() (recommended)

    # If JOB_T0 isn't set, fall back to per-worker start time (less accurate vs walltime)
    if t_global0 <= 0:
        t_global0 = time.time()

    # Don't start work if we're too close to time limit
    if max_seconds > 0:
        elapsed = time.time() - t_global0
        if elapsed > (max_seconds - safety):
            print(f"[pid {os.getpid()}] near walltime (elapsed={elapsed:.1f}s), skipping shard_start={shard_start}",
                  flush=True)
            return 0

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    shard_idx = int(shard_start) // int(shard_size)
    shard_path = outdir / f"{prefix}_shard_{shard_idx:06d}.npz"

    # Skip if already done (restart-safe)
    if shard_path.exists():
        # Return shard_count so progress can still advance if you want,
        # or return 0 if you only want "new work" counted.
        print(f"[pid {os.getpid()}] exists, skip {shard_path}", flush=True)
        return 0

    # mmap params to avoid copying big arrays into each process
    params = np.load(params_npy, mmap_mode="r")

    t0 = time.time()
    try:
        print(f"[pid {os.getpid()}] start shard {shard_idx} "
              f"range=[{shard_start},{shard_start+shard_count})", flush=True)

        generate_shards(
            model=model,
            thetas=params,
            outdir=outdir,
            shard_size=shard_size,
            start=int(shard_start),
            count=int(shard_count),
            prefix=prefix,
            overwrite=False,
        )

        dt = time.time() - t0
        print(f"[pid {os.getpid()}] done shard {shard_idx} in {dt/60:.2f} min", flush=True)
        return int(shard_count)

    except Exception as e:
        print(f"[pid {os.getpid()}] ERROR shard {shard_idx}: {e}", flush=True)
        traceback.print_exc()
        # Returning 0 keeps the pool going; raising would crash the whole run.
        return 0


def load_phase_bank(filename):
    """
    Load full phase bank from an HDF5 file created with variable-length datasets.
    Returns a dict: { survey_name : [np.ndarray of phases] }
    """
    bank = {}

    with h5py.File(filename, "r") as f:
        for survey in f.keys():
            grp = f[survey]
            dset = grp["phases"]

            # Read all cadences into memory
            phase_list = []
            for i in range(len(dset)):
                phase_list.append(np.array(dset[i], dtype=np.float64))

            bank[survey] = phase_list

    return bank

def load_noise_bank(filename):
    """
    Load full noise bank from an HDF5 file created with variable-length datasets.
    Returns a dict: { survey_name : [np.ndarray of errs] }
    """
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


def build_model(rng_seed: int):
    """
    Construct EBModel ONCE per process.
    IMPORTANT: Each SLURM rank gets its own process + its own EBModel/bundle.
    """

    config_path = '/global/homes/j/jrblaum/software/ebsbi/docs/config_linear_times_sequential.yml'
    conf = Config(config_path)
    
    # ---- YOU must adapt these to your codebase ----
    eb_path = conf.eb_path         # Path to data/resources used by EBModel
    params_dict = conf.labels_dict     # Ordered dict mapping parameter names -> ranges/metadata

    phase_bank = load_phase_bank("/global/cfs/cdirs/m2218/jrblaum/research/eclipsing_binaries_lfi/data/phase_bank.h5py")
    noise_bank = load_noise_bank("/global/cfs/cdirs/m2218/jrblaum/research/eclipsing_binaries_lfi/data/noise_bank.h5py") 

    model = EBModel(
        eb_path=eb_path,
        params_dict=params_dict,
        phase_bank=phase_bank,
        noise_bank=noise_bank,
        rng=rng_seed,
        amortized=True,
        system_weights="length",
    )

    # Create bundle once per worker process (reused thereafter)
    # Use the same passbands you use for LCs in training:
    passbands = ("SDSS:g", "Johnson:V", "ZTF:g", "ZTF:r")
    model._ensure_bundle_initialized(passbands=passbands)

    return model, passbands


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params-npy", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--shard-size", type=int, default=256)
    ap.add_argument("--prefix", default="training")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--count", type=int, default=None)
    ap.add_argument("--n-procs", type=int, default=None, help="number of worker processes")
    ap.add_argument("--chunk", type=int, default=64, help="how many samples per worker job")
    ap.add_argument("--max-seconds", type=int, default=None,
                help="max wall time in seconds for this run (worker stops before starting a new shard)")
    ap.add_argument("--safety-seconds", type=int, default=600,
                help="stop this many seconds before max-seconds")
    args = ap.parse_args()

    params = np.load(args.params_npy, mmap_mode="r")
    N = len(params)

    start = int(args.start)
    end = N if args.count is None else min(N, start + int(args.count))
    if start >= end:
        print("Nothing to do: start>=end")
        return

    indices = np.arange(start, end, dtype=int)

    nprocs = args.n_procs or int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
    nprocs = max(1, nprocs)

    # make jobs as shard-aligned (one job per shard)
    jobs = []
    aligned_start = (start // args.shard_size) * args.shard_size
    for shard_start in range(aligned_start, end, args.shard_size):
        shard_end = min(shard_start + args.shard_size, end)
        if shard_end <= start:
            continue  # skip shards fully before requested start
        # NOTE: this job writes the whole shard [shard_start, shard_end)
        jobs.append((args.params_npy, args.outdir, args.shard_size, args.prefix, int(shard_start), int(shard_end - shard_start)))


    ctx = mp.get_context("spawn")  # IMPORTANT for PHOEBE stability
    with ctx.Pool(processes=nprocs, initializer=_init_worker) as pool:

        # give each process a distinct seed by re-calling initializer is not trivial;
        # easiest: use a wrapper that passes unique ids:
        # Instead: create pool without initializer, and use "initializer" trick via env var.
        # For now, keep same seed; cadence randomness is still per-process if EBModel uses process RNG.
        done = 0
        for k in pool.imap_unordered(_do_chunk, jobs):
            done += k
            print(f"done {done}/{len(indices)}")

if __name__ == "__main__":
    os.environ["JOB_T0"] = str(time.time())
    main()