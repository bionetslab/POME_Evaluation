#!/usr/bin/env python
"""Impute the simulated-missingness files with HI-VAE (Nazabal et al., 2020).

Runs the authors' original TensorFlow-1 implementation
(github.com/probabilistic-learning/HI-VAE, commit 1ac9c31, unmodified; baked into
``container/hivae.def`` at ``$HIVAE_REPO``) on every masked file of the benchmark and
writes the imputed tables next to the other baselines:

    input : data/imputation_data/missforest/{DATASET}/simulated_data/masked_values_{r}_{run}.tsv
    output: data/imputation_data/hivae/{DATASET}/imputed_data/masked_values_{r}_{run}.tsv

Inputs are the MissForest copies of the masked files because those are the only ones
stored as samples x variables with NaN for every missing entry (the k-NN/AutoComplete
``simulated_data`` of HANCOCK/TCGA_LUAD are in POME's variables x samples layout).
Outputs are samples x variables with the original row/column labels, so
``scripts/generate_imputation_results.py`` scores them like the KNN/MissForest files.

Per file:
1. Variable types (HI-VAE needs one per column): every variable in the dataset's
   categorical list -> ``cat`` (observed categories re-coded 0..K-1, decoded back on
   output); continuous variables -> ``pos`` (log-normal) if all observed values are
   >= 0, else ``real`` (Gaussian). ``count``/``ordinal`` are not used.
2. Every NaN (originally missing and simulated) is passed as ``--true_miss_file``,
   so it is excluded from training and filled in the reconstruction.
3. ``main_scripts.py`` is run twice like the authors' ``script_HIVAE.sh``: training,
   then a single full-batch pass restoring the network. Its
   ``Results/<name>/<name>_data_reconstruction.csv`` holds the observed values plus
   the mode of each missing entry's likelihood (HI-VAE's imputation).

Hyperparameters default to the authors' example script: model_HIVAE_inputDropout,
dim z=2, y=5, s=10, batch size 100, 200 epochs. The one deviation: the training batch
grows for very sparse tables (see ``training_batch_size``), because the original code
breaks when a mini-batch has no observed value of a variable. The TF graph seed and numpy seed are
set to ``--seed`` (the original code sets none), so a run is repeatable up to
multithreaded-float nondeterminism.

TensorFlow 1.15 has no CUDA build for current GPUs; this runs on CPU. Files are
processed in parallel with ``--jobs`` worker processes, each pinned to its own set of
cores (TensorFlow sizes its thread pools from the CPU affinity mask). Scratch files
go to ``--workdir`` (default ``$TMPDIR`` or /tmp) and are deleted after each file.

Run from the repo root inside the HI-VAE image:

    POME_EVAL_SIF=$(ws_find pome_eval)/containers/hivae.sif \\
        container/run.sh scripts/impute_hivae.py --datasets hancock luad --jobs 8
    ... --ratios 0.1 --runs 0 --dry-run     # list what would be done
"""

import argparse
import glob
import multiprocessing
import os
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMPUTATION_ROOT = os.path.join(PROJECT_ROOT, "data", "imputation_data")
INPUT_METHOD_DIR = "missforest"          # where the samples x variables inputs live
OUTPUT_METHOD_DIR = "hivae"

# dataset CLI key -> (on-disk label, categorical-variable list inside the input dir)
DATASETS = {
    "hancock": ("HANCOCK", "cat_variables.csv"),
    "luad": ("TCGA_LUAD", "TCGA_LUAD_cat_vars.csv"),
    "mimic": ("MIMIC", "mimic_cat_variables.csv"),
}


# --- child mode: run the original main_scripts.py with seeds set ------------------
def _child_main(argv):
    """``impute_hivae.py --_child SEED CPUS -- <main_scripts.py args>`` (cwd = workdir).

    Pins the process to CPUS (comma list) *before* TensorFlow is imported, seeds numpy,
    and wraps graph_new.HVAE_graph so the graph main_scripts.py builds gets a graph-level
    seed. main_scripts.py is then executed unmodified as __main__.
    """
    seed = int(argv[0])
    cpus = [int(c) for c in argv[1].split(",") if c != ""]
    script_args = argv[3:] if argv[2] == "--" else argv[2:]
    if cpus and hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, cpus)

    import random
    import runpy

    import numpy as np

    repo = os.environ.get("HIVAE_REPO", "/opt/HI-VAE")
    sys.path.insert(0, repo)
    random.seed(seed)
    np.random.seed(seed)

    import tensorflow as tf
    import graph_new

    original_graph = graph_new.HVAE_graph

    def seeded_graph(*args, **kwargs):
        # Called inside `with tf.Graph().as_default()` in main_scripts.py.
        tf.set_random_seed(seed)
        return original_graph(*args, **kwargs)

    graph_new.HVAE_graph = seeded_graph
    sys.argv = [os.path.join(repo, "main_scripts.py")] + script_args
    runpy.run_path(sys.argv[0], run_name="__main__")


# --- parent mode -------------------------------------------------------------------
def _encode(df, cat_vars):
    """Return (data matrix, types rows, per-column categories or None)."""
    import numpy as np

    data = np.empty(df.shape, dtype=float)
    types, categories = [], []
    for j, col in enumerate(df.columns):
        values = df[col].to_numpy(dtype=float)
        observed = values[~np.isnan(values)]
        if observed.size == 0:
            raise ValueError("column %r has no observed value" % col)
        if col in cat_vars:
            cats = np.unique(observed)
            codes = np.full(values.shape, np.nan)
            obs = ~np.isnan(values)
            codes[obs] = np.searchsorted(cats, values[obs])
            data[:, j] = codes
            types.append(("cat", len(cats), len(cats)))
            categories.append(cats)
        else:
            data[:, j] = values
            types.append(("pos" if observed.min() >= 0 else "real", 1, ""))
            categories.append(None)
    return data, types, categories


def _write_inputs(workdir, data, types):
    import numpy as np

    with open(os.path.join(workdir, "data.csv"), "w") as f:
        for row in data:
            f.write(",".join("nan" if np.isnan(v) else repr(float(v)) for v in row) + "\n")
    with open(os.path.join(workdir, "data_types.csv"), "w") as f:
        f.write("type,dim,nclass\n")
        for t, dim, ncls in types:
            f.write("%s,%d,%s\n" % (t, dim, ncls))
    rows, cols = np.where(np.isnan(data))
    with open(os.path.join(workdir, "true_miss.csv"), "w") as f:
        for r, c in zip(rows, cols):
            f.write("%d,%d\n" % (r + 1, c + 1))        # the code expects 1-based indices


def _run_child(workdir, seed, cpus, script_args, log):
    cmd = [sys.executable, os.path.abspath(__file__), "--_child", str(seed),
           ",".join(str(c) for c in cpus), "--"] + script_args
    res = subprocess.run(cmd, cwd=workdir, stdout=log, stderr=subprocess.STDOUT)
    if res.returncode != 0:
        raise RuntimeError("main_scripts.py exited with %d (log: %s)"
                           % (res.returncode, log.name))


def training_batch_size(data, opts):
    """``max(--batch-size, ceil(--min-obs-per-batch * N / fewest observed in a column))``.

    HI-VAE normalizes every variable with the mean/variance of its *observed* values in
    the current mini-batch (VAE_functions.batch_normalization); a batch without a single
    observed value of some variable gives NaN parameters that poison the whole network.
    The authors' datasets have no variables that sparse, MIMIC does (e.g. 73 of 4428
    observed). The batch is enlarged until the sparsest variable has
    ``--min-obs-per-batch`` observations per batch in expectation (P(none) ~ e^-20),
    capped at N. With the defaults HANCOCK/TCGA_LUAD mostly keep 100 (up to ~200 at
    50% simulated missingness), MIMIC uses ~600-1450.
    """
    import math

    import numpy as np

    n = data.shape[0]
    fewest = int((~np.isnan(data)).sum(axis=0).min())
    needed = int(math.ceil(opts.min_obs_per_batch * n / float(fewest)))
    return min(n, max(opts.batch_size, needed))


_WORKER_CPUS = []     # cores owned by this worker process (set by _init_worker)


def _init_worker(cpu_queue):
    global _WORKER_CPUS
    _WORKER_CPUS = cpu_queue.get()


def impute_file(in_path, out_path, cat_vars, opts):
    """Impute one masked file; returns (out_path, seconds)."""
    import numpy as np
    import pandas as pd

    cpus = _WORKER_CPUS
    t0 = time.time()
    df = pd.read_csv(in_path, sep="\t", index_col=0)
    data, types, categories = _encode(df, cat_vars)

    batch_size = training_batch_size(data, opts)
    workdir = tempfile.mkdtemp(prefix="hivae_", dir=opts.workdir)
    try:
        _write_inputs(workdir, data, types)
        name = "hivae"
        common = [
            "--model_name", opts.model,
            "--data_file", "data.csv", "--types_file", "data_types.csv",
            "--miss_file", "no_extra_mask.csv",            # absent -> no extra mask
            "--true_miss_file", "true_miss.csv",
            "--dim_latent_z", str(opts.dim_z), "--dim_latent_y", str(opts.dim_y),
            "--dim_latent_s", str(opts.dim_s), "--save_file", name,
            "--display", str(max(1, opts.epochs // 10)),          # log every 10% of epochs
        ]
        with open(os.path.join(workdir, "main_scripts.log"), "w") as log:
            _run_child(workdir, opts.seed, cpus, common + [
                "--batch_size", str(batch_size), "--epochs", str(opts.epochs)], log)
            _run_child(workdir, opts.seed, cpus, common + [
                "--batch_size", str(len(df)), "--epochs", "1",
                "--train", "0", "--restore", "1"], log)
        rec_path = os.path.join(workdir, "Results", name, name + "_data_reconstruction.csv")
        rec = np.loadtxt(rec_path, delimiter=",", ndmin=2)
        if rec.shape != df.shape:
            raise RuntimeError("reconstruction has shape %s, expected %s" % (rec.shape, df.shape))

        out = np.empty(df.shape, dtype=float)
        for j, cats in enumerate(categories):
            if cats is None:
                out[:, j] = rec[:, j]
            else:
                out[:, j] = cats[np.clip(np.rint(rec[:, j]).astype(int), 0, len(cats) - 1)]
        observed = ~np.isnan(df.to_numpy(dtype=float))
        out[observed] = df.to_numpy(dtype=float)[observed]   # keep observed values exact
        if np.isnan(out).any():
            raise RuntimeError("%d imputed values are NaN" % int(np.isnan(out).sum()))

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        tmp_out = out_path + ".tmp"
        pd.DataFrame(out, index=df.index, columns=df.columns).to_csv(tmp_out, sep="\t")
        os.replace(tmp_out, out_path)
    except Exception:
        kept = os.path.join(opts.workdir, "failed_" + os.path.basename(workdir))
        shutil.move(workdir, kept)
        raise RuntimeError("HI-VAE failed on %s; scratch kept at %s" % (in_path, kept))
    shutil.rmtree(workdir, ignore_errors=True)
    return out_path, time.time() - t0


def _ratio_run(path):
    # masked_values_{ratio}_{run}.tsv
    stem = os.path.basename(path)[:-len(".tsv")]
    _, _, ratio, run = stem.split("_")
    return float(ratio), int(float(run))


def collect_tasks(opts):
    import pandas as pd

    tasks = []
    for key in opts.datasets:
        label, cat_file = DATASETS[key]
        in_dir = os.path.join(IMPUTATION_ROOT, INPUT_METHOD_DIR, label)
        files = sorted(glob.glob(os.path.join(in_dir, "simulated_data", "masked_values_*.tsv")))
        if not files:
            print("[skip] %s: no inputs under %s" % (label, in_dir))
            continue
        cat_vars = set(pd.read_csv(os.path.join(in_dir, cat_file))["cat_var"])
        for path in files:
            ratio, run = _ratio_run(path)
            if opts.ratios and ratio not in opts.ratios:
                continue
            if opts.runs and run not in opts.runs:
                continue
            out = os.path.join(IMPUTATION_ROOT, OUTPUT_METHOD_DIR, label, "imputed_data",
                               os.path.basename(path))
            if os.path.exists(out) and not opts.overwrite:
                continue
            tasks.append((path, out, cat_vars))
    return tasks


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--_child":
        _child_main(sys.argv[2:])
        return

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datasets", nargs="+", choices=list(DATASETS), default=list(DATASETS))
    p.add_argument("--ratios", nargs="+", type=float, help="only these missingness ratios")
    p.add_argument("--runs", nargs="+", type=int, help="only these simulation runs")
    p.add_argument("--model", default="model_HIVAE_inputDropout",
                   choices=["model_HIVAE_inputDropout", "model_HIVAE_factorized"])
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=100,
                   help="minimum training batch size (see --min-obs-per-batch)")
    p.add_argument("--min-obs-per-batch", type=float, default=20,
                   help="enlarge the batch until every variable has this many observed "
                        "values per batch in expectation")
    p.add_argument("--dim-z", type=int, default=2)
    p.add_argument("--dim-y", type=int, default=5)
    p.add_argument("--dim-s", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--jobs", type=int, default=1, help="files imputed in parallel")
    p.add_argument("--workdir", default=os.environ.get("TMPDIR") or "/tmp",
                   help="scratch directory (node-local; default $TMPDIR or /tmp)")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    opts = p.parse_args()

    tasks = collect_tasks(opts)
    print("%d file(s) to impute with HI-VAE" % len(tasks))
    if opts.dry_run:
        for in_path, out, _ in tasks:
            print("  %s -> %s" % (os.path.relpath(in_path, PROJECT_ROOT),
                                   os.path.relpath(out, PROJECT_ROOT)))
        return
    if not tasks:
        return

    available = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") \
        else list(range(os.cpu_count() or 1))
    jobs = max(1, min(opts.jobs, len(tasks), len(available)))
    per_job = len(available) // jobs
    cpu_sets = [available[i * per_job:(i + 1) * per_job] for i in range(jobs)]
    print("%d worker(s) x %d core(s)" % (jobs, per_job))

    cpu_queue = multiprocessing.Manager().Queue()
    for cpu_set in cpu_sets:
        cpu_queue.put(cpu_set)

    failures = 0
    with ProcessPoolExecutor(max_workers=jobs, initializer=_init_worker,
                             initargs=(cpu_queue,)) as pool:
        futures = {pool.submit(impute_file, in_path, out, cat_vars, opts): in_path
                   for in_path, out, cat_vars in tasks}
        for fut in as_completed(futures):
            try:
                out, secs = fut.result()
                print("[done %5.0fs] %s" % (secs, os.path.relpath(out, PROJECT_ROOT)), flush=True)
            except Exception as exc:                       # keep going, report at the end
                failures += 1
                print("[FAILED] %s: %s" % (os.path.relpath(futures[fut], PROJECT_ROOT), exc),
                      flush=True)
    if failures:
        sys.exit("%d of %d file(s) failed" % (failures, len(tasks)))


if __name__ == "__main__":
    main()
