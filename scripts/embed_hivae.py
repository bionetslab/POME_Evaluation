#!/usr/bin/env python
"""HI-VAE sample embeddings (Nazabal et al., 2020) for the unsupervised clustering analysis.

Competitor of POME in ``scripts/generate_unsupervised_figure.py``. HI-VAE handles
heterogeneous, incomplete tables natively, so it is trained on the full datasets with their
original missingness (the same data POME sees, read from POME's input graphs via
``pome_evaluation.tabular_inputs``). Training is the one of the imputation benchmark
(``scripts/impute_hivae.py``, whose helpers are reused): the authors' original TF1 code
(probabilistic-learning/HI-VAE @ 1ac9c31, unmodified, in ``hivae.sif``),
model_HIVAE_inputDropout, variable types cat / pos / real as there, dim y = 5, s = 10,
batch size 100 (enlarged for very sparse tables: MIMIC), 200 epochs, seeds set to the run.

The one change w.r.t. the imputation benchmark is the latent size: dim z = the embedding
size (16, 32, 64). The sample embedding is HI-VAE's deterministic test-time encoding,
``samples_test['z']`` = mean of q(z | x, s*) with s* = argmax q(s | x), computed in the
authors' full-batch restore pass (``--train 0 --restore 1``, samples in input order). The
original code does not write z, so the child process wraps
``read_functions.samples_concatenation`` (called once in that pass) to save it; nothing else
is touched.

    output: data/embeddings_competitors/{DATASET}/embeddings/{DATASET}_HIVAE_{dim}_{run}.tsv
            (samples x dim, sample IDs as index)

TF 1.15 runs on CPU; fits run in parallel (``--jobs``), each pinned to its own cores.

    POME_EVAL_SIF=$(ws_find pome_eval)/containers/hivae.sif \\
        container/run.sh scripts/embed_hivae.py --jobs 16
    ... --datasets HANCOCK --dims 16 --runs 0 --dry-run
"""

import argparse
import multiprocessing
import os
import shutil
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
import impute_hivae as ih  # noqa: E402
from pome_evaluation.tabular_inputs import DATASETS, embedding_path, load_table  # noqa: E402

METHOD = "HIVAE"


# --- child mode --------------------------------------------------------------------
def _child_main(argv):
    """``embed_hivae.py --_child Z_OUT SEED CPUS -- <main_scripts.py args>``."""
    z_out = argv[0]
    if z_out:
        repo = os.environ.get("HIVAE_REPO", "/opt/HI-VAE")
        sys.path.insert(0, repo)
        import numpy as np
        import read_functions                       # numpy only, no TensorFlow import

        original = read_functions.samples_concatenation

        def saving_concatenation(samples):
            out = original(samples)                 # (s, z, y, x)
            np.save(z_out, out[1])
            return out

        read_functions.samples_concatenation = saving_concatenation
    ih._child_main(argv[1:])


def _run_child(workdir, z_out, seed, cpus, script_args, log):
    import subprocess

    cmd = [sys.executable, os.path.abspath(__file__), "--_child", z_out, str(seed),
           ",".join(str(c) for c in cpus), "--"] + script_args
    res = subprocess.run(cmd, cwd=workdir, stdout=log, stderr=subprocess.STDOUT)
    if res.returncode != 0:
        raise RuntimeError("main_scripts.py exited with %d (log: %s)" % (res.returncode, log.name))


# --- parent mode -------------------------------------------------------------------
def embed(dataset, dim, run, out_path, opts):
    import numpy as np
    import pandas as pd

    t0 = time.time()
    table, cat_cols = load_table(dataset)
    data, types, _ = ih._encode(table, set(cat_cols))
    batch_size = ih.training_batch_size(data, opts)
    workdir = tempfile.mkdtemp(prefix="hivae_emb_", dir=opts.workdir)
    try:
        ih._write_inputs(workdir, data, types)
        name = "hivae"
        common = [
            "--model_name", opts.model,
            "--data_file", "data.csv", "--types_file", "data_types.csv",
            "--miss_file", "no_extra_mask.csv",
            "--true_miss_file", "true_miss.csv",
            "--dim_latent_z", str(dim), "--dim_latent_y", str(opts.dim_y),
            "--dim_latent_s", str(opts.dim_s), "--save_file", name,
            "--display", str(max(1, opts.epochs // 10)),
        ]
        z_file = os.path.join(workdir, "z.npy")
        with open(os.path.join(workdir, "main_scripts.log"), "w") as log:
            _run_child(workdir, "", run, ih._WORKER_CPUS, common + [
                "--batch_size", str(batch_size), "--epochs", str(opts.epochs)], log)
            _run_child(workdir, z_file, run, ih._WORKER_CPUS, common + [
                "--batch_size", str(len(table)), "--epochs", "1",
                "--train", "0", "--restore", "1"], log)
        z = np.load(z_file)
        if z.shape != (len(table), dim):
            raise RuntimeError("z has shape %s, expected %s" % (z.shape, (len(table), dim)))
        if not np.isfinite(z).all():
            raise RuntimeError("z has non-finite values")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        df = pd.DataFrame(z, index=table.index, columns=["dim_%d" % i for i in range(dim)])
        df.index.name = "sample"
        df.to_csv(out_path + ".tmp", sep="\t")
        os.replace(out_path + ".tmp", out_path)
    except Exception:
        kept = os.path.join(opts.workdir, "failed_" + os.path.basename(workdir))
        shutil.move(workdir, kept)
        raise RuntimeError("HI-VAE failed on %s dim %d run %d; scratch kept at %s"
                           % (dataset, dim, run, kept))
    shutil.rmtree(workdir, ignore_errors=True)
    return out_path, batch_size, time.time() - t0


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--_child":
        _child_main(sys.argv[2:])
        return

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS)
    p.add_argument("--dims", nargs="+", type=int, default=[16, 32, 64])
    p.add_argument("--runs", nargs="+", type=int, default=list(range(10)))
    p.add_argument("--model", default="model_HIVAE_inputDropout",
                   choices=["model_HIVAE_inputDropout", "model_HIVAE_factorized"])
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=100)
    p.add_argument("--min-obs-per-batch", type=float, default=20)
    p.add_argument("--dim-y", type=int, default=5)
    p.add_argument("--dim-s", type=int, default=10)
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--workdir", default=os.environ.get("TMPDIR") or "/tmp")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    opts = p.parse_args()

    tasks = []
    for dataset in opts.datasets:
        for dim in opts.dims:
            for run in opts.runs:
                out = embedding_path(dataset, METHOD, dim, run)
                if opts.overwrite or not os.path.exists(out):
                    tasks.append((dataset, dim, run, out))
    print("%d HI-VAE embedding(s) to compute" % len(tasks))
    if opts.dry_run:
        for t in tasks:
            print("  %s dim %d run %d -> %s" % (t[0], t[1], t[2], os.path.relpath(t[3], PROJECT_ROOT)))
        return
    if not tasks:
        return

    available = sorted(os.sched_getaffinity(0))
    jobs = max(1, min(opts.jobs, len(tasks), len(available)))
    per_job = len(available) // jobs
    cpu_queue = multiprocessing.Manager().Queue()
    for i in range(jobs):
        cpu_queue.put(available[i * per_job:(i + 1) * per_job])
    print("%d worker(s) x %d core(s)" % (jobs, per_job))

    failures = 0
    with ProcessPoolExecutor(max_workers=jobs, initializer=ih._init_worker,
                             initargs=(cpu_queue,)) as pool:
        futures = {pool.submit(embed, d, dim, run, out, opts): (d, dim, run)
                   for d, dim, run, out in tasks}
        for fut in as_completed(futures):
            try:
                out, bs, secs = fut.result()
                print("[done %5.0fs, batch %d] %s" % (secs, bs, os.path.relpath(out, PROJECT_ROOT)),
                      flush=True)
            except Exception as exc:
                failures += 1
                print("[FAILED] %s: %s" % (futures[fut], exc), flush=True)
    if failures:
        sys.exit("%d of %d embedding(s) failed" % (failures, len(tasks)))


if __name__ == "__main__":
    main()
