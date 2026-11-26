#!/usr/bin/env python
"""Inductive HI-VAE embeddings (train-fit, test-transform) for the linear-probing analysis.

Competitor of POME in ``scripts/linear_probe_inductive_embeddings.py``. It is the train/test
version of ``scripts/embed_hivae.py`` (the clustering competitor), with identical training: the
authors' unmodified TF1 code in ``hivae.sif``, model_HIVAE_inputDropout, types cat / pos / real,
dim z = embedding size, y = 5, s = 10, batch 100 (enlarged for sparse tables), 200 epochs,
seeds = run. The input is the train/test splits that POME's inductive embeddings use
(``data/train_test_splits/<dataset>/split_NN_{train,test}_pome.tsv``).

Per (split, dim, run):

1. **Train** HI-VAE on the training split only. Variable types and the categories of each
   categorical variable come from the training split.
2. **Training embedding**: the authors' full-batch restore pass (``--train 0 --restore 1``) over
   the training samples, ``z`` = mean of q(z | x, s*). This is exactly what ``embed_hivae.py``
   does on the full data.
3. **Test embedding**: one more restore pass of the *same frozen network*, over the training
   samples followed by the test samples, keeping the test rows. No weights are updated. The test
   samples have to be passed together with the training samples for two reasons, both in the
   authors' code:
   * ``read_functions.read_data`` re-codes each categorical column by ``np.unique`` of the values
     present in the file, so a test-only file lacking some category would scramble the codes.
     With the training rows present every training category occurs, and codes stay aligned.
   * HI-VAE normalizes real/positive variables with the mean/variance of the observed values *in
     the current batch* (``VAE_functions.batch_normalization``). With the training rows in the
     batch these statistics are dominated by the training data (80 %) rather than taken from
     the test split alone.
   Test values of a categorical variable never seen in training are treated as missing.

    output: output/inductive_competitors/hivae/<dataset>/split_NN/dim_D/run_RR_{train,test}.csv

TF 1.15, CPU; fits run in parallel (``--jobs``), each pinned to its own cores:

    sbatch --job-name=ind_hivae --time=04:00:00 \\
        --export=POME_EVAL_SIF=$(ws_find pome_eval)/containers/hivae.sif \\
        container/submit.sh scripts/embed_hivae_inductive.py --jobs 16
    ... --datasets hancock --splits 0 --dims 16 --runs 0 --dry-run
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
import embed_hivae as eh  # noqa: E402
import impute_hivae as ih  # noqa: E402
from pome_evaluation.tabular_inputs import (SPLIT_DATASETS, inductive_paths,  # noqa: E402
                                            load_split_tables, split_ids, write_embedding_csv)

METHOD = "hivae"


def encode_like_train(test, types, categories):
    """Encode ``test`` with the training types/categories; unseen categories -> NaN."""
    import numpy as np

    data = np.empty(test.shape, dtype=float)
    n_unseen = 0
    for j, col in enumerate(test.columns):
        values = test[col].to_numpy(dtype=float)
        if types[j][0] == "cat":
            cats = categories[j]
            obs = ~np.isnan(values)
            known = obs & np.isin(values, cats)
            n_unseen += int((obs & ~known).sum())
            codes = np.full(values.shape, np.nan)
            codes[known] = np.searchsorted(cats, values[known])
            data[:, j] = codes
        else:
            if types[j][0] == "pos" and np.nanmin(values) < 0:
                raise ValueError("negative test value in positive column %r" % col)
            data[:, j] = values
    return data, n_unseen


def embed(dataset, split, dim, run, opts):
    import numpy as np

    t0 = time.time()
    train, test, cat_cols = load_split_tables(dataset, split)
    train_data, types, categories = ih._encode(train, set(cat_cols))
    test_data, n_unseen = encode_like_train(test, types, categories)
    batch_size = ih.training_batch_size(train_data, opts)
    workdir = tempfile.mkdtemp(prefix="hivae_ind_", dir=opts.workdir)
    try:
        common = [
            "--model_name", opts.model,
            "--data_file", "data.csv", "--types_file", "data_types.csv",
            "--miss_file", "no_extra_mask.csv",
            "--true_miss_file", "true_miss.csv",
            "--dim_latent_z", str(dim), "--dim_latent_y", str(opts.dim_y),
            "--dim_latent_s", str(opts.dim_s), "--save_file", "hivae",
            "--display", str(max(1, opts.epochs // 10)),
        ]

        def restore(data, z_file, log):
            ih._write_inputs(workdir, data, types)
            eh._run_child(workdir, z_file, run, ih._WORKER_CPUS, common + [
                "--batch_size", str(len(data)), "--epochs", "1",
                "--train", "0", "--restore", "1"], log)
            z = np.load(z_file)
            if z.shape != (len(data), dim):
                raise RuntimeError("z has shape %s, expected %s" % (z.shape, (len(data), dim)))
            return z

        with open(os.path.join(workdir, "main_scripts.log"), "w") as log:
            ih._write_inputs(workdir, train_data, types)
            eh._run_child(workdir, "", run, ih._WORKER_CPUS, common + [
                "--batch_size", str(batch_size), "--epochs", str(opts.epochs)], log)
            z_train = restore(train_data, os.path.join(workdir, "z_train.npy"), log)
            z_both = restore(np.vstack([train_data, test_data]),
                             os.path.join(workdir, "z_both.npy"), log)
        train_out, test_out = inductive_paths(METHOD, dataset, split, dim, run)
        write_embedding_csv(z_train, train.index, train_out)
        write_embedding_csv(z_both[len(train):], test.index, test_out)
    except Exception as exc:
        kept = os.path.join(opts.workdir, "failed_" + os.path.basename(workdir))
        shutil.move(workdir, kept)
        raise RuntimeError("HI-VAE failed on %s split %d dim %d run %d (%s); scratch kept at %s"
                           % (dataset, split, dim, run, exc, kept))
    shutil.rmtree(workdir, ignore_errors=True)
    return test_out, batch_size, n_unseen, time.time() - t0


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--_child":
        eh._child_main(sys.argv[2:])
        return

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datasets", nargs="+", choices=SPLIT_DATASETS, default=list(SPLIT_DATASETS))
    p.add_argument("--splits", nargs="+", type=int, default=None, help="default: all")
    p.add_argument("--dims", nargs="+", type=int, default=[16, 32, 64])
    p.add_argument("--runs", nargs="+", type=int, default=list(range(5)))
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
        for split in (opts.splits or split_ids(dataset)):
            for dim in opts.dims:
                for run in opts.runs:
                    paths = inductive_paths(METHOD, dataset, split, dim, run)
                    if opts.overwrite or not all(os.path.exists(q) for q in paths):
                        tasks.append((dataset, split, dim, run))
    print("%d inductive HI-VAE embedding(s) to compute" % len(tasks))
    if opts.dry_run:
        for t in tasks:
            print("  %s split %02d dim %d run %d" % t)
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
        futures = {pool.submit(embed, d, s, dim, run, opts): (d, s, dim, run)
                   for d, s, dim, run in tasks}
        for fut in as_completed(futures):
            try:
                out, bs, n_unseen, secs = fut.result()
                print("[done %5.0fs, batch %d, unseen test categories %d] %s"
                      % (secs, bs, n_unseen, os.path.relpath(out, PROJECT_ROOT)), flush=True)
            except Exception as exc:
                failures += 1
                print("[FAILED] %s: %s" % (futures[fut], exc), flush=True)
    if failures:
        sys.exit("%d of %d embedding(s) failed" % (failures, len(tasks)))


if __name__ == "__main__":
    main()
