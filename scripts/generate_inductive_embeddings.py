"""Compute inductive (train-fit, test-transform) embeddings for every dataset
and train/test split under ``data/train_test_splits/``.

For each split we produce embeddings with two methods:

POME  (``*_train_pome.tsv`` / ``*_test_pome.tsv``, graph format)
    Fit an ``Embedder`` in **inductive mode** (``inductive=True``) on the
    training split, read off the transductive embeddings for the training
    samples via ``get_embeddings()``, and embed the unseen test samples
    inductively via ``transform()`` (single forward pass of the frozen trained
    encoder, no retraining).

    POME is used with its inductive defaults (POME >= 1.7): fit() first selects the
    number of training epochs by early stopping on the smoothed held-out link AUC of
    a 3-fold sample-holdout CV within the 2000-epoch budget, then trains on all
    training samples for that many epochs (``generate_inductive_stopping_ablation.py``
    / ``inductive_stopping_criteria_ablation.py`` / ``analyze_inductive_epoch_probes.py``).
    ``--epochs`` overrides the budget (default: POME's, 2000).

UMAP  (``*_train_umap.csv`` / ``*_test_umap.csv``, sample format)
    Fit UMAP on the training split and apply it to the unseen test samples.
    Mirrors ``src/pome_evaluation/embed_UMAP_combined.py`` but inductively.

    **Requires the NA-aware umap fork** (``~/Projects/umap-na-aware.git``, exposed
    by ``container/{run,submit}.sh`` via ``UMAP_REPO``), not stock umap-learn.
    Two additions of that fork are load-bearing here and the script aborts if
    they are missing rather than silently producing different embeddings:

    - *pairwise-removal (NaN-aware) distances* -- ``parallel_nan_euclidean`` /
      ``parallel_nan_hamming`` compute distances over the features observed in
      *both* vectors, in ``fit()`` and in the bipartite ``transform()``. They are
      selected by passing ``ensure_all_finite="allow-nan"`` with metric
      ``euclidean`` / ``hamming``, so each component model that is later
      intersected is built on missingness-aware distances.
    - ``umap.transform_combined()`` -- embeds unseen samples into an existing
      *combined* (intersected) space, which the fit-once ``a * b`` artifact
      cannot do on its own.

    - HANCOCK / TCGA-LUAD: a numeric UMAP (euclidean, on RobustScaler-scaled
      continuous features) and a categorical UMAP (hamming) are fit *separately*,
      each at the full ``dim`` components, then combined by intersecting their
      fuzzy graphs (``numeric_mapper * cat_mapper``) -- exactly as the original
      transductive pipeline. The train embedding is the intersected model's
      ``embedding_``; the test embedding is produced by ``umap.transform_combined``,
      which rebuilds that intersection for unseen samples (per-modality bipartite
      graphs -> graph intersection -> optimise against the joint embedding). Train
      and test therefore share the same ``dim``-column intersected space.
    - MIMIC IV: numeric-only (its UMAP data has no categorical columns), so a
      single numeric mapper is used at full ``dim``; the test set is embedded with
      the mapper's ordinary inductive ``transform()``. => output has ``dim`` columns.

For each split and method, embeddings are computed for dimensions 16, 32, 64,
and for each dimension over 10 independent runs (distinct random seeds).

Variable-type modes (``--modes``)
--------------------------------
By default both methods see *all* variables (``combined``). The two restricted
modes embed a variable-type subset of the very same splits, which is what the
data-integration analysis compares against:

    combined      all variables (default)
    numeric_only  continuous variables only
    cat_only      categorical variables only

The restriction is applied to the split files themselves -- POME's graph format
is row-filtered on its ``type`` column (``cont`` / ``cat``), the UMAP sample
matrices are column-filtered on the dataset's ``cat_var`` list -- so the sample
partitions are byte-for-byte the ones used by the combined runs and the three
modes are directly comparable. This reproduces the transductive inputs
``*_wo_targets_{numeric_only,cat_only}_{graph.tsv,UMAP.csv}``, which are exact
subsets of the combined inputs. For a restricted UMAP mode only one modality
remains, so the fuzzy-graph intersection is skipped and the single mapper's
ordinary inductive ``transform()`` embeds the test samples (mirroring
``src/pome_evaluation/embed_UMAP_{numeric_only,cat_only}.py``).

    The restricted UMAP modes depend on the fork's NaN handling: with stock
    umap-learn a single-modality ``transform()`` returns non-finite coordinates
    for every test sample carrying a NaN (HANCOCK numeric_only split 00: 76/153
    test samples, TCGA-LUAD: 92/113), which the linear probe would then have to
    drop. The fork's pairwise-removal distances embed all of them, so every mode
    is scored on the full test split -- check ``n_test_nonfinite`` in the probing
    results if you suspect the wrong umap was picked up.

Run from the project root with the POME-enabled environment, e.g.:

    conda run -n torch python scripts/generate_inductive_embeddings.py
    conda run -n torch python scripts/generate_inductive_embeddings.py --dry-run
    conda run -n torch python scripts/generate_inductive_embeddings.py \
        --datasets hancock --methods pome --epochs 200
    conda run -n torch python scripts/generate_inductive_embeddings.py \
        --datasets hancock luad_noleak --modes numeric_only cat_only --dims 64

Outputs (each run writes a ``_train`` and a ``_test`` CSV, sample-indexed):

    output/inductive/{method}/{dataset}/split_{NN}/dim_{D}/run_{R}_{train,test}.csv
    output/inductive/{method}/{dataset}/{mode}/split_{NN}/dim_{D}/run_{R}_{train,test}.csv

The ``combined`` mode keeps the original (mode-less) layout so existing outputs
stay valid; the restricted modes add one path segment.

POME is used with its own defaults (POME >= 1.7, the final version) and, unless ``--output-dir``
is given, writes to ``output/inductive_pome`` so that the published POME embeddings under
``output/inductive`` are not overwritten; UMAP writes to ``output/inductive``.

For POME, the epoch count and wall-clock time of every run are additionally logged to

    <output-dir>/pome/tuned_epochs.csv

with columns ``dataset, mode, split, dim, run, optimal_epochs, max_epochs, seconds``
(``optimal_epochs``: epochs selected by POME's early stopping and trained;
``max_epochs``: the budget). It is
upserted once per computed run (so it survives an interrupted run), and skipped
runs leave their existing rows untouched.

Because that upsert rewrites the whole file, **concurrent jobs must not share
it**. Sharded submissions pass a unique ``--log-tag`` (writing
``tuned_epochs.<tag>.csv``) and fold the shards back together afterwards with
``scripts/merge_tuned_epochs.py``. ``container/submit_pome_modes.sh`` fans the
POME work out over one job per (dataset x mode x split-group) and sets the tags
for you.

Existing outputs are skipped unless ``--overwrite`` is passed, so the script is
resumable.
"""

import argparse
import re
import time
import warnings
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import RobustScaler

# --- Configuration -----------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SPLITS_ROOT = PROJECT_ROOT / "data" / "train_test_splits"
INPUT_DATASETS = PROJECT_ROOT / "data" / "input_datasets"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "output" / "inductive"
DEFAULT_POME_OUTPUT_ROOT = PROJECT_ROOT / "output" / "inductive_pome"

# luad_noleak = luad without its outcome/follow-up variables (scripts/make_luad_noleak_splits.py)
DATASETS = ("hancock", "mimic", "luad_noleak")
METHODS = ("pome", "umap")
MODES = ("combined", "numeric_only", "cat_only")
DIMENSIONS = (16, 32, 64, 128)
N_RUNS = 10

# Variable-type restriction per mode. POME's graph format carries the type of
# each variable (row) in its `type` column; a restricted mode keeps only the
# rows of the matching type.
TYPE_COL = "type"
GRAPH_TYPE_BY_MODE = {"numeric_only": "cont", "cat_only": "cat"}

# Per-dataset categorical-variable lists (column `cat_var`), used to split the
# UMAP sample matrices into numeric vs categorical feature blocks.
CAT_VAR_FILES = {
    "hancock": "hancock_cat_variables.csv",
    "luad_noleak": "TCGA_LUAD_cat_vars.csv",   # only columns present are used
    "mimic": "mimic_aggregated_cat_variables.csv",
}
# Datasets whose categorical UMAP is deactivated (numeric-only), per the
# original transductive pipeline.
NUMERIC_ONLY_DATASETS = {"mimic"}

# POME hyperparameters (mirrors evaluate_inductive_ood.py in the POME repo).
NA_ENCODING = -99.0
DEFAULT_EPOCHS = None  # POME's default inductive budget (2000) with early stopping
DEFAULT_BINS = 15
DEFAULT_DISCRETIZATION = "z"
DEFAULT_SEED = 42

_SPLIT_RE = re.compile(r"split_(\d+)_train_(pome|umap)\.(tsv|csv)$")


# --- Helpers -----------------------------------------------------------------
def discover_splits(dataset: str, method: str) -> list[int]:
    """Return sorted split ids that have a training file for ``method``."""
    ext = "tsv" if method == "pome" else "csv"
    paths = (SPLITS_ROOT / dataset).glob(f"split_*_train_{method}.{ext}")
    ids = []
    for p in paths:
        m = _SPLIT_RE.search(p.name)
        if m:
            ids.append(int(m.group(1)))
    return sorted(ids)


def standardize_columns(emb: pd.DataFrame) -> pd.DataFrame:
    """Rename embedding columns to dim_0 .. dim_{k-1}, preserving the index."""
    emb = emb.copy()
    emb.columns = [f"dim_{i}" for i in range(emb.shape[1])]
    return emb


def mode_root(out_root: Path, method: str, dataset: str, mode: str) -> Path:
    """Return the directory holding the ``split_*`` trees of one mode.

    ``combined`` keeps the original mode-less layout so previously computed
    embeddings remain discoverable; restricted modes add a ``{mode}`` segment.
    """
    base = out_root / method / dataset
    return base if mode == "combined" else base / mode


def run_paths(out_root: Path, method: str, dataset: str,
              split_id: int, dim: int, run: int,
              mode: str = "combined") -> tuple[Path, Path]:
    """Return (train_csv, test_csv) output paths for one run."""
    d = (mode_root(out_root, method, dataset, mode)
         / f"split_{split_id:02d}" / f"dim_{dim}")
    return d / f"run_{run:02d}_train.csv", d / f"run_{run:02d}_test.csv"


# Columns of the epochs log written for inductive POME runs. `seconds` is
# the wall-clock cost of the whole fit (training + transform),
# which is what job walltimes have to be sized from.
EPOCHS_LOG_COLS = ["dataset", "mode", "split", "dim", "run",
                   "optimal_epochs", "max_epochs", "seconds"]


def epochs_log_path(out_root: Path, log_tag: str = "") -> Path:
    """Path of the CSV that records the epoch count and runtime of each POME run.

    ``record_tuned_epochs`` rewrites this file on every run, so two jobs sharing
    it would clobber each other's rows. Sharded jobs therefore pass a unique
    ``--log-tag`` and write ``tuned_epochs.<tag>.csv``; merge the shards
    afterwards with ``scripts/merge_tuned_epochs.py``.
    """
    name = f"tuned_epochs.{log_tag}.csv" if log_tag else "tuned_epochs.csv"
    return out_root / "pome" / name


def record_tuned_epochs(out_root: Path, dataset: str, split_id: int, dim: int,
                        run: int, optimal: int, cap: int,
                        mode: str = "combined", log_tag: str = "",
                        seconds: float | None = None) -> None:
    """Upsert one (dataset, mode, split, dim, run) row into the tuned-epochs log.

    Loads the existing CSV (if any), replaces any prior row with the same key,
    appends the new record, and rewrites the sorted file. Written incrementally
    (once per computed run) so the information survives an interrupted run.
    """
    log_path = epochs_log_path(out_root, log_tag)
    row = {"dataset": dataset, "mode": mode, "split": split_id, "dim": dim,
           "run": run, "optimal_epochs": int(optimal), "max_epochs": int(cap),
           "seconds": None if seconds is None else round(float(seconds), 1)}
    if log_path.exists():
        df = pd.read_csv(log_path)
        # Logs written before modes existed hold combined runs only.
        if "mode" not in df.columns:
            df["mode"] = "combined"
        key = ((df["dataset"] == dataset) & (df["mode"] == mode)
               & (df["split"] == split_id) & (df["dim"] == dim)
               & (df["run"] == run))
        df = df[~key]
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row], columns=EPOCHS_LOG_COLS)
    df = df.sort_values(["dataset", "mode", "split", "dim", "run"]).reset_index(drop=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(log_path, index=False)


# --- POME --------------------------------------------------------------------
def load_graph(path: Path) -> pd.DataFrame:
    """Load a graph-format split: rows = variables, cols = samples + 'type'."""
    return pd.read_csv(path, sep="\t", index_col=0)


def filter_graph_by_mode(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    """Keep only the variable rows of ``mode``'s type (``combined`` = all).

    Reproduces the transductive ``*_wo_targets_{numeric_only,cat_only}_graph.tsv``
    inputs, which are exact row subsets of the combined graph.
    """
    if mode == "combined":
        return df
    return df[df[TYPE_COL] == GRAPH_TYPE_BY_MODE[mode]]


def embed_pome_split(dataset: str, split_id: int, dims, n_runs: int,
                     epochs: int, bins: int, discretization: str,
                     seed_base: int, device: str,
                     out_root: Path, overwrite: bool,
                     mode: str = "combined", log_tag: str = "") -> None:
    from pome.gnn_embedding import Embedder, make_deterministic

    split_dir = SPLITS_ROOT / dataset
    train_df = filter_graph_by_mode(
        load_graph(split_dir / f"split_{split_id:02d}_train_pome.tsv"), mode)
    test_df = filter_graph_by_mode(
        load_graph(split_dir / f"split_{split_id:02d}_test_pome.tsv"), mode)
    if train_df.empty:
        print(f"    [warn] pome {dataset} split {split_id:02d}: no "
              f"{GRAPH_TYPE_BY_MODE[mode]} variables -- skipping mode {mode}")
        return

    for dim in dims:
        for run in range(n_runs):
            train_out, test_out = run_paths(
                out_root, "pome", dataset, split_id, dim, run, mode)
            if not overwrite and train_out.exists() and test_out.exists():
                print(f"    [skip] pome {dataset} split {split_id:02d} "
                      f"dim {dim} run {run:02d}")
                continue

            t_start = time.perf_counter()
            make_deterministic(seed_base + run)
            # Inductive mode (POME >= 1.7): fit() selects the epochs by CV early
            # stopping within the budget, then trains on all training samples; the
            # test samples are embedded inductively via transform() (one frozen pass).
            embedder = Embedder(
                embedding_dimension=dim,
                epochs=epochs,
                bins_per_continuous=bins,
                discretization_type=discretization,
                na_encoding=NA_ENCODING,
                device=device,
                inductive=True,
            )
            embedder.fit(train_df)
            train_emb, *_ = embedder.get_embeddings()  # transductive (train)
            test_emb = embedder.transform(test_df)     # inductive (test)

            train_out.parent.mkdir(parents=True, exist_ok=True)
            standardize_columns(train_emb).to_csv(train_out)
            standardize_columns(test_emb).to_csv(test_out)
            elapsed = time.perf_counter() - t_start
            record_tuned_epochs(out_root, dataset, split_id, dim, run,
                                embedder.num_epochs, embedder.max_epochs, mode, log_tag, elapsed)
            # The per-run wall clock is what job walltimes get sized from, so
            # keep it in the job log as well as the epochs CSV.
            print(f"    [ok]   pome {dataset} split {split_id:02d} dim {dim} "
                  f"run {run:02d}  epochs {embedder.num_epochs}  {elapsed:.1f}s  "
                  f"train {train_emb.shape} test {test_emb.shape}", flush=True)


# --- UMAP --------------------------------------------------------------------
def split_numeric_categorical(df: pd.DataFrame, dataset: str):
    """Split a UMAP sample matrix into (continuous, categorical) column blocks.

    For numeric-only datasets the categorical block is empty regardless of the
    cat-var file.
    """
    if dataset in NUMERIC_ONLY_DATASETS:
        cat_cols = []
    else:
        cat_path = INPUT_DATASETS / CAT_VAR_FILES[dataset]
        cat_vars = set(pd.read_csv(cat_path)["cat_var"])
        cat_cols = [c for c in df.columns if c in cat_vars]
    cont_cols = [c for c in df.columns if c not in set(cat_cols)]
    return cont_cols, cat_cols


def import_na_aware_umap():
    """Import umap, insisting on the NA-aware fork.

    Stock umap-learn imports fine but silently lacks both the pairwise-removal
    distances and ``transform_combined``; falling back to it would produce
    embeddings that are not comparable to the rest of the analysis. Fail loudly
    instead, naming the knob that fixes it.
    """
    import umap
    import umap.distances as umap_dist

    missing = [name for name, obj in (
        ("transform_combined", umap), ("parallel_nan_euclidean", umap_dist),
        ("parallel_nan_hamming", umap_dist)) if not hasattr(obj, name)]
    if missing:
        raise SystemExit(
            f"The imported umap ({umap.__file__}) is missing {missing} -- this "
            f"is stock umap-learn, not the NA-aware fork. Run through "
            f"container/run.sh or container/submit.sh (they put "
            f"$UMAP_REPO, default ~/Projects/umap-na-aware.git, first on "
            f"PYTHONPATH), or set UMAP_REPO to the fork's location.")
    return umap


def embed_umap_split(dataset: str, split_id: int, dims, n_runs: int,
                     out_root: Path, overwrite: bool,
                     mode: str = "combined") -> None:
    umap = import_na_aware_umap()

    split_dir = SPLITS_ROOT / dataset
    train_df = pd.read_csv(
        split_dir / f"split_{split_id:02d}_train_umap.csv", index_col=0)
    test_df = pd.read_csv(
        split_dir / f"split_{split_id:02d}_test_umap.csv", index_col=0)
    # Align test columns to the training feature order.
    test_df = test_df[train_df.columns]

    cont_cols, cat_cols = split_numeric_categorical(train_df, dataset)
    # A restricted mode drops one of the two feature blocks -- the column subset
    # is exactly the transductive `*_wo_targets_{numeric_only,cat_only}_UMAP.csv`.
    if mode == "numeric_only":
        cat_cols = []
    elif mode == "cat_only":
        cont_cols = []
    use_num, use_cat = len(cont_cols) > 0, len(cat_cols) > 0
    if not (use_num or use_cat):
        print(f"    [warn] umap {dataset} split {split_id:02d}: no features "
              f"left -- skipping mode {mode}")
        return

    if use_num:
        # RobustScaler is fit on the training continuous features only, then
        # applied to both train and test (proper inductive scaling).
        scaler = RobustScaler().fit(train_df[cont_cols].to_numpy())
        train_num = scaler.transform(train_df[cont_cols].to_numpy())
        test_num = scaler.transform(test_df[cont_cols].to_numpy())
    if use_cat:
        train_cat = train_df[cat_cols].to_numpy()
        test_cat = test_df[cat_cols].to_numpy()

    for dim in dims:
        # Both modalities are fit at the full `dim`; the fuzzy-graph intersection
        # produces a single `dim`-column joint embedding (np.min of the two
        # n_components), so no dimensionality splitting is needed.
        for run in range(n_runs):
            train_out, test_out = run_paths(
                out_root, "umap", dataset, split_id, dim, run, mode)
            if not overwrite and train_out.exists() and test_out.exists():
                print(f"    [skip] umap {dataset} split {split_id:02d} "
                      f"dim {dim} run {run:02d}")
                continue

            # One mapper per active modality, each at the full target
            # dimensionality: numeric = euclidean on the scaled continuous
            # block, categorical = hamming on the raw categorical block.
            # `ensure_all_finite="allow-nan"` is what routes both metrics
            # through the fork's pairwise-removal distances, so the models that
            # are intersected below are themselves missingness-aware.
            mappers, test_blocks = [], []
            if use_num:
                mappers.append(umap.UMAP(n_components=dim, random_state=run).fit(
                    train_num.copy(), ensure_all_finite="allow-nan"))
                test_blocks.append(test_num.copy())
            if use_cat:
                mappers.append(umap.UMAP(
                    n_components=dim, metric="hamming", random_state=run).fit(
                    train_cat.copy(), ensure_all_finite="allow-nan"))
                test_blocks.append(test_cat.copy())

            if len(mappers) == 2:
                # Combine the two fuzzy graphs by intersection (matching the
                # transductive pipeline). transform_combined() reproduces that
                # intersection for unseen test samples -- one bipartite
                # (new x train) graph per modality, folded together with the
                # same operator, then optimised against the fixed combined
                # embedding -- so train and test share the same joint space.
                combined = mappers[0] * mappers[1]
                train_arr = combined.embedding_
                test_arr = umap.transform_combined(
                    mappers, combined, test_blocks,
                    op="intersection",            # matches the `*` above
                    ensure_all_finite="allow-nan")
            else:
                # Single modality (numeric-only dataset, or a restricted mode):
                # plain inductive transform, no intersection. Still NaN-aware.
                train_arr = mappers[0].embedding_
                test_arr = mappers[0].transform(
                    test_blocks[0], ensure_all_finite="allow-nan")

            train_emb = pd.DataFrame(train_arr, index=train_df.index)
            test_emb = pd.DataFrame(test_arr, index=test_df.index)

            train_out.parent.mkdir(parents=True, exist_ok=True)
            standardize_columns(train_emb).to_csv(train_out)
            standardize_columns(test_emb).to_csv(test_out)
            print(f"    [ok]   umap {dataset} split {split_id:02d} dim {dim} "
                  f"run {run:02d}  train {train_emb.shape} test {test_emb.shape}")


# --- Driver ------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS,
                        default=list(DATASETS))
    parser.add_argument("--methods", nargs="+", choices=METHODS,
                        default=list(METHODS))
    parser.add_argument("--modes", nargs="+", choices=MODES,
                        default=["combined"],
                        help="variable-type subsets to embed: combined (all "
                             "variables, default), numeric_only, cat_only. The "
                             "restricted modes feed the data-integration "
                             "analysis and write to a {mode}/ subdirectory")
    parser.add_argument("--dims", nargs="+", type=int, default=list(DIMENSIONS))
    parser.add_argument("--runs", type=int, default=N_RUNS)
    parser.add_argument("--splits", nargs="+", type=int, default=None,
                        help="split ids to process (default: all discovered)")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS,
                        help="POME inductive training budget; early stopping selects "
                             "the epochs within it (default: POME's, 2000)")
    parser.add_argument("--bins", type=int, default=DEFAULT_BINS,
                        help="POME bins_per_continuous (default: 15)")
    parser.add_argument("--discretization", default=DEFAULT_DISCRETIZATION,
                        help="POME discretization_type (default: z)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="base seed; run r uses seed+r (default: 42)")
    parser.add_argument("--log-tag", default="",
                        help="suffix for the POME epochs/runtime log "
                             "(tuned_epochs.<tag>.csv). REQUIRED when several "
                             "jobs run concurrently -- the log is rewritten on "
                             "every run, so a shared file loses rows. Merge the "
                             "shards with scripts/merge_tuned_epochs.py")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="output root (default: output/inductive_pome for POME, "
                             "output/inductive for UMAP)")
    parser.add_argument("--overwrite", action="store_true",
                        help="recompute and overwrite existing embedding files")
    parser.add_argument("--dry-run", action="store_true",
                        help="list the work to be done and exit")
    args = parser.parse_args()

    device = "cpu"
    if "pome" in args.methods:
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            pass

    print(f"Splits root: {SPLITS_ROOT}")
    out_root = {"pome": args.output_dir or DEFAULT_POME_OUTPUT_ROOT,
                "umap": args.output_dir or DEFAULT_OUTPUT_ROOT}
    for method in args.methods:
        print(f"Output root ({method}): {out_root[method]}")
    print(f"Datasets: {args.datasets} | methods: {args.methods} | "
          f"modes: {args.modes} | dims: {args.dims} | runs: {args.runs}")
    if "pome" in args.methods:
        print(f"POME: inductive mode, early stopping within budget epochs={args.epochs or 2000} "
              f"bins={args.bins} discretization={args.discretization} "
              f"device={device}")

    plan = []  # (method, dataset, mode, [split_ids])
    for method in args.methods:
        for dataset in args.datasets:
            found = discover_splits(dataset, method)
            ids = found if args.splits is None else [s for s in args.splits
                                                     if s in found]
            if not ids:
                print(f"  [warn] no {method} splits for {dataset}")
                continue
            for mode in args.modes:
                plan.append((method, dataset, mode, ids))
                n = len(ids) * len(args.dims) * args.runs
                print(f"  {method:4s} {dataset:8s} {mode:12s}: splits {ids} "
                      f"-> {n} runs ({n * 2} files)")

    if args.dry_run:
        total = sum(len(ids) * len(args.dims) * args.runs
                    for _, _, _, ids in plan)
        print(f"\n[dry-run] {total} runs total; nothing computed.")
        return

    warnings.filterwarnings("ignore")
    for method, dataset, mode, ids in plan:
        for split_id in ids:
            print(f"\n[{method} | {dataset} | {mode} | split {split_id:02d}]")
            if method == "pome":
                embed_pome_split(
                    dataset, split_id, args.dims, args.runs, args.epochs,
                    args.bins, args.discretization, args.seed,
                    device, out_root["pome"], args.overwrite,
                    mode=mode, log_tag=args.log_tag)
            else:
                embed_umap_split(
                    dataset, split_id, args.dims, args.runs,
                    out_root["umap"], args.overwrite, mode=mode)

    print("\nDone.")


if __name__ == "__main__":
    main()
