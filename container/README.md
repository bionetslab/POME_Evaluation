# Container setup for POME_Evaluation on Helma (NHR@FAU)

Apptainer + Slurm workflow for reproducing the POME evaluation figures and
analyses on the Helma cluster. The heavy dependency stack is baked into one
`.sif` image (1 inode, no conda inode pressure); both project sources
(`POME_Evaluation` and `POME`) are bind-mounted at run time, so editing code
never triggers a rebuild.

Built from `HELMA_SETUP_HANDOFF.md`; this is the evaluation-side superset of the
POME runtime container.

## Layout

| Where | What |
|-------|------|
| `container/` (this dir, in the repo) | recipe + wrappers, version-controlled |
| `/hnvme/workspace/y105dc10-pome_eval/containers/pome_eval.sif` | the built image (kept off `$HOME`) |
| `/hnvme/workspace/y105dc10-pome_eval/{logs,runs,data}` | Slurm logs, job scratch, heavy-data staging |
| repo `data/`, `output/` | scripts read/write these **relative to the repo root** |

## One-time build (on a frontend, e.g. helma4/helma5)

```bash
bash container/build.sh          # builds -> workspace/containers/pome_eval.sif
```

## Everyday use — re-run one analysis or plotting step (`run.sh`)

Runs on the **current node** (a frontend for CPU work), no Slurm. This is the
common path: imputation figures, survival, linear probing, and every plotting
notebook are **CPU-only**.

```bash
# a python script
container/run.sh scripts/generate_imputation_figure.py --dim 32

# execute a notebook in place (re-runs all cells, saves outputs back)
container/run.sh scripts/plot_fig4_unsupervised.ipynb
container/run.sh --html scripts/plot_fig4_unsupervised.ipynb   # also export HTML

# interactive Jupyter Lab (prints the SSH tunnel command to use)
container/run.sh lab --port 8890

# a shell / raw python inside the container
container/run.sh shell
container/run.sh python -c "import pome, torch; print(torch.__version__)"

# add --gpu to expose a GPU (only if you're on a GPU node / inside an salloc)
container/run.sh --gpu scripts/generate_embeddings.py
```

## GPU-heavy scripts — submit to H200 (`submit.sh`)

Only these import `pome`/torch and want a GPU:
`generate_embeddings.py`, `generate_inductive_embeddings.py`,
`generate_inductive_epoch_snapshots.py`,
`src/pome_evaluation/impute_graph_based.py`,
`scripts/generate_imputation_epoch_results.py`.

```bash
sbatch container/submit.sh scripts/generate_inductive_embeddings.py [args...]
squeue --me -o "%.10i %.9P %.2t %.10M %C %b %R"
scontrol show job <id> | grep -iE "Partition=|TresPerNode|JobState"
```

**Always H200, never H100** — the AI.BAY project `y105dc` has 0 H100
GPU-minutes on the default `mq_health` QOS, so H100 requests pend forever.
`submit.sh` already targets `--partition=h200`. Its `#SBATCH --time` defaults to
6h; override per-job on the command line (e.g. `sbatch --time=24:00:00 …`).

**Imputation grid.** `src/pome_evaluation/impute_graph_based.py` sweeps a
`{z,nonlinear} × {7,11,15} × {16,32,64}` grid per dataset, scoped by
`--discretizations/--bins/--dims/--datasets`. To fan it out across jobs (each
capped at 24h), use the helper `container/submit_imputation_grid.sh`, which
emits one job per `(discretization, bins, dim)` cell (18 total):

```bash
DRY_RUN=1 container/submit_imputation_grid.sh   # preview the sbatch commands
container/submit_imputation_grid.sh             # submit all 18
```

### Fast job I/O — stage outputs to node-local NVMe (`--stage-out`)

These jobs emit thousands of small files, and the repo lives on NFS `$HOME`
(small-file writes are slow there). Pass `--stage-out` with the repo-relative
output dir(s): writes go to node-local NVMe (`$TMPDIR`) during the run and are
bulk-copied back into the repo at the end.

```bash
sbatch container/submit.sh --stage-out data/imputation_data/pome_based \
    src/pome_evaluation/impute_graph_based.py
sbatch container/submit.sh --stage-out output/inductive \
    scripts/generate_inductive_embeddings.py
```

> **Pass it as the flag, not a `STAGE_OUT=…` env var.** `submit.sh` sets
> `#SBATCH --export=NONE`, so submit-time environment variables are **not**
> propagated into the job (and Slurm forbids whitelisting vars alongside
> `NONE`) — a bare `STAGE_OUT=… sbatch …` is silently dropped and the job
> writes straight to NFS. Command-line args always reach the batch script, so
> `--stage-out` is the reliable path. (A `STAGE_OUT` env var is still honored as
> a fallback when `submit.sh` runs *inside* an existing allocation.)

The copy-back runs inside the allocation (via an EXIT trap, so it fires even if
the job fails) because `$TMPDIR` is wiped at job end — leave a few minutes of
walltime headroom. Staged dirs are treated as **write-only sinks**: their
existing contents are hidden from the job while staged, so only use it for dirs
the script writes (not ones it also reads inputs from).

## Notes

- `pome` comes from `~/Projects/POME.git/src` (bind-mounted on `PYTHONPATH`).
  Override with `POME_REPO=/path/to/POME.git`. For a frozen image instead,
  uncomment `pip install pome-py` in `pome_eval.def` and rebuild.
- `umap` comes from the **NA-aware fork** `~/Projects/umap-na-aware.git`, which
  shadows the stock `umap-learn` inside the image. Override with
  `UMAP_REPO=/path/to/umap-na-aware.git`. The fork keeps its package contents at
  the repo root, so `run.sh`/`submit.sh` create a shim dir `$WS/pythonpath/umap`
  (a symlink to the fork) and put it first on `PYTHONPATH`; delete that symlink
  to go back to stock umap-learn. The inductive UMAP embeddings need the fork's
  pairwise-removal NaN distances and `transform_combined()`, and
  `generate_inductive_embeddings.py` aborts if it imports the stock package.
- The stack is pinned to **numpy<2 / pandas 2.x** (required by `napypi` and
  `lifelines`; also dodges the pandas-3 API break). `napypi` is installed
  `--no-deps` (see `pome_eval.def`).
- Reproduced outputs stay in the repo on `$HOME` (git-diffable against the
  committed versions). At the repo's ~12–15 GB / ~10–15k-file scale this fits
  comfortably under the `$HOME` quota (100 GB / 500k inodes); the I/O concern is
  the small-file *write storm* during jobs, which `--stage-out` handles above.
- `scxmatch`/`anndata` from the handoff are **not** used by this repo, so no
  `git`/`build-essential` in `%post`.
