#!/bin/bash -l
#SBATCH --job-name=pome_eval
#SBATCH --partition=h200           # AI.BAY project y105dc has budget on H200, NOT H100
#SBATCH --gres=gpu:h200:1          # 1 H200 -> 32 CPU cores + 192 GB host RAM, 141 GB VRAM
#SBATCH --time=06:00:00            # <=24h on the h200 partition
#SBATCH --export=NONE              # NHR@FAU best practice: clean job environment
# NOTE: default QOS 'mq_health' grants gres/gpu:h200 minutes but gres/gpu:h100=0,
# so requesting h100 pends forever on 'QOSGrpGRESMinutes'. Use h200.
# Slurm logs -> workspace/logs. #SBATCH directives can't expand variables, so
# this is the resolved absolute path of the pome_eval workspace.
#SBATCH --output=/hnvme/workspace/y105dc10-pome_eval/logs/%x-%j.out
#SBATCH --error=/hnvme/workspace/y105dc10-pome_eval/logs/%x-%j.err
#
# Use this ONLY for the GPU-heavy scripts (they import `pome`/torch):
#     generate_embeddings.py, generate_inductive_embeddings.py,
#     generate_inductive_epoch_snapshots.py, generate_imputation_epoch_results.py,
#     src/pome_evaluation/impute_graph_based.py
# Everything else (imputation figures, survival, linear probing, all plotting
# notebooks) is CPU-only -- run those with container/run.sh, no Slurm needed.
#
# Submit (paths are resolved from this script's location; run from anywhere):
#     sbatch container/submit.sh scripts/generate_inductive_embeddings.py [args...]
# Everything after the script name is forwarded to `python <script> <args...>`.
#
# Outputs land in the repo (scripts use repo-relative data/ and output/). These
# jobs emit thousands of small files, and the repo lives on NFS $HOME -- so for
# efficiency, redirect that write storm to node-local NVMe and bulk-copy back at
# the end with --stage-out (see the "stage heavy OUTPUT dirs" block below):
#     sbatch container/submit.sh --stage-out data/imputation_data/pome_based \
#         src/pome_evaluation/impute_graph_based.py
#     sbatch container/submit.sh --stage-out output/inductive \
#         scripts/generate_inductive_embeddings.py
#
# IMPORTANT: pass staging via the --stage-out FLAG, not a STAGE_OUT env var.
# #SBATCH --export=NONE (below) means submit-time environment variables are NOT
# propagated into the job -- and Slurm forbids whitelisting vars alongside NONE
# -- so a bare "STAGE_OUT=... sbatch container/submit.sh ..." is silently
# dropped. Command-line args always reach the batch script, so the flag works.
# (A STAGE_OUT env var is still honored as a fallback when submit.sh runs inside
# an existing allocation, e.g. via run.sh --gpu.)

set -euo pipefail
unset SLURM_EXPORT_ENV             # required with --export=NONE (see Helma docs)

# ---- Paths (resolved from this script; SLURM copies it, so fall back too) --
SETUP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd || echo "")"
REPO="${POME_EVAL_REPO:-}"
if [[ -z "$REPO" ]]; then
    if [[ -n "$SETUP_DIR" && -d "$SETUP_DIR/../scripts" ]]; then
        REPO="$(cd "$SETUP_DIR/.." && pwd)"
    else
        REPO="$HOME/Projects/POME_Evaluation.git"
    fi
fi
POME_REPO="${POME_REPO:-$HOME/Projects/POME.git}"
UMAP_REPO="${UMAP_REPO:-$HOME/Projects/umap-na-aware.git}"
WS="$(ws_find pome_eval)"
SIF="${POME_EVAL_SIF:-$WS/containers/pome_eval.sif}"

# ---- NA-aware umap fork ----------------------------------------------------
# The image ships stock umap-learn (no pairwise-removal NaN distances, no
# transform_combined). The fork imports itself absolutely (`import umap.*`) but
# keeps its package contents at the repo root, so expose it through a shim
# directory holding a `umap` symlink, first on PYTHONPATH. Mirrors run.sh.
UMAP_SHIM=""
if [[ -f "$UMAP_REPO/umap_.py" ]]; then
    UMAP_SHIM="$WS/pythonpath"
    mkdir -p "$UMAP_SHIM"
    ln -sfn "$UMAP_REPO" "$UMAP_SHIM/umap"
else
    echo ">> WARNING: NA-aware umap not at $UMAP_REPO -- using stock umap-learn" >&2
fi

# ---- Wrapper options (must precede the script name) ------------------------
# --stage-out DIR : repo-relative OUTPUT dir(s) to stage on node-local NVMe.
# Passed as a flag (not an env var) so it survives #SBATCH --export=NONE; the
# STAGE_OUT env var is still honored as a fallback for in-allocation use.
STAGE_OUT="${STAGE_OUT:-}"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --stage-out)   STAGE_OUT="${2:?--stage-out needs a directory}"; shift 2 ;;
        --stage-out=*) STAGE_OUT="${1#*=}"; shift ;;
        --)            shift; break ;;
        -*)            echo "submit.sh: unknown wrapper option '$1'" >&2; exit 2 ;;
        *)             break ;;
    esac
done

SCRIPT="${1:?usage: sbatch submit.sh [--stage-out DIR] <python_script> [args...]}"; shift || true

# ---- CPU threads: match the 32 cores that come with one H200 ---------------
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"

echo ">> node=$(hostname)  job=${SLURM_JOB_ID:-<none>}"
echo ">> sif=${SIF}"
echo ">> repo=${REPO}  pome=${POME_REPO}  umap=${UMAP_SHIM:-<stock>}  ws=${WS}"
nvidia-smi -L || true

# ---- Run -------------------------------------------------------------------
#  --nv        : expose the H200 to the container
#  --pwd $REPO : repo-relative 'data'/'output' resolve into the repo tree
#  --bind $WS  : /hnvme workspace ($HOME is bound by default)
BINDS=(--bind "$REPO" --bind "$WS")
[[ "$POME_REPO" == "$HOME"/* ]] || BINDS+=(--bind "$POME_REPO")
[[ -z "$UMAP_SHIM" || "$UMAP_REPO" == "$HOME"/* ]] || BINDS+=(--bind "$UMAP_REPO")
PYPATH="${UMAP_SHIM:+$UMAP_SHIM:}$REPO/src:$POME_REPO/src"
# Packages added after the image was built (e.g. tabicl for the TabICL embeddings),
# installed with `pip install --no-deps --target $WS/pylib ...` inside the image.
[[ -d "$WS/pylib" ]] && PYPATH="$PYPATH:$WS/pylib"

# ---- Optional: stage heavy OUTPUT dirs to node-local NVMe ($TMPDIR) --------
# Imputation/embedding jobs emit thousands of small files; writing those to the
# NFS $HOME repo during the run is slow (small-file metadata latency). Pass
# --stage-out with a space-separated list of repo-relative OUTPUT dirs: writes
# are redirected to fast local NVMe and bulk-copied back to the repo at the end
# (the copy also runs on early exit -- $TMPDIR is wiped when the job ends, so it
# MUST happen inside the allocation; leave a few minutes of walltime headroom).
# These dirs are treated as write-only sinks: their existing contents are hidden
# from the job while staged. Example:
#   sbatch container/submit.sh --stage-out data/imputation_data/pome_based \
#       src/pome_evaluation/impute_graph_based.py
STAGE_ROOT="${TMPDIR:-/tmp}/pome_eval_stage.${SLURM_JOB_ID:-$$}"
declare -a STAGED=()
copy_back() {
    local d src
    for d in "${STAGED[@]}"; do
        src="$STAGE_ROOT/$d"
        [[ -d "$src" ]] || continue
        echo ">> Copying $d back to repo ($(du -sh "$src" 2>/dev/null | cut -f1))"
        mkdir -p "$REPO/$d"
        if command -v rsync >/dev/null 2>&1; then
            rsync -a "$src"/ "$REPO/$d"/ || echo ">> WARNING: rsync of $d failed"
        else
            cp -a "$src"/. "$REPO/$d"/ || echo ">> WARNING: cp of $d failed"
        fi
    done
}
if [[ -n "${STAGE_OUT:-}" ]]; then
    trap copy_back EXIT
    for d in $STAGE_OUT; do
        mkdir -p "$STAGE_ROOT/$d" "$REPO/$d"
        BINDS+=(--bind "$STAGE_ROOT/$d:$REPO/$d")
        STAGED+=("$d")
        echo ">> Staging OUTPUT $d -> $STAGE_ROOT/$d (node-local NVMe)"
    done
fi

apptainer run --nv "${BINDS[@]}" --pwd "$REPO" \
    --env PYTHONPATH="$PYPATH" \
    "$SIF" "$SCRIPT" "$@"

echo ">> Finished. Outputs under ${REPO}/output (and data/ for embeddings)."
