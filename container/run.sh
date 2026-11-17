#!/bin/bash -l
# Run a single POME_Evaluation script or notebook inside the container -- the
# everyday "re-run one analysis/plotting step" tool. Runs on the CURRENT node
# (a frontend for CPU-only work, or inside an salloc/job for GPU work); it does
# NOT submit to Slurm. For GPU-heavy embedding generation use submit.sh instead.
#
# Usage:
#   container/run.sh [options] <script.py>   [args...]   # run a python script
#   container/run.sh [options] <notebook.ipynb>          # execute a notebook
#   container/run.sh [options] lab                       # launch Jupyter Lab
#   container/run.sh [options] shell                     # interactive shell
#   container/run.sh [options] python -c '...'           # raw passthrough
#
# Options:
#   --gpu             expose the GPU (adds `apptainer --nv`); default is CPU
#   --to FMT          nbconvert output format (default: notebook, in-place)
#   --html            shorthand for `--to html`
#   --port N          port for `lab` (default 8888)
#   -h, --help        show this help
#
# Paths (override via env): POME_EVAL_REPO, POME_REPO, UMAP_REPO, POME_EVAL_SIF.
# UMAP_REPO points at the NA-aware umap fork (default ~/Projects/umap-na-aware.git);
# it shadows the stock umap-learn in the image via PYTHONPATH.
# Scripts read `data/` and write `output/` RELATIVE TO THE REPO ROOT, so the
# container always runs with the repo as its working directory.
set -euo pipefail

# --- locate everything relative to this script -----------------------------
SETUP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${POME_EVAL_REPO:-$(cd "$SETUP_DIR/.." && pwd)}"          # the eval repo root
POME_REPO="${POME_REPO:-$HOME/Projects/POME.git}"              # provides `import pome`
UMAP_REPO="${UMAP_REPO:-$HOME/Projects/umap-na-aware.git}"     # provides `import umap`
WS="$(ws_find pome_eval)"
SIF="${POME_EVAL_SIF:-$WS/containers/pome_eval.sif}"

# --- parse options ----------------------------------------------------------
NV=""; TO="notebook"; PORT="8888"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu)   NV="--nv"; shift ;;
    --to)    TO="$2"; shift 2 ;;
    --html)  TO="html"; shift ;;
    --port)  PORT="$2"; shift 2 ;;
    -h|--help) sed -n '2,26p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    --) shift; break ;;
    *) break ;;
  esac
done
[[ $# -ge 1 ]] || { echo "error: nothing to run (see --help)" >&2; exit 2; }

[[ -f "$SIF" ]] || { echo "error: image not found: $SIF (run container/build.sh)" >&2; exit 1; }
[[ -d "$POME_REPO/src/pome" ]] || echo ">> warning: pome source not at $POME_REPO/src (set POME_REPO)" >&2

# --- NA-aware umap fork ------------------------------------------------------
# The image ships stock umap-learn, which has neither the pairwise-removal
# (NaN-aware) euclidean/hamming distances nor `transform_combined`. The fork's
# modules import each other absolutely (`import umap.distances as dist`), so it
# must be importable as a package literally named `umap` -- but its package
# contents sit at the repo ROOT. Materialise a shim directory holding a `umap`
# symlink to the repo and put that first on PYTHONPATH, ahead of site-packages.
UMAP_SHIM=""
if [[ -f "$UMAP_REPO/umap_.py" ]]; then
  UMAP_SHIM="$WS/pythonpath"
  mkdir -p "$UMAP_SHIM"
  ln -sfn "$UMAP_REPO" "$UMAP_SHIM/umap"
else
  echo ">> warning: NA-aware umap source not at $UMAP_REPO (set UMAP_REPO);" >&2
  echo ">>          falling back to the stock umap-learn inside the image" >&2
fi

# --- assemble the apptainer invocation -------------------------------------
#  --pwd $REPO      : repo-relative 'data'/'output' paths resolve correctly
#  --bind           : $HOME is bound by default (repo + POME live there); the
#                     /hnvme workspace is not, so bind it explicitly
#  PYTHONNOUSERSITE : baked into the image, so host site-packages never leak in
BINDS=(--bind "$REPO" --bind "$WS")
[[ "$POME_REPO" == "$HOME"/* ]] || BINDS+=(--bind "$POME_REPO")   # bind only if outside $HOME
[[ -z "$UMAP_SHIM" || "$UMAP_REPO" == "$HOME"/* ]] || BINDS+=(--bind "$UMAP_REPO")
PYPATH="${UMAP_SHIM:+$UMAP_SHIM:}$REPO/src:$POME_REPO/src"
APPTAINER=(apptainer run $NV "${BINDS[@]}" --pwd "$REPO"
           --env PYTHONPATH="$PYPATH"
           "$SIF")

cmd="$1"; shift || true
case "$cmd" in
  shell)
    exec apptainer shell $NV "${BINDS[@]}" --pwd "$REPO" \
         --env PYTHONPATH="$PYPATH" "$SIF"
    ;;
  lab)
    echo ">> Jupyter Lab on ${HOSTNAME}:${PORT}  (repo=$REPO)"
    echo ">> From your laptop:  ssh -N -L ${PORT}:${HOSTNAME}:${PORT} ${USER}@${HOSTNAME}.nhr.fau.de"
    exec "${APPTAINER[@]}" -m jupyter lab --no-browser \
         --ip=127.0.0.1 --port="$PORT" --notebook-dir="$REPO"
    ;;
  python)
    exec "${APPTAINER[@]}" "$@"             # runscript is `python`, so forward args
    ;;
  jupyter|nbconvert)
    exec "${APPTAINER[@]}" -m "$cmd" "$@"   # python -m jupyter|nbconvert ...
    ;;
  bash)
    exec apptainer exec $NV "${BINDS[@]}" --pwd "$REPO" \
         --env PYTHONPATH="$PYPATH" "$SIF" bash "$@"
    ;;
  *.ipynb)
    nb="$cmd"
    [[ -f "$REPO/$nb" || -f "$nb" ]] || { echo "error: notebook not found: $nb" >&2; exit 1; }
    echo ">> Executing notebook: $nb  (--to $TO)"
    exec "${APPTAINER[@]}" -m jupyter nbconvert --to "$TO" --execute \
         $([[ "$TO" == notebook ]] && echo --inplace) \
         --ExecutePreprocessor.timeout=-1 "$nb"
    ;;
  *.py)
    echo ">> Running: $cmd $*"
    exec "${APPTAINER[@]}" "$cmd" "$@"
    ;;
  *)
    echo "error: don't know how to run '$cmd' (expected *.py, *.ipynb, lab, shell, python ...)" >&2
    exit 2
    ;;
esac
