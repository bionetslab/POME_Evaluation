#!/bin/bash -l
# Build the POME_Evaluation Apptainer image on a Helma FRONTEND (not in a job).
#
#   bash container/build.sh
#
# Why the frontend: NHR@FAU only allows container builds on the AlmaLinux
# frontends (Fritz/Alex/Helma/Woody). We point Apptainer's scratch at fast
# node-local disk (thousands of unpack files would otherwise burn the
# workspace's limited inodes) and keep only the final single-file .sif in the
# pome_eval workspace.
set -euo pipefail

# --- locate paths relative to this script (works from anywhere) -------------
SETUP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS="$(ws_find pome_eval)"                      # /hnvme/workspace/<user>-pome_eval
SIF="${WS}/containers/pome_eval.sif"
mkdir -p "${WS}/containers"

# Large caches/temp on local disk (many inodes) -- NOT on Lustre/workspace.
export APPTAINER_CACHEDIR="${TMPDIR:-/tmp}/${USER}-apptainer-cache"
export APPTAINER_TMPDIR="${TMPDIR:-/tmp}/${USER}-apptainer-tmp"
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"

echo ">> Building ${SIF}"
cd "$SETUP_DIR"     # so 'requirements.txt' in the %files section resolves

# If you hit 'Could not write info to setgroups: Permission denied', swap the
# next line for: /apps/singularity/apptainer-wrapper.sh build "$SIF" pome_eval.def
apptainer build --force "$SIF" pome_eval.def

echo ">> Done. Image: ${SIF}"
apptainer inspect "$SIF" | sed 's/^/   /'

# Free the (now redundant) layer cache to reclaim local disk.
apptainer cache clean --force >/dev/null 2>&1 || true
