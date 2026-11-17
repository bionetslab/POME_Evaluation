#!/bin/bash -l
#
# Submit the POME imputation grid as 18 independent Slurm jobs, one per
# (discretization, bins, dim) cell. Each job runs all datasets for its cell and
# is capped at 24h (override with TIME=). Existing imputed .tsv files are
# skipped, so re-submitting only fills gaps.
#
#   container/submit_imputation_grid.sh                 # submit all 18
#   DRY_RUN=1 container/submit_imputation_grid.sh       # print sbatch cmds only
#   DATASETS="MIMIC" container/submit_imputation_grid.sh   # one dataset
#   TIME=12:00:00 DIMS="32" container/submit_imputation_grid.sh  # scope down
#
# Env overrides: DISCS, BINS, DIMS (space-separated lists), DATASETS (forwarded
# to the script), TIME (Slurm walltime), DRY_RUN (1 = echo, don't submit).

set -euo pipefail

SETUP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUBMIT="$SETUP_DIR/submit.sh"
SCRIPT="src/pome_evaluation/impute_graph_based.py"

DISCS="${DISCS:-z nonlinear}"
BINS="${BINS:-7 11 15}"
DIMS="${DIMS:-16 32 64}"
TIME="${TIME:-24:00:00}"
DRY_RUN="${DRY_RUN:-0}"

# Stage the write storm to node-local NVMe; jobs write disjoint imputed_* dirs.
# Passed as submit.sh's --stage-out flag (survives #SBATCH --export=NONE, which
# an env var would not). Set STAGE_OUT="" to disable staging.
STAGE_OUT="${STAGE_OUT-data/imputation_data/pome_based}"
stage_args=()
[[ -n "$STAGE_OUT" ]] && stage_args=(--stage-out "$STAGE_OUT")

# Optional dataset scoping, forwarded to the Python script.
ds_args=()
[[ -n "${DATASETS:-}" ]] && ds_args=(--datasets $DATASETS)

n=0
for disc in $DISCS; do
  for b in $BINS; do
    for dim in $DIMS; do
      n=$((n + 1))
      cmd=(sbatch --time="$TIME" --job-name="impute_${disc}_${b}_${dim}"
           "$SUBMIT" "${stage_args[@]}" "$SCRIPT"
           --discretizations "$disc" --bins "$b" --dims "$dim" "${ds_args[@]}")
      if [[ "$DRY_RUN" == "1" ]]; then
        echo "${cmd[*]}"
      else
        echo ">> [$n] submitting disc=$disc bins=$b dim=$dim"
        "${cmd[@]}"
      fi
    done
  done
done

echo ">> ${n} job(s) $([[ "$DRY_RUN" == "1" ]] && echo listed || echo submitted)."
