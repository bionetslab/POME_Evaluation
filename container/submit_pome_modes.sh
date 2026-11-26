#!/bin/bash -l
# Fan out the inductive POME embedding generation over several Slurm jobs.
#
# POME's inductive epoch tuning runs a sample-holdout CV inside EVERY fit (see
# `Embedder(inductive=True)`), so one fit is far more expensive than a plain
# training run and a single job covering all datasets/modes/splits can blow past
# submit.sh's 6 h walltime. This script shards the work along
# (dataset x mode x split-group) and sbatches one job per shard, so the shards
# run concurrently and each stays short.
#
# Usage:
#   container/submit_pome_modes.sh [options]
#
# Options (all have defaults suited to the data-integration figure):
#   --datasets "hancock luad_noleak"        datasets to cover
#   --modes    "numeric_only cat_only"   variable-type modes to cover
#   --dims     "64"                  embedding sizes
#   --runs     5                     runs per (split, dim)
#   --epochs   2000                  fixed training epochs (POME >= 1.5, no early stopping)
#   --splits   "0 1 2 3 4"           restrict to these split ids (default: all
#                                    found); use to submit only the shards that
#                                    are not already queued/running
#   --splits-per-job 5               split ids per job (10 splits -> 2 jobs each)
#   --time     06:00:00              walltime per job
#   --output-dir DIR                 embedding root (default: the generator's,
#                                    output/inductive_pome14)
#   --only-first                     submit just the first shard (to time a fit)
#   --dry-run                        print the sbatch commands, submit nothing
#
# POME >= 1.5 trains inductive mode to this fixed budget without early stopping
# (POME's and the generator's default is 2000).
#
# Each job gets a unique --log-tag so the per-run tuned-epochs logs never race
# (generate_inductive_embeddings.py rewrites that file after every run). When
# ALL jobs have finished, fold the shards back together:
#
#   container/run.sh scripts/merge_tuned_epochs.py
#
# The generator skips existing outputs, so re-running this script after a
# failed/timed-out shard only recomputes what is missing.
set -euo pipefail

SETUP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SETUP_DIR/.." && pwd)"

DATASETS="hancock luad_noleak"
MODES="numeric_only cat_only"
DIMS="64"
RUNS=5
EPOCHS=2000
SPLITS=""
SPLITS_PER_JOB=5
WALLTIME="06:00:00"
OUTPUT_DIR=""
DRY_RUN=0
ONLY_FIRST=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --datasets)       DATASETS="$2"; shift 2 ;;
        --modes)          MODES="$2"; shift 2 ;;
        --dims)           DIMS="$2"; shift 2 ;;
        --runs)           RUNS="$2"; shift 2 ;;
        --epochs)         EPOCHS="$2"; shift 2 ;;
        --splits)         SPLITS="$2"; shift 2 ;;
        --splits-per-job) SPLITS_PER_JOB="$2"; shift 2 ;;
        --time)           WALLTIME="$2"; shift 2 ;;
        --output-dir)     OUTPUT_DIR="$2"; shift 2 ;;
        --only-first)     ONLY_FIRST=1; shift ;;
        --dry-run)        DRY_RUN=1; shift ;;
        -h|--help)        sed -n '2,30p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

# Split ids come from the split files themselves, so the sharding follows
# whatever data/train_test_splits/ actually contains.
n_jobs=0
for dataset in $DATASETS; do
    mapfile -t split_ids < <(
        find "$REPO/data/train_test_splits/$dataset" -name 'split_*_train_pome.tsv' \
            -printf '%f\n' 2>/dev/null | sed -E 's/split_0*([0-9]+)_train_pome\.tsv/\1/' | sort -n
    )
    if [[ -n "$SPLITS" ]]; then
        # Keep only the requested ids, preserving the discovered (sorted) order.
        keep=()
        for s in "${split_ids[@]}"; do
            for want in $SPLITS; do
                [[ "$s" == "$want" ]] && keep+=("$s") && break
            done
        done
        split_ids=("${keep[@]}")
    fi
    if [[ ${#split_ids[@]} -eq 0 ]]; then
        echo ">> warning: no pome splits for $dataset -- skipping" >&2
        continue
    fi

    for mode in $MODES; do
        for ((i = 0; i < ${#split_ids[@]}; i += SPLITS_PER_JOB)); do
            chunk=("${split_ids[@]:i:SPLITS_PER_JOB}")
            tag="${dataset}_${mode}_s${chunk[0]}-${chunk[-1]}"
            cmd=(sbatch
                 --job-name="pome_${tag}"
                 --time="$WALLTIME"
                 --export=PYTHONHASHSEED=0
                 "$SETUP_DIR/submit.sh"
                 scripts/generate_inductive_embeddings.py
                 --methods pome
                 --datasets "$dataset"
                 --modes "$mode"
                 --dims $DIMS
                 --runs "$RUNS"
                 --epochs "$EPOCHS"
                 --splits "${chunk[@]}"
                 --log-tag "$tag"
                 ${OUTPUT_DIR:+--output-dir "$OUTPUT_DIR"})
            if [[ $DRY_RUN -eq 1 ]]; then
                printf '%q ' "${cmd[@]}"; echo
            else
                "${cmd[@]}"
            fi
            n_jobs=$((n_jobs + 1))
            if [[ $ONLY_FIRST -eq 1 ]]; then
                echo ">> --only-first: stopping after one shard."
                echo ">> Time a fit from its log, then submit the rest:"
                echo ">>     grep '\[ok\]' \$(ws_find pome_eval)/logs/pome_${tag}-*.out"
                break 3
            fi
        done
    done
done

echo
if [[ $DRY_RUN -eq 1 ]]; then
    echo ">> [dry-run] $n_jobs job(s) would be submitted; nothing was queued."
else
    echo ">> Submitted $n_jobs job(s). Watch with: squeue --me"
    echo ">> When ALL have finished, merge the tuned-epoch logs:"
    echo ">>     container/run.sh scripts/merge_tuned_epochs.py"
fi
