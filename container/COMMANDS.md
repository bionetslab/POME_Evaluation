# POME_Evaluation — job command cheatsheet

Run from the repo root. Full docs: `container/README.md`.

## Build (once, on a frontend)
```bash
bash container/build.sh
```

## Re-run one script / notebook (CPU, no Slurm — the common case)
```bash
container/run.sh scripts/generate_imputation_figure.py --dim 32   # a script
container/run.sh scripts/plot_fig4_unsupervised.ipynb             # a notebook (in place)
container/run.sh --html scripts/plot_fig4_unsupervised.ipynb      # + export HTML
container/run.sh lab --port 8890                                  # Jupyter Lab (prints tunnel cmd)
container/run.sh shell                                            # shell in the container
```

## GPU jobs (embeddings / imputation → H200)
```bash
# STAGE_OUT redirects the output dir's write storm to node-local NVMe, copies back at end
STAGE_OUT="data/imputation_data/pome_based" \
    sbatch container/submit.sh src/pome_evaluation/impute_graph_based.py
STAGE_OUT="output/inductive" \
    sbatch container/submit.sh scripts/generate_inductive_embeddings.py
```

## Monitor
```bash
squeue --me -o "%.10i %.9P %.2t %.10M %C %b %R"
scontrol show job <id> | grep -iE "Partition=|TresPerNode|JobState"
tail -f /hnvme/workspace/y105dc10-pome_eval/logs/pome_eval-<id>.out
scancel <id>
```

## Notes
- Always **H200**, never H100 (0 H100 budget → pends forever).
- Only `generate_*embeddings*`, `generate_imputation_epoch_results.py` and
  `impute_graph_based.py` need the GPU; everything
  else is CPU — use `run.sh`.
