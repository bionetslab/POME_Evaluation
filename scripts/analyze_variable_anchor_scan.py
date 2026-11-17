"""Summary of the e_j anchor scan (trainable variable encodings anchored to their spectral init).

Compares, on MIMIC incl. the two targets, the variable representations of

  stable_k8                  frozen encoding table (e_j fixed at the spectral init; deterministic)
  stable_k8_learnvar         trainable e_j, no anchor
  stable_k8_learnvar_a{w}    trainable e_j with penalty w * mean_j ||e_j - e_j_init||^2

per encoding size:
  drift          median over lab variables of cos(e_j, e_j_init) and ||e_j - e_j_init|| / ||e_j_init||
                 (sample and value rows are frozen, so all runs share the init's coordinate frame)
  reproducibility  lab-lab cosine-matrix Spearman between seeds; Leiden ARI between seeds
  semantics      Spearman of target-to-lab cosine with CatBoost importance (mean over seeds, and
                 fraction of seeds with permutation p < 0.05), Aplasia and NF

Prerequisites (per config): generate_variable_representations.py --pome-config CONFIG, then
analyze_variable_representation_stability.py --pome-config CONFIG --importance-dir data/input_datasets

    container/run.sh scripts/analyze_variable_anchor_scan.py

Output: output/variable_embeddings/variable_anchor_scan.csv (+ printed table)
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REP_ROOT = PROJECT_ROOT / "data" / "embeddings" / "MIMIC"
STAB_ROOT = PROJECT_ROOT / "output" / "variable_embeddings"
CONFIGS = ["stable_k8", "stable_k8_learnvar_a100", "stable_k8_learnvar_a10",
           "stable_k8_learnvar_a1", "stable_k8_learnvar_a0.1", "stable_k8_learnvar"]
OPTION = "centroid_input"


def drift(config, dim):
    init_dir = REP_ROOT / "variable_representations_stable_k8" / f"dim_{dim}" / "run_0"
    e0 = pd.read_csv(init_dir / "raw.tsv", sep="\t", index_col=0)
    labs = ~e0.index.astype(str).str.startswith("label_")
    out = []
    for run_dir in sorted((REP_ROOT / f"variable_representations_{config}" / f"dim_{dim}")
                          .glob("run_*")):
        e = pd.read_csv(run_dir / "raw.tsv", sep="\t", index_col=0).loc[e0.index]
        A, B = e0.to_numpy()[labs], e.to_numpy()[labs]
        na = np.linalg.norm(A, axis=1)
        cos = (A * B).sum(1) / (na * np.linalg.norm(B, axis=1))
        out.append((np.median(cos), np.median(np.linalg.norm(B - A, axis=1) / na)))
    return np.mean(out, axis=0) if out else (np.nan, np.nan)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--configs", nargs="+", default=CONFIGS)
    p.add_argument("--output", type=Path, default=STAB_ROOT / "variable_anchor_scan.csv")
    args = p.parse_args()

    rows = []
    for config in args.configs:
        stab_dir = STAB_ROOT / f"representation_stability_{config}"
        if not stab_dir.exists():
            print(f"[skip] {config}: no {stab_dir.name}")
            continue
        stab = pd.read_csv(stab_dir / "stability_summary.csv")
        stab = stab[stab["option"] == OPTION]
        imp = pd.read_csv(stab_dir / "importance_correlation.csv")
        imp = imp[imp["option"] == OPTION]
        for dim in sorted(stab["dim"].unique()):
            s = stab[stab["dim"] == dim].set_index("metric")["mean"]
            cos0, rel = drift(config, dim)
            row = dict(config=config, dim=dim, n_runs=int(imp[imp["dim"] == dim]["run"].nunique()),
                       cos_to_init=cos0, rel_change=rel,
                       lablab_rho=s.get("lablab_spearman", np.nan),
                       leiden_ari=s.get("leiden_ari", np.nan))
            for target in ("Aplasia", "NF"):
                sub = imp[(imp["dim"] == dim) & (imp["target"] == target)]
                row[f"imp_rho_{target}"] = sub["spearman"].mean()
                row[f"imp_rho_sd_{target}"] = sub["spearman"].std()
                row[f"imp_p05_{target}"] = (sub["perm_p"] < 0.05).mean()
            rows.append(row)
    res = pd.DataFrame(rows)
    res.to_csv(args.output, index=False)
    with pd.option_context("display.width", 220):
        print(res.round(3).to_string(index=False))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
