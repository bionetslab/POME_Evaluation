"""Invariance of MIMIC variable encodings to POME's exact parameter symmetries (supplement).

A value node's input is normalize(e_j + e_v), so several transformations of the trained encoding
table leave the fitted model exactly unchanged (same sample embeddings, decoder scores, loss):

    cat_shift     e_j += a, e_c -= a for the class codes c of each categorical variable j
    cat_rescale   e_j + e_c -> lambda_c (e_j + e_c) per categorical value node
    num_shift     e_j += a for all numeric variables, e_b -= a for all (shared) bin encodings
    global_scale  all variable/value encodings *= s
    rotation      all encodings -> Q e, absorbed by the first GAT layer (W -> W Q^T)
    all           all of the above combined

For each fitted model and transformation this script reports the maximum absolute change of the
variable-variable cosine similarity matrix (all 102 variables incl. the two targets) for the raw
encodings e_j (POME <= 1.1) and the value-node centroids returned by POME >= 1.2, plus the change
of the sample embeddings as a control that the model itself is unchanged.

Fit + measure (GPU):  sbatch container/submit.sh scripts/variable_encoding_invariance.py
Plot only (CPU):      container/run.sh scripts/variable_encoding_invariance.py --plot-only
Output:  output/variable_embeddings/variable_encoding_invariance.{csv,pdf,png}
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
OUT = PROJECT_ROOT / "output" / "variable_embeddings" / "variable_encoding_invariance"
TRANSFORMS = ["cat_shift", "cat_rescale", "num_shift", "global_scale", "rotation", "all"]
TRANSFORM_LABELS = {"cat_shift": "Categorical\nshift", "cat_rescale": "Categorical\nrescaling",
                    "num_shift": "Numeric\nshift", "global_scale": "Global\nscale",
                    "rotation": "Rotation", "all": "All\ncombined"}
SERIES = {"raw": "Original (raw $e_j$)", "centroid_input": "Revised (value-node centroids)",
          "sample_embeddings": "Sample embeddings (control)"}
FLOOR = 1e-8   # zero changes are shown at this value on the log axis


def cosine_matrix(X):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    return X @ X.T


def representations(embedder):
    """Raw e_j from the live weights, revised centroids, and sample embeddings."""
    import torch
    device = embedder.model.node_embeddings.weight.device
    with torch.no_grad():
        latent, _, _ = embedder.model.get_embeddings(embedder._graph_data.edge_index.to(device))
    embedder._all_embeddings = latent.detach().cpu()   # fit() also stores them on the CPU
    samples, centroids, _, _ = embedder.get_embeddings()
    model = embedder.model
    raw = model.node_embeddings.weight.detach()[model.variable_embeddings_ids].cpu().numpy()
    return dict(raw=raw, centroid_input=centroids.to_numpy(), sample_embeddings=samples.to_numpy())


def apply_transform(embedder, kind, rng):
    """Apply one exact symmetry in place to the trained encoding table."""
    import torch
    model = embedder.model
    W = model.node_embeddings.weight.data
    dev, dt = W.device, W.dtype
    n2e = model.node_to_embeddings.to(dev)
    n = len(embedder._sample_node_dict)
    var_rows = dict(zip(embedder._variable_names, model.variable_embeddings_ids.tolist()))
    value_n2e = n2e[n:]
    scale = float(W[n:].norm(dim=1).mean())

    def random_vector():
        return torch.tensor(rng.normal(size=W.size(1)) * scale, dtype=dt, device=dev)

    for var, row in var_rows.items():
        if var in embedder._cont_vars:
            continue
        codes = value_n2e[value_n2e[:, 0] == row, 1]
        if kind in ("cat_shift", "all"):
            a = random_vector()
            W[row] += a
            W[codes] -= a
        if kind in ("cat_rescale", "all"):
            for c in codes.tolist():
                lam = float(np.exp(rng.uniform(np.log(0.2), np.log(5))))
                W[c] = lam * (W[row] + W[c]) - W[row]
    if kind in ("num_shift", "all"):
        rows = torch.tensor([var_rows[v] for v in embedder._variable_names
                             if v in embedder._cont_vars], device=dev)
        a = random_vector()
        W[rows] += a
        W[model.bin_embedding_ids.long().to(dev)] -= a
    if kind in ("global_scale", "all"):
        W[n:] *= 3.7
    if kind in ("rotation", "all"):
        Q, _ = np.linalg.qr(rng.normal(size=(W.size(1), W.size(1))))
        Q = torch.tensor(Q, dtype=dt, device=dev)
        W.copy_(W @ Q.T)
        lin = model.encoder.convs[0].lin.weight
        lin.data = lin.data @ Q.T


def measure(dims, seeds, epochs, device):
    import torch
    from pome import Embedder, make_deterministic
    from generate_variable_representations import NA_ENCODING, load_mimic_with_targets

    device = device if (device == "cpu" or torch.cuda.is_available()) else "cpu"
    data = load_mimic_with_targets()
    rows = []
    for dim in dims:
        for seed in seeds:
            print(f"[fit] dim {dim} seed {seed}", flush=True)
            make_deterministic(seed)
            embedder = Embedder(embedding_dimension=dim, epochs=epochs, na_encoding=NA_ENCODING,
                                device=device)
            embedder.fit(data.copy())
            state = {k: v.clone() for k, v in embedder.model.state_dict().items()}
            base = representations(embedder)
            base_cos = {k: cosine_matrix(base[k]) for k in ("raw", "centroid_input")}
            rng = np.random.default_rng(seed)
            for kind in TRANSFORMS:
                embedder.model.load_state_dict(state)
                apply_transform(embedder, kind, rng)
                reps = representations(embedder)
                for k in ("raw", "centroid_input"):
                    change = np.abs(cosine_matrix(reps[k]) - base_cos[k]).max()
                    rows.append(dict(dim=dim, seed=seed, transform=kind, series=k,
                                     max_abs_change=float(change)))
                # Control: sample embeddings are compared via cosines too (rotation-invariant).
                change = np.abs(cosine_matrix(reps["sample_embeddings"])
                                - cosine_matrix(base["sample_embeddings"])).max()
                rows.append(dict(dim=dim, seed=seed, transform=kind, series="sample_embeddings",
                                 max_abs_change=float(change)))
            embedder.model.load_state_dict(state)
    df = pd.DataFrame(rows)
    df.to_csv(OUT.with_suffix(".csv"), index=False)
    return df


def plot(df):
    import matplotlib.pyplot as plt
    import seaborn as sns

    df = df.assign(value=df["max_abs_change"].clip(lower=FLOOR),
                   label=df["series"].map(SERIES),
                   transform_label=df["transform"].map(TRANSFORM_LABELS))
    print(df.groupby(["series", "transform"])["max_abs_change"].max().unstack()[TRANSFORMS]
          .to_string(float_format=lambda x: f"{x:.1e}"))
    palette = dict(zip(SERIES.values(), ["#9e9e9e", sns.color_palette("tab10")[0], "#4d4d4d"]))
    fig, ax = plt.subplots(figsize=(9, 4.5), layout="constrained")
    sns.stripplot(data=df, x="transform_label", y="value", hue="label",
                  order=[TRANSFORM_LABELS[t] for t in TRANSFORMS], hue_order=list(SERIES.values()),
                  palette=palette, dodge=True, size=6, alpha=0.8, jitter=0.15, ax=ax)
    ax.set_yscale("log")
    ax.set_ylim(FLOOR / 3, 10)
    ticks = [FLOOR] + [10.0 ** k for k in range(-6, 1, 2)]
    ax.set_yticks(ticks, ["0"] + [f"$10^{{{k}}}$" for k in range(-6, 1, 2)])
    ax.minorticks_off()
    ax.set_xlabel("Model-preserving transformation of the trained encodings", fontsize=13)
    ax.set_ylabel("Max. absolute change of\ncosine similarities", fontsize=13)
    ax.tick_params(labelsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    handles, labels = ax.get_legend_handles_labels()
    ax.get_legend().remove()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, fontsize=11, frameon=False)
    fig.savefig(OUT.with_suffix(".pdf"))
    fig.savefig(OUT.with_suffix(".png"), dpi=150)
    print(f"Wrote {OUT.with_suffix('.pdf')}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dims", nargs="+", type=int, default=[16, 32, 64])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--plot-only", action="store_true", help="re-plot the saved CSV")
    args = parser.parse_args()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df = (pd.read_csv(OUT.with_suffix(".csv")) if args.plot_only
          else measure(args.dims, args.seeds, args.epochs, args.device))
    plot(df)


if __name__ == "__main__":
    main()
