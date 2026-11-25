"""Does the input ("self-loop") feature of unseen samples matter in POME's inductive mode?

``Embedder.transform()`` adds each new sample as a receive-only node whose own input feature is a
zero vector; its embedding is then meant to come entirely from the value nodes it connects to.
Training samples, by contrast, enter the encoder with their (fixed) spectral coordinate x_i as
input. This script checks systematically whether the choice of the new-sample input changes the
embedding, by re-running the frozen inductive forward pass with different inputs:

  zero      the current implementation (all-zero input)
  spectral  the data-derived alternative: the sample's out-of-sample (Nystroem) spectral
            coordinate, i.e. exactly the input it would have had as a training sample
            (x_i = unit(sum_v A_hat_iv V_v S^-1/2), from the training graph's SVD)
  random    a random unit vector per sample (stress test: arbitrary input)
  own       training samples only: their actual training input x_i (isolates the effect of the
            receive-only edges from the effect of the input)

Per dataset x dim x split x training length E (full fit, cosine lr over E, no stopping) it records
  * training samples: cosine(transductive embedding, inductive re-embedding) for every input
  * test samples: cosine(zero-input embedding, embedding with another input)
  * test samples: linear-probe AP per target for every input (logreg on the transductive
    training embeddings, as in generate_inductive_stopping_ablation.py)

GPU part (one job per dataset; PYTHONHASHSEED=0, POME >= 1.5). The reported results were computed
with the POME branch inductive-revision (stopping_criterion="none"), whose fixed-epoch fits are
bit-identical to POME main >= 1.5:
    sbatch --job-name=init_hancock \\
        --export=POME_REPO=$HOME/Projects/POME.git,PYTHONHASHSEED=0 \\
        container/submit.sh scripts/evaluate_inductive_init.py --datasets hancock
Summary + figure (CPU, after all jobs):
    container/run.sh scripts/evaluate_inductive_init.py --summarize

Inputs:  data/train_test_splits/{dataset}/split_{NN}_{train,test}_pome.tsv, data/input_datasets
Outputs: output/inductive_init_check/{dataset}_cosines.csv, {dataset}_probes.csv (resumable),
         summary_cosines.csv, summary_probes.csv, figure_inductive_init.pdf/.png
"""
import argparse
import contextlib
import io
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from linear_probe_inductive_embeddings import load_binary_targets, probe_one  # noqa: E402
from generate_inductive_stopping_ablation import embedder, load_graph, SPLITS_ROOT  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT = PROJECT_ROOT / "output" / "inductive_init_check"
DATASETS = {"hancock": "HANCOCK", "luad": "TCGA-LUAD", "mimic": "MIMIC-IV"}
EPOCHS = [1, 25, 100, 500, 2000]
INITS = ["zero", "spectral", "random"]


def unit(m):
    import torch
    return m / m.norm(p=2, dim=1, keepdim=True).clamp(min=1e-8)


def nystroem_inputs(emb, pairs_per_sample):
    """Out-of-sample spectral coordinates of new samples, normalized like the training rows."""
    import torch
    from pome.utils import _spectral_components
    g = emb._graph_data
    ns, nn_ = g.num_samples, g.num_nodes
    if not hasattr(emb, "_nys_cache"):  # SVD of the training graph, once per fit
        U, S, V = _spectral_components(g.unique_edges, ns, nn_, emb.embedding_dimension)
        src, dst = g.unique_edges.detach().cpu().numpy()
        pairs = np.unique(np.stack([src, dst]), axis=1)
        deg_v = np.maximum(np.bincount(pairs[1] - ns, minlength=nn_ - ns), 1).astype(float)
        emb._nys_cache = (U, S, V, deg_v)
    U, S, V, deg_v = emb._nys_cache
    d = emb.embedding_dimension
    X = np.zeros((len(pairs_per_sample), d))
    for i, sp in enumerate(pairs_per_sample):
        v = np.unique([node for _, node in sp]) - ns
        a = deg_v[v] ** -0.5 / np.sqrt(len(v))          # row i of D_s^-1/2 A D_v^-1/2
        X[i, :U.shape[1]] = (a[:, None] * V[v]).sum(0) / np.sqrt(S)   # = U_i S^1/2
    return unit(torch.tensor(X, dtype=torch.float32))


def inductive_embed(emb, pairs_per_sample, new_inputs):
    """Frozen receive-only forward pass (as Embedder._inductive_forward) with given new inputs."""
    import torch
    device = next(emb.model.parameters()).device
    m = emb.model
    with torch.no_grad():
        c1 = m.node_embeddings(m.node_to_embeddings[:, 0].to(device))
        c2 = m.node_embeddings(m.node_to_embeddings[:, 1].to(device))
        base = c1 + c2
        n0 = emb._graph_data.num_nodes
        aug = torch.cat([base, new_inputs.to(device, base.dtype)], dim=0)
        ei = emb._augmented_edge_index(pairs_per_sample, n0, device)
        latent = emb._frozen_encode(aug, ei)
    return latent[n0:].cpu().numpy()


def row_cos(a, b):
    return (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))


def run_one(dataset, dim, split, epochs, targets, device, seed):
    import torch
    from pome.gnn_embedding import make_deterministic
    split_dir = SPLITS_ROOT / dataset
    train_df = load_graph(split_dir / f"split_{split:02d}_train_pome.tsv")
    test_df = load_graph(split_dir / f"split_{split:02d}_test_pome.tsv")
    make_deterministic(seed)
    from pome.gnn_embedding import Embedder
    # On the inductive-revision branch, disable its stopping rule (= POME main >= 1.5).
    branch = "stopping_criterion" in Embedder.__init__.__code__.co_varnames
    emb = embedder(dim, epochs, device, **({"stopping_criterion": "none"} if branch else {}))
    with contextlib.redirect_stdout(io.StringIO()):
        emb.fit(train_df.copy())
    meta = dict(dataset=dataset, dim=dim, split=split, epochs=epochs)

    train_emb = emb.get_embeddings()[0]
    train_emb.index = train_emb.index.astype(str)
    tr_samples = [c for c in train_df.columns if c != emb.type_column]
    te_samples = [c for c in test_df.columns if c != emb.type_column]
    tr_in, te_in = train_df.drop(columns=[emb.type_column]), test_df.drop(columns=[emb.type_column])
    tr_pairs = [emb._sample_value_node_pairs(tr_in, s) for s in tr_samples]
    te_pairs = [emb._sample_value_node_pairs(te_in, s) for s in te_samples]
    rng = torch.Generator().manual_seed(1000 + split)
    # number of value nodes per sample (the self-loop is one of degree + 1 messages per layer)
    meta["degree_train"] = float(np.mean([len({v for _, v in sp}) for sp in tr_pairs]))
    meta["degree_test"] = float(np.mean([len({v for _, v in sp}) for sp in te_pairs]))

    # Training-sample inputs as used in fit(): sample rows of the encoding table.
    m = emb.model
    with torch.no_grad():
        h0 = (m.node_embeddings(m.node_to_embeddings[:, 0]) + m.node_embeddings(m.node_to_embeddings[:, 1]))
        own = unit(h0[:emb._graph_data.num_samples].cpu())
    sample_order = [emb._sample_node_dict[s] for s in tr_samples]
    own = own[sample_order]

    def inputs(kind, pairs, n):
        if kind == "zero":
            return torch.zeros(n, dim)
        if kind == "spectral":
            return nystroem_inputs(emb, pairs)
        if kind == "random":
            return unit(torch.randn(n, dim, generator=rng))
        return own

    Z_tr = train_emb.loc[[str(s) for s in tr_samples]].to_numpy()
    rows, probe_rows = [], []
    # sanity: Nystroem reproduces the training samples' own spectral input
    nys_own = float(row_cos(nystroem_inputs(emb, tr_pairs).numpy(), own.numpy()).mean())
    for kind in INITS + ["own"]:
        T = inductive_embed(emb, tr_pairs, inputs(kind, tr_pairs, len(tr_samples)))
        c = row_cos(T, Z_tr)
        rows.append(dict(meta, samples="train", init=kind, reference="transductive",
                         cos_mean=c.mean(), cos_p05=np.quantile(c, 0.05), cos_min=c.min(),
                         nystroem_vs_own_input=nys_own))
    te = {}
    for kind in INITS:
        te[kind] = inductive_embed(emb, te_pairs, inputs(kind, te_pairs, len(te_samples)))
        if kind != "zero":
            c = row_cos(te[kind], te["zero"])
            rows.append(dict(meta, samples="test", init=kind, reference="zero",
                             cos_mean=c.mean(), cos_p05=np.quantile(c, 0.05), cos_min=c.min(),
                             nystroem_vs_own_input=nys_own))
        test_emb = pd.DataFrame(te[kind], index=[str(s) for s in te_samples],
                                columns=train_emb.columns)
        for target, y in targets.items():
            res = probe_one(train_emb, test_emb, y, standardize=False)
            if res is not None:
                probe_rows.append(dict(meta, init=kind, target=target,
                                       average_precision=res["average_precision"]))
    # the production transform() must equal the zero-input pass
    ref = emb.transform(test_df.copy()).loc[[str(s) for s in te_samples]].to_numpy()
    assert np.allclose(ref, te["zero"], atol=1e-5), "zero-input pass differs from transform()"
    return rows, probe_rows


def generate(args):
    import torch
    warnings.filterwarnings("ignore", category=UserWarning, module="torch_geometric")
    warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    args.out.mkdir(parents=True, exist_ok=True)
    for dataset in args.datasets:
        cos_path, probe_path = args.out / f"{dataset}_cosines.csv", args.out / f"{dataset}_probes.csv"
        cos_df = pd.read_csv(cos_path) if cos_path.exists() else pd.DataFrame()
        probe_df = pd.read_csv(probe_path) if probe_path.exists() else pd.DataFrame()
        done = set() if cos_df.empty else set(map(tuple, cos_df[["dim", "split", "epochs"]].values))
        targets = load_binary_targets(dataset)
        for dim in args.dims:
            for split in args.splits:
                for epochs in args.epochs:
                    if (dim, split, epochs) in done:
                        continue
                    t0 = time.perf_counter()
                    rows, prows = run_one(dataset, dim, split, epochs, targets, device, args.seed)
                    cos_df = pd.concat([cos_df, pd.DataFrame(rows)], ignore_index=True)
                    probe_df = pd.concat([probe_df, pd.DataFrame(prows)], ignore_index=True)
                    cos_df.to_csv(cos_path, index=False)
                    probe_df.to_csv(probe_path, index=False)
                    r = pd.DataFrame(rows).set_index(["samples", "init"])["cos_mean"]
                    print(f"[ok] {dataset} d{dim} split {split:02d} E{epochs:5d} "
                          f"{time.perf_counter() - t0:5.0f}s  train cos zero/spec/rand/own "
                          f"{r['train', 'zero']:.4f}/{r['train', 'spectral']:.4f}/"
                          f"{r['train', 'random']:.4f}/{r['train', 'own']:.4f}  test cos vs zero "
                          f"spec/rand {r['test', 'spectral']:.4f}/{r['test', 'random']:.4f}",
                          flush=True)


def summarize(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import wilcoxon

    cos = pd.concat([pd.read_csv(f) for f in sorted(args.out.glob("*_cosines.csv"))])
    probes = pd.concat([pd.read_csv(f) for f in sorted(args.out.glob("*_probes.csv"))])
    cs = cos.groupby(["samples", "init", "dataset", "dim", "epochs"]).agg(
        cos_mean=("cos_mean", "mean"), cos_p05=("cos_p05", "mean"), cos_min=("cos_min", "min"),
        splits=("split", "nunique")).reset_index()
    cs.to_csv(args.out / "summary_cosines.csv", index=False)

    key = ["dataset", "dim", "split", "epochs", "target"]
    w = probes.pivot_table(index=key, columns="init", values="average_precision").reset_index()
    rows = []
    for (ds, dim, ep), g in w.groupby(["dataset", "dim", "epochs"]):
        for kind in ("spectral", "random"):
            diff = (g[kind] - g["zero"]).to_numpy()
            # paired over splits (target-mean per split)
            per_split = g.assign(d=g[kind] - g["zero"]).groupby("split")["d"].mean()
            p = float(wilcoxon(per_split).pvalue) if (per_split != 0).any() else np.nan
            rows.append(dict(dataset=ds, dim=dim, epochs=ep, init=kind, ap_zero=g["zero"].mean(),
                             ap_init=g[kind].mean(), delta_ap=diff.mean(),
                             max_abs_delta=np.abs(diff).max(), p_wilcoxon=p))
    ps = pd.DataFrame(rows)
    ps.to_csv(args.out / "summary_probes.csv", index=False)
    with pd.option_context("display.width", 220, "display.max_rows", 500):
        print("== cosine similarity (mean over samples, then splits) ==")
        print(cs.pivot_table(index=["dataset", "dim", "epochs"], columns=["samples", "init"],
                             values="cos_mean").round(4).to_string())
        print("\n== linear-probe AP of test samples: other input - zero input ==")
        print(ps.round(4).to_string(index=False))

    # Figure: (a) train re-embedding vs transductive, (b) test: other input vs zero, (c) delta AP
    colors = {"hancock": "#2a78d6", "luad": "#eb6834", "mimic": "#1baf7a"}
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4), gridspec_kw=dict(wspace=0.45))
    ls = {32: "-", 64: "--"}
    panels = [("train", "zero", "a  training samples\nzero-input re-embedding\nvs. transductive"),
              ("test", "spectral", "b  unseen samples\nspectral vs. zero input")]
    for ax, (samples, init, title) in zip(axes[:2], panels):
        for (ds, dim), g in cs[(cs.samples == samples) & (cs.init == init)].groupby(["dataset", "dim"]):
            ax.plot(g.epochs, g.cos_mean, ls.get(dim, ":"), marker="o", ms=3, color=colors[ds], lw=1.2)
        ax.set_title(title, fontsize=7, loc="left", fontweight="bold")
    ax = axes[2]
    for (ds, dim), g in ps[ps.init == "spectral"].groupby(["dataset", "dim"]):
        ax.plot(g.epochs, g.delta_ap, ls.get(dim, ":"), marker="o", ms=3, color=colors[ds], lw=1.2)
    ax.axhline(0, color="#8a8986", lw=0.6)
    ax.set_ylabel("\u0394 AP (mean over targets, splits)", fontsize=7)
    for a in axes[:2]:
        a.set_ylabel("cosine similarity (mean)", fontsize=7)
    ax.set_title("c  unseen samples\nprobe AP, spectral \u2212 zero", fontsize=7, loc="left", fontweight="bold")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("training epochs", fontsize=7)
        ax.tick_params(labelsize=6)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    handles = [plt.Line2D([], [], color=c, marker="o", ms=3, label=DATASETS[d]) for d, c in colors.items()]
    handles += [plt.Line2D([], [], color="#555", ls=l, label=f"d={k}") for k, l in ls.items()]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=5,
               fontsize=6, frameon=False)
    for ext in ("pdf", "png"):
        fig.savefig(args.out / f"figure_inductive_init.{ext}", dpi=300, bbox_inches="tight")
    print(f"wrote {args.out}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--datasets", nargs="+", default=list(DATASETS))
    p.add_argument("--dims", type=int, nargs="+", default=[32, 64])
    p.add_argument("--splits", type=int, nargs="+", default=list(range(10)))
    p.add_argument("--epochs", type=int, nargs="+", default=EPOCHS)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--summarize", action="store_true")
    args = p.parse_args()
    summarize(args) if args.summarize else generate(args)


if __name__ == "__main__":
    main()
