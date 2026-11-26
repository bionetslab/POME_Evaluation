"""Does the input of unseen samples, or how many are transformed together, matter in POME's inductive mode?

``Embedder.transform()`` adds each new sample as a receive-only node (edges point only from the
training value nodes to the new sample) whose own input feature is a zero vector. Two properties
follow, which this script checks empirically on fully trained models (2000 epochs, POME's inductive
default):

  * the new-sample input should hardly matter, because its own input is one of the ~degree + 1
    messages a sample node aggregates in each GAT layer:
      zero    the current implementation (all-zero input)
      random  a random unit vector per sample (stress test: arbitrary input); one vector per
              sample, kept fixed across batch fractions
  * new samples cannot influence each other (receive-only), so the number of samples passed to
    ``transform()`` at once should not matter at all. The samples are therefore split into random
    chunks of a fraction f of the set (``--fractions``, at least one sample per chunk), and every
    chunk is embedded by its own frozen forward pass.

Per dataset x dim x split (one fit per unit, seed ``--seed``) and per input x fraction it records

  * training samples: cosine(transductive embedding, inductive re-embedding through transform())
  * test samples: cosine(production transform(), i.e. zero input with all test samples at once,
    embedding with this input and fraction)
  * test samples: linear-probe AP per target (raw logreg fit on the transductive training
    embeddings, ``probe_one(standardize=False)``) and its difference to the production embedding
  * the train / test embeddings (``emb/{dataset}/dim_{D}/split_{NN}.npz``) for the Rosenbaum
    cross-match test (scXmatch) of test vs. training samples, run by ``--crossmatch``

Figure (``--summarize``): top row (a) training-sample re-embedding vs. transductive, (b) unseen
samples vs. the production transform(), (c) probe AP difference, each against the fraction of
samples per transform() call (zero input: filled, random input: open markers); bottom row (d-f)
cross-match z per dataset on the same axis, with the model-free references of
``evaluate_inductive_ood.py`` (raw values, spectral coordinates; grey lines).

GPU part (one job per dataset; PYTHONHASHSEED=0, POME >= 1.5):
    sbatch --job-name=init_hancock --export=PYTHONHASHSEED=0 \\
        container/submit.sh scripts/evaluate_inductive_init.py --datasets hancock
Cross-match (scXmatch environment, see evaluate_inductive_ood.py; resumable):
    $WORK/envs/scxmatch/bin/python scripts/evaluate_inductive_init.py --crossmatch
Summary + figure (CPU):
    container/run.sh scripts/evaluate_inductive_init.py --summarize

Inputs:  data/train_test_splits/{dataset}/split_{NN}_{train,test}_pome.tsv, data/input_datasets,
         output/inductive_ood/ood_crossmatch_results.csv (model-free references)
Outputs: output/inductive_init_check/{dataset}_cosines.csv, {dataset}_probes.csv (resumable),
         emb/ (npz, untracked), crossmatch.csv, summary_cosines.csv, summary_probes.csv,
         summary_crossmatch.csv, figure_inductive_init.pdf/.png
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT = PROJECT_ROOT / "output" / "inductive_init_check"
OOD_RESULTS = PROJECT_ROOT / "output" / "inductive_ood" / "ood_crossmatch_results.csv"
DATASETS = {"hancock": "HANCOCK", "luad_noleak": "TCGA-LUAD", "mimic": "MIMIC-IV"}
EPOCHS = 2000
INITS = ["zero", "random"]
FRACTIONS = [0.01, 0.05, 0.1, 0.25, 0.5, 1.0]


def unit(m):
    return m / m.norm(p=2, dim=1, keepdim=True).clamp(min=1e-8)


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


def chunked_embed(emb, pairs, inputs, frac, seed):
    """Embed the samples in random chunks of a fraction `frac` (>= 1 sample), one pass per chunk."""
    n = len(pairs)
    size = max(1, int(round(frac * n)))
    order = np.random.default_rng(seed).permutation(n)
    out = np.empty((n, emb.embedding_dimension))
    for start in range(0, n, size):
        idx = order[start:start + size]
        out[idx] = inductive_embed(emb, [pairs[i] for i in idx], inputs[idx])
    return out


def row_cos(a, b):
    return (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))


def run_one(dataset, dim, split, args, targets, device):
    import torch
    from generate_inductive_stopping_ablation import embedder, load_graph, SPLITS_ROOT
    from linear_probe_inductive_embeddings import probe_one
    from pome.gnn_embedding import make_deterministic
    split_dir = SPLITS_ROOT / dataset
    train_df = load_graph(split_dir / f"split_{split:02d}_train_pome.tsv")
    test_df = load_graph(split_dir / f"split_{split:02d}_test_pome.tsv")
    make_deterministic(args.seed)
    emb = embedder(dim, args.epochs, device)
    with contextlib.redirect_stdout(io.StringIO()):
        emb.fit(train_df.copy())
    meta = dict(dataset=dataset, dim=dim, split=split, epochs=args.epochs)

    train_emb = emb.get_embeddings()[0]
    train_emb.index = train_emb.index.astype(str)
    tr_samples = [c for c in train_df.columns if c != emb.type_column]
    te_samples = [c for c in test_df.columns if c != emb.type_column]
    tr_in, te_in = train_df.drop(columns=[emb.type_column]), test_df.drop(columns=[emb.type_column])
    tr_pairs = [emb._sample_value_node_pairs(tr_in, s) for s in tr_samples]
    te_pairs = [emb._sample_value_node_pairs(te_in, s) for s in te_samples]
    gen = torch.Generator().manual_seed(1000 + split)
    inputs = {"train": {"zero": torch.zeros(len(tr_samples), dim),
                        "random": unit(torch.randn(len(tr_samples), dim, generator=gen))},
              "test": {"zero": torch.zeros(len(te_samples), dim),
                       "random": unit(torch.randn(len(te_samples), dim, generator=gen))}}

    Z_tr = train_emb.loc[[str(s) for s in tr_samples]].to_numpy()
    te_index = [str(s) for s in te_samples]
    prod = emb.transform(test_df.copy()).loc[te_index].to_numpy()   # production transform()
    rows, probe_rows, arrays = [], [], {"train_transductive": Z_tr}

    def probes(Z):
        test_emb = pd.DataFrame(Z, index=te_index, columns=train_emb.columns)
        out = {}
        for target, y in targets.items():
            res = probe_one(train_emb, test_emb, y, standardize=False)   # raw logreg
            if res is not None:
                out[target] = res["average_precision"]
        return out

    ap_prod = probes(prod)
    for kind in INITS:
        for frac in args.fractions:
            seed = 10_000 * split + int(round(1000 * frac))
            T = chunked_embed(emb, tr_pairs, inputs["train"][kind], frac, seed)
            c = row_cos(T, Z_tr)
            rows.append(dict(meta, samples="train", init=kind, fraction=frac,
                             reference="transductive", cos_mean=c.mean(),
                             cos_p05=np.quantile(c, 0.05), cos_min=c.min()))
            E = chunked_embed(emb, te_pairs, inputs["test"][kind], frac, seed + 1)
            c = row_cos(E, prod)
            rows.append(dict(meta, samples="test", init=kind, fraction=frac,
                             reference="production", cos_mean=c.mean(),
                             cos_p05=np.quantile(c, 0.05), cos_min=c.min()))
            for target, ap in probes(E).items():
                probe_rows.append(dict(meta, init=kind, fraction=frac, target=target,
                                       average_precision=ap, ap_production=ap_prod[target]))
            arrays[f"test_{kind}_f{frac:g}"] = E
    # the zero-input single pass over all test samples must equal transform()
    assert np.allclose(arrays["test_zero_f1"], prod, atol=1e-5), "zero pass differs from transform()"
    npz = args.out / "emb" / dataset / f"dim_{dim}" / f"split_{split:02d}.npz"
    npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz, train_index=np.array([str(s) for s in tr_samples]),
                        test_index=np.array(te_index),
                        **{k: v.astype(np.float32) for k, v in arrays.items()})
    return rows, probe_rows


def generate(args):
    import torch
    from linear_probe_inductive_embeddings import load_binary_targets
    warnings.filterwarnings("ignore", category=UserWarning, module="torch_geometric")
    warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    args.out.mkdir(parents=True, exist_ok=True)
    for dataset in args.datasets:
        cos_path, probe_path = args.out / f"{dataset}_cosines.csv", args.out / f"{dataset}_probes.csv"
        cos_df = pd.read_csv(cos_path) if cos_path.exists() else pd.DataFrame()
        probe_df = pd.read_csv(probe_path) if probe_path.exists() else pd.DataFrame()
        done = set() if cos_df.empty else set(map(tuple, cos_df[["dim", "split"]].values))
        targets = load_binary_targets(dataset)
        for dim in args.dims:
            for split in args.splits:
                if (dim, split) in done:
                    continue
                t0 = time.perf_counter()
                rows, prows = run_one(dataset, dim, split, args, targets, device)
                cos_df = pd.concat([cos_df, pd.DataFrame(rows)], ignore_index=True)
                probe_df = pd.concat([probe_df, pd.DataFrame(prows)], ignore_index=True)
                cos_df.to_csv(cos_path, index=False)
                probe_df.to_csv(probe_path, index=False)
                r = pd.DataFrame(rows).set_index(["samples", "init", "fraction"])["cos_mean"]
                print(f"[ok] {dataset} d{dim} split {split:02d} {time.perf_counter() - t0:5.0f}s  "
                      f"min cos train zero/random {r['train', 'zero'].min():.4f}/"
                      f"{r['train', 'random'].min():.4f}  test zero/random "
                      f"{r['test', 'zero'].min():.4f}/{r['test', 'random'].min():.4f}", flush=True)


def crossmatch(args):
    """Cross-match z of test vs. training (transductive) embeddings per input x fraction."""
    import anndata as ad
    import scxmatch
    from evaluate_inductive_ood import crossmatch as xm
    out_path = args.out / "crossmatch.csv"
    done = pd.read_csv(out_path) if out_path.exists() else pd.DataFrame()
    key = ["dataset", "dim", "split", "init", "fraction"]
    done_keys = set(map(tuple, done[key].astype(str).to_numpy())) if len(done) else set()
    rows = []
    for f in sorted((args.out / "emb").glob("*/dim_*/split_*.npz")):
        dataset, dim, split = f.parts[-3], int(f.parts[-2][4:]), int(f.stem[6:])
        z = np.load(f)
        ref = pd.DataFrame(z["train_transductive"], index=z["train_index"])
        for name in z.files:
            if not name.startswith("test_") or name == "test_index":
                continue
            init, frac = name[5:].rsplit("_f", 1)
            meta = dict(dataset=dataset, dim=dim, split=split, init=init, fraction=float(frac))
            if tuple(str(meta[c]) for c in key) in done_keys:
                continue
            test = pd.DataFrame(z[name], index=z["test_index"])
            k = None if len(ref) + len(test) <= args.max_full else args.k
            res = xm(scxmatch, ad, ref, test, k)
            rows.append(dict(meta, n_reference=len(ref), n_test=len(test), k=k if k else "full",
                             p_value=res["p_value"], z_score=res["z_score"]))
            print(f"{dataset:12s} d{dim} split {split:02d} {init:6s} f={frac:5s} "
                  f"z={res['z_score']:+.2f} p={res['p_value']:.3g}", flush=True)
            pd.concat([done, pd.DataFrame(rows)], ignore_index=True).to_csv(out_path, index=False)
    print(f"wrote {out_path}")


def summarize(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import wilcoxon

    def read(pattern):
        return pd.concat([pd.read_csv(f) for f in sorted(args.out.glob(pattern))
                          if not f.name.startswith("summary_")], ignore_index=True)

    cos, probes = read("*_cosines.csv"), read("*_probes.csv")
    grp = ["samples", "init", "dataset", "dim", "fraction"]
    cs = cos.groupby(grp).agg(cos_mean=("cos_mean", "mean"), cos_p05=("cos_p05", "mean"),
                              cos_min=("cos_min", "min"), splits=("split", "nunique")).reset_index()
    cs.to_csv(args.out / "summary_cosines.csv", index=False)

    probes["delta"] = probes.average_precision - probes.ap_production
    rows = []
    for (ds, dim, init, frac), g in probes.groupby(["dataset", "dim", "init", "fraction"]):
        per_split = g.groupby("split")["delta"].mean()   # target mean per split, paired
        p = float(wilcoxon(per_split).pvalue) if (per_split.abs() > 1e-12).any() else np.nan
        rows.append(dict(dataset=ds, dim=dim, init=init, fraction=frac,
                         ap=g.average_precision.mean(), ap_production=g.ap_production.mean(),
                         delta_ap=g.delta.mean(), max_abs_delta=g.delta.abs().max(), p_wilcoxon=p))
    ps = pd.DataFrame(rows)
    ps.to_csv(args.out / "summary_probes.csv", index=False)

    xm_path = args.out / "crossmatch.csv"
    xm = pd.read_csv(xm_path) if xm_path.exists() else pd.DataFrame()
    if len(xm):
        xs = xm.groupby(["dataset", "dim", "init", "fraction"]).agg(
            z_mean=("z_score", "mean"), z_sem=("z_score", "sem"), splits=("split", "nunique"),
            significant=("p_value", lambda p: int((p < 0.05).sum()))).reset_index()
        xs.to_csv(args.out / "summary_crossmatch.csv", index=False)
    with pd.option_context("display.width", 220, "display.max_rows", 500):
        print("== cosine similarity (mean over samples, then splits) ==")
        print(cs.pivot_table(index=["dataset", "dim", "fraction"], columns=["samples", "init"],
                             values="cos_mean").round(5).to_string())
        print("\n== linear-probe AP of test samples vs. production transform() ==")
        print(ps.round(4).to_string(index=False))
        if len(xm):
            print("\n== cross-match z, test vs. training samples ==")
            print(xs.pivot_table(index=["dataset", "dim", "fraction"], columns="init",
                                 values="z_mean").round(2).to_string())

    # ---- figure ---------------------------------------------------------------------------
    colors = {"hancock": "#2a78d6", "luad_noleak": "#eb6834", "mimic": "#1baf7a"}
    ls = {32: "-", 64: "--"}
    fill = {"zero": True, "random": False}
    muted = "#8a8986"
    fig = plt.figure(figsize=(7.2, 5.0))
    gs = fig.add_gridspec(2, 3, hspace=0.75, wspace=0.5)
    top = [fig.add_subplot(gs[0, j]) for j in range(3)]
    bottom = [fig.add_subplot(gs[1, j]) for j in range(3)]

    def line(ax, g, y, color, dim, init):
        ax.plot(g.fraction, g[y], ls.get(dim, ":"), color=color, lw=1.1,
                marker="o", ms=3.5, mfc=color if fill[init] else "white", mec=color, mew=0.9)

    panels = [("train", "cos_mean", "a  training samples\nre-embedding vs. transductive",
               "cosine similarity (mean)"),
              ("test", "cos_mean", "b  unseen samples\nvs. production transform()",
               "cosine similarity (mean)")]
    for ax, (samples, y, title, ylab) in zip(top[:2], panels):
        for (ds, dim, init), g in cs[cs.samples == samples].groupby(["dataset", "dim", "init"]):
            line(ax, g.sort_values("fraction"), y, colors[ds], dim, init)
        ax.set_title(title, fontsize=7, loc="left", fontweight="bold")
        ax.set_ylabel(ylab, fontsize=7)
    ax = top[2]
    for (ds, dim, init), g in ps.groupby(["dataset", "dim", "init"]):
        line(ax, g.sort_values("fraction"), "delta_ap", colors[ds], dim, init)
    ax.axhline(0, color=muted, lw=0.6)
    ax.set_ylabel("Δ AP vs. production\n(mean over targets, splits)", fontsize=7)
    ax.set_title("c  unseen samples\nprobe AP difference", fontsize=7, loc="left",
                 fontweight="bold")

    ref = pd.read_csv(OOD_RESULTS) if OOD_RESULTS.exists() else pd.DataFrame()
    if len(ref):
        ref = ref[(ref.comparison == "main") & ref.config.isin(["raw_onehot", "spectral"])]
    for ax, (ds, dsl) in zip(bottom, DATASETS.items()):
        if len(xm):
            for (dim, init), g in xs[xs.dataset == ds].groupby(["dim", "init"]):
                g = g.sort_values("fraction")
                line(ax, g, "z_mean", colors[ds], dim, init)
        for cfg, lsty in (("raw_onehot", ":"), ("spectral", "-.")):
            r = ref[(ref.dataset == ds) & (ref.config == cfg)] if len(ref) else ref
            if len(r):
                ax.axhline(r.z_score.mean(), color=muted, ls=lsty, lw=0.9)
        ax.axhline(0, color=muted, lw=0.5)
        ax.set_title(f"{'def'[list(DATASETS).index(ds)]}  {dsl}\ncross-match, test vs. train",
                     fontsize=7, loc="left", fontweight="bold")
        ax.set_ylabel("cross-match z (mean over splits)", fontsize=7)
    for ax in top + bottom:
        ax.set_xscale("log")
        ax.set_xticks(FRACTIONS, [f"{f:g}" for f in FRACTIONS])
        ax.minorticks_off()
        ax.set_xlabel("fraction of samples per transform() call", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.ticklabel_format(axis="y", useOffset=False)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    handles = [plt.Line2D([], [], color=c, marker="o", ms=3.5, label=DATASETS[d])
               for d, c in colors.items()]
    handles += [plt.Line2D([], [], color="#555", ls=l, label=f"d={k}") for k, l in ls.items()]
    handles += [plt.Line2D([], [], ls="", marker="o", ms=4, color="#555", label="zero input"),
                plt.Line2D([], [], ls="", marker="o", ms=4, mfc="white", mec="#555", mew=0.9,
                           label="random input"),
                plt.Line2D([], [], color=muted, ls=":", label="raw values (model-free)"),
                plt.Line2D([], [], color=muted, ls="-.", label="spectral coords. (model-free)")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.03), ncol=5,
               fontsize=6, frameon=False, handlelength=3)
    for ext in ("pdf", "png"):
        fig.savefig(args.out / f"figure_inductive_init.{ext}", dpi=300, bbox_inches="tight")
    print(f"wrote {args.out}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--datasets", nargs="+", default=list(DATASETS))
    p.add_argument("--dims", type=int, nargs="+", default=[32, 64])
    p.add_argument("--splits", type=int, nargs="+", default=list(range(10)))
    p.add_argument("--fractions", type=float, nargs="+", default=FRACTIONS)
    p.add_argument("--epochs", type=int, default=EPOCHS, help="training epochs (default 2000)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--max-full", type=int, default=10000,
                   help="cross-match: exact matching up to this many samples, else k-NN graph")
    p.add_argument("--k", type=int, default=100)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--crossmatch", action="store_true")
    mode.add_argument("--summarize", action="store_true")
    args = p.parse_args()
    if args.crossmatch:
        crossmatch(args)
    elif args.summarize:
        summarize(args)
    else:
        generate(args)


if __name__ == "__main__":
    main()
