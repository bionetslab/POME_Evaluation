"""Does the input of unseen samples, or how many are transformed together, matter in POME's inductive mode?

``Embedder.transform()`` adds each new sample as a receive-only node (edges point only from the
training value nodes to the new sample) whose own input feature is a zero vector. Two properties
follow, which this script checks empirically on models trained with POME's inductive defaults
(POME >= 1.7: early stopping on the smoothed held-out link AUC within the 2000-epoch budget; the
selected number of epochs is recorded per fit):

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
  * test samples: cosine(default transform(), i.e. zero input with all test samples in one call,
    embedding with this input and fraction)
  * test samples: linear-probe AP per target (raw logreg fit on the transductive training
    embeddings, ``probe_one(standardize=False)``) and its difference to the default transform() embedding
  * the train / test embeddings (``emb/{dataset}/dim_{D}/split_{NN}.npz``) for the Rosenbaum
    cross-match test (scXmatch) of test vs. training samples, run by ``--crossmatch``

Figure (``--summarize``, embedding size ``FIGURE_DIM`` = 32 only; the summary CSVs keep all
dims): top row (a) cosine of the training samples' re-embedding to their transductive embedding,
against the fraction of samples per transform() call (0.01, 0.1, 1; zero input: filled, random
input: open markers, joined by a vertical line; datasets slightly offset horizontally), (b) bars
of the mean absolute linear-probe AP difference between random and zero input of the unseen
samples (over targets and splits) per fraction and dataset, legend in the
third slot; bottom row (c-e) box plots of the cross-match z (unseen vs. training samples) over the
10 splits per dataset and embedding size (all dims run, e.g. 16/32/64; all unseen samples in one
transform() call; zero input filled, random outlined), with the model-free references of
``evaluate_inductive_ood.py`` (raw values; spectral coordinates per embedding size where
available; mean over splits, grey lines). The
unseen-sample cosine to the default transform() and the AP differences with their Wilcoxon p
are in summary_cosines.csv and summary_probes.csv.

GPU part (one job per dataset; PYTHONHASHSEED=0, POME >= 1.7):
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
FIGURE_DIM = 32  # embedding size shown in the figure
OOD_RESULTS = PROJECT_ROOT / "output" / "inductive_ood" / "ood_crossmatch_results.csv"
DATASETS = {"hancock": "HANCOCK", "luad_noleak": "TCGA-LUAD", "mimic": "MIMIC-IV"}
EPOCHS = None  # POME's inductive default: early stopping within the 2000-epoch budget
INITS = ["zero", "random"]
FRACTIONS = [0.01, 0.05, 0.1, 0.25, 0.5, 1.0]
FIGURE_FRACTIONS = [0.01, 0.1, 1.0]  # fractions shown in the figure (no differences between any)


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
    meta = dict(dataset=dataset, dim=dim, split=split, epochs=emb.num_epochs)  # trained epochs

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
    prod = emb.transform(test_df.copy()).loc[te_index].to_numpy()   # default transform()
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
                         delta_ap=g.delta.mean(), mean_abs_delta=g.delta.abs().mean(),
                         max_abs_delta=g.delta.abs().max(), p_wilcoxon=p))
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
        print("\n== linear-probe AP of test samples vs. default transform() ==")
        print(ps.round(4).to_string(index=False))
        if len(xm):
            print("\n== cross-match z, test vs. training samples ==")
            print(xs.pivot_table(index=["dataset", "dim", "fraction"], columns="init",
                                 values="z_mean").round(2).to_string())

    # ---- figure ---------------------------------------------------------------------------
    colors = {"hancock": "#2a78d6", "luad_noleak": "#eb6834", "mimic": "#1baf7a"}
    cs = cs[(cs.dim == FIGURE_DIM) & cs.fraction.isin(FIGURE_FRACTIONS)]
    ps = ps[(ps.dim == FIGURE_DIM) & ps.fraction.isin(FIGURE_FRACTIONS)]
    if len(xm):
        xs = xs[(xs.dim == FIGURE_DIM) & xs.fraction.isin(FIGURE_FRACTIONS)]
    fill = {"zero": True, "random": False}
    muted = "#8a8986"
    fig = plt.figure(figsize=(7.2, 5.0))
    gs = fig.add_gridspec(2, 3, hspace=0.75, wspace=0.75)
    top = [fig.add_subplot(gs[0, j]) for j in range(3)]
    bottom = [fig.add_subplot(gs[1, j]) for j in range(3)]

    def line(ax, g, y, color, dim, init):
        ax.plot(g.fraction, g[y], "-", color=color, lw=1.1,
                marker="o", ms=3.5, mfc=color if fill[init] else "white", mec=color, mew=0.9)

    # panels a-c: per dataset and fraction, zero (filled) and random (open) input joined by a
    # vertical line; the datasets side by side around each fraction (log x, multiplicative offset)
    offset = {ds: 10 ** (0.14 * (k - 1)) for k, ds in enumerate(colors)}

    def pairs(ax, g, y):
        for ds, gd in g.groupby("dataset"):
            w = gd.pivot_table(index="fraction", columns="init", values=y)
            x = w.index.to_numpy() * offset[ds]
            ax.vlines(x, w.zero, w.random, color=colors[ds], lw=1.0, zorder=2)
            ax.plot(x, w.zero, "o", ls="", ms=3.5, color=colors[ds], zorder=3)
            # transparent face: where zero and random coincide, the filled point stays visible
            ax.plot(x, w.random, "o", ls="", ms=5, mfc="none", mec=colors[ds], mew=0.9,
                    zorder=4)

    # a: training samples, cosine to their transductive embedding; b: unseen samples, absolute
    # probe AP (the unseen-sample cosine to the default transform() is in summary_cosines.csv)
    ax = top[0]
    pairs(ax, cs[cs.samples == "train"], "cos_mean")
    ax.set_title("Training samples: re-embedding\nvs. transductive embedding", fontsize=7)
    ax.set_ylabel("Cosine similarity (mean)", fontsize=7)
    # b: mean |AP(random) - AP(zero)| over targets x splits, bars per fraction and dataset
    ax = top[1]
    pr = ps[ps.init == "random"]
    width = 0.8 / len(colors)
    for k, (ds, col) in enumerate(colors.items()):
        g = pr[pr.dataset == ds].set_index("fraction").reindex(FIGURE_FRACTIONS)
        ax.bar(np.arange(len(FIGURE_FRACTIONS)) + (k - 1) * width, g.mean_abs_delta, width,
               color=col)
    ax.set_xticks(range(len(FIGURE_FRACTIONS)), [f"{f:g}" for f in FIGURE_FRACTIONS])
    ax.set_title("Downstream effect of unseen\nsample initialization", fontsize=7)
    ax.set_ylabel("|Δ AP| (mean over targets, splits)", fontsize=7)
    ax.set_xlabel("Fraction of samples per transform() call", fontsize=7)
    ax.tick_params(labelsize=6)
    plots = [top[0]]

    ref = pd.read_csv(OOD_RESULTS) if OOD_RESULTS.exists() else pd.DataFrame()
    if len(ref):
        ref = ref[(ref.comparison == "main") & ref.config.isin(["raw_onehot", "spectral"])]
    for ax, (ds, dsl) in zip(bottom, DATASETS.items()):
        # c-e: box plots of the per-split z (10 splits) per embedding size, zero
        # (filled) left of random (outlined) input; all unseen samples in one transform() call
        # (the fraction makes no difference)
        if len(xm):
            xd = xm[(xm.dataset == ds) & np.isclose(xm.fraction, 1.0)]
            dims = sorted(xd.dim.unique())
            for i, d in enumerate(dims):
                for init, dx in (("zero", -0.2), ("random", 0.2)):
                    z = xd.loc[(xd.dim == d) & (xd.init == init), "z_score"].to_numpy()
                    if not len(z):
                        continue
                    bp = ax.boxplot([z], positions=[i + dx], widths=0.32, patch_artist=True,
                                    showfliers=False, medianprops=dict(color="black", lw=1.0),
                                    boxprops=dict(lw=0.8), whiskerprops=dict(lw=0.8),
                                    capprops=dict(lw=0.8))
                    bp["boxes"][0].set_facecolor(colors[ds] if fill[init] else "white")
                    bp["boxes"][0].set_alpha(0.55 if fill[init] else 1)
            ax.set_xticks(range(len(dims)), [str(d) for d in dims])
            ax.set_xlim(-0.6, len(dims) - 0.4)
        # model-free references (mean over splits): raw values across the panel, spectral
        # coordinates (dimension-dependent) as a segment at each embedding size it was run for
        r = ref[ref.dataset == ds] if len(ref) else ref
        if len(r):
            raw = r[r.config == "raw_onehot"]
            if len(raw):
                ax.axhline(raw.z_score.mean(), color=muted, ls=":", lw=0.9)
            if len(xm):
                for i, d in enumerate(dims):
                    sp = r[(r.config == "spectral") & (r.dim == d)]
                    if len(sp):
                        ax.hlines(sp.z_score.mean(), i - 0.45, i + 0.45, color=muted, ls="-.",
                                  lw=0.9)
        ax.set_title(f"Out-of-distribution\nanalysis ({dsl})", fontsize=7)
        ax.set_ylabel("Cross-match z", fontsize=7)
        ax.set_xlabel("Embedding size d", fontsize=7)
        ax.tick_params(labelsize=6)
    for ax in plots:
        ax.set_xscale("log")
        ax.set_xticks(FIGURE_FRACTIONS, [f"{f:g}" for f in FIGURE_FRACTIONS])
        ax.set_xlim(FIGURE_FRACTIONS[0] / 2.2, FIGURE_FRACTIONS[-1] * 2.2)
        ax.minorticks_off()
        ax.set_xlabel("Fraction of samples per transform() call", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.ticklabel_format(axis="y", useOffset=False)
    for ax, key in zip(top[:2] + bottom, "abcde"):
        ax.text(-0.32, 1.22, key, transform=ax.transAxes, fontsize=10, fontweight="bold",
                va="top", ha="left")
    handles = [plt.Line2D([], [], color=c, marker="o", ms=3.5, label=DATASETS[d])
               for d, c in colors.items()]
    handles += [plt.Line2D([], [], ls="", label=""),  # pad: columns fill top to bottom
                plt.Line2D([], [], ls="", marker="o", ms=4, color="#555", label="Zero input"),
                plt.Line2D([], [], ls="", marker="o", ms=4, mfc="white", mec="#555", mew=0.9,
                           label="Random input"),
                plt.Line2D([], [], color=muted, ls=":", label="Raw values (model-free)"),
                plt.Line2D([], [], color=muted, ls="-.", label="Spectral coords. (model-free)")]
    handles = [h for h in handles if h.get_label()]  # one column: no padding needed
    top[2].axis("off")
    top[2].legend(handles=handles, loc="center", fontsize=6, frameon=True, fancybox=False,
                  edgecolor="black", handlelength=3)
    for ext in ("pdf", "png"):
        fig.savefig(args.out / f"figure_inductive_init.{ext}", dpi=300, bbox_inches="tight")
    print(f"wrote {args.out}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--datasets", nargs="+", default=list(DATASETS))
    p.add_argument("--dims", type=int, nargs="+", default=[32, 64])
    p.add_argument("--splits", type=int, nargs="+", default=list(range(10)))
    p.add_argument("--fractions", type=float, nargs="+", default=FRACTIONS)
    p.add_argument("--epochs", type=int, default=EPOCHS,
                   help="training budget (default: POME's, 2000 with early stopping)")
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
