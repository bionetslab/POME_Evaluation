"""Validate POME's label-free inductive stopping criteria against downstream generalization.

Reviewer question: does the (RankMe-based) stopping criterion select an epoch count with good
*downstream* quality, and how were its tolerance and persistence (patience) chosen?

Consumes the output of ``generate_inductive_stopping_ablation.py``: per dataset x dim x split
(the 10 fixed train/test splits of data/train_test_splits; one training seed each), the label-free
CV trace (3-fold sample-holdout CV *within the training split*, over 2000 epochs, without stopping)
and the downstream quality of full fits trained for E epochs, E on a grid. Downstream quality is
the linear-probe average precision (AP): an L2 logistic regression per binary clinical target,
fit on the transductive embeddings of the training split and scored on the inductive (transform())
embeddings of the held-out test split -- no CV, the test split is never seen by POME or the probe.
Targets (``linear_probe_inductive_embeddings.DATASET_TARGETS``): HANCOCK recurrence, survival
status, RFS event; TCGA-LUAD disease-free, disease-specific survival and progression-free status;
MIMIC-IV aplasia, neutropenic fever. "AP" without a target is the mean over a dataset's targets.

For every stopping rule -- criterion x tolerance x patience -- the rule is applied to the fold-mean
signal of the CV trace with ``pome.utils.select_stopping_epoch`` (the code ``fit()`` runs, so the
selected epoch is the one POME would train for), and its downstream outcome AP(E_sel) is read off
the grid (linear interpolation in log E). Criteria:

  rankme_gap  matched train-vs-held-out RankMe gap, "onset" rule (current POME; tol absolute)
  val_loss    held-out link-prediction BCE of the inductively embedded CV samples, "best" rule
              (classical early stopping; tol = relative minimum improvement)
  val_auc     held-out link-prediction ROC-AUC, "best" rule on -AUC (tol absolute)
  fixed       a fixed epoch count (no CV); fixed 2000 = what POME 1.4 trains (its RankMe rule
              never fires)

Per rule and split: regret = AP of the best grid epoch (oracle, uses test labels) - AP(E_sel), and
gain = AP(E_sel) - AP(2000). Tolerance/patience are chosen to minimize the mean regret (datasets
weighted equally, both dims); leave-one-dataset-out (choose on two datasets, evaluate on the third)
checks that the choice transfers. Agreement of each signal with downstream quality is measured
as the Spearman correlation (per split) between the signal at epoch E (oriented so that higher =
better) and AP(E) over the grid.

AP per target vs. epochs and the reproducibility of AP maxima: ``analyze_inductive_epoch_probes.py``.

Run from the project root (CPU; needs ``pome.utils.select_stopping_epoch`` of the POME branch
inductive-revision, and the CV traces of ``generate_inductive_stopping_ablation.py --cv-trace``):
    POME_REPO=$HOME/Projects/POME-inductive-revision \\
        container/run.sh scripts/analyze_inductive_stopping_ablation.py

Inputs:  output/inductive_stopping_ablation/{dataset}/dim_{D}/split_{NN}/
         {cv_trace,grid_probes}.csv, optional ood_crossmatch.csv
Outputs (output/inductive_stopping_ablation/):
  rule_results.csv      per rule x dataset x dim x split: selected epoch, AP, regret, gain
  rule_summary.csv      per rule: mean regret / gain / epochs per dataset and overall
  signal_agreement.csv  Spearman(signal, AP) per signal x dataset x dim x split
  lodo_selection.csv    leave-one-dataset-out choice of tol/patience per criterion
  stopping_validation.pdf/.png      AP and CV signals vs epochs, selected epochs
  stopping_regret_heatmaps.pdf/.png regret per criterion x tolerance x patience
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr, wilcoxon  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ROOT = PROJECT_ROOT / "output" / "inductive_stopping_ablation"
DATASETS = {"hancock": "HANCOCK", "luad_noleak": "TCGA-LUAD", "mimic": "MIMIC-IV"}
BUDGET = 2000
EVAL_EVERY = 10
PATIENCES = [1, 2, 3, 5, 10, 20]
CRITERIA = {
    # name: signal column, orientation (+1: lower is better), rule mode, relative tol, tol grid
    "rankme_gap": dict(signal="overfit_index", sign=1, mode="onset", relative=False,
                       tols=[0.0, 0.0025, 0.005, 0.01, 0.02, 0.05], label="RankMe gap"),
    "val_loss": dict(signal="val_bce", sign=1, mode="best", relative=True,
                     tols=[0.0, 0.001, 0.0025, 0.005, 0.01, 0.02], label="held-out link loss"),
    "val_auc": dict(signal="val_auc", sign=-1, mode="best", relative=False,
                    tols=[0.0, 0.0005, 0.001, 0.0025, 0.005, 0.01], label="held-out link AUC"),
}
CURRENT = ("rankme_gap", 0.05, 3)  # POME <= 1.4 defaults (overfit_tol, cv_patience)
KEY = ["dataset", "dim", "split"]


def load(root):
    traces = pd.concat([pd.read_csv(f) for f in sorted(root.glob("*/dim_*/split_*/cv_trace.csv"))],
                       ignore_index=True)
    probes = pd.concat([pd.read_csv(f) for f in sorted(root.glob("*/dim_*/split_*/grid_probes.csv"))],
                       ignore_index=True)
    signals = [c["signal"] for c in CRITERIA.values()] + ["rankme_train", "rankme_val"]
    trace = traces.groupby(KEY + ["epoch"])[signals].mean().reset_index()   # fold mean
    ap = probes.groupby(KEY + ["epochs"])["average_precision"].mean().rename("ap").reset_index()
    return trace, ap, probes


def ap_at(curve, epoch):
    """AP at `epoch`, interpolated linearly in log(epoch) between grid fits."""
    return float(np.interp(np.log(epoch), np.log(curve.index.to_numpy(float)), curve.to_numpy()))


def apply_rules(trace, ap):
    from pome.utils import select_stopping_epoch
    rows = []
    ap_groups = {k: g.set_index("epochs")["ap"].sort_index() for k, g in ap.groupby(KEY)}
    for k, tr in trace.groupby(KEY):
        if k not in ap_groups:
            continue
        curve = ap_groups[k]
        oracle, ap_budget = curve.max(), ap_at(curve, BUDGET)
        base = dict(zip(KEY, k), oracle_ap=oracle, oracle_epochs=int(curve.idxmax()),
                    ap_budget=ap_budget)
        tr = tr.sort_values("epoch")
        for name, c in CRITERIA.items():
            vals = c["sign"] * tr[c["signal"]].to_numpy()
            for tol in c["tols"]:
                for pat in PATIENCES:
                    e = max(select_stopping_epoch(tr["epoch"].to_numpy(), vals, BUDGET,
                                                  mode=c["mode"], tol=tol, patience=pat,
                                                  relative=c["relative"]), EVAL_EVERY)
                    rows.append(dict(base, criterion=name, tol=tol, patience=pat, epochs=e,
                                     ap=ap_at(curve, e)))
        for e in curve.index:
            rows.append(dict(base, criterion="fixed", tol=np.nan, patience=int(e), epochs=int(e),
                             ap=float(curve.loc[e])))
    res = pd.DataFrame(rows)
    res["regret"] = res["oracle_ap"] - res["ap"]
    res["gain_vs_budget"] = res["ap"] - res["ap_budget"]
    # The current implementation trains 2000 epochs (its RankMe rule never fires).
    res["gain_vs_current"] = res["gain_vs_budget"]
    return res


def summarize(res):
    rule = ["criterion", "tol", "patience"]
    def wilcoxon_p(x):
        x = x.dropna()
        return float(wilcoxon(x).pvalue) if len(x) > 5 and (x != 0).any() else np.nan

    per_ds = res.groupby(rule + ["dataset"], dropna=False).agg(
        regret=("regret", "mean"), gain=("gain_vs_budget", "mean"), ap=("ap", "mean"),
        vs_current=("gain_vs_current", "mean"), p_vs_current=("gain_vs_current", wilcoxon_p),
        epochs=("epochs", "median")).reset_index()
    wide = per_ds.pivot_table(index=rule, columns="dataset",
                              values=["regret", "gain", "ap", "vs_current", "p_vs_current",
                                      "epochs"], dropna=False)
    wide.columns = [f"{v}_{d}" for v, d in wide.columns]
    overall = per_ds.groupby(rule, dropna=False)[["regret", "gain", "ap", "vs_current"]].mean()
    overall.columns = [f"{c}_mean" for c in overall.columns]
    # Worst dataset regret: a robust choice should not fail badly on any dataset.
    overall["regret_worst"] = per_ds.groupby(rule, dropna=False)["regret"].max()
    # pivot_table(dropna=False) creates rows for tol x patience combinations of other criteria
    return (wide.join(overall).dropna(subset=["regret_mean"]).reset_index()
            .sort_values("regret_mean"))


def lodo(res):
    """Leave-one-dataset-out choice of (tol, patience) per criterion."""
    rows = []
    rule = ["tol", "patience"]
    for crit, g in res[res.criterion != "fixed"].groupby("criterion"):
        per = g.groupby(rule + ["dataset"])["regret"].mean().unstack("dataset")
        for held in per.columns:
            train = per.drop(columns=held).mean(axis=1)
            tol, pat = train.idxmin()
            rows.append(dict(criterion=crit, held_out=held, tol=tol, patience=pat,
                             train_regret=train.min(), held_out_regret=per.loc[(tol, pat), held],
                             held_out_best_regret=per[held].min()))
    return pd.DataFrame(rows)


def agreement(trace, ap):
    rows = []
    for k, tr in trace.groupby(KEY):
        a = ap[(ap[KEY] == pd.Series(dict(zip(KEY, k)))).all(axis=1)].sort_values("epochs")
        if len(a) < 3:
            continue
        tr = tr.sort_values("epoch")
        for name, c in CRITERIA.items():
            # Signal at the grid epochs (CV checks every 10 epochs; interpolate), higher = better.
            s = -c["sign"] * np.interp(a["epochs"], tr["epoch"], tr[c["signal"]])
            rho = spearmanr(s, a["ap"]).statistic
            rows.append(dict(zip(KEY, k), signal=name, spearman=rho))
    return pd.DataFrame(rows)


def best_rule(summary, crit):
    s = summary[summary.criterion == crit]
    r = s.sort_values("regret_mean").iloc[0]
    return r["tol"], int(r["patience"])


def plot_validation(trace, ap, res, rules, out):
    """Per dataset: AP vs epochs (oracle curve) with the CV signals and the selected epochs."""
    fig, axes = plt.subplots(len(DATASETS), 4, figsize=(15, 3.1 * len(DATASETS)), squeeze=False)
    colors = {32: "#1f77b4", 64: "#d62728"}
    for i, (ds, dsl) in enumerate(DATASETS.items()):
        for dim, col in colors.items():
            a = ap[(ap.dataset == ds) & (ap.dim == dim)]
            if a.empty:
                continue
            m = a.groupby("epochs")["ap"].agg(["mean", "sem"])
            ax = axes[i, 0]
            ax.plot(m.index, m["mean"], "o-", color=col, ms=3, label=f"d={dim}")
            ax.fill_between(m.index, m["mean"] - m["sem"], m["mean"] + m["sem"], color=col,
                            alpha=.15, lw=0)
            sel = res[(res.dataset == ds) & (res.dim == dim)]
            for (crit, tol, pat), mk in zip(rules, ("v", "^", "s", "D")):
                r = sel[(sel.criterion == crit) & np.isclose(sel.tol.fillna(-1), tol)
                        & (sel.patience == pat)]
                if not r.empty:
                    ax.plot(r.epochs.median(), r.ap.mean(), mk, color=col, mec="k", ms=8)
            t = trace[(trace.dataset == ds) & (trace.dim == dim)]
            for j, name in enumerate(CRITERIA, start=1):
                g = t.groupby("epoch")[CRITERIA[name]["signal"]].agg(["mean", "sem"])
                axes[i, j].plot(g.index, g["mean"], color=col, lw=1)
                axes[i, j].fill_between(g.index, g["mean"] - g["sem"], g["mean"] + g["sem"],
                                        color=col, alpha=.15, lw=0)
        axes[i, 0].set_ylabel(f"{dsl}\nprobe AP, test split\n(mean over targets)")
        for j in range(4):
            axes[i, j].set_xscale("log")
            if i == len(DATASETS) - 1:
                axes[i, j].set_xlabel("training epochs")
        for j, name in enumerate(CRITERIA, start=1):
            if i == 0:
                axes[i, j].set_title(f"label-free: {CRITERIA[name]['label']}\n"
                                     "(3-fold CV within the training split)", fontsize=10)
    axes[0, 0].set_title("downstream: linear probe\n(mean \u00b1 SEM over 10 train/test splits)",
                         fontsize=10)
    handles = [plt.Line2D([], [], marker=mk, ls="", color="grey", mec="k",
                          label=f"{c} tol={t:g} pat={p}") for (c, t, p), mk in zip(rules, "v^sD")]
    axes[0, 0].legend(handles=axes[0, 0].get_legend_handles_labels()[0] + handles, fontsize=7)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out / f"stopping_validation.{ext}", dpi=200)
    plt.close(fig)


def plot_heatmaps(res, out):
    sub = res
    fig, axes = plt.subplots(1, len(CRITERIA) + 1, figsize=(4.6 * (len(CRITERIA) + 1), 3.6))
    fig.subplots_adjust(wspace=0.5)
    per = sub.groupby(["criterion", "tol", "patience", "dataset"], dropna=False)["regret"].mean()
    per = per.groupby(["criterion", "tol", "patience"], dropna=False).mean()
    vmax = float(per.max())
    for ax, (name, c) in zip(axes, CRITERIA.items()):
        h = per.loc[name].unstack("patience").reindex(index=c["tols"], columns=PATIENCES)
        im = ax.imshow(h.to_numpy(), cmap="viridis_r", vmin=0, vmax=vmax, aspect="auto")
        for (r, cc), v in np.ndenumerate(h.to_numpy()):
            ax.text(cc, r, f"{v:.3f}", ha="center", va="center", fontsize=7,
                    color="w" if v > vmax / 2 else "k")
        ax.set_xticks(range(len(PATIENCES)), PATIENCES)
        ax.set_yticks(range(len(c["tols"])), [f"{t:g}" for t in c["tols"]])
        ax.set_xlabel("patience (checks of 10 epochs)")
        ax.set_ylabel("tolerance" + (" (relative)" if c["relative"] else ""))
        ax.set_title(c["label"])
    f = per.loc["fixed"].droplevel("tol")
    axes[-1].bar(range(len(f)), f.to_numpy(), color="grey")
    axes[-1].set_xticks(range(len(f)), [str(e) for e in f.index], rotation=60, fontsize=7)
    axes[-1].set_ylim(0, vmax)
    axes[-1].set_xlabel("fixed epochs")
    axes[-1].set_title("no early stopping")
    fig.colorbar(im, ax=axes[:-1], shrink=.8, label="mean AP regret vs. oracle epoch")
    fig.suptitle("Downstream regret of stopping rules (mean over datasets, dims, splits)",
                 fontsize=10)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"stopping_regret_heatmaps.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--root", type=Path, default=ROOT)
    args = p.parse_args()
    out = args.root

    trace, ap, probes = load(args.root)
    res = apply_rules(trace, ap)
    res.to_csv(out / "rule_results.csv", index=False)
    summary = summarize(res)
    summary.to_csv(out / "rule_summary.csv", index=False)
    agree = agreement(trace, ap)
    agree.to_csv(out / "signal_agreement.csv", index=False)
    lo = lodo(res)
    lo.to_csv(out / "lodo_selection.csv", index=False)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    cols = ["criterion", "tol", "patience", "regret_mean", "regret_worst", "vs_current_mean"] \
        + [f"p_vs_current_{d}" for d in DATASETS] + [f"regret_{d}" for d in DATASETS] \
        + [f"epochs_{d}" for d in DATASETS]
    cols = [c for c in cols if c in summary]
    print("\n== Best rules per criterion (mean regret over datasets) ==")
    print(summary.groupby("criterion").head(3)[cols].round(4).to_string(index=False))
    cur = summary[(summary.criterion == CURRENT[0]) & np.isclose(summary.tol, CURRENT[1])
                  & (summary.patience == CURRENT[2])]
    print("\n== Current rule (rankme_gap tol 0.05, patience 3) ==")
    print(cur[cols].round(4).to_string(index=False))

    print("\n== Spearman(signal, downstream AP) over the epoch grid (median over splits) ==")
    print(agree.groupby(["signal", "dataset"])["spearman"].median()
          .unstack("dataset").round(2).to_string())

    print("\n== Leave-one-dataset-out choice of tol/patience ==")
    print(lo.round(4).to_string(index=False))

    ood_path = args.root / "ood_crossmatch.csv"
    if ood_path.exists():
        ood = pd.read_csv(ood_path)
        print("\n== Cross-match train vs test embeddings (z ~ 0: in-distribution) ==")
        print(ood.groupby(["dataset", "dim", "epochs"]).agg(
            z_mean=("z_score", "mean"), z_sem=("z_score", "sem"),
            significant=("p_value", lambda q: int((q < 0.05).sum())),
            splits=("split", "nunique")).round(2).unstack("epochs").to_string())

    rules = [CURRENT] + [(c, *best_rule(summary, c)) for c in CRITERIA]
    plot_validation(trace, ap, res, rules, out)
    plot_heatmaps(res, out)
    print(f"\nwrote outputs to {out}")


if __name__ == "__main__":
    main()
