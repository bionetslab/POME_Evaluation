"""Plotting helpers for imputation average-rank figures."""
from __future__ import annotations

from pathlib import Path
import string

import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
from matplotlib.lines import Line2D
import seaborn as sns

from .imputation_analysis import POME_DATASET_LABELS, POME_DATASET_ORDER, POME_DIM_ORDER


def _plot_rank_panel(ax, ranks_df, title, xylabelsize, ticklabelsize, titlefontsize):
    sns.lineplot(
        data=ranks_df,
        x="na_ratio",
        y="avg_rank",
        hue="method",
        ax=ax,
        legend=False,
        marker="o",
        markersize=9,
    )
    ax.set_xlabel("Simulated missingness ratio", fontsize=xylabelsize)
    ax.set_ylabel("Average rank ↓", fontsize=xylabelsize)
    ax.tick_params(axis="x", labelsize=ticklabelsize)
    ax.tick_params(axis="y", labelsize=ticklabelsize)
    ax.set_yticks([1, 2, 3, 4])
    ax.set_title(title, fontsize=titlefontsize)


def _plot_violin_panel(
    ax,
    subset,
    y,
    ylabel,
    title,
    method_order,
    method_label_map,
    palette,
    xylabelsize,
    ticklabelsize,
    titlefontsize,
):
    subset_plot = subset.copy()
    subset_plot["method_display"] = subset_plot["method"].map(method_label_map).fillna(subset_plot["method"])
    display_order = [method_label_map.get(method, method) for method in method_order]

    sns.violinplot(
        data=subset_plot,
        x="method_display",
        y=y,
        order=display_order,
        hue="method",
        hue_order=method_order,
        palette=palette,
        ax=ax,
        cut=0,
        density_norm="width",
        inner="quart",
        linewidth=1,
        legend=False,
    )
    ax.set_xlabel("")
    ax.set_ylabel(ylabel, fontsize=xylabelsize)
    ax.tick_params(axis="x", labelsize=ticklabelsize, rotation=45)
    for tick in ax.get_xticklabels():
        tick.set_ha("right")
    ax.tick_params(axis="y", labelsize=ticklabelsize)
    ax.set_title(title, fontsize=titlefontsize)


def plot_imputation_results(
    cont_ranks,
    cat_ranks,
    metric_distributions,
    output_path: str = "imputation_ranks_with_competitors_32.pdf",
    figsize: tuple[float, float] = (16, 10),
):
    """Create an imputation figure with rank panels and dataset-split violin panels."""
    fig, axes = plt.subplot_mosaic(
        [["a", "b", "c", "d"], ["e", "f", "g", "h"]],
        figsize=figsize,
        layout="constrained",
    )

    dataset_order = ["HANCOCK", "MIMIC IV", "TCGA-LUAD"]
    method_order = sorted(metric_distributions["method"].unique())
    method_label_map = {"KNN": r"$k$-NN"}
    palette = dict(zip(method_order, sns.color_palette("tab10", n_colors=len(method_order))))

    labelfontsize = 20
    titlefontsize = 20
    ticklabelsize = 16
    xylabelsize = 18
    legendfontsize = 16
    legendtitlefontsize = 18
    offset = transforms.ScaledTranslation(-36 / 72, 27 / 72, fig.dpi_scale_trans)

    _plot_rank_panel(
        ax=axes["a"],
        ranks_df=cat_ranks,
        title="Categorical variables\n(all datasets)",
        xylabelsize=xylabelsize,
        ticklabelsize=ticklabelsize,
        titlefontsize=titlefontsize,
    )

    b_axes = ["b", "c", "d"]
    d_axes = ["f", "g", "h"]

    for axis_name, dataset in zip(b_axes, dataset_order):
        subset = metric_distributions[metric_distributions["dataset"] == dataset]
        _plot_violin_panel(
            ax=axes[axis_name],
            subset=subset,
            y="acc_cat",
            ylabel="Multiclass accuracy",
            title=f"Categorical variables\n({dataset})",
            method_order=method_order,
            method_label_map=method_label_map,
            palette=palette,
            xylabelsize=xylabelsize,
            ticklabelsize=ticklabelsize,
            titlefontsize=titlefontsize,
        )

    _plot_rank_panel(
        ax=axes["e"],
        ranks_df=cont_ranks,
        title="Numeric variables\n(all datasets)",
        xylabelsize=xylabelsize,
        ticklabelsize=ticklabelsize,
        titlefontsize=titlefontsize,
    )

    for axis_name, dataset in zip(d_axes, dataset_order):
        subset = metric_distributions[metric_distributions["dataset"] == dataset]
        _plot_violin_panel(
            ax=axes[axis_name],
            subset=subset,
            y="mae_cont",
            ylabel="Mean absolute error",
            title=f"Numeric variables\n({dataset})",
            method_order=method_order,
            method_label_map=method_label_map,
            palette=palette,
            xylabelsize=xylabelsize,
            ticklabelsize=ticklabelsize,
            titlefontsize=titlefontsize,
        )

    handles = [
        Line2D([0], [0], color=palette[method], lw=4, label=method)
        for method in method_order
    ]
    labels = [method_label_map.get(method, method) for method in method_order]

    fig.legend(
        handles,
        labels,
        title="Imputation method",
        loc="upper center",
        bbox_to_anchor=(0.5, 1.1),
        ncol=min(len(labels), 8),
        fontsize=legendfontsize,
        title_fontsize=legendtitlefontsize,
        frameon=True,
        fancybox=False,
    )

    axes["a"].text(0.0, 1.0, "a", transform=axes["a"].transAxes + offset, fontsize=labelfontsize, fontweight="bold")
    axes["b"].text(0.0, 1.0, "b", transform=axes["b"].transAxes + offset, fontsize=labelfontsize, fontweight="bold")
    axes["c"].text(0.0, 1.0, "c", transform=axes["c"].transAxes + offset, fontsize=labelfontsize, fontweight="bold")
    axes["d"].text(0.0, 1.0, "d", transform=axes["d"].transAxes + offset, fontsize=labelfontsize, fontweight="bold")
    axes["e"].text(0.0, 1.0, "e", transform=axes["e"].transAxes + offset, fontsize=labelfontsize, fontweight="bold")
    axes["f"].text(0.0, 1.0, "f", transform=axes["f"].transAxes + offset, fontsize=labelfontsize, fontweight="bold")
    axes["g"].text(0.0, 1.0, "g", transform=axes["g"].transAxes + offset, fontsize=labelfontsize, fontweight="bold")
    axes["h"].text(0.0, 1.0, "h", transform=axes["h"].transAxes + offset, fontsize=labelfontsize, fontweight="bold")

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, format="pdf", bbox_inches="tight")
    return fig, axes


def _annotate_panel_letters(axes, fig, labelfontsize: int):
    offset = transforms.ScaledTranslation(-36 / 72, 27 / 72, fig.dpi_scale_trans)
    for idx, ax in enumerate(axes.flat):
        ax.text(
            0.0,
            1.0,
            string.ascii_lowercase[idx],
            transform=ax.transAxes + offset,
            fontsize=labelfontsize,
            fontweight="bold",
        )


def plot_binning_effects_results(
    data,
    output_path: str = "output/imputation_binning_effects_pome.pdf",
    figsize: tuple[float, float] = (16, 12),
):
    """Create a 3x4 box-plot figure for POME binning effects."""
    fig, axes = plt.subplots(3, 4, figsize=figsize, layout="constrained")

    dataset_order = POME_DATASET_ORDER
    dataset_labels = POME_DATASET_LABELS
    nbins_order = [7, 11, 15]
    strategy_labels = {
        "z": r"$z$-score binning",
        "nonlinear": "non-linear binning",
    }
    pome_color = sns.color_palette("tab10", n_colors=10)[3]
    red_shades = sns.light_palette(pome_color, n_colors=3)
    strategy_colors = {
        "z": red_shades[2],
        "nonlinear": red_shades[1],
    }

    labelfontsize = 20
    titlefontsize = 20
    ticklabelsize = 16
    xylabelsize = 18

    col_specs = [
        ("acc_cat", "Multiclass accuracy", "z"),
        ("acc_cat", "Multiclass accuracy", "nonlinear"),
        ("mae_cont", "Mean absolute error", "z"),
        ("mae_cont", "Mean absolute error", "nonlinear"),
    ]

    for row, dataset_key in enumerate(dataset_order):
        for col, (metric, ylabel, strategy) in enumerate(col_specs):
            ax = axes[row, col]
            subset = data[
                (data["dataset"] == dataset_key)
                & (data["strategy"] == strategy)
            ][["nbins", metric]].dropna()

            sns.boxplot(
                data=subset,
                x="nbins",
                y=metric,
                order=nbins_order,
                ax=ax,
                linewidth=1,
                color=strategy_colors[strategy],
            )

            ax.set_xlabel("Number of bins", fontsize=xylabelsize)
            ax.set_ylabel(f"{ylabel}\n({dataset_labels[dataset_key]})", fontsize=xylabelsize)
            ax.tick_params(axis="x", labelsize=ticklabelsize)
            ax.tick_params(axis="y", labelsize=ticklabelsize)

            metric_title = "Categorical variables" if metric == "acc_cat" else "Numeric variables"
            ax.set_title(f"{metric_title}\n({strategy_labels[strategy]})", fontsize=titlefontsize)

    _annotate_panel_letters(axes=axes, fig=fig, labelfontsize=labelfontsize)

    flat_axes = list(axes.flat)
    for idx in range(0, len(flat_axes), 2):
        ax1 = flat_axes[idx]
        ax2 = flat_axes[idx + 1]
        ymin = min(ax1.get_ylim()[0], ax2.get_ylim()[0])
        ymax = max(ax1.get_ylim()[1], ax2.get_ylim()[1])
        ax1.set_ylim(ymin, ymax)
        ax2.set_ylim(ymin, ymax)

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, format="pdf", bbox_inches="tight")
    return fig, axes


def _plot_dim_violin_panel(
    ax,
    subset,
    y: str,
    ylabel: str,
    title: str,
    xylabelsize: int,
    ticklabelsize: int,
    titlefontsize: int,
):
    subset_plot = subset.copy()
    subset_plot["dim_display"] = subset_plot["dim"].astype(str)
    pome_color = sns.color_palette("tab10", n_colors=10)[3]

    sns.violinplot(
        data=subset_plot,
        x="dim_display",
        y=y,
        order=[str(dim) for dim in POME_DIM_ORDER],
        ax=ax,
        cut=0,
        density_norm="width",
        inner="quart",
        linewidth=1,
        color=pome_color,
    )
    ax.set_xlabel("")
    ax.set_ylabel(ylabel, fontsize=xylabelsize)
    ax.tick_params(axis="x", labelsize=ticklabelsize, rotation=45)
    for tick in ax.get_xticklabels():
        tick.set_ha("right")
    ax.tick_params(axis="y", labelsize=ticklabelsize)
    ax.set_title(title, fontsize=titlefontsize)


def plot_imputation_dim_results(
    metric_distributions,
    output_path: str = "output/imputation_dims_pome_z15.pdf",
    figsize: tuple[float, float] = (12, 8),
):
    """Create a 2x3 violin figure for POME embedding-size effects."""
    fig, axes = plt.subplots(2, 3, figsize=figsize, layout="constrained")

    dataset_order = [POME_DATASET_LABELS[key] for key in POME_DATASET_ORDER]
    labelfontsize = 20
    titlefontsize = 20
    ticklabelsize = 16
    xylabelsize = 18

    for col, dataset in enumerate(dataset_order):
        subset = metric_distributions[metric_distributions["dataset"] == dataset]

        _plot_dim_violin_panel(
            ax=axes[0, col],
            subset=subset,
            y="acc_cat",
            ylabel="Multiclass accuracy",
            title=f"Categorical variables\n({dataset})",
            xylabelsize=xylabelsize,
            ticklabelsize=ticklabelsize,
            titlefontsize=titlefontsize,
        )

        _plot_dim_violin_panel(
            ax=axes[1, col],
            subset=subset,
            y="mae_cont",
            ylabel="Mean absolute error",
            title=f"Numeric variables\n({dataset})",
            xylabelsize=xylabelsize,
            ticklabelsize=ticklabelsize,
            titlefontsize=titlefontsize,
        )

    _annotate_panel_letters(axes=axes, fig=fig, labelfontsize=labelfontsize)

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, format="pdf", bbox_inches="tight")
    return fig, axes
