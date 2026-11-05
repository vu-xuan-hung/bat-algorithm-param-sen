"""Figures generated from saved results. Nothing here runs an algorithm.

Figures are written to ``<results>/figures/``, separate from the data files. The exact
numbers behind each figure are in ``summary.csv`` and ``runs.csv``.

Figures use matplotlib's object-oriented API, so no global pyplot state is touched.
Series colours follow a validated categorical order and are assigned by level order.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ba_clustering.results import load_runs, read_manifest

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
SEQUENTIAL_MAP = "Blues"
LINE_WIDTH = 1.5  # 2 px at the 96 px-per-inch convention
MARKER_SIZE = 6.0  # 8 px
DPI = 200

Records = Sequence[Mapping[str, Any]]


def _new_figure(width: float, height: float) -> Figure:
    return Figure(figsize=(width, height), dpi=DPI, facecolor=SURFACE, layout="constrained")


def _style_axes(axes: Axes) -> None:
    axes.set_facecolor(SURFACE)
    axes.grid(True, color=GRID, linewidth=0.6)
    axes.set_axisbelow(True)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color(GRID)
    axes.tick_params(colors=TEXT_SECONDARY, labelsize=8, length=0)
    axes.xaxis.label.set_color(TEXT_SECONDARY)
    axes.yaxis.label.set_color(TEXT_SECONDARY)
    axes.title.set_color(TEXT_PRIMARY)


def _completed(records: Records, **filters: Any) -> list[Mapping[str, Any]]:
    return [
        record
        for record in records
        if record["status"] == "completed"
        and all(record[key] == value for key, value in filters.items())
    ]


def _kmeans_mean(records: Records, dataset: str) -> float | None:
    values = [record["sse"] for record in _completed(records, dataset=dataset, algorithm="kmeans")]
    return float(np.mean(values)) if values else None


def _draw_kmeans_line(axes: Axes, value: float | None) -> None:
    if value is not None:
        axes.axhline(
            value, color=TEXT_SECONDARY, linewidth=1.0, linestyle="--", label="K-Means mean"
        )


def _mean_trace(members: Records) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Return evaluations and mean best-so-far SSE, over the rows all runs share."""
    length = min(len(member["history"]["best_fitness"]) for member in members)
    traces = np.array([member["history"]["best_fitness"][:length] for member in members])
    evaluations = np.array(members[0]["history"]["n_evaluations"][:length])
    return evaluations, traces.mean(axis=0)


def plot_main_effects(
    records: Records, manifest: Mapping[str, Any], dataset: str, path: Path
) -> None:
    """Small multiples of mean SSE by level for each OFAT factor, with SD error bars."""
    factors: Mapping[str, Sequence[float]] = manifest["spec"]["factors"]
    base: Mapping[str, Any] = manifest["spec"]["ba_base"]
    n_columns = min(3, len(factors))
    n_rows = int(np.ceil(len(factors) / n_columns))
    figure = _new_figure(3.3 * n_columns, 2.6 * n_rows + 0.5)
    all_axes = figure.subplots(n_rows, n_columns, squeeze=False, sharey=True).ravel()
    kmeans = _kmeans_mean(records, dataset)
    reference = _completed(records, dataset=dataset, algorithm="ba", condition_id="reference")
    for axes, (name, declared) in zip(all_axes, factors.items(), strict=False):
        levels = sorted(set(declared) | {float(base[name])})
        means, spreads = [], []
        for level in levels:
            members = (
                reference
                if level == base[name]
                else _completed(
                    records,
                    dataset=dataset,
                    algorithm="ba",
                    condition_id=f"{name}={level:g}",
                )
            )
            values = [member["sse"] for member in members]
            means.append(np.mean(values) if values else np.nan)
            spreads.append(np.std(values, ddof=1) if len(values) > 1 else 0.0)
        positions = np.arange(len(levels))
        axes.errorbar(
            positions, means, yerr=spreads, color=SERIES[0], linewidth=LINE_WIDTH,
            marker="o", markersize=MARKER_SIZE, capsize=0, elinewidth=1.0,
        )  # fmt: skip
        reference_index = levels.index(float(base[name]))
        axes.plot(
            positions[reference_index], means[reference_index], marker="o",
            markersize=MARKER_SIZE + 3, markerfacecolor="none",
            markeredgecolor=TEXT_PRIMARY, linestyle="none", label="Reference",
        )  # fmt: skip
        _draw_kmeans_line(axes, kmeans)
        axes.set_xticks(positions, [f"{level:g}" for level in levels])
        axes.set_xlabel(name)
        _style_axes(axes)
    for axes in all_axes[len(factors) :]:
        axes.set_visible(False)
    for axes in all_axes[::n_columns]:
        axes.set_ylabel("SSE (mean ± SD across seeds)", fontsize=8)
    handles, labels = all_axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside upper right", frameon=False, fontsize=8, ncols=2)
    figure.suptitle(
        f"SSE by parameter level, {dataset}", color=TEXT_PRIMARY, fontsize=11, x=0.02, ha="left"
    )
    figure.savefig(path, facecolor=SURFACE)


def plot_factor_convergence(
    records: Records, manifest: Mapping[str, Any], dataset: str, factor: str, path: Path
) -> None:
    """Mean best-so-far SSE against objective evaluations, one line per level of ``factor``."""
    base_value = float(manifest["spec"]["ba_base"][factor])
    levels = sorted(set(manifest["spec"]["factors"][factor]) | {base_value})
    figure = _new_figure(6.2, 3.6)
    axes = figure.subplots()
    for index, level in enumerate(levels):
        condition = "reference" if level == base_value else f"{factor}={level:g}"
        members = _completed(records, dataset=dataset, algorithm="ba", condition_id=condition)
        if not members:
            continue
        evaluations, mean = _mean_trace(members)
        label = f"{factor}={level:g}" + (" (reference)" if level == base_value else "")
        axes.plot(
            evaluations, mean, color=SERIES[index % len(SERIES)], linewidth=LINE_WIDTH, label=label
        )
    _draw_kmeans_line(axes, _kmeans_mean(records, dataset))
    axes.set_xlabel("Objective evaluations")
    axes.set_ylabel("Best-so-far SSE (mean across seeds)")
    axes.set_title(f"Convergence by {factor}, {dataset}", loc="left", fontsize=11)
    axes.legend(frameon=False, fontsize=8)
    _style_axes(axes)
    figure.savefig(path, facecolor=SURFACE)


def plot_baseline(records: Records, dataset: str, path: Path) -> None:
    """Final SSE per seed for each algorithm, with the mean marked."""
    figure = _new_figure(4.6, 3.2)
    axes = figure.subplots()
    groups = [
        (
            "BA (reference)",
            _completed(records, dataset=dataset, algorithm="ba", condition_id="reference"),
        ),
        ("K-Means", _completed(records, dataset=dataset, algorithm="kmeans")),
    ]
    jitter = np.random.default_rng(0)  # Cosmetic horizontal jitter only.
    labels = []
    for index, (name, members) in enumerate(groups):
        values = np.array([member["sse"] for member in members], dtype=float)
        labels.append(name)
        if values.size == 0:
            continue
        offsets = jitter.uniform(-0.12, 0.12, size=values.size)
        axes.scatter(
            index + offsets, values, s=36, color=SERIES[index], edgecolors=SURFACE,
            linewidths=1.0, label=f"{name}: each seed", zorder=3,
        )  # fmt: skip
        axes.hlines(
            values.mean(),
            index - 0.28,
            index + 0.28,
            color=TEXT_PRIMARY,
            linewidth=LINE_WIDTH,
            zorder=4,
        )
    axes.plot([], [], color=TEXT_PRIMARY, linewidth=LINE_WIDTH, label="Mean")
    axes.set_xticks(range(len(labels)), labels)
    axes.set_xlim(-0.6, len(labels) - 0.4)
    axes.set_ylabel("Final SSE")
    axes.set_title(f"Final SSE by algorithm, {dataset}", loc="left", fontsize=11)
    axes.legend(frameon=False, fontsize=8)
    _style_axes(axes)
    figure.savefig(path, facecolor=SURFACE)


def interaction_matrix(
    records: Records, manifest: Mapping[str, Any], dataset: str, factors: Sequence[str]
) -> tuple[list[float], list[float], np.ndarray[Any, Any]]:
    """Return row levels, column levels, and mean SSE per cell for the first two factors.

    Cells with no completed runs are NaN.
    """
    first, second = factors[0], factors[1]
    base = manifest["spec"]["ba_base"]
    rows = sorted(manifest["spec"]["factors"][first])
    columns = sorted(manifest["spec"]["factors"][second])
    matrix = np.full((len(rows), len(columns)), np.nan)
    for i, row in enumerate(rows):
        for j, column in enumerate(columns):
            is_reference = row == base[first] and column == base[second]
            condition = "reference" if is_reference else f"{first}={row:g}|{second}={column:g}"
            members = _completed(records, dataset=dataset, algorithm="ba", condition_id=condition)
            if members:
                matrix[i, j] = np.mean([member["sse"] for member in members])
    return rows, columns, matrix


def plot_interaction(
    records: Records, manifest: Mapping[str, Any], dataset: str, factors: Sequence[str], path: Path
) -> None:
    """Heatmap of mean SSE over the first two grid factors, with every cell labelled."""
    first, second = factors[0], factors[1]
    rows, columns, matrix = interaction_matrix(records, manifest, dataset, factors)
    figure = _new_figure(4.8, 3.6)
    axes = figure.subplots()
    image = axes.imshow(matrix, cmap=SEQUENTIAL_MAP, aspect="auto", origin="lower")
    midpoint = np.nanmean(matrix)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if np.isfinite(matrix[i, j]):
                color = SURFACE if matrix[i, j] > midpoint else TEXT_PRIMARY
                axes.text(
                    j, i, f"{matrix[i, j]:.1f}", ha="center", va="center", fontsize=8, color=color
                )
    axes.set_xticks(range(len(columns)), [f"{value:g}" for value in columns])
    axes.set_yticks(range(len(rows)), [f"{value:g}" for value in rows])
    axes.set_xlabel(second)
    axes.set_ylabel(first)
    axes.set_title(f"Mean SSE, {first} by {second}, {dataset}", loc="left", fontsize=11)
    _style_axes(axes)
    axes.grid(False)
    colorbar = figure.colorbar(image, ax=axes)
    colorbar.outline.set_visible(False)
    colorbar.ax.tick_params(labelsize=8, colors=TEXT_SECONDARY, length=0)
    figure.savefig(path, facecolor=SURFACE)


def generate_plots(results_directory: str | Path) -> list[Path]:
    """Create every figure relevant to an experiment's design and return their paths.

    Existing figures are regenerated, because they derive entirely from saved data.

    Raises:
        FileNotFoundError: If the directory has no manifest.
    """
    directory = Path(results_directory)
    manifest = read_manifest(directory)
    records = load_runs(directory)
    figures = directory / "figures"
    figures.mkdir(exist_ok=True)
    design = manifest["spec"]["design"]
    factors = list(manifest["spec"]["factors"])
    written: list[Path] = []
    for dataset in manifest["spec"]["datasets"]:
        if not _completed(records, dataset=dataset):
            continue
        if design == "ofat":
            path = figures / f"main_effects_{dataset}.png"
            plot_main_effects(records, manifest, dataset, path)
            written.append(path)
            for factor in factors:
                path = figures / f"convergence_{dataset}_{factor}.png"
                plot_factor_convergence(records, manifest, dataset, factor, path)
                written.append(path)
        elif design == "grid":
            path = figures / f"interaction_{dataset}_{factors[0]}_{factors[1]}.png"
            plot_interaction(records, manifest, dataset, factors, path)
            written.append(path)
        else:
            path = figures / f"baseline_{dataset}.png"
            plot_baseline(records, dataset, path)
            written.append(path)
    return written
