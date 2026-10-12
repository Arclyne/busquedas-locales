"""Figuras del reporte a partir de los CSV que genera main.cpp.

- figures/convergencia.png: mediana e intercuartil del mejor f(x) contra las evaluaciones.
- figures/dispersion.png: diagramas de caja del f(x) final de las 30 ejecuciones.
"""

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIRECTORY = Path("results")
FIGURES_DIRECTORY = Path("figures")
TARGET_ERROR = 1e-8
EVALUATIONS_PER_DIMENSION = 10000

ALGORITHMS = [
    ("HillClimbing", "Hill Climbing", "HC", "#2a78d6", "-"),
    ("SteepestAscent", "Steepest Ascent", "SAHC", "#eb6834", "--"),
    ("SteepestAscentWithReplacement", "Steepest Ascent with Replacement", "SAHC-R", "#1baf7a", "-."),
    ("SimulatedAnnealing", "Simulated Annealing", "SA", "#c98500", ":"),
]
DIMENSIONS = [10, 30, 50, 100]
TEXT_COLOR = "#52514e"
GRID_COLOR = "#e4e3df"


def configure_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times"],
        "mathtext.fontset": "stix",
        "font.size": 8,
        "axes.edgecolor": TEXT_COLOR,
        "axes.linewidth": 0.6,
        "xtick.color": TEXT_COLOR,
        "ytick.color": TEXT_COLOR,
    })


def style_axis(axis):
    axis.set_yscale("log")
    axis.grid(True, axis="y", which="major", color=GRID_COLOR, linewidth=0.5)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)


def target_line(axis):
    axis.axhline(TARGET_ERROR, color=TEXT_COLOR, linewidth=0.7, linestyle=(0, (1, 2)), zorder=1)


def plot_convergence():
    traces = defaultdict(lambda: defaultdict(dict))
    evaluations = {}
    with (RESULTS_DIRECTORY / "convergence.csv").open() as file:
        for row in csv.DictReader(file):
            dimensions = int(row["Dimensions"])
            checkpoint = int(row["Checkpoint"])
            traces[(row["Algorithm"], dimensions)][int(row["Run"])][checkpoint] = float(row["BestFx"])
            evaluations[(dimensions, checkpoint)] = int(row["Evaluations"])

    figure, axes = plt.subplots(2, 2, figsize=(6.2, 4.6), sharey=True)
    for axis, dimensions in zip(axes.flat, DIMENSIONS):
        for key, label, _, color, line_style in ALGORITHMS:
            runs = traces[(key, dimensions)]
            checkpoints = sorted(next(iter(runs.values())).keys())
            matrix = np.array([[runs[run][c] for c in checkpoints] for run in sorted(runs)])
            x = np.array([evaluations[(dimensions, c)] for c in checkpoints])

            lower, upper = np.percentile(matrix, [25, 75], axis=0)
            axis.fill_between(x, lower, upper, color=color, alpha=0.15, linewidth=0)
            axis.plot(x, np.median(matrix, axis=0), color=color, linestyle=line_style, linewidth=1.4, label=label)

        target_line(axis)
        style_axis(axis)
        axis.set_ylim(1e-10, 1e4)
        axis.set_xlim(0, EVALUATIONS_PER_DIMENSION * dimensions)
        axis.set_title(f"D = {dimensions}", fontsize=9)
        axis.ticklabel_format(axis="x", style="sci", scilimits=(0, 0), useMathText=True)
        axis.xaxis.get_offset_text().set_fontsize(7)

    for axis in axes[1]:
        axis.set_xlabel("Evaluaciones de la función objetivo")
    for axis in axes[:, 0]:
        axis.set_ylabel("Mejor f(x) (mediana)")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=7.5,
                  bbox_to_anchor=(0.5, -0.01), handlelength=2.6)
    figure.tight_layout(rect=(0, 0.05, 1, 1))
    figure.savefig(FIGURES_DIRECTORY / "convergencia.png", dpi=300)
    plt.close(figure)


def plot_dispersion():
    final = defaultdict(list)
    with (RESULTS_DIRECTORY / "runs.csv").open() as file:
        for row in csv.DictReader(file):
            final[(row["Algorithm"], int(row["Dimensions"]))].append(float(row["BestFx"]))

    figure, axes = plt.subplots(1, 4, figsize=(6.2, 2.5), sharey=True)
    for axis, dimensions in zip(axes, DIMENSIONS):
        data = [final[(key, dimensions)] for key, *_ in ALGORITHMS]
        boxes = axis.boxplot(data, widths=0.6, patch_artist=True, showfliers=True,
                             medianprops={"color": "#0b0b0b", "linewidth": 1.0},
                             whiskerprops={"color": TEXT_COLOR, "linewidth": 0.7},
                             capprops={"color": TEXT_COLOR, "linewidth": 0.7},
                             flierprops={"marker": "o", "markersize": 2.5, "markeredgewidth": 0.5,
                                         "markeredgecolor": TEXT_COLOR})
        for patch, (_, _, _, color, _) in zip(boxes["boxes"], ALGORITHMS):
            patch.set_facecolor(color)
            patch.set_alpha(0.55)
            patch.set_edgecolor(color)

        target_line(axis)
        style_axis(axis)
        axis.set_ylim(1e-9, 1e-6)
        axis.set_title(f"D = {dimensions}", fontsize=9)
        axis.set_xticks(range(1, len(ALGORITHMS) + 1))
        axis.set_xticklabels([short for _, _, short, _, _ in ALGORITHMS], rotation=45, ha="right", fontsize=7)

    axes[0].set_ylabel("f(x) final")
    figure.tight_layout()
    figure.savefig(FIGURES_DIRECTORY / "dispersion.png", dpi=300)
    plt.close(figure)


def main():
    configure_style()
    FIGURES_DIRECTORY.mkdir(exist_ok=True)
    plot_convergence()
    plot_dispersion()
    print(f"Figures saved to {FIGURES_DIRECTORY}/")


if __name__ == "__main__":
    main()
