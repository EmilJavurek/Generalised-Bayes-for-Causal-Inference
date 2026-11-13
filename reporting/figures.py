"""Plot a completed comparison: python -m reporting.figures calibration.

Reads the current raw/summary CSVs and writes disposable PDF figures and a
plain LaTeX summary table inside that comparison's results directory.
"""

from __future__ import annotations

import argparse
import json
import os

from gbci.paths import PROJECT_ROOT, RESULTS_DIR

os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".cache" / "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd

STUDIES = (
    "ate",
    "cate",
    "likelihood_comparison",
    "calibration",
    "prior_strength",
    "cross_fitting",
    "nuisance_learners",
    "bart_comparison",
    "ihdp",
)


def _save(fig, output, name):
    folder = output / "figures"
    folder.mkdir(exist_ok=True)
    fig.savefig(folder / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def _heatmap(frame, row, column, metric, output, name, *, logarithmic=False):
    values = frame.pivot(index=row, columns=column, values=metric).sort_index()
    assert np.isfinite(values.to_numpy()).all(), f"Missing cells in {name}"
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    options = (
        {"norm": LogNorm(0.1, 10), "cmap": "RdYlGn_r"}
        if logarithmic
        else {"vmin": 0, "vmax": 1, "cmap": "viridis"}
    )
    im = ax.imshow(values.to_numpy(), aspect="auto", **options)
    ax.set_xticks(range(len(values.columns)), labels=values.columns)
    ax.set_yticks(range(len(values.index)), labels=values.index)
    ax.set_xlabel(column.replace("_", " "))
    ax.set_ylabel(row.replace("_", " "))
    for i, line in enumerate(values.to_numpy()):
        for j, value in enumerate(line):
            ax.text(
                j,
                i,
                f"{value:.2g}" if logarithmic else f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=7,
                color="black" if logarithmic or value > 0.55 else "white",
            )
    fig.colorbar(im, ax=ax, label=metric.replace("_", " "))
    _save(fig, output, name)


def _comparison(summary, category, method, output, name):
    categories, methods = list(summary[category].unique()), list(summary[method].unique())
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.0))
    width = 0.8 / len(methods)
    for i, label in enumerate(methods):
        arm = summary.loc[summary[method] == label].set_index(category).reindex(categories)
        assert arm.index.is_unique and arm["coverage"].notna().all()
        x = np.arange(len(categories)) + (i - (len(methods) - 1) / 2) * width
        for ax, metric in zip(axes, ("coverage", "mse", "mean_ci_length")):
            ax.bar(x, arm[metric], width=width, label=str(label).replace("_", " "))
        if {"coverage_mc_lo", "coverage_mc_hi"} <= set(arm):
            axes[0].errorbar(
                x,
                arm.coverage,
                yerr=[arm.coverage - arm.coverage_mc_lo, arm.coverage_mc_hi - arm.coverage],
                fmt="none",
                color="black",
                linewidth=0.7,
                capsize=2,
            )
    for ax, title in zip(axes, ("Coverage", "Posterior-mean MSE", "Mean interval length")):
        ax.set_xticks(range(len(categories)), labels=[str(c).replace("_", " ") for c in categories])
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.2)
    axes[0].axhline(0.95, color="black", linestyle="--", linewidth=0.8)
    axes[0].set_ylim(0, 1.08)
    axes[-1].legend(fontsize=7)
    fig.tight_layout()
    _save(fig, output, name)


def ate(output, raw, summary, settings):
    baseline = summary.loc[summary.n == settings["baseline_n"]]
    _heatmap(baseline, "method", "dgp", "coverage", output, "coverage")
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    for method, arm in summary.loc[summary.dgp == 1].groupby("method"):
        arm = arm.sort_values("n")
        ax.plot(arm.n, arm.mean_ci_length, marker=".", label=method.upper())
    ax.set(xlabel="Sample size", ylabel="Mean 95% interval length")
    ax.grid(alpha=0.2)
    ax.legend()
    _save(fig, output, "interval_length")


def cate(output, raw, summary, settings):
    for model, arm in summary.groupby("model"):
        _heatmap(arm, "method", "dgp", "coverage", output, f"coverage_{model}")
    curve = pd.read_csv(output / "example_curve.csv")
    fig, ax = plt.subplots(figsize=(5.5, 3.3))
    example = curve.loc[curve.method == "aipw"]
    for model, arm in example.groupby("model"):
        arm = arm.sort_values("x0")
        ax.plot(arm.x0, arm["mean"], label=model)
        ax.fill_between(arm.x0, arm.ci_lo, arm.ci_hi, alpha=0.15)
    truth = example.loc[example.model == "analytic"].sort_values("x0")
    ax.plot(truth.x0, truth.truth, color="black", linestyle="--", label="true CATE")
    ax.set(xlabel="First covariate (others fixed)", ylabel="CATE")
    ax.legend()
    _save(fig, output, "cate_slice")


def calibration(output, raw, summary, settings):
    scaled = summary.loc[
        summary.calibration.str.startswith("analytic_x_")
        | summary.calibration.eq("analytic_plugin")
    ].copy()
    scaled["scale"] = scaled.calibration.map(
        lambda value: (
            1.0 if value == "analytic_plugin" else float(value.removeprefix("analytic_x_"))
        )
    )
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    for color, (n, group) in zip(
        plt.rcParams["axes.prop_cycle"].by_key()["color"], scaled.groupby("n")
    ):
        for dgp, arm in group.groupby("dgp"):
            arm = arm.sort_values("scale")
            ax.plot(
                arm.scale,
                arm.coverage,
                color=color,
                linestyle="--" if dgp == 6 else "-",
                alpha=0.7 if dgp == 6 else 0.2,
            )
        typical = group.loc[group.dgp != 6].groupby("scale").coverage.median()
        ax.plot(
            typical.index, typical, marker=".", color=color, label=f"n={n}, median excluding D6"
        )
    ax.axhline(0.95, color="black", linestyle=":", linewidth=0.8)
    ax.set(xlabel="Multiplier of analytic learning rate", ylabel="Coverage", ylim=(0, 1.03))
    ax.legend(fontsize=7)
    _save(fig, output, "calibration_scaling")


def prior_strength(output, raw, summary, settings):
    comparison = pd.read_csv(output / "mse_comparison.csv")
    _heatmap(
        comparison,
        "prior_sd",
        "prior_offset",
        "mse_ratio",
        output,
        "prior_strength",
        logarithmic=True,
    )


def cross_fitting(output, raw, summary, settings):
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    for dgp, arm in summary.loc[summary.scope == "folds"].groupby("dgp"):
        arm = arm.sort_values("folds")
        ax.errorbar(
            arm.folds,
            arm.coverage,
            yerr=[arm.coverage - arm.coverage_mc_lo, arm.coverage_mc_hi - arm.coverage],
            marker=".",
            capsize=2,
            label=f"D{dgp}",
        )
    ax.axhline(0.95, color="black", linestyle="--", linewidth=0.8)
    ax.set(xlabel="Cross-fitting folds", ylabel="Coverage", ylim=(0, 1.03))
    ax.legend()
    _save(fig, output, "cross_fitting")


def nuisance_learners(output, raw, summary, settings):
    extension = pd.read_csv(output / "all_dgp_summary.csv")
    _heatmap(extension, "learner", "dgp", "coverage", output, "nuisance_coverage")


def likelihood_comparison(output, raw, summary, settings):
    _comparison(summary, "scenario", "method", output, "likelihood_comparison")


def bart_comparison(output, raw, summary, settings):
    _comparison(summary, "dgp", "method", output, "bart_comparison")


def ihdp(output, raw, summary, settings):
    _comparison(summary, "bundle", "method", output, "ihdp")


def run(study: str):
    output = RESULTS_DIR / study
    raw = pd.read_csv(output / "raw.csv")
    summary = pd.read_csv(output / "summary.csv")
    settings = json.loads((output / "settings.json").read_text(encoding="utf-8"))
    columns = [
        name
        for name in (
            "dgp",
            "bundle",
            "scenario",
            "n",
            "method",
            "model",
            "learner",
            "calibration",
            "prior_sd",
            "prior_offset",
            "scope",
            "folds",
            "partition",
            "replications",
            "coverage",
            "coverage_mc_lo",
            "coverage_mc_hi",
            "coverage_mc_se",
            "mse",
            "mean_ci_length",
            "mean_runtime_seconds",
            "mean_calibration_runtime_seconds",
        )
        if name in summary
    ]
    (output / "summary.tex").write_text(
        summary[columns].to_latex(index=False, escape=True, float_format=lambda x: f"{x:.4g}"),
        encoding="utf-8",
    )
    globals()[study](output, raw, summary, settings)
    print(f"Figures and summary table saved in {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", choices=(*STUDIES, "all"))
    selected = parser.parse_args().study
    for study in STUDIES if selected == "all" else (selected,):
        run(study)
