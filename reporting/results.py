"""Monte Carlo summaries and result files."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from gbci.paths import RESULTS_DIR


def new_run_directory(name: str) -> Path:
    output = RESULTS_DIR / name
    if output.exists():
        raise FileExistsError(
            f"Results already exist: {output}. Delete deliberately or use another GBCI_RESULTS_DIR."
        )
    output.mkdir(parents=True)
    return output


def summarize(frame: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    assert not frame.empty, "No replications to summarize"
    result = []
    for keys, group in frame.groupby(group_columns, sort=True, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_columns, keys))
        assert not group["rep"].duplicated().any(), (
            "Repeated dataset within a summary group; include the comparison arm in group_columns"
        )
        row["replications"] = int(group["rep"].nunique())
        row["rows"] = len(group)
        if "covered" in group:
            assert group["covered"].isin((True, False)).all()
            successes = int(group["covered"].sum())
            interval = binomtest(successes, len(group)).proportion_ci(method="exact")
            row.update(
                coverage=successes / len(group),
                coverage_mc_lo=float(interval.low),
                coverage_mc_hi=float(interval.high),
            )
        if "pointwise_coverage" in group:
            values = group["pointwise_coverage"]
            assert np.isfinite(values).all() and values.between(0, 1).all()
            row["coverage"] = float(values.mean())
            row["coverage_mc_se"] = (
                float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else np.nan
            )
        for source, target in (
            ("sq_error", "mse"),
            ("mse", "mse"),
            ("ci_length", "mean_ci_length"),
            ("mean_ci_length", "mean_ci_length"),
            ("post_sd", "mean_post_sd"),
            ("runtime_seconds", "mean_runtime_seconds"),
            ("omega", "mean_omega"),
            ("sd_efficiency_ratio", "mean_sd_efficiency_ratio"),
            ("calibration_runtime_seconds", "mean_calibration_runtime_seconds"),
        ):
            if source in group:
                values = pd.to_numeric(group[source], errors="raise")
                assert np.isfinite(values).all(), f"Missing or nonfinite {source}"
                row[target] = float(values.mean())
                row[f"{target}_mc_se"] = (
                    float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else np.nan
                )
        result.append(row)
    return pd.DataFrame(result)


def save_results(
    output: Path, rows, settings: dict, group_columns: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.DataFrame(rows)
    assert not raw.empty, "No results were produced"
    summary = summarize(raw, group_columns)
    raw.to_csv(output / "raw.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    (output / "settings.json").write_text(
        json.dumps(settings, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(f"Saved {len(raw)} rows to {output}")
    return raw, summary
