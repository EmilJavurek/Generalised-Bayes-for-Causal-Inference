"""CATE GP comparison; metrics average over fresh covariates within each dataset."""

import numpy as np
import pandas as pd

from gbci.simulation import simulate, stable_seed
from gbci.estimation import cross_fit_phi, omega_analytic, pseudo_outcome
from gbci.posterior import fit_cate_gp
from experiments.config import load_config
from reporting.results import new_run_directory, save_results


def run(s):
    assert s["repetitions"] > 0 and s["evaluation_n"] > 0
    assert s["example_points"] >= 2
    assert all(
        np.isfinite(s[key]) for key in ("example_min", "example_max", "example_other_covariates")
    )
    assert s["example_min"] < s["example_max"]
    output = new_run_directory("cate")
    rows, example = [], []
    for dgp in s["dgps"]:
        for rep in range(s["repetitions"]):
            seed = stable_seed(s["seed"], dgp, s["n"], rep)
            data = simulate(dgp, s["n"], stable_seed(seed, "data"))
            evaluation = simulate(dgp, s["evaluation_n"], stable_seed(seed, "evaluation"))
            x_eval = evaluation["X"]
            truth = np.asarray(data["cate_true"](x_eval), dtype=float).reshape(-1)
            assert truth.shape == (s["evaluation_n"],) and np.isfinite(truth).all()
            if dgp == s["example_dgp"] and rep == 0:
                grid = np.full(
                    (s["example_points"], x_eval.shape[1]),
                    s["example_other_covariates"],
                    dtype=float,
                )
                grid[:, 0] = np.linspace(s["example_min"], s["example_max"], s["example_points"])
                grid_truth = np.asarray(data["cate_true"](grid), dtype=float).reshape(-1)
                assert grid_truth.shape == (s["example_points"],) and np.isfinite(grid_truth).all()
            fit = cross_fit_phi(
                data["X"],
                data["A"],
                data["Y"],
                learner=s["learner"],
                method="aipw",
                folds=s["folds"],
                split_seed=stable_seed(seed, "split"),
                nuisance_seed=stable_seed(seed, "nuisance"),
                clip_eps=s["clip_eps"],
                lam_prop=s["lam_prop"],
                lam_outcome=s["lam_outcome"],
            )
            for method in s["methods"]:
                phi = pseudo_outcome(
                    method, data["A"], data["Y"], fit["e_hat"], fit["mu0_hat"], fit["mu1_hat"]
                )
                omega = omega_analytic(phi)
                models = fit_cate_gp(
                    data["X"],
                    phi,
                    omega,
                    inducing_points=s["inducing_points"],
                    kernel=s["kernel"],
                    lengthscale=s["lengthscale"],
                    variance=s["variance"],
                    jitter=s["jitter"],
                    epochs=s["epochs"],
                    learning_rate=s["learning_rate"],
                    seed=stable_seed(seed, method, "gp"),
                )
                for model_name in ("analytic", "variational"):
                    mean, variance = models[model_name].predict(x_eval)
                    mean, variance = np.asarray(mean).reshape(-1), np.asarray(variance).reshape(-1)
                    assert mean.shape == variance.shape == truth.shape
                    assert (
                        np.isfinite(mean).all()
                        and np.isfinite(variance).all()
                        and (variance >= 0).all()
                    )
                    half = 1.959963984540054 * np.sqrt(variance)
                    lo, hi = mean - half, mean + half
                    rows.append(
                        {
                            "dgp": dgp,
                            "n": s["n"],
                            "rep": rep,
                            "method": method,
                            "model": model_name,
                            "omega": float(omega),
                            "pointwise_coverage": float(np.mean((lo <= truth) & (truth <= hi))),
                            "mean_ci_length": float(np.mean(hi - lo)),
                            "mse": float(np.mean((mean - truth) ** 2)),
                        }
                    )
                    if dgp == s["example_dgp"] and rep == 0:
                        grid_mean, grid_var = models[model_name].predict(grid)
                        grid_mean = np.asarray(grid_mean).reshape(-1)
                        grid_var = np.asarray(grid_var).reshape(-1)
                        assert grid_mean.shape == grid_var.shape == grid_truth.shape
                        assert (
                            np.isfinite(grid_mean).all()
                            and np.isfinite(grid_var).all()
                            and (grid_var >= 0).all()
                        )
                        grid_half = 1.959963984540054 * np.sqrt(grid_var)
                        for i in range(len(grid_truth)):
                            example.append(
                                {
                                    "method": method,
                                    "model": model_name,
                                    "index": i,
                                    **{f"x{j}": float(grid[i, j]) for j in range(grid.shape[1])},
                                    "truth": grid_truth[i],
                                    "mean": grid_mean[i],
                                    "ci_lo": grid_mean[i] - grid_half[i],
                                    "ci_hi": grid_mean[i] + grid_half[i],
                                }
                            )
    save_results(output, rows, s, ["dgp", "n", "method", "model"])
    if example:
        pd.DataFrame(example).to_csv(output / "example_curve.csv", index=False)
    print(f"Wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    run(load_config("cate"))
