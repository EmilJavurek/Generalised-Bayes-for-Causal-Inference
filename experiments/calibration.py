"""Calibration sweep and efficiency comparison."""

import numpy as np
from time import perf_counter
from gbci.simulation import simulate, stable_seed
from gbci.estimation import (
    cross_fit_phi,
    omega_analytic,
    omega_bootstrap_fixed_phi,
    omega_bootstrap_refit,
)
from gbci.posterior import gaussian_ate
from experiments.config import load_config
from reporting.results import new_run_directory, save_results


def run(s):
    out = new_run_directory("calibration")
    rows = []
    for dgp in s["dgps"]:
        for n in s["sample_sizes"]:
            for rep in range(s["replications"]):
                seed = stable_seed(s["seed"], "calibration", dgp, n, rep)
                data = simulate(dgp, n, stable_seed(seed, "data"))
                X, A, Y = data["X"], data["A"], data["Y"]
                phi = cross_fit_phi(
                    X,
                    A,
                    Y,
                    learner=s["learner"],
                    method=s["method"],
                    clip_eps=s["clip_eps"],
                    lam_prop=s["lam_prop"],
                    lam_outcome=s["lam_outcome"],
                    folds=s["folds"],
                    split_seed=stable_seed(seed, "split"),
                    nuisance_seed=stable_seed(seed, "nuisance"),
                )["phi"]
                start = perf_counter()
                analytic = omega_analytic(phi)
                analytic_time = perf_counter() - start
                start = perf_counter()
                fixed = omega_bootstrap_fixed_phi(
                    phi, draws=s["bootstrap_draws"], seed=stable_seed(seed, "bootstrap")
                )
                fixed_time = perf_counter() - start
                timings = {
                    "fixed_omega_1": 0.0,
                    "analytic_plugin": analytic_time,
                    "bootstrap_fixed_phi": fixed_time,
                }
                arms = {
                    "fixed_omega_1": 1.0,
                    "analytic_plugin": analytic,
                    "bootstrap_fixed_phi": fixed,
                }
                arms.update({f"analytic_x_{v:g}": analytic * v for v in s["omega_scales"]})
                timings.update({f"analytic_x_{v:g}": analytic_time for v in s["omega_scales"]})
                if s["refit_bootstrap"]:
                    start = perf_counter()
                    arms["bootstrap_refit_mean_phi_variance"] = omega_bootstrap_refit(
                        X,
                        A,
                        Y,
                        draws=s["bootstrap_draws"],
                        seed=stable_seed(seed, "refit"),
                        learner=s["learner"],
                        method=s["method"],
                        folds=s["folds"],
                        clip_eps=s["clip_eps"],
                        lam_prop=s["lam_prop"],
                        lam_outcome=s["lam_outcome"],
                    )
                    timings["bootstrap_refit_mean_phi_variance"] = perf_counter() - start
                for arm, omega in arms.items():
                    post = gaussian_ate(
                        phi, omega, prior_mean=s["prior_mean"], prior_sd=s["prior_sd"]
                    )
                    truth = data["ate_true"]
                    rows.append(
                        {
                            "dgp": dgp,
                            "n": n,
                            "rep": rep,
                            "calibration": arm,
                            "omega": omega,
                            "calibration_runtime_seconds": timings[arm],
                            "ate_true": truth,
                            **post,
                            "covered": post["ci_lo"] <= truth <= post["ci_hi"],
                            "sq_error": (post["post_mean"] - truth) ** 2,
                            "sd_efficiency_ratio": post["post_sd"]
                            / np.sqrt(np.var(phi, ddof=1) / n),
                        }
                    )
    raw, _ = save_results(out, rows, s, ["dgp", "n", "calibration"])
    raw[raw.calibration == "analytic_plugin"].groupby(["dgp", "n"], as_index=False).agg(
        mean_sd_efficiency_ratio=("sd_efficiency_ratio", "mean"),
        min_sd_efficiency_ratio=("sd_efficiency_ratio", "min"),
        max_sd_efficiency_ratio=("sd_efficiency_ratio", "max"),
    ).to_csv(out / "efficiency_summary.csv", index=False)
    return out


if __name__ == "__main__":
    run(load_config("calibration"))
