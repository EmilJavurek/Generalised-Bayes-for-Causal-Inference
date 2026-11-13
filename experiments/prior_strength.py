"""Informative prior strength and offset experiment."""

import numpy as np
from gbci.simulation import simulate, stable_seed
from gbci.estimation import cross_fit_phi, omega_analytic
from gbci.posterior import gaussian_ate
from experiments.config import load_config
from reporting.results import new_run_directory, save_results


def run(s):
    out = new_run_directory("prior_strength")
    rows = []
    for rep in range(s["replications"]):
        seed = stable_seed(s["seed"], "prior_strength", rep)
        data = simulate(s["dgp"], s["n"], stable_seed(seed, "data"))
        phi = cross_fit_phi(
            data["X"],
            data["A"],
            data["Y"],
            learner=s["learner"],
            method=s["method"],
            clip_eps=s["clip_eps"],
            lam_prop=s["lam_prop"],
            lam_outcome=s["lam_outcome"],
            folds=s["folds"],
            split_seed=stable_seed(seed, "split"),
            nuisance_seed=stable_seed(seed, "nuisance"),
        )["phi"]
        omega = omega_analytic(phi)
        truth = data["ate_true"]
        for sd in s["prior_sds"]:
            for offset in s["prior_offsets"]:
                post = gaussian_ate(phi, omega, prior_mean=truth + offset, prior_sd=sd)
                rows.append(
                    {
                        "rep": rep,
                        "prior_sd": sd,
                        "prior_offset": offset,
                        "omega": omega,
                        "ate_true": truth,
                        **post,
                        "covered": post["ci_lo"] <= truth <= post["ci_hi"],
                        "sq_error": (post["post_mean"] - truth) ** 2,
                        "frequentist_sq_error": (np.mean(phi) - truth) ** 2,
                    }
                )
    raw, _ = save_results(out, rows, s, ["prior_sd", "prior_offset"])
    view = raw.groupby(["prior_sd", "prior_offset"], as_index=False).agg(
        posterior_mean_mse=("sq_error", "mean"),
        frequentist_mse=("frequentist_sq_error", "mean"),
        mean_ci_length=("ci_length", "mean"),
    )
    assert (view.frequentist_mse > 0).all(), "A zero reference MSE makes the ratio undefined"
    view["mse_ratio"] = view.posterior_mean_mse / view.frequentist_mse
    view["inside_sqrt2_boundary"] = view.prior_offset.abs() < np.sqrt(2) * view.prior_sd
    view["inside_finite_sample_boundary"] = (
        view.prior_offset**2 < 2 * view.prior_sd**2 + view.frequentist_mse
    )
    view.to_csv(out / "mse_comparison.csv", index=False)
    return out


if __name__ == "__main__":
    run(load_config("prior_strength"))
