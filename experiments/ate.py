"""Simulated ATE strategy and sample-size comparisons on shared datasets."""

import numpy as np

from gbci.simulation import simulate, stable_seed
from gbci.estimation import cross_fit_phi, omega_analytic, pseudo_outcome
from gbci.posterior import gaussian_ate
from experiments.config import load_config
from reporting.results import new_run_directory, save_results


def run(s):
    assert s["repetitions"] > 0
    designs = sorted(
        {(dgp, s["baseline_n"]) for dgp in s["dgps"]} | {(1, n) for n in s["sample_sizes"]}
    )
    output = new_run_directory("ate")
    rows = []
    for dgp, n in designs:
        for rep in range(s["repetitions"]):
            seed = stable_seed(s["seed"], dgp, n, rep)
            data = simulate(dgp, n, stable_seed(seed, "data"))
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
                post = gaussian_ate(phi, omega, prior_mean=s["prior_mean"], prior_sd=s["prior_sd"])
                truth = float(data["ate_true"])
                rows.append(
                    {
                        "dgp": dgp,
                        "n": n,
                        "rep": rep,
                        "method": method,
                        "ate_true": truth,
                        "phi_mean": float(np.mean(phi)),
                        "omega": float(omega),
                        **post,
                        "covered": int(post["ci_lo"] <= truth <= post["ci_hi"]),
                        "sq_error": float((post["post_mean"] - truth) ** 2),
                    }
                )
    save_results(output, rows, s, ["dgp", "n", "method"])
    print(f"Wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    run(load_config("ate"))
