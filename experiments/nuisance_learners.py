"""Nuisance learner comparison and shared Bayesian/all-DGP views."""

from gbci.simulation import simulate, stable_seed
from gbci.estimation import cross_fit_phi, omega_analytic, propensity_diagnostics
from gbci.posterior import gaussian_ate
from experiments.config import load_config
from reporting.results import new_run_directory, save_results, summarize


def run(s):
    out = new_run_directory("nuisance_learners")
    rows = []
    for dgp in s["dgps"]:
        reps = s["primary_replications"] if dgp in s["primary_dgps"] else s["other_replications"]
        for rep in range(reps):
            seed = stable_seed(s["seed"], "nuisance_learners", dgp, rep)
            data = simulate(dgp, s["n"], stable_seed(seed, "data"))
            X, A, Y = data["X"], data["A"], data["Y"]
            split, nuisance = stable_seed(seed, "split"), stable_seed(seed, "nuisance")
            baseline = cross_fit_phi(
                X,
                A,
                Y,
                learner=s["baseline_learner"],
                method=s["method"],
                clip_eps=s["clip_eps"],
                lam_prop=s["lam_prop"],
                lam_outcome=s["lam_outcome"],
                folds=s["folds"],
                split_seed=split,
                nuisance_seed=nuisance,
            )
            omega = omega_analytic(baseline["phi"])
            for learner in s["learners"]:
                fit = (
                    baseline
                    if learner == s["baseline_learner"]
                    else cross_fit_phi(
                        X,
                        A,
                        Y,
                        learner=learner,
                        method=s["method"],
                        clip_eps=s["clip_eps"],
                        lam_prop=s["lam_prop"],
                        lam_outcome=s["lam_outcome"],
                        folds=s["folds"],
                        split_seed=split,
                        nuisance_seed=stable_seed(nuisance, learner),
                        mlp_max_iter=s["mlp_max_iter"],
                    )
                )
                post = gaussian_ate(
                    fit["phi"], omega, prior_mean=s["prior_mean"], prior_sd=s["prior_sd"]
                )
                truth = data["ate_true"]
                rows.append(
                    {
                        "dgp": dgp,
                        "n": s["n"],
                        "rep": rep,
                        "learner": learner,
                        "omega": omega,
                        "ate_true": truth,
                        **post,
                        **propensity_diagnostics(A, fit["e_hat"], clip_eps=s["clip_eps"]),
                        "covered": post["ci_lo"] <= truth <= post["ci_hi"],
                        "sq_error": (post["post_mean"] - truth) ** 2,
                    }
                )
    raw, summary = save_results(out, rows, s, ["dgp", "n", "learner"])
    summarize(raw[raw.rep < s["extension_replications"]], ["dgp", "n", "learner"]).to_csv(
        out / "all_dgp_summary.csv", index=False
    )
    summary[summary.learner == "bayesian_gp"].to_csv(out / "bayesian_summary.csv", index=False)
    return out


if __name__ == "__main__":
    run(load_config("nuisance_learners"))
