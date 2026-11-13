"""Paired BART response-surface and generalized Bayes comparison."""

import numpy as np
from time import perf_counter
import pymc as pm
import pymc_bart as pmb
from pymc_bart.utils import _sample_posterior
from gbci.simulation import simulate, stable_seed
from gbci.estimation import cross_fit_phi, omega_analytic
from gbci.posterior import gaussian_ate
from experiments.config import load_config
from reporting.results import new_run_directory, save_results


def bart_posterior(X, A, Y, seed, trees, draws, tune):

    observed = np.column_stack([A, X])
    counterfactual = np.vstack(
        [np.column_stack([np.ones(len(X)), X]), np.column_stack([np.zeros(len(X)), X])]
    )
    with pm.Model():
        design = pm.Data("design", observed)
        mu = pmb.BART("mu", X=design, Y=Y, m=trees)
        tree_store = mu.owner.op.all_trees
        manager = getattr(tree_store, "_manager", None)
        try:
            pm.Normal("outcome", mu=mu, sigma=1.0, observed=Y, shape=mu.shape)
            pm.sample(
                draws=draws,
                tune=tune,
                chains=1,
                cores=1,
                random_seed=seed,
                progressbar=False,
                compute_convergence_checks=False,
                return_inferencedata=True,
            )
            predictions = _sample_posterior(
                tree_store,
                X=counterfactual,
                rng=np.random.default_rng(stable_seed(seed, "predict")),
                size=draws,
                shape=1,
            ).squeeze(axis=-1)
        finally:
            if manager is not None:
                manager.shutdown()
    assert predictions.shape == (draws, 2 * len(X)) and np.isfinite(predictions).all()
    effects = predictions[:, : len(X)] - predictions[:, len(X) :]
    weights = np.random.default_rng(stable_seed(seed, "covariate_distribution")).exponential(
        size=effects.shape
    )
    weights /= weights.sum(axis=1, keepdims=True)
    ate = np.sum(weights * effects, axis=1)
    lo, hi = np.quantile(ate, [0.025, 0.975])
    return {
        "post_mean": float(ate.mean()),
        "post_var": float(ate.var(ddof=1)),
        "post_sd": float(ate.std(ddof=1)),
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "ci_length": float(hi - lo),
    }


def run(s):
    out = new_run_directory("bart_comparison")
    rows = []
    for dgp in s["dgps"]:
        for rep in range(s["replications"]):
            seed = stable_seed(s["seed"], "bart_comparison", dgp, rep)
            data = simulate(dgp, s["n"], stable_seed(seed, "data"))
            X, A, Y = data["X"], data["A"], data["Y"]
            start = perf_counter()
            bart = bart_posterior(
                X, A, Y, stable_seed(seed, "bart"), s["trees"], s["draws"], s["tune"]
            )
            bart_time = perf_counter() - start
            start = perf_counter()
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
            generalized = gaussian_ate(
                phi, omega_analytic(phi), prior_mean=s["prior_mean"], prior_sd=s["prior_sd"]
            )
            generalized_time = perf_counter() - start
            truth = data["ate_true"]
            for method, post in (("bart", bart), ("generalized_bayes_aipw", generalized)):
                rows.append(
                    {
                        "dgp": dgp,
                        "n": s["n"],
                        "rep": rep,
                        "method": method,
                        "ate_true": truth,
                        "runtime_seconds": bart_time if method == "bart" else generalized_time,
                        **post,
                        "covered": post["ci_lo"] <= truth <= post["ci_hi"],
                        "sq_error": (post["post_mean"] - truth) ** 2,
                    }
                )
    raw, _ = save_results(out, rows, s, ["dgp", "n", "method"])
    paired = raw.pivot(
        index=["dgp", "n", "rep"], columns="method", values=["sq_error", "ci_length"]
    )
    for metric in ("sq_error", "ci_length"):
        paired[(metric, "bart_minus_generalized")] = (
            paired[(metric, "bart")] - paired[(metric, "generalized_bayes_aipw")]
        )
    paired.to_csv(out / "paired_differences.csv")
    return out


if __name__ == "__main__":
    run(load_config("bart_comparison"))
