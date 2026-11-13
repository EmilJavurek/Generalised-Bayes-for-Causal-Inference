"""Standard likelihood versus generalized Bayes under outcome misspecification."""

import warnings

import numpy as np
from scipy.special import expit
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold
from gbci.simulation import stable_seed
from gbci.estimation import omega_analytic, pseudo_outcome
from gbci.posterior import gaussian_ate
from experiments.config import load_config
from reporting.results import new_run_directory, save_results


def run(s):
    out = new_run_directory("likelihood_comparison")
    rows = []
    for scenario, severity in s["scenarios"].items():
        for rep in range(s["replications"]):
            seed = stable_seed(s["seed"], "likelihood_comparison", scenario, rep)
            rng = np.random.default_rng(stable_seed(seed, "data"))
            mean = severity * np.array([1.0, -1.0])
            X = rng.standard_normal((s["n"], 2)) + mean
            A = rng.binomial(1, expit(X @ np.array([0.6 + 0.2 * severity, -0.6 - 0.2 * severity])))
            tau = 1.0 + X @ (severity * np.array([0.5, -0.5]))
            Y = X @ np.ones(2) + A * tau + rng.standard_normal(s["n"])
            truth = float(1.0 + mean @ (severity * np.array([0.5, -0.5])))
            features = np.column_stack([np.ones(len(X)), X])
            design = np.column_stack([features[:, 0], A, features[:, 1:]])
            covariance = np.linalg.inv(
                design.T @ design + np.eye(design.shape[1]) / s["prior_sd"] ** 2
            )
            prior = np.full(design.shape[1], s["prior_mean"])
            coefficient = covariance @ (design.T @ Y + prior / s["prior_sd"] ** 2)
            sd = float(np.sqrt(covariance[1, 1]))
            mu = float(coefficient[1])
            z = 1.959963984540054
            standard = {
                "post_mean": mu,
                "post_var": sd**2,
                "post_sd": sd,
                "ci_lo": mu - z * sd,
                "ci_hi": mu + z * sd,
                "ci_length": 2 * z * sd,
            }
            e, m0, m1 = (np.empty(len(X)) for _ in range(3))
            for train, test in KFold(
                n_splits=s["folds"], shuffle=True, random_state=stable_seed(seed, "split")
            ).split(X):
                with warnings.catch_warnings():
                    warnings.simplefilter("error", ConvergenceWarning)
                    propensity = LogisticRegression(
                        C=1.0 / s["lam_prop"], fit_intercept=False, solver="lbfgs", max_iter=500
                    ).fit(features[train], A[train])
                outcome = Ridge(alpha=s["lam_outcome"], fit_intercept=False).fit(
                    design[train], Y[train]
                )
                e[test] = np.clip(
                    propensity.predict_proba(features[test])[:, 1], s["clip_eps"], 1 - s["clip_eps"]
                )
                d0 = np.column_stack([features[test, 0], np.zeros(len(test)), features[test, 1:]])
                d1 = np.column_stack([features[test, 0], np.ones(len(test)), features[test, 1:]])
                m0[test], m1[test] = outcome.predict(d0), outcome.predict(d1)
            phi = pseudo_outcome(s["method"], A, Y, e, m0, m1)
            assert np.isfinite(phi).all()
            omega = omega_analytic(phi)
            generalized = gaussian_ate(
                phi, omega, prior_mean=s["prior_mean"], prior_sd=s["prior_sd"]
            )
            for method, post in (
                ("standard_bayes", standard),
                ("generalized_bayes_aipw", generalized),
            ):
                rows.append(
                    {
                        "scenario": scenario,
                        "severity": severity,
                        "n": s["n"],
                        "rep": rep,
                        "method": method,
                        "ate_true": truth,
                        **post,
                        "covered": post["ci_lo"] <= truth <= post["ci_hi"],
                        "sq_error": (post["post_mean"] - truth) ** 2,
                    }
                )
    save_results(out, rows, s, ["scenario", "severity", "n", "method"])
    return out


if __name__ == "__main__":
    run(load_config("likelihood_comparison"))
