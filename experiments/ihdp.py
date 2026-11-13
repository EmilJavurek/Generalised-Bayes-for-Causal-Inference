"""IHDP ATE: train-unit nuisance fitting and test-unit potential-outcome truth.

The overlapping bundles are reported separately, not pooled."""

import numpy as np

from gbci.paths import DATASETS_DIR
from gbci.simulation import stable_seed
from gbci.estimation import cross_fit_phi, omega_analytic, pseudo_outcome
from gbci.posterior import gaussian_ate
from experiments.config import load_config
from reporting.results import new_run_directory, save_results


def run(s):
    assert s["max_reps"] is None or s["max_reps"] > 0
    output = new_run_directory("ihdp")
    rows = []
    for bundle, filename in s["bundles"].items():
        with np.load(DATASETS_DIR / "IHDP" / f"{filename}.train.npz") as train:
            x, a, y = (np.asarray(train[k], dtype=np.float64) for k in ("x", "t", "yf"))
        with np.load(DATASETS_DIR / "IHDP" / f"{filename}.test.npz") as test:
            mu0, mu1 = (np.asarray(test[k], dtype=np.float64) for k in ("mu0", "mu1"))
        assert x.ndim == 3 and a.shape == y.shape == (x.shape[0], x.shape[2])
        assert mu0.ndim == 2 and mu0.shape == mu1.shape and mu0.shape[1] == x.shape[2]
        assert all(np.isfinite(v).all() for v in (x, a, y, mu0, mu1))
        assert np.isin(a, (0, 1)).all()
        count = x.shape[2] if s["max_reps"] is None else min(s["max_reps"], x.shape[2])
        for rep in range(count):
            X, A, Y = x[:, :, rep], a[:, rep], y[:, rep]
            truth = float(np.mean(mu1[:, rep] - mu0[:, rep]))
            seed = stable_seed(s["seed"], bundle, rep)
            fit = cross_fit_phi(
                X,
                A,
                Y,
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
                phi = pseudo_outcome(method, A, Y, fit["e_hat"], fit["mu0_hat"], fit["mu1_hat"])
                omega = omega_analytic(phi)
                post = gaussian_ate(phi, omega, prior_mean=s["prior_mean"], prior_sd=s["prior_sd"])
                rows.append(
                    {
                        "bundle": bundle,
                        "rep": rep,
                        "method": method,
                        "n": len(X),
                        "ate_true": truth,
                        "phi_mean": float(np.mean(phi)),
                        "phi_var": float(np.var(phi, ddof=1)),
                        "omega": float(omega),
                        **post,
                        "covered": int(post["ci_lo"] <= truth <= post["ci_hi"]),
                        "sq_error": float((post["post_mean"] - truth) ** 2),
                    }
                )
    save_results(output, rows, s, ["bundle", "method"])
    print(f"Wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    run(load_config("ihdp"))
