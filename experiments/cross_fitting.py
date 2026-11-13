"""K-fold and partition sensitivity with a shared K=5 learning rate."""

from gbci.simulation import simulate, stable_seed
from gbci.estimation import cross_fit_phi, omega_analytic
from gbci.posterior import gaussian_ate
from experiments.config import load_config
from reporting.results import new_run_directory, save_results


def run(s):
    out = new_run_directory("cross_fitting")
    rows = []
    for dgp in s["dgps"]:
        for rep in range(s["replications"]):
            seed = stable_seed(s["seed"], "cross_fitting", dgp, rep)
            data = simulate(dgp, s["n"], stable_seed(seed, "data"))
            X, A, Y = data["X"], data["A"], data["Y"]
            split, nuisance = stable_seed(seed, "split"), stable_seed(seed, "nuisance")
            fit_options = dict(
                learner=s["learner"],
                method=s["method"],
                clip_eps=s["clip_eps"],
                lam_prop=s["lam_prop"],
                lam_outcome=s["lam_outcome"],
                nuisance_seed=nuisance,
            )
            baseline = cross_fit_phi(
                X, A, Y, folds=s["calibration_folds"], split_seed=split, **fit_options
            )
            omega = omega_analytic(baseline["phi"])
            arms = []
            for k in s["folds"]:
                fit = (
                    baseline
                    if k == s["calibration_folds"]
                    else cross_fit_phi(X, A, Y, folds=k, split_seed=split, **fit_options)
                )
                arms.append(("folds", k, 0, fit))
            if rep < s["partition_replications"]:
                for p in range(1, s["extra_partitions"] + 1):
                    fit = cross_fit_phi(
                        X,
                        A,
                        Y,
                        folds=s["calibration_folds"],
                        split_seed=stable_seed(split, "partition", p),
                        **fit_options,
                    )
                    arms.append(("partition", s["calibration_folds"], p, fit))
            for scope, folds, partition, fit in arms:
                post = gaussian_ate(
                    fit["phi"], omega, prior_mean=s["prior_mean"], prior_sd=s["prior_sd"]
                )
                truth = data["ate_true"]
                rows.append(
                    {
                        "dgp": dgp,
                        "n": s["n"],
                        "rep": rep,
                        "scope": scope,
                        "folds": folds,
                        "partition": partition,
                        "omega": omega,
                        "ate_true": truth,
                        **post,
                        "covered": post["ci_lo"] <= truth <= post["ci_hi"],
                        "sq_error": (post["post_mean"] - truth) ** 2,
                    }
                )
    raw, _ = save_results(out, rows, s, ["dgp", "n", "scope", "folds", "partition"])
    random = raw[
        (
            (raw.scope == "partition")
            | ((raw.scope == "folds") & (raw.folds == s["calibration_folds"]))
        )
        & (raw.rep < s["partition_replications"])
    ]
    spread = random.groupby(["dgp", "n", "rep"], as_index=False).agg(
        partition_sd=("post_mean", "std"),
        partition_range=("post_mean", lambda x: x.max() - x.min()),
    )
    spread.to_csv(out / "partition_spread.csv", index=False)
    return out


if __name__ == "__main__":
    run(load_config("cross_fitting"))
