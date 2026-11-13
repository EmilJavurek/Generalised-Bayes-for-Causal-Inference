"""Nuisance estimation, causal pseudo-outcomes, and learning-rate calibration.

All statistical arrays use NumPy float64. The bootstrap functions implement
different targets and deliberately have different names.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import BayesianRidge, LogisticRegression, Ridge
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from gbci.simulation import stable_seed


def _data(X, A, Y) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X, A, Y = np.asarray(X, dtype=np.float64), np.asarray(A), np.asarray(Y, dtype=np.float64)
    assert X.ndim == 2 and len(X) >= 2 and X.shape[1] > 0, "X must be a covariate matrix"
    assert A.shape == Y.shape == (len(X),), "Treatment/outcome shapes do not match X"
    assert np.isfinite(X).all() and np.isfinite(A).all() and np.isfinite(Y).all(), (
        "Nonfinite observations"
    )
    assert np.isin(A, (0, 1)).all(), "Treatment must be binary before conversion"
    return X, A.astype(int), Y


def feature_map(X: np.ndarray) -> np.ndarray:
    """Fixed enriched ridge features: X, X², sin(X), interaction, intercept."""
    X = np.asarray(X, dtype=np.float64)
    assert X.ndim == 2 and np.isfinite(X).all()
    extra = X[:, 0] * X[:, 1] if X.shape[1] >= 2 else X[:, 0] ** 3
    return np.column_stack([X, X**2, np.sin(X), extra, np.ones(len(X))])


def linear_features(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    assert X.ndim == 2 and np.isfinite(X).all()
    return np.column_stack([np.ones(len(X)), X])


def fit_nuisance(
    X,
    A,
    Y,
    *,
    learner: str = "ridge",
    seed: int,
    clip_eps: float = 0.01,
    lam_prop: float = 1.0,
    lam_outcome: float = 1.0,
    mlp_max_iter: int = 1000,
) -> Callable[[np.ndarray], dict[str, np.ndarray]]:
    """Fit a propensity model and one outcome model per treatment arm."""
    X, A, Y = _data(X, A, Y)
    assert 0 < clip_eps < 0.5 and lam_prop > 0 and lam_outcome >= 0
    assert np.bincount(A, minlength=2).min() >= 2, "Too few observations in a treatment arm"
    transform = (
        feature_map
        if learner == "ridge"
        else linear_features
        if learner == "linear"
        else np.asarray
    )
    features = transform(X)
    regression_propensity = False

    if learner in ("ridge", "linear"):
        propensity = LogisticRegression(C=1 / lam_prop, fit_intercept=False, max_iter=500)
        outcomes = [Ridge(alpha=lam_outcome, fit_intercept=False) for _ in range(2)]
    elif learner == "tree":
        parameters = dict(
            learning_rate=0.05,
            max_iter=100,
            max_leaf_nodes=8,
            min_samples_leaf=20,
            l2_regularization=1.0,
        )
        propensity = HistGradientBoostingClassifier(**parameters, random_state=seed)
        outcomes = [
            HistGradientBoostingRegressor(
                **parameters, random_state=stable_seed(seed, "outcome", a)
            )
            for a in (0, 1)
        ]
    elif learner == "mlp":
        assert mlp_max_iter > 0
        parameters = dict(
            hidden_layer_sizes=(12,),
            activation="tanh",
            solver="adam",
            alpha=0.01,
            batch_size=64,
            learning_rate_init=0.005,
            max_iter=mlp_max_iter,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=10,
        )
        propensity = make_pipeline(StandardScaler(), MLPClassifier(**parameters, random_state=seed))
        outcomes = [
            make_pipeline(
                StandardScaler(),
                MLPRegressor(**parameters, random_state=stable_seed(seed, "outcome", a)),
            )
            for a in (0, 1)
        ]
    elif learner == "bayesian_gp":
        # Posterior-mean plug-ins from an RBF random-feature approximation.
        gamma, components = 0.5 / max(X.shape[1], 1), min(96, max(48, 4 * X.shape[1]))

        def rff_model(model_seed):
            return make_pipeline(
                StandardScaler(),
                RBFSampler(gamma=gamma, n_components=components, random_state=model_seed),
                BayesianRidge(),
            )

        propensity = rff_model(seed)
        outcomes = [rff_model(stable_seed(seed, "outcome", a)) for a in (0, 1)]
        regression_propensity = True
    else:
        raise ValueError(f"Unknown nuisance learner: {learner}")

    # A fit that reaches its iteration limit is a failed fit, not a completed row.
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        propensity.fit(features, A)
        for treatment, model in enumerate(outcomes):
            mask = A == treatment
            minimum = (
                30
                if learner == "mlp"
                else 20
                if learner == "tree"
                else 10
                if learner == "bayesian_gp"
                else 2
            )
            assert mask.sum() >= minimum, f"Too few observations for {learner} in arm {treatment}"
            model.fit(features[mask], Y[mask])

    def predict(X_new: np.ndarray) -> dict[str, np.ndarray]:
        X_new = np.asarray(X_new, dtype=np.float64)
        assert X_new.ndim == 2 and X_new.shape[1] == X.shape[1] and np.isfinite(X_new).all()
        features_new = transform(X_new)
        e = (
            propensity.predict(features_new)
            if regression_propensity
            else propensity.predict_proba(features_new)[:, 1]
        )
        # Clipping is the specified estimator, not a repair for nonfinite predictions.
        assert np.isfinite(e).all(), "Nonfinite propensity predictions"
        prediction = {"e_hat": np.clip(np.asarray(e, dtype=np.float64), clip_eps, 1 - clip_eps)}
        prediction.update(
            {
                f"mu{a}_hat": np.asarray(model.predict(features_new), dtype=np.float64)
                for a, model in enumerate(outcomes)
            }
        )
        for name, values in prediction.items():
            assert values.shape == (len(X_new),) and np.isfinite(values).all(), (
                f"Invalid {name} predictions"
            )
        return prediction

    return predict


def pseudo_outcome(method: str, A, Y, e_hat, mu0_hat, mu1_hat) -> np.ndarray:
    A, Y, e, m0, m1 = [
        np.asarray(value, dtype=np.float64) for value in (A, Y, e_hat, mu0_hat, mu1_hat)
    ]
    assert A.ndim == 1 and all(value.shape == A.shape for value in (Y, e, m0, m1))
    assert all(np.isfinite(value).all() for value in (A, Y, e, m0, m1))
    assert np.isin(A, (0, 1)).all() and ((e > 0) & (e < 1)).all()
    if method == "aipw":
        phi = m1 - m0 + A * (Y - m1) / e - (1 - A) * (Y - m0) / (1 - e)
    elif method == "ipw":
        phi = A * Y / e - (1 - A) * Y / (1 - e)
    elif method == "ra":
        phi = m1 - m0
    else:
        raise ValueError(f"Unknown pseudo-outcome method: {method}")
    assert np.isfinite(phi).all(), "Nonfinite pseudo-outcomes"
    return phi


def cross_fit_phi(
    X,
    A,
    Y,
    *,
    learner: str = "ridge",
    method: str = "aipw",
    folds: int = 5,
    split_seed: int,
    nuisance_seed: int,
    clip_eps: float = 0.01,
    lam_prop: float = 1.0,
    lam_outcome: float = 1.0,
    mlp_max_iter: int = 1000,
) -> dict[str, np.ndarray]:
    X, A, Y = _data(X, A, Y)
    assert 2 <= folds <= len(X), "Cross-fitting requires between 2 and n folds"
    predictions = {key: np.full(len(X), np.nan) for key in ("e_hat", "mu0_hat", "mu1_hat")}
    for fold, (train, heldout) in enumerate(
        KFold(folds, shuffle=True, random_state=split_seed).split(X)
    ):
        predictor = fit_nuisance(
            X[train],
            A[train],
            Y[train],
            learner=learner,
            seed=stable_seed(nuisance_seed, fold),
            clip_eps=clip_eps,
            lam_prop=lam_prop,
            lam_outcome=lam_outcome,
            mlp_max_iter=mlp_max_iter,
        )
        for key, values in predictor(X[heldout]).items():
            predictions[key][heldout] = values
    assert all(np.isfinite(value).all() for value in predictions.values()), (
        "Unfilled cross-fitting predictions"
    )
    return {"phi": pseudo_outcome(method, A, Y, **predictions), **predictions}


def omega_analytic(phi: np.ndarray) -> float:
    phi = np.asarray(phi, dtype=np.float64)
    assert phi.ndim == 1 and len(phi) >= 2 and np.isfinite(phi).all()
    variance = float(phi.var(ddof=1))
    assert np.isfinite(variance) and variance > 0, (
        "Pseudo-outcome variance must be finite and positive"
    )
    return 1 / variance


def omega_bootstrap_fixed_phi(phi: np.ndarray, *, draws: int, seed: int) -> float:
    """1 / (n Var_boot(mean(phi*))); nuisance fits and scores remain fixed."""
    phi = np.asarray(phi, dtype=np.float64)
    assert phi.ndim == 1 and len(phi) >= 2 and np.isfinite(phi).all() and draws >= 2
    rng = np.random.default_rng(seed)
    means = np.array([phi[rng.integers(0, len(phi), len(phi))].mean() for _ in range(draws)])
    variance = float(means.var(ddof=1))
    assert variance > 0 and np.isfinite(variance), "Invalid fixed-score bootstrap variance"
    return 1 / (len(phi) * variance)


def omega_bootstrap_refit(
    X,
    A,
    Y,
    *,
    draws: int,
    seed: int,
    learner: str = "ridge",
    method: str = "aipw",
    folds: int = 5,
    clip_eps: float = 0.01,
    lam_prop: float = 1.0,
    lam_outcome: float = 1.0,
    mlp_max_iter: int = 1000,
) -> float:
    """Refitted-score target: 1 / mean_boot(Var(phi*_refitted, ddof=0)).

    Each resampled dataset gets new nuisance fits and cross-fitted scores.
    This is NOT the variance of bootstrap means or coverage-matching calibration.
    The population-variance convention is retained as part of this definition.
    """
    X, A, Y = _data(X, A, Y)
    assert draws >= 2
    rng = np.random.default_rng(seed)
    variances = []
    for draw in range(draws):
        index = rng.integers(0, len(X), len(X))
        fit = cross_fit_phi(
            X[index],
            A[index],
            Y[index],
            learner=learner,
            method=method,
            folds=folds,
            split_seed=stable_seed(seed, draw, "split"),
            nuisance_seed=stable_seed(seed, draw, "nuisance"),
            clip_eps=clip_eps,
            lam_prop=lam_prop,
            lam_outcome=lam_outcome,
            mlp_max_iter=mlp_max_iter,
        )
        variances.append(fit["phi"].var(ddof=0))
    variance = float(np.mean(variances))
    assert variance > 0 and np.isfinite(variance), "Invalid refitted bootstrap variance"
    return 1 / variance


def propensity_diagnostics(A, e_hat, *, clip_eps: float = 0.01) -> dict[str, float]:
    A, e = np.asarray(A), np.asarray(e_hat, dtype=np.float64)
    assert A.shape == e.shape and np.isin(A, (0, 1)).all() and np.isfinite(e).all()
    assert ((e > 0) & (e < 1)).all()
    weights = np.where(A == 1, 1 / e, 1 / (1 - e))
    return {
        "propensity_clip_rate": float(np.mean((e <= clip_eps) | (e >= 1 - clip_eps))),
        "propensity_min": float(e.min()),
        "propensity_max": float(e.max()),
        "effective_weight_max": float(weights.max()),
        "effective_sample_size": float(weights.sum() ** 2 / np.square(weights).sum()),
    }
