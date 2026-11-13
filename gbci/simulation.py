"""Synthetic back-door settings D1-D9 and reproducible experiment seeds."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Optional

import numpy as np

Array = np.ndarray
DGPResult = dict[str, Any]


def _sigmoid(x: Array) -> Array:
    return 1 / (1 + np.exp(-x))


def simulate_dgp_1(
    n: int,
    seed: Optional[int] = None,
    *,
    beta: Optional[Array] = None,
    tau: float = 1.0,
    gamma: Optional[Array] = None,
) -> DGPResult:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 2))
    beta = np.array([0.5, -0.5]) if beta is None else np.array(beta)
    gamma = np.array([1.0, 1.0]) if gamma is None else np.array(gamma)
    e = _sigmoid(X @ beta)
    A = rng.binomial(1, e)
    Y = tau * A + X @ gamma + rng.standard_normal(n)

    def cate_true(x: Array) -> Array:
        return np.full((x.shape[0],), tau)

    return dict(X=X, A=A, Y=Y, ate_true=tau, cate_true=cate_true)


def simulate_dgp_2(
    n: int,
    seed: Optional[int] = None,
    *,
    beta: Optional[Array] = None,
    theta0: float = 1.0,
    theta: Optional[Array] = None,
    gamma: Optional[Array] = None,
) -> DGPResult:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 2))
    beta = np.array([0.5, -0.5]) if beta is None else np.array(beta)
    theta = np.array([0.5, -0.3]) if theta is None else np.array(theta)
    gamma = np.array([1.0, 1.0]) if gamma is None else np.array(gamma)
    e = _sigmoid(X @ beta)
    A = rng.binomial(1, e)
    Y = (theta0 + X @ theta) * A + X @ gamma + rng.standard_normal(n)
    ate_true = theta0

    def cate_true(x: Array) -> Array:
        return theta0 + x @ theta

    return dict(X=X, A=A, Y=Y, ate_true=ate_true, cate_true=cate_true)


def simulate_dgp_3(
    n: int,
    seed: Optional[int] = None,
    *,
    mu: Optional[Array] = None,
    beta: Optional[Array] = None,
    theta0: float = 1.0,
    theta: Optional[Array] = None,
    gamma: Optional[Array] = None,
) -> DGPResult:
    rng = np.random.default_rng(seed)
    mu = np.array([0.5, 0.0]) if mu is None else np.array(mu)
    X = rng.standard_normal((n, 2)) + mu
    beta = np.array([0.5, -0.5]) if beta is None else np.array(beta)
    theta = np.array([0.5, -0.3]) if theta is None else np.array(theta)
    gamma = np.array([1.0, 1.0]) if gamma is None else np.array(gamma)
    e = _sigmoid(X @ beta)
    A = rng.binomial(1, e)
    Y = (theta0 + X @ theta) * A + X @ gamma + rng.standard_normal(n)
    ate_true = theta0 + np.dot(theta, mu)

    def cate_true(x: Array) -> Array:
        return theta0 + x @ theta

    return dict(X=X, A=A, Y=Y, ate_true=ate_true, cate_true=cate_true)


def simulate_dgp_4(
    n: int,
    seed: Optional[int] = None,
    *,
    beta: Optional[Array] = None,
    alpha0: float = 1.0,
    alpha1: float = 1.0,
) -> DGPResult:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 2))
    beta = np.array([0.5, -0.5]) if beta is None else np.array(beta)
    e = _sigmoid(X @ beta)
    A = rng.binomial(1, e)
    f = X[:, 0] ** 2 + np.sin(X[:, 1])
    Y = (alpha0 + alpha1 * X[:, 0] * X[:, 1]) * A + f + rng.standard_normal(n)
    ate_true = alpha0

    def cate_true(x: Array) -> Array:
        return alpha0 + alpha1 * x[:, 0] * x[:, 1]

    return dict(X=X, A=A, Y=Y, ate_true=ate_true, cate_true=cate_true)


def simulate_dgp_5(
    n: int, seed: Optional[int] = None, *, beta: Optional[Dict[str, float]] = None, tau: float = 1.0
) -> DGPResult:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 2))
    if beta is None:
        beta = dict(b0=0.0, b1=0.5, b2=0.5, b3=0.5)
    z = beta["b0"] + beta["b1"] * X[:, 0] + beta["b2"] * X[:, 0] ** 2 + beta["b3"] * np.sin(X[:, 1])
    e = _sigmoid(z)
    A = rng.binomial(1, e)
    h = X[:, 0] + 0.5 * X[:, 0] ** 2 + 0.5 * np.sin(X[:, 1])
    Y = tau * A + h + rng.standard_normal(n)

    def cate_true(x: Array) -> Array:
        return np.full((x.shape[0],), tau)

    return dict(X=X, A=A, Y=Y, ate_true=tau, cate_true=cate_true)


def simulate_dgp_6(
    n: int, seed: Optional[int] = None, *, tau: float = 1.0, gamma: Optional[Array] = None
) -> DGPResult:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 2))
    e = _sigmoid(3.5 + 3 * X[:, 0])
    A = rng.binomial(1, e)
    gamma = np.array([1.0, 1.0]) if gamma is None else np.array(gamma)
    Y = tau * A + X @ gamma + rng.standard_normal(n)

    def cate_true(x: Array) -> Array:
        return np.full((x.shape[0],), tau)

    return dict(X=X, A=A, Y=Y, ate_true=tau, cate_true=cate_true)


def simulate_dgp_7(
    n: int,
    seed: Optional[int] = None,
    *,
    tau: float = 1.0,
    beta: Optional[Array] = None,
    gamma: Optional[Array] = None,
    nu: float = 3,
) -> DGPResult:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 2))
    beta = np.array([0.5, -0.5]) if beta is None else np.array(beta)
    gamma = np.array([1.0, 1.0]) if gamma is None else np.array(gamma)
    e = _sigmoid(X @ beta)
    A = rng.binomial(1, e)
    sigma = np.exp(0.5 * X[:, 0])
    eta = rng.standard_t(df=nu, size=n)
    Y = tau * A + X @ gamma + sigma * eta

    def cate_true(x: Array) -> Array:
        return np.full((x.shape[0],), tau)

    return dict(X=X, A=A, Y=Y, ate_true=tau, cate_true=cate_true)


def simulate_dgp_8(
    n: int, seed: Optional[int] = None, *, p: int = 20, s: int = 5, tau: float = 1.0
) -> DGPResult:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    beta = np.zeros(p)
    gamma = np.zeros(p)
    beta[:s] = np.linspace(0.2, 1.0, s)
    gamma[:s] = np.linspace(1.0, 0.2, s)
    e = _sigmoid(X @ beta)
    A = rng.binomial(1, e)
    Y = tau * A + X @ gamma + rng.standard_normal(n)

    def cate_true(x: Array) -> Array:
        return np.full((x.shape[0],), tau)

    return dict(X=X, A=A, Y=Y, ate_true=tau, cate_true=cate_true)


def simulate_dgp_9(n: int, seed: Optional[int] = None) -> DGPResult:
    rng = np.random.default_rng(seed)
    X = rng.random((n, 5))
    e = _sigmoid(-0.5 + X[:, 0] - 0.25 * X[:, 1] + 0.25 * X[:, 2])
    A = rng.binomial(1, e)
    mu = (
        10 * np.sin(np.pi * X[:, 0] * X[:, 1])
        + 20 * (X[:, 2] - 0.5) ** 2
        + 10 * X[:, 3]
        + 5 * X[:, 4]
    )
    tau_x = 1 + X[:, 0] / (X[:, 1] + 0.1)
    Y = mu + tau_x * A + rng.standard_normal(n)
    ate_true = 1 + 0.5 * np.log(11)

    def cate_true(x: Array) -> Array:
        return 1 + x[:, 0] / (x[:, 1] + 0.1)

    return dict(X=X, A=A, Y=Y, ate_true=ate_true, cate_true=cate_true)


_DGPS = {
    1: simulate_dgp_1,
    2: simulate_dgp_2,
    3: simulate_dgp_3,
    4: simulate_dgp_4,
    5: simulate_dgp_5,
    6: simulate_dgp_6,
    7: simulate_dgp_7,
    8: simulate_dgp_8,
    9: simulate_dgp_9,
}


def simulate(dgp: int, n: int, seed: int) -> DGPResult:
    """Draw one dataset, with an ATE truth and a callable CATE truth."""
    assert dgp in _DGPS, f"Unknown synthetic setting: {dgp}"
    assert n >= 2, "At least two observations are required"
    data = _DGPS[dgp](n=n, seed=seed)
    assert data["X"].shape[0] == len(data["A"]) == len(data["Y"]) == n
    assert all(np.isfinite(data[key]).all() for key in ("X", "A", "Y"))
    assert np.isfinite(data["ate_true"])
    return data


def stable_seed(seed: int, *parts: Any) -> int:
    """Independent named random streams for data, folds, learners and inference."""
    payload = json.dumps([seed, *parts], separators=(",", ":"), default=str)
    return int(hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8], 16)
