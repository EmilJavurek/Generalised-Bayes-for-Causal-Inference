"""
Current file: src.py
"""

# ----------------------------------------------
# imports

import csv
import hashlib
import json
import os
import sys
import time
import shutil
import itertools
import traceback
import numpy as np
import matplotlib.pyplot as plt
import torch
import sklearn
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression, Ridge
from tqdm import tqdm
from scipy.stats import binomtest

# ----------------------------------------------
# DGPs

## Backdoor-adjustment (X,A,Y)
from typing import Any, Callable, Dict, List, Optional, Iterable

Array = np.ndarray
CATEFn = Callable[[Array], Array]
DGPResult = Dict[str, Any]




'''
SIGNATURE:
def simulate_dgp_N(
    n: int,
    seed: Optional[int] = None,
    **params: Any,
) -> DGPResult:
    """
    Simulate dataset for DGP N.
    Returns a dict with keys:
        X         : (n, d) covariate matrix array
        A         : (n,) treatment array
        Y         : (n,) outcome array
        ate_true  : float ground-truth ATE
        cate_true : CATEFn, maps X: (n, d) -> (n,)
    """
'''

#NOTE: unconfounded is making a mess with RA estimator since then pseudo-outcomes are constant without variance -> finite omega fails. 
# === DGP 0: unconfounded, homoskedastic ===============================
# def simulate_dgp_0(n: int, seed: Optional[int] = None, *, tau: float = 2.0, mu: float = 0.0, sigma: float = 1.0, p: float = 0.5) -> DGPResult:
#     """
#     Simulate data: Y = mu + tau*A + eps, with A~Bernoulli(p), eps~N(0,sigma^2).
#     Returns X (empty features), A, Y, ate_true, and cate_true(X).
#     """
#     rng = np.random.default_rng(seed)
#     X = np.zeros((n, 0))
#     A = rng.binomial(1, p, size=n)
#     eps = rng.normal(0, sigma, size=n)
#     Y = mu + tau * A + eps
#     def cate_true(x: Array) -> Array:
#         return np.full((x.shape[0],), tau)
#     return dict(X=X, A=A, Y=Y, ate_true=tau, cate_true=cate_true)

def _sigmoid(x: Array) -> Array:
    return 1 / (1 + np.exp(-x))


def _clip_with_sign(x: Array, min_abs: float) -> Array:
    """
    Clip values away from zero while preserving sign.
    """
    sign = np.sign(x)
    sign[sign == 0] = 1.0
    return sign * np.maximum(np.abs(x), min_abs)

# === DGP 1: Linear, confounded, homoskedastic ===============================
def simulate_dgp_1(n: int, seed: Optional[int] = None, *, beta: Optional[Array] = None, tau: float = 1.0, gamma: Optional[Array] = None) -> DGPResult:
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


# === DGP 2: Linear heterogeneity (mean-zero X) ==============================
def simulate_dgp_2(n: int, seed: Optional[int] = None, *, beta: Optional[Array] = None, theta0: float = 1.0, theta: Optional[Array] = None, gamma: Optional[Array] = None) -> DGPResult:
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


# === DGP 3: Linear heterogeneity, nonzero mean ==============================
def simulate_dgp_3(n: int, seed: Optional[int] = None, *, mu: Optional[Array] = None, beta: Optional[Array] = None, theta0: float = 1.0, theta: Optional[Array] = None, gamma: Optional[Array] = None) -> DGPResult:
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


# === DGP 4: Nonlinear outcome (quadratic + interaction) =====================
def simulate_dgp_4(n: int, seed: Optional[int] = None, *, beta: Optional[Array] = None, alpha0: float = 1.0, alpha1: float = 1.0) -> DGPResult:
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


# === DGP 5: Nonlinear propensity ============================================
def simulate_dgp_5(n: int, seed: Optional[int] = None, *, beta: Optional[Dict[str, float]] = None, tau: float = 1.0) -> DGPResult:
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


# === DGP 6: Limited overlap ================================================
def simulate_dgp_6(n: int, seed: Optional[int] = None, *, tau: float = 1.0, gamma: Optional[Array] = None) -> DGPResult:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 2))
    e = _sigmoid(3.5 + 3 * X[:, 0])
    A = rng.binomial(1, e)
    gamma = np.array([1.0, 1.0]) if gamma is None else np.array(gamma)
    Y = tau * A + X @ gamma + rng.standard_normal(n)

    def cate_true(x: Array) -> Array:
        return np.full((x.shape[0],), tau)

    return dict(X=X, A=A, Y=Y, ate_true=tau, cate_true=cate_true)


# === DGP 7: Heteroskedastic heavy-tailed noise =============================
def simulate_dgp_7(n: int, seed: Optional[int] = None, *, tau: float = 1.0, beta: Optional[Array] = None, gamma: Optional[Array] = None, nu: float = 3) -> DGPResult:
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


# === DGP 8: High-dimensional sparse confounding ============================
def simulate_dgp_8(n: int, seed: Optional[int] = None, *, p: int = 20, s: int = 5, tau: float = 1.0) -> DGPResult:
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


# === DGP 9: Friedman-style nonlinear CATE ==================================
def simulate_dgp_9(n: int, seed: Optional[int] = None) -> DGPResult:
    rng = np.random.default_rng(seed)
    X = rng.random((n, 5))
    e = _sigmoid(-0.5 + X[:, 0] - 0.25 * X[:, 1] + 0.25 * X[:, 2])
    A = rng.binomial(1, e)
    mu = 10 * np.sin(np.pi * X[:, 0] * X[:, 1]) + 20 * (X[:, 2] - 0.5) ** 2 + 10 * X[:, 3] + 5 * X[:, 4]
    tau_x = 1 + X[:, 0] / (X[:, 1] + 0.1)
    Y = mu + tau_x * A + rng.standard_normal(n)
    ate_true = 1 + 0.5 * np.log(11)

    def cate_true(x: Array) -> Array:
        return 1 + x[:, 0] / (x[:, 1] + 0.1)

    return dict(X=X, A=A, Y=Y, ate_true=ate_true, cate_true=cate_true)


## Instrumental variables DGPs


# === IV-DGP 0: constant effect, randomized Z, strong monotone first stage ===
def simulate_dgp_iv_0(n, seed=None, d=2,
                      pi=0.5,
                      alpha0=0.0, alphaX=(0.1, -0.1), alphaZ=4.0,
                      gamma0=0.0, gammaX=(1.0, 1.0),
                      tau=2.0, sigma=0.2):
    """
    Binary IV DGP with constant treatment effect.
    Z randomized, A generated via threshold-logit with shared U => monotonicity if alphaZ>0.
    Y = gamma0 + gammaX^T X + tau*A + eps, eps ~ N(0, sigma^2).
    True IV-ATE = tau.
    """
    if seed is not None:
        np.random.seed(seed)

    X = np.random.randn(n, d)
    Z = np.random.binomial(1, pi, size=n)

    alphaX = np.asarray(alphaX, dtype=float)
    gammaX = np.asarray(gammaX, dtype=float)

    U = np.random.rand(n)  # shared latent for threshold model
    p = _sigmoid(alpha0 + X @ alphaX + alphaZ * Z)
    A = (U < p).astype(int)

    eps = np.random.normal(0, sigma, size=n)
    Y = gamma0 + X @ gammaX + tau * A + eps

    def cate_true(Xnew):
        return np.full(Xnew.shape[0], tau, dtype=float)

    return dict(X=X, Z=Z, A=A, Y=Y, ate_true=float(tau), cate_true=cate_true)



# === IV-DGP 1: heterogeneous effect, nonlinear baseline outcome, pi(x) ===
def simulate_dgp_iv_1(n, seed=None,
                      mu=(0.5, -0.2),
                      alpha0=-0.3, alphaX=(0.4, -0.3), alphaZ=2.0,
                      delta0=0.0, deltaX=(0.2, -0.2),
                      theta0=1.0, theta=(0.6, -0.3),
                      sigma=0.2):
    """
    Binary IV DGP with effect heterogeneity and X-dependent instrument assignment.
    X ~ N(mu, I_2)
    Z ~ Bernoulli(pi(X)), pi(X)=sigmoid(delta0 + deltaX^T X)
    A via threshold-logit with shared U => monotonicity if alphaZ>0
    Y = g(X) + tau(X)*A + eps, eps ~ N(0, sigma^2)
    g(X)=X1^2 + sin(X2), tau(X)=theta0 + theta^T X
    True IV-ATE = E[tau(X)] = theta0 + theta^T mu
    """
    if seed is not None:
        np.random.seed(seed)

    mu = np.asarray(mu, dtype=float)
    X = np.random.randn(n, 2) + mu

    deltaX = np.asarray(deltaX, dtype=float)
    piX = _sigmoid(delta0 + X @ deltaX)
    Z = np.random.binomial(1, piX, size=n)

    alphaX = np.asarray(alphaX, dtype=float)
    U = np.random.rand(n)
    p = _sigmoid(alpha0 + X @ alphaX + alphaZ * Z)
    A = (U < p).astype(int)

    # nonlinear baseline + heterogeneous effect
    g = X[:, 0] ** 2 + np.sin(X[:, 1])
    theta = np.asarray(theta, dtype=float)
    tauX = theta0 + X @ theta

    eps = np.random.normal(0, sigma, size=n)
    Y = g + tauX * A + eps

    ate_true = float(theta0 + theta @ mu)

    def cate_true(Xnew):
        Xnew = np.asarray(Xnew)
        return theta0 + Xnew @ theta

    return dict(X=X, Z=Z, A=A, Y=Y, ate_true=ate_true, cate_true=cate_true)



# === IV-DGP 2: weak/variable first stage + limited instrument overlap + heavy tails ===
def simulate_dgp_iv_2(n, seed=None, p=10,
                      # instrument assignment: limited overlap in Z
                      c0=0.3, c1=0.3,
                      # weak instrument in first stage
                      alpha0=-0.2, alphaZ=1.2, alphaX_scale=0.3,
                      # outcome
                      tau=1.0, nu=3):
    """
    Hard IV DGP:
    X ~ N(0, I_p)
    Z ~ Bernoulli(pi(X)) with pi(X)=sigmoid(c0 + c1*X1)  (often near 0/1 => limited overlap)
    A via threshold-logit with small alphaZ (weak first stage), shared U => monotonicity if alphaZ>0
    Y = g(X) + tau*A + eps, eps = exp(0.3*X1) * t_nu  (heteroskedastic heavy tails)
    True IV-ATE = tau.
    """
    if seed is not None:
        np.random.seed(seed)

    X = np.random.randn(n, p)

    piX = _sigmoid(c0 + c1 * X[:, 0])
    Z = np.random.binomial(1, piX, size=n)

    # sparse-ish alphaX: only first few covariates matter
    alphaX = np.zeros(p, dtype=float)
    alphaX[:3] = alphaX_scale * np.array([0.8, -0.6, 0.4])

    U = np.random.rand(n)
    pA = _sigmoid(alpha0 + X @ alphaX + alphaZ * Z)
    A = (U < pA).astype(int)

    g = 0.5 * X[:, 0] + 0.25 * (X[:, 1] ** 2) - 0.25 * np.sin(X[:, 2])
    sigmaX = np.exp(0.3 * X[:, 0])
    eps = sigmaX * np.random.standard_t(df=nu, size=n)

    Y = g + tau * A + eps

    def cate_true(Xnew):
        return np.full(Xnew.shape[0], tau, dtype=float)

    return dict(X=X, Z=Z, A=A, Y=Y, ate_true=float(tau), cate_true=cate_true)





# ----------------------------------------------
# Nuisance estimation

## Individual models 

## Helper: simple feature map (polynomial + sine)
def build_features(X: Array) -> Array:
    """
    Build a modest feature map Phi(X): [X, X^2, sin(X), 1].
    Args:
        X: (n, d) covariate matrix
    Returns:
        Phi(X): (n, d') feature matrix with intercept
    """
    X2 = X ** 2
    Xs = np.sin(X)
    ones = np.ones((X.shape[0], 1))
    if X.shape[1] >= 2:
        x_extra = (X[:, 0] * X[:, 1])[:, None]
    elif X.shape[1] == 1:
        x_extra = (X[:, 0] ** 3)[:, None]
    else:
        x_extra = np.zeros((X.shape[0], 1))
    Phi = np.concatenate([X, X2, Xs, x_extra, ones], axis=1)
    return Phi



## Composite wrapper

## Unified nuisance estimator (prediction functions)
def fit_nuisance_models(
    X: Array,
    A: Array,
    Y: Array,
    lam_prop: float = 1.0,
    lam_outcome: float = 1.0,
    clip_eps: float = 1e-2,
    seed: Optional[int] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Fit propensity and outcome models and return a predictor for new X.
    Returns a dict with a predict(X_new) callable and fitted weights.
    """
    Phi = build_features(X)

    # Propensity
    c_val = 1.0 / lam_prop if lam_prop > 0 else 1e12
    logit = LogisticRegression(
        C=c_val,
        fit_intercept=False,
        solver="lbfgs",
        max_iter=500,
        verbose=1 if verbose else 0,
        # l1_ratio=0.0,
    )
    logit.fit(Phi, A)
    w_e = logit.coef_.ravel()

    # Outcome regressions (T-learner)
    Phi0 = Phi[A == 0]
    Phi1 = Phi[A == 1]
    Y0 = Y[A == 0]
    Y1 = Y[A == 1]

    ridge0 = Ridge(alpha=lam_outcome, fit_intercept=False)
    ridge1 = Ridge(alpha=lam_outcome, fit_intercept=False)
    ridge0.fit(Phi0, Y0)
    ridge1.fit(Phi1, Y1)
    w0 = ridge0.coef_
    w1 = ridge1.coef_

    def predict(X_new: Array) -> Dict[str, Array]:
        Phi_new = build_features(X_new)
        e_hat = _sigmoid(Phi_new @ w_e)
        e_hat = np.clip(e_hat, clip_eps, 1 - clip_eps)
        mu0_hat = Phi_new @ w0
        mu1_hat = Phi_new @ w1
        return dict(e_hat=e_hat, mu0_hat=mu0_hat, mu1_hat=mu1_hat)

    return dict(predict=predict, w_e=w_e, w0=w0, w1=w1, clip_eps=clip_eps)


def fit_nuisance_models_noisy(
    X: Array,
    A: Array,
    Y: Array,
    *,
    base_nuisance_fn_REGISTRATION: str = "default",
    base_nuisance_params: Optional[Dict[str, Any]] = None,
    noise: Optional[Dict[str, float]] = None,
    seed: Optional[int] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Fit base nuisances and inject Gaussian noise into predictions.
    """
    base_nuisance_params = dict(base_nuisance_params or {})
    noise = noise or {}
    base_fn = _resolve_from_registry(base_nuisance_fn_REGISTRATION, _NUISANCE_REGISTRY, "base_nuisance_fn_REGISTRATION")
    if "verbose" not in base_nuisance_params:
        base_nuisance_params["verbose"] = verbose
    if "seed" not in base_nuisance_params:
        base_nuisance_params["seed"] = seed
    base_models = base_fn(X, A, Y, **base_nuisance_params)
    rng = np.random.default_rng(seed)

    mu_0_mean = float(noise.get("mu0_mean", 0.5))
    mu_1_mean = float(noise.get("mu1_mean", 0.0))
    e_mean = float(noise.get("e_mean", 0.0))

    mu0_sd = float(noise.get("mu0_sd", 0.5))
    mu1_sd = float(noise.get("mu1_sd", 0.5))
    e_sd = float(noise.get("e_sd", 0.2))
    clip_eps = float(noise.get("clip_eps", base_models.get("clip_eps", 1e-2)))

    def predict(X_new: Array) -> Dict[str, Array]:
        nuisances = base_models["predict"](X_new)
        e_hat = nuisances["e_hat"] + rng.normal(e_mean, e_sd, size=nuisances["e_hat"].shape)
        mu0_hat = nuisances["mu0_hat"] + rng.normal(mu_0_mean, mu0_sd, size=nuisances["mu0_hat"].shape)
        mu1_hat = nuisances["mu1_hat"] + rng.normal(mu_1_mean, mu1_sd, size=nuisances["mu1_hat"].shape)
        e_hat = np.clip(e_hat, clip_eps, 1 - clip_eps)
        return dict(e_hat=e_hat, mu0_hat=mu0_hat, mu1_hat=mu1_hat)

    return dict(
        predict=predict,
        w_e=base_models.get("w_e"),
        w0=base_models.get("w0"),
        w1=base_models.get("w1"),
        clip_eps=base_models.get("clip_eps", clip_eps),
    )


def fit_nuisance_models_iv(
    X: Array,
    Z: Array,
    A: Array,
    Y: Array,
    lam_pi: float = 1.0,
    lam_y: float = 1.0,
    lam_a: float = 1.0,
    clip_eps: float = 1e-2,
    seed: Optional[int] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Fit IV nuisances: pi(x), m_Y(z,x), m_A(z,x).
    """
    Phi = build_features(X)

    # Instrument propensity pi(x)
    c_val = 1.0 / lam_pi if lam_pi > 0 else 1e12
    logit_z = LogisticRegression(
        C=c_val,
        fit_intercept=False,
        solver="lbfgs",
        max_iter=500,
        verbose=1 if verbose else 0,
    )
    logit_z.fit(Phi, Z)
    w_pi = logit_z.coef_.ravel()

    # Outcome regressions by instrument
    Phi0 = Phi[Z == 0]
    Phi1 = Phi[Z == 1]
    Y0 = Y[Z == 0]
    Y1 = Y[Z == 1]
    A0 = A[Z == 0]
    A1 = A[Z == 1]

    ridge0 = Ridge(alpha=lam_y, fit_intercept=False)
    ridge1 = Ridge(alpha=lam_y, fit_intercept=False)
    ridge0.fit(Phi0, Y0)
    ridge1.fit(Phi1, Y1)
    w_y0 = ridge0.coef_
    w_y1 = ridge1.coef_

    c_val_a = 1.0 / lam_a if lam_a > 0 else 1e12
    logit_a0 = LogisticRegression(
        C=c_val_a,
        fit_intercept=False,
        solver="lbfgs",
        max_iter=500,
        verbose=1 if verbose else 0,
    )
    logit_a1 = LogisticRegression(
        C=c_val_a,
        fit_intercept=False,
        solver="lbfgs",
        max_iter=500,
        verbose=1 if verbose else 0,
    )
    logit_a0.fit(Phi0, A0)
    logit_a1.fit(Phi1, A1)
    w_a0 = logit_a0.coef_.ravel()
    w_a1 = logit_a1.coef_.ravel()

    def predict(X_new: Array) -> Dict[str, Array]:
        Phi_new = build_features(X_new)
        pi_hat = _sigmoid(Phi_new @ w_pi)
        pi_hat = np.clip(pi_hat, clip_eps, 1 - clip_eps)
        m_y0_hat = Phi_new @ w_y0
        m_y1_hat = Phi_new @ w_y1
        m_a0_hat = _sigmoid(Phi_new @ w_a0)
        m_a1_hat = _sigmoid(Phi_new @ w_a1)
        return dict(
            pi_hat=pi_hat,
            m_y0_hat=m_y0_hat,
            m_y1_hat=m_y1_hat,
            m_a0_hat=m_a0_hat,
            m_a1_hat=m_a1_hat,
        )

    return dict(
        predict=predict,
        w_pi=w_pi,
        w_y0=w_y0,
        w_y1=w_y1,
        w_a0=w_a0,
        w_a1=w_a1,
        clip_eps=clip_eps,
    )


def fit_nuisance_models_iv_fancy(
    X: Array,
    Z: Array,
    A: Array,
    Y: Array,
    lam_pi: float = 1.0,
    lam_y: float = 1.0,
    lam_a: float = 1.0,
    clip_eps: float = 1e-2,
    seed: Optional[int] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Fancy IV nuisances using nonlinear models with Z interactions.
    Interface-compatible with fit_nuisance_models_iv.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

    X = np.asarray(X)
    Z = np.asarray(Z)
    A = np.asarray(A)
    Y = np.asarray(Y)
    n = X.shape[0]

    Phi = build_features(X)
    Z_col = Z.reshape(-1, 1)
    Phi_z = np.concatenate([Phi, Z_col, Phi * Z_col], axis=1)

    pi_const = None
    pi_model = None
    if np.unique(Z).size < 2:
        pi_const = float(np.mean(Z))
    else:
        pi_model = HistGradientBoostingClassifier(
            max_depth=4,
            learning_rate=0.05,
            max_iter=300,
            min_samples_leaf=max(20, int(0.005 * n)),
            l2_regularization=0.0,
            early_stopping=True,
            random_state=seed,
        )
        pi_model.fit(Phi, Z)

    y_model = HistGradientBoostingRegressor(
        max_depth=4,
        learning_rate=0.05,
        max_iter=400,
        min_samples_leaf=max(20, int(0.005 * n)),
        l2_regularization=0.0,
        early_stopping=True,
        random_state=seed,
    )
    y_model.fit(Phi_z, Y)

    a_const = None
    a_model = None
    if np.unique(A).size < 2:
        a_const = float(np.mean(A))
    else:
        a_model = HistGradientBoostingClassifier(
            max_depth=4,
            learning_rate=0.05,
            max_iter=300,
            min_samples_leaf=max(20, int(0.005 * n)),
            l2_regularization=0.0,
            early_stopping=True,
            random_state=seed,
        )
        a_model.fit(Phi_z, A)

    def _predict_y(X_new: Array, z_value: int) -> Array:
        Phi_new = build_features(X_new)
        z_col = np.full((Phi_new.shape[0], 1), z_value, dtype=Phi_new.dtype)
        Phi_new_z = np.concatenate([Phi_new, z_col, Phi_new * z_col], axis=1)
        return y_model.predict(Phi_new_z)

    def _predict_a(X_new: Array, z_value: int) -> Array:
        if a_model is None:
            return np.full(X_new.shape[0], a_const)
        Phi_new = build_features(X_new)
        z_col = np.full((Phi_new.shape[0], 1), z_value, dtype=Phi_new.dtype)
        Phi_new_z = np.concatenate([Phi_new, z_col, Phi_new * z_col], axis=1)
        return a_model.predict_proba(Phi_new_z)[:, 1]

    def predict(X_new: Array) -> Dict[str, Array]:
        X_new = np.asarray(X_new)
        Phi_new = build_features(X_new)
        if pi_model is None:
            pi_hat = np.full(X_new.shape[0], pi_const)
        else:
            pi_hat = pi_model.predict_proba(Phi_new)[:, 1]
        pi_hat = np.clip(pi_hat, clip_eps, 1 - clip_eps)

        m_y0_hat = _predict_y(X_new, 0)
        m_y1_hat = _predict_y(X_new, 1)
        m_a0_hat = _predict_a(X_new, 0)
        m_a1_hat = _predict_a(X_new, 1)
        m_a0_hat = np.clip(m_a0_hat, clip_eps, 1 - clip_eps)
        m_a1_hat = np.clip(m_a1_hat, clip_eps, 1 - clip_eps)
        def _sanitize(arr: Array, name: str) -> Array:
            arr = np.asarray(arr, dtype=float)
            if not np.all(np.isfinite(arr)):
                finite = np.isfinite(arr)
                fallback = float(np.nanmedian(arr[finite])) if np.any(finite) else 0.0
                arr = np.where(finite, arr, fallback)
                if verbose:
                    print(f"fit_nuisance_models_iv_fancy: sanitized non-finite values in {name}")
            return arr
        pi_hat = _sanitize(pi_hat, "pi_hat")
        m_y0_hat = _sanitize(m_y0_hat, "m_y0_hat")
        m_y1_hat = _sanitize(m_y1_hat, "m_y1_hat")
        m_a0_hat = _sanitize(m_a0_hat, "m_a0_hat")
        m_a1_hat = _sanitize(m_a1_hat, "m_a1_hat")
        return dict(
            pi_hat=pi_hat,
            m_y0_hat=m_y0_hat,
            m_y1_hat=m_y1_hat,
            m_a0_hat=m_a0_hat,
            m_a1_hat=m_a1_hat,
        )

    if verbose:
        print("fit_nuisance_models_iv_fancy: fitted HGB models with Z interactions")

    return dict(
        predict=predict,
        w_pi=None,
        w_y0=None,
        w_y1=None,
        w_a0=None,
        w_a1=None,
        clip_eps=clip_eps,
    )


def fit_nuisance_models_iv_noisy(
    X: Array,
    Z: Array,
    A: Array,
    Y: Array,
    *,
    base_nuisance_fn_REGISTRATION: str = "default_iv",
    base_nuisance_params: Optional[Dict[str, Any]] = None,
    noise: Optional[Dict[str, float]] = None,
    seed: Optional[int] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Fit IV nuisances and inject Gaussian noise into predictions.
    """
    base_nuisance_params = dict(base_nuisance_params or {})
    noise = noise or {}
    base_fn = _resolve_from_registry(base_nuisance_fn_REGISTRATION, _NUISANCE_IV_REGISTRY, "base_nuisance_fn_REGISTRATION")
    if "verbose" not in base_nuisance_params:
        base_nuisance_params["verbose"] = verbose
    if "seed" not in base_nuisance_params:
        base_nuisance_params["seed"] = seed
    base_models = base_fn(X, Z, A, Y, **base_nuisance_params)
    rng = np.random.default_rng(seed)

    pi_mean = float(noise.get("pi_mean", 0.0))
    my0_mean = float(noise.get("m_y0_mean", 0.0))
    my1_mean = float(noise.get("m_y1_mean", 0.0))
    ma0_mean = float(noise.get("m_a0_mean", 0.0))
    ma1_mean = float(noise.get("m_a1_mean", 0.0))

    pi_sd = float(noise.get("pi_sd", 0.05))
    my0_sd = float(noise.get("m_y0_sd", 0.5))
    my1_sd = float(noise.get("m_y1_sd", 0.5))
    ma0_sd = float(noise.get("m_a0_sd", 0.1))
    ma1_sd = float(noise.get("m_a1_sd", 0.1))
    clip_eps = float(noise.get("clip_eps", base_models.get("clip_eps", 1e-2)))

    def predict(X_new: Array) -> Dict[str, Array]:
        nuisances = base_models["predict"](X_new)
        pi_hat = nuisances["pi_hat"] + rng.normal(pi_mean, pi_sd, size=nuisances["pi_hat"].shape)
        m_y0_hat = nuisances["m_y0_hat"] + rng.normal(my0_mean, my0_sd, size=nuisances["m_y0_hat"].shape)
        m_y1_hat = nuisances["m_y1_hat"] + rng.normal(my1_mean, my1_sd, size=nuisances["m_y1_hat"].shape)
        m_a0_hat = nuisances["m_a0_hat"] + rng.normal(ma0_mean, ma0_sd, size=nuisances["m_a0_hat"].shape)
        m_a1_hat = nuisances["m_a1_hat"] + rng.normal(ma1_mean, ma1_sd, size=nuisances["m_a1_hat"].shape)
        pi_hat = np.clip(pi_hat, clip_eps, 1 - clip_eps)
        return dict(
            pi_hat=pi_hat,
            m_y0_hat=m_y0_hat,
            m_y1_hat=m_y1_hat,
            m_a0_hat=m_a0_hat,
            m_a1_hat=m_a1_hat,
        )

    return dict(
        predict=predict,
        w_pi=base_models.get("w_pi"),
        w_y0=base_models.get("w_y0"),
        w_y1=base_models.get("w_y1"),
        w_a0=base_models.get("w_a0"),
        w_a1=base_models.get("w_a1"),
        clip_eps=base_models.get("clip_eps", clip_eps),
    )


## Fit nuisances on (X, A, Y) and predict on X_eval (default: X)
def estimate_nuisances_pred(
    X: Array,
    A: Array,
    Y: Array,
    X_eval: Optional[Array] = None,
    lam_prop: float = 1.0,
    lam_outcome: float = 1.0,
    clip_eps: float = 1e-2,
    seed: Optional[int] = None,
    verbose: bool = False,
) -> Dict[str, Array]:
    """
    Fit nuisances on (X, A, Y) and predict on X_eval (default: X).
    Useful for cross-fitting by passing X_eval as the held-out fold.
    """
    models = fit_nuisance_models(
        X, A, Y, lam_prop=lam_prop, lam_outcome=lam_outcome, clip_eps=clip_eps, seed=seed, verbose=verbose
    )
    X_eval = X if X_eval is None else X_eval
    return models["predict"](X_eval)


## Checking fit

## Quick sanity diagnostics for fitted nuisance functions
def check_nuisances(
    X: Array,
    A: Array,
    Y: Array,
    e_hat: Array,
    mu0_hat: Array,
    mu1_hat: Array,
    max_points: int = 2000,
    seed: Optional[int] = None,
) -> None:
    """
    Quick sanity diagnostics for fitted nuisance functions.
    """
    n = len(A)
    if n > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, max_points, replace=False)
        X, A, Y, e_hat, mu0_hat, mu1_hat = (
            X[idx],
            A[idx],
            Y[idx],
            e_hat[idx],
            mu0_hat[idx],
            mu1_hat[idx],
        )

    a1 = e_hat[A == 1]
    a0 = e_hat[A == 0]
    if len(a1) > 0 and len(a0) > 0:
        auc_like = np.mean(a1[:, None] > a0[None, :])
    else:
        auc_like = np.nan

    fig, axs = plt.subplots(1, 3, figsize=(12, 3.5))
    bins = np.linspace(0, 1, 25)
    axs[0].hist(e_hat[A == 0], bins=bins, alpha=0.6, label="A=0")
    axs[0].hist(e_hat[A == 1], bins=bins, alpha=0.6, label="A=1")
    axs[0].set_title(f"Propensity histograms (AUC~={auc_like:.2f})")
    axs[0].set_xlabel("e_hat")
    axs[0].legend()

    axs[1].scatter(mu0_hat[A == 0], Y[A == 0], s=8, alpha=0.5, color="C0")
    lo, hi = np.percentile(Y, [1, 99])
    axs[1].plot([lo, hi], [lo, hi], "k--", lw=1)
    axs[1].set_title("mu0_hat vs. Y (A=0)")
    axs[1].set_xlabel("mu0_hat")
    axs[1].set_ylabel("Y")

    axs[2].scatter(mu1_hat[A == 1], Y[A == 1], s=8, alpha=0.5, color="C1")
    axs[2].plot([lo, hi], [lo, hi], "k--", lw=1)
    axs[2].set_title("mu1_hat vs. Y (A=1)")
    axs[2].set_xlabel("mu1_hat")
    axs[2].set_ylabel("Y")

    plt.tight_layout()
    plt.show()

    mse0 = np.mean((Y[A == 0] - mu0_hat[A == 0]) ** 2) if np.any(A == 0) else np.nan
    mse1 = np.mean((Y[A == 1] - mu1_hat[A == 1]) ** 2) if np.any(A == 1) else np.nan
    overlap = np.mean((e_hat > 0.05) & (e_hat < 0.95))
    print(f"MSE(mu0): {mse0:.3f},   MSE(mu1): {mse1:.3f}")
    print(f"Propensity mean: {e_hat.mean():.3f}, std: {e_hat.std():.3f}, AUC~={auc_like:.3f}, overlap: {overlap:.3f}")

# ----------------------------------------------
# Loss functions

## pseudo-outcomes

def compute_phi_aipw(
    X: Array,
    A: Array,
    Y: Array,
    e_hat: Array,
    mu0_hat: Array,
    mu1_hat: Array,
    device: Optional[str] = None,
) -> torch.Tensor:
    """
    Compute the AIPW pseudo-outcomes phi_i^{AIPW}(W_i; eta).
    Returns a 1D torch tensor (n,).
    """
    dev = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    A_t = torch.tensor(A, dtype=torch.float32, device=dev)
    Y_t = torch.tensor(Y, dtype=torch.float32, device=dev)
    e_t = torch.tensor(e_hat, dtype=torch.float32, device=dev)
    m0_t = torch.tensor(mu0_hat, dtype=torch.float32, device=dev)
    m1_t = torch.tensor(mu1_hat, dtype=torch.float32, device=dev)

    phi = A_t * (Y_t - m1_t) / e_t - (1 - A_t) * (Y_t - m0_t) / (1 - e_t) + m1_t - m0_t
    return phi


def compute_phi_ipw(
    X: Array,
    A: Array,
    Y: Array,
    e_hat: Array,
    mu0_hat: Array,
    mu1_hat: Array,
    device: Optional[str] = None,
) -> torch.Tensor:
    """
    Compute the IPW pseudo-outcomes phi_i^{IPW}(W_i; eta).
    Returns a 1D torch tensor (n,).
    Extra inputs are accepted for signature consistency and may be unused.
    """
    dev = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    A_t = torch.tensor(A, dtype=torch.float32, device=dev)
    Y_t = torch.tensor(Y, dtype=torch.float32, device=dev)
    e_t = torch.tensor(e_hat, dtype=torch.float32, device=dev)

    phi = A_t * Y_t / e_t - (1 - A_t) * Y_t / (1 - e_t)
    return phi


def compute_phi_ra(
    X: Array,
    A: Array,
    Y: Array,
    e_hat: Array,
    mu0_hat: Array,
    mu1_hat: Array,
    device: Optional[str] = None,
) -> torch.Tensor:
    """
    Compute the RA pseudo-outcomes phi_i^{RA}(W_i; eta).
    Returns a 1D torch tensor (n,).
    Extra inputs are accepted for signature consistency and may be unused.
    """
    dev = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    m0_t = torch.tensor(mu0_hat, dtype=torch.float32, device=dev)
    m1_t = torch.tensor(mu1_hat, dtype=torch.float32, device=dev)

    phi = m1_t - m0_t
    return phi


def compute_phi_wald_iv(
    X: Array,
    Z: Array,
    A: Array,
    Y: Array,
    pi_hat: Array,
    m_y0_hat: Array,
    m_y1_hat: Array,
    m_a0_hat: Array,
    m_a1_hat: Array,
    *,
    clip_delta_a: float = 0.05, #1e-2,
    device: Optional[str] = None,
) -> torch.Tensor:
    """
    Wald (plug-in) IV pseudo-outcomes: Delta_Y(X) / Delta_A(X).
    """
    delta_y = m_y1_hat - m_y0_hat
    delta_a = m_a1_hat - m_a0_hat
    delta_a = _clip_with_sign(delta_a, clip_delta_a)
    phi = delta_y / delta_a
    return torch.tensor(phi, dtype=torch.float32, device=device)


def compute_phi_dr_iv(
    X: Array,
    Z: Array,
    A: Array,
    Y: Array,
    pi_hat: Array,
    m_y0_hat: Array,
    m_y1_hat: Array,
    m_a0_hat: Array,
    m_a1_hat: Array,
    *,
    clip_eps: float = 1e-2,
    clip_delta_a: float = 0.05, #  1e-2,
    winsor_q: float = 0.05,
    device: Optional[str] = None,
) -> torch.Tensor:
    """
    DR-IV pseudo-outcomes based on the identified ATE.
    """
    pi_hat = np.clip(pi_hat, clip_eps, 1 - clip_eps)
    delta_y = m_y1_hat - m_y0_hat
    delta_a = m_a1_hat - m_a0_hat
    delta_a = _clip_with_sign(delta_a, clip_delta_a)

    m_yz = np.where(Z == 1, m_y1_hat, m_y0_hat)
    m_az = np.where(Z == 1, m_a1_hat, m_a0_hat)
    # h = (Z - pi_hat) / (pi_hat * (1 - pi_hat))
    h = np.where(Z == 0, -1 / (1 - pi_hat), 1 / pi_hat) # maybe more computationally stable

    term_y = (h / delta_a) * (Y - m_yz)
    term_a = -(delta_y * h / (delta_a ** 2)) * (A - m_az)
    if winsor_q and winsor_q > 0:
        lo_y, hi_y = np.quantile(term_y, [winsor_q, 1 - winsor_q])
        lo_a, hi_a = np.quantile(term_a, [winsor_q, 1 - winsor_q])
        term_y = np.clip(term_y, lo_y, hi_y)
        term_a = np.clip(term_a, lo_a, hi_a)
    phi = delta_y / delta_a + term_y + term_a
    return torch.tensor(phi, dtype=torch.float32, device=device)


# Cross-fitting wrapper for pseudo-outcomes
def compute_phi_and_nuisances(
    X: Array,
    A: Array,
    Y: Array,
    phi_fn: Callable[[Array, Array, Array, Array, Array, Array, Optional[str]], torch.Tensor],
    n_splits: Optional[int] = None,
    *,
    seed: Optional[int] = None,
    nuisance_seed: Optional[int] = None,
    nuisance_fn: Callable[[Array, Array, Array], Dict[str, Any]],
    nuisance_params: Optional[Dict[str, Any]] = {},
    device: Optional[str] = None,
    verbose: bool = False,
    return_models: bool = False,
) -> Dict[str, Any]:
    """
    Compute pseudo-outcomes with optional cross-fitting and return nuisance predictions.
    """
    # Input validation    
    if n_splits is not None and n_splits <= 0:
        raise ValueError("n_splits must be a positive integer when provided")

    effective_nuisance_seed = nuisance_seed if nuisance_seed is not None else seed

    # No cross-fitting
    if n_splits is None or n_splits == 1:
        models = nuisance_fn(
            X,
            A,
            Y,
            seed=effective_nuisance_seed,
            **nuisance_params,
            verbose=verbose,
        )
        nuisances = models["predict"](X)
        e_hat = nuisances["e_hat"]
        mu0_hat = nuisances["mu0_hat"]
        mu1_hat = nuisances["mu1_hat"]
        phi = phi_fn(X, A, Y, e_hat, mu0_hat, mu1_hat, device=device)
        nuisance_models = None
        if return_models:
            nuisance_models = {"w_e": models["w_e"], "w0": models["w0"], "w1": models["w1"], "clip_eps": models["clip_eps"]}
        return dict(phi=phi, nuisances=nuisances, nuisance_models=nuisance_models, cross_fitted=False)

    # Cross-fitting
    dev = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dev_str = device if device is not None else dev.type
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_seeds = None
    if effective_nuisance_seed is not None:
        fold_seeds = np.random.SeedSequence(effective_nuisance_seed).spawn(n_splits)
    phi_all = torch.empty(len(X), dtype=torch.float32, device=dev)
    e_hat_all = np.empty(len(X), dtype=float)
    mu0_hat_all = np.empty(len(X), dtype=float)
    mu1_hat_all = np.empty(len(X), dtype=float)
    nuisance_models = [] if return_models else None

    for fold_idx, (train_index, test_index) in enumerate(kf.split(X)):
        X_train, X_test = X[train_index], X[test_index]
        A_train, A_test = A[train_index], A[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]

        fold_seed = None
        if fold_seeds is not None:
            fold_seed = int(fold_seeds[fold_idx].generate_state(1, dtype=np.uint32)[0])
        models = nuisance_fn(
            X_train,
            A_train,
            Y_train,
            seed=fold_seed,
            **nuisance_params,
            verbose=verbose,
        )
        nuisances = models["predict"](X_test)
        e_hat_fold = nuisances["e_hat"]
        mu0_hat_fold = nuisances["mu0_hat"]
        mu1_hat_fold = nuisances["mu1_hat"]

        phi = phi_fn(X_test, A_test, Y_test, e_hat_fold, mu0_hat_fold, mu1_hat_fold, device=dev_str)
        phi_all[test_index] = phi.to(device=dev)
        e_hat_all[test_index] = e_hat_fold
        mu0_hat_all[test_index] = mu0_hat_fold
        mu1_hat_all[test_index] = mu1_hat_fold

        if return_models:
            nuisance_models.append(
                {"w_e": models["w_e"], "w0": models["w0"], "w1": models["w1"], "clip_eps": models["clip_eps"]}
            )

    nuisances_all = {"e_hat": e_hat_all, "mu0_hat": mu0_hat_all, "mu1_hat": mu1_hat_all}
    return dict(phi=phi_all, nuisances=nuisances_all, nuisance_models=nuisance_models, cross_fitted=True)


def compute_phi_and_nuisances_iv(
    X: Array,
    Z: Array,
    A: Array,
    Y: Array,
    phi_fn: Callable[[Array, Array, Array, Array, Array, Array, Array, Array, Optional[str]], torch.Tensor],
    n_splits: Optional[int] = None,
    *,
    seed: Optional[int] = None,
    nuisance_seed: Optional[int] = None,
    nuisance_fn: Callable[[Array, Array, Array, Array], Dict[str, Any]],
    nuisance_params: Optional[Dict[str, Any]] = {},
    device: Optional[str] = None,
    verbose: bool = False,
    return_models: bool = False,
) -> Dict[str, Any]:
    """
    Compute IV pseudo-outcomes with optional cross-fitting and return nuisance predictions.
    """
    if n_splits is not None and n_splits <= 0:
        raise ValueError("n_splits must be a positive integer when provided")

    effective_nuisance_seed = nuisance_seed if nuisance_seed is not None else seed

    if n_splits is None or n_splits == 1:
        models = nuisance_fn(
            X,
            Z,
            A,
            Y,
            seed=effective_nuisance_seed,
            **nuisance_params,
            verbose=verbose,
        )
        nuisances = models["predict"](X)
        pi_hat = nuisances["pi_hat"]
        m_y0_hat = nuisances["m_y0_hat"]
        m_y1_hat = nuisances["m_y1_hat"]
        m_a0_hat = nuisances["m_a0_hat"]
        m_a1_hat = nuisances["m_a1_hat"]
        phi = phi_fn(X, Z, A, Y, pi_hat, m_y0_hat, m_y1_hat, m_a0_hat, m_a1_hat, device=device)
        nuisance_models = None
        if return_models:
            nuisance_models = {
                "w_pi": models["w_pi"],
                "w_y0": models["w_y0"],
                "w_y1": models["w_y1"],
                "w_a0": models["w_a0"],
                "w_a1": models["w_a1"],
                "clip_eps": models["clip_eps"],
            }
        return dict(phi=phi, nuisances=nuisances, nuisance_models=nuisance_models, cross_fitted=False)

    dev = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dev_str = device if device is not None else dev.type
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_seeds = None
    if effective_nuisance_seed is not None:
        fold_seeds = np.random.SeedSequence(effective_nuisance_seed).spawn(n_splits)

    phi_all = torch.empty(len(X), dtype=torch.float32, device=dev)
    pi_hat_all = np.empty(len(X))
    m_y0_hat_all = np.empty(len(X))
    m_y1_hat_all = np.empty(len(X))
    m_a0_hat_all = np.empty(len(X))
    m_a1_hat_all = np.empty(len(X))
    nuisance_models = [] if return_models else None

    for fold_id, (train_index, test_index) in enumerate(kf.split(X)):
        X_train, X_test = X[train_index], X[test_index]
        Z_train, Z_test = Z[train_index], Z[test_index]
        A_train, A_test = A[train_index], A[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]

        fold_seed = None
        if fold_seeds is not None:
            fold_seed = _seed_sequence_to_int(fold_seeds[fold_id])

        models = nuisance_fn(
            X_train,
            Z_train,
            A_train,
            Y_train,
            seed=fold_seed,
            **nuisance_params,
            verbose=verbose,
        )
        nuisances = models["predict"](X_test)
        pi_hat_fold = nuisances["pi_hat"]
        m_y0_fold = nuisances["m_y0_hat"]
        m_y1_fold = nuisances["m_y1_hat"]
        m_a0_fold = nuisances["m_a0_hat"]
        m_a1_fold = nuisances["m_a1_hat"]

        phi = phi_fn(X_test, Z_test, A_test, Y_test, pi_hat_fold, m_y0_fold, m_y1_fold, m_a0_fold, m_a1_fold, device=dev_str)
        phi_all[test_index] = phi.to(device=dev)
        pi_hat_all[test_index] = pi_hat_fold
        m_y0_hat_all[test_index] = m_y0_fold
        m_y1_hat_all[test_index] = m_y1_fold
        m_a0_hat_all[test_index] = m_a0_fold
        m_a1_hat_all[test_index] = m_a1_fold

        if return_models:
            nuisance_models.append(
                {
                    "w_pi": models["w_pi"],
                    "w_y0": models["w_y0"],
                    "w_y1": models["w_y1"],
                    "w_a0": models["w_a0"],
                    "w_a1": models["w_a1"],
                    "clip_eps": models["clip_eps"],
                }
            )

    nuisances_all = {
        "pi_hat": pi_hat_all,
        "m_y0_hat": m_y0_hat_all,
        "m_y1_hat": m_y1_hat_all,
        "m_a0_hat": m_a0_hat_all,
        "m_a1_hat": m_a1_hat_all,
    }
    return dict(phi=phi_all, nuisances=nuisances_all, nuisance_models=nuisance_models, cross_fitted=True)

# def compute_phi(
#     X: Array,
#     A: Array,
#     Y: Array,
#     phi_fn: Callable[[Array, Array, Array, Array, Array, Array, Optional[str]], torch.Tensor],
#     n_splits: Optional[int] = None,
#     *,
#     seed: int = 42,
#     nuisance_fn: Callable[[Array, Array, Array], Dict[str, Any]],
#     nuisance_params: Optional[Dict[str, Any]] = None,
#     device: Optional[str] = None,
#     verbose: bool = False,
# ) -> torch.Tensor:
#     """
#     Compute pseudo-outcomes with optional cross-fitting.
#     """
#     return compute_phi_and_nuisances(
#         X,
#         A,
#         Y,
#         phi_fn=phi_fn,
#         n_splits=n_splits,
#         seed=seed,
#         nuisance_fn=nuisance_fn,
#         nuisance_params=nuisance_params,
#         device=device,
#         verbose=verbose,
#         return_models=False,
#     )["phi"]


## Loss constructors (factories)

def make_mse_from_phi(phi: torch.Tensor) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Construct loss from precomputed pseudo-outcomes - for ATE
    NOTE: uses squared mean error instead of mean squared error -> difference is a constant factor w.r.t. theta but it's easier to compute loss gradients at runtime
    """
    phi_mean = phi.mean()

    def loss(theta: torch.Tensor) -> torch.Tensor:
        psi = phi_mean - theta
        return psi ** 2 / 2.0

    return loss

def _as_tensor(
    x: Any,
    *,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    if torch.is_tensor(x):
        if device is None and dtype is None:
            return x
        return x.to(device=device or x.device, dtype=dtype or x.dtype)
    return torch.tensor(x, device=device, dtype=dtype or torch.float32)


def rbf_kernel(
    X1: Any,
    X2: Any,
    *,
    lengthscale: float = 1.0,
    variance: float = 1.0,
) -> torch.Tensor:
    X1_t = _as_tensor(X1)
    X2_t = _as_tensor(X2, device=X1_t.device, dtype=X1_t.dtype)

    X1_norm = (X1_t ** 2).sum(dim=1, keepdim=True)
    X2_norm = (X2_t ** 2).sum(dim=1, keepdim=True).T
    dists = X1_norm + X2_norm - 2.0 * (X1_t @ X2_t.T)
    return variance * torch.exp(-0.5 * dists / (lengthscale ** 2))


def _pairwise_sq_dists(X1: torch.Tensor, X2: torch.Tensor) -> torch.Tensor:
    X1_norm = (X1 ** 2).sum(dim=1, keepdim=True)
    X2_norm = (X2 ** 2).sum(dim=1, keepdim=True).T
    dists = X1_norm + X2_norm - 2.0 * (X1 @ X2.T)
    return torch.clamp(dists, min=0.0)


def _kernel_matrix(
    X1: Any,
    X2: Any,
    *,
    kernel: str = "rbf",
    lengthscale: float = 1.0,
    variance: float = 1.0,
    rq_alpha: float = 1.0,
) -> torch.Tensor:
    X1_t = _as_tensor(X1)
    X2_t = _as_tensor(X2, device=X1_t.device, dtype=X1_t.dtype)
    kernel = kernel.lower()

    if kernel == "rbf":
        d2 = _pairwise_sq_dists(X1_t, X2_t)
        return variance * torch.exp(-0.5 * d2 / (lengthscale ** 2))
    if kernel == "matern32":
        d2 = _pairwise_sq_dists(X1_t, X2_t)
        r = torch.sqrt(d2 + 1e-12) / lengthscale
        sqrt3 = np.sqrt(3.0)
        return variance * (1.0 + sqrt3 * r) * torch.exp(-sqrt3 * r)
    if kernel == "matern52":
        d2 = _pairwise_sq_dists(X1_t, X2_t)
        r = torch.sqrt(d2 + 1e-12) / lengthscale
        sqrt5 = np.sqrt(5.0)
        return variance * (1.0 + sqrt5 * r + 5.0 * r ** 2 / 3.0) * torch.exp(-sqrt5 * r)
    if kernel == "rq":
        d2 = _pairwise_sq_dists(X1_t, X2_t)
        return variance * (1.0 + d2 / (2.0 * rq_alpha * (lengthscale ** 2))) ** (-rq_alpha)
    raise ValueError(f"Unknown kernel '{kernel}'. Supported: rbf, matern32, matern52, rq")


def _fit_mean_function(
    X: Array,
    y: Array,
    *,
    mean_type: str = "none",
    mean_alpha: float = 1e-3,
    mean_value: Optional[float] = None,
) -> Dict[str, Any]:
    mean_type = str(mean_type or "none").lower()
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)

    if mean_type == "none":
        def mean_fn(X_new: Array) -> Array:
            return np.zeros(X_new.shape[0], dtype=float)
        return dict(mean_type=mean_type, mean_value=0.0, mean_coef=None, mean_fn=mean_fn)

    if mean_type == "constant":
        const = float(np.mean(y)) if mean_value is None else float(mean_value)
        def mean_fn(X_new: Array) -> Array:
            return np.full(X_new.shape[0], const, dtype=float)
        return dict(mean_type=mean_type, mean_value=const, mean_coef=None, mean_fn=mean_fn)

    if mean_type == "linear":
        if X.size == 0 or X.shape[1] == 0:
            const = float(np.mean(y)) if mean_value is None else float(mean_value)
            def mean_fn(X_new: Array) -> Array:
                return np.full(X_new.shape[0], const, dtype=float)
            return dict(mean_type="constant", mean_value=const, mean_coef=None, mean_fn=mean_fn)

        X_aug = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
        d = X_aug.shape[1]
        alpha = float(mean_alpha)
        ridge = alpha * np.eye(d)
        ridge[-1, -1] = 0.0  # do not penalize intercept
        coef = np.linalg.solve(X_aug.T @ X_aug + ridge, X_aug.T @ y)

        def mean_fn(X_new: Array) -> Array:
            X_new = np.asarray(X_new, dtype=float)
            Xn = np.concatenate([X_new, np.ones((X_new.shape[0], 1))], axis=1)
            return Xn @ coef

        return dict(mean_type=mean_type, mean_value=float(coef[-1]), mean_coef=coef[:-1], mean_fn=mean_fn)

    raise ValueError(f"Unknown mean_type '{mean_type}'. Supported: none, constant, linear")


class GPPosterior:
    """
    Analytical GP posterior for CATE estimation with an RBF kernel.
    """

    def __init__(
        self,
        X_train: Any,
        y_train: Any,
        *,
        kernel: str = "rbf",
        lengthscale: float = 1.0,
        variance: float = 1.0,
        rq_alpha: float = 1.0,
        noise_var: float = 1.0,
        jitter: float = 1e-6,
        device: Optional[str] = None,
    ) -> None:
        dev = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        X_t = _as_tensor(X_train, device=dev, dtype=torch.float32)
        y_t = _as_tensor(y_train, device=dev, dtype=torch.float32).view(-1)

        self.X_train = X_t
        self.y_train = y_t
        self.kernel = str(kernel).lower()
        self.lengthscale = float(lengthscale)
        self.variance = float(variance)
        self.rq_alpha = float(rq_alpha)
        self.noise_var = float(noise_var)
        self.jitter = float(jitter)

        K = _kernel_matrix(
            X_t,
            X_t,
            kernel=self.kernel,
            lengthscale=self.lengthscale,
            variance=self.variance,
            rq_alpha=self.rq_alpha,
        )
        n = X_t.shape[0]
        K = K + (self.noise_var + self.jitter) * torch.eye(n, device=dev, dtype=X_t.dtype)
        self.L = torch.linalg.cholesky(K)
        self.alpha = torch.cholesky_solve(y_t[:, None], self.L).squeeze(1)

    def predict(self, X_test: Any, *, return_var: bool = False):
        X_s = _as_tensor(X_test, device=self.X_train.device, dtype=self.X_train.dtype)
        K_s = _kernel_matrix(
            X_s,
            self.X_train,
            kernel=self.kernel,
            lengthscale=self.lengthscale,
            variance=self.variance,
            rq_alpha=self.rq_alpha,
        )
        mean = K_s @ self.alpha

        if not return_var:
            return mean

        v = torch.linalg.solve_triangular(self.L, K_s.T, upper=False)
        K_ss_diag = torch.full((X_s.shape[0],), self.variance, device=mean.device, dtype=mean.dtype)
        var = K_ss_diag - (v ** 2).sum(dim=0)
        var = torch.clamp(var, min=1e-6)
        return mean, var


class InducingPointGP(torch.nn.Module):
    """
    Sparse GP with inducing points for variational inference.
    """

    def __init__(
        self,
        *,
        X_train: Any,
        M: int = 20,
        kernel: str = "rbf",
        lengthscale: float = 1.0,
        variance: float = 1.0,
        rq_alpha: float = 1.0,
        jitter: float = 1e-6,
        device: Optional[str] = None,
    ) -> None:
        super().__init__()
        dev = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        X_t = _as_tensor(X_train, device=dev, dtype=torch.float32)
        n = X_t.shape[0]
        M = int(M)
        if M <= 0:
            raise ValueError("M must be positive for inducing points")
        M = min(M, n)

        perm = torch.randperm(n, device=dev)
        Z_init = X_t[perm[:M]].clone()

        self.Z = torch.nn.Parameter(Z_init)
        self.mu = torch.nn.Parameter(torch.zeros(M, device=dev))
        self.L_raw = torch.nn.Parameter(torch.eye(M, device=dev))

        self.kernel = str(kernel).lower()
        self.lengthscale = float(lengthscale)
        self.variance = float(variance)
        self.rq_alpha = float(rq_alpha)
        self.jitter = float(jitter)

    def vi_parameters(self) -> List[torch.nn.Parameter]:
        return [self.Z, self.mu, self.L_raw]

    def get_L(self) -> torch.Tensor:
        L = torch.tril(self.L_raw)
        diag = torch.diagonal(L)
        diag = torch.exp(diag) + 1e-6
        L = L - torch.diag(torch.diagonal(L)) + torch.diag(diag)
        return L

    def _kzz_cholesky(self) -> torch.Tensor:
        K_zz = _kernel_matrix(
            self.Z,
            self.Z,
            kernel=self.kernel,
            lengthscale=self.lengthscale,
            variance=self.variance,
            rq_alpha=self.rq_alpha,
        )
        K_zz = K_zz + self.jitter * torch.eye(self.Z.shape[0], device=self.Z.device, dtype=self.Z.dtype)
        return torch.linalg.cholesky(K_zz)

    def predict_mean_var(self, X_test: Any) -> Dict[str, torch.Tensor]:
        X_t = _as_tensor(X_test, device=self.Z.device, dtype=self.Z.dtype)
        K_xz = _kernel_matrix(
            X_t,
            self.Z,
            kernel=self.kernel,
            lengthscale=self.lengthscale,
            variance=self.variance,
            rq_alpha=self.rq_alpha,
        )

        L = self._kzz_cholesky()
        mu_t = self.mu[:, None]
        alpha = torch.cholesky_solve(mu_t, L)
        mean = (K_xz @ alpha).squeeze(1)

        L_q = self.get_L()
        Sigma = L_q @ L_q.T
        eye = torch.eye(self.Z.shape[0], device=self.Z.device, dtype=self.Z.dtype)
        K_zz_inv = torch.cholesky_solve(eye, L)
        K_zz = _kernel_matrix(
            self.Z,
            self.Z,
            kernel=self.kernel,
            lengthscale=self.lengthscale,
            variance=self.variance,
            rq_alpha=self.rq_alpha,
        )
        C = K_zz_inv @ (K_zz - Sigma) @ K_zz_inv
        K_xz_C = K_xz @ C
        K_xx_diag = torch.full((X_t.shape[0],), self.variance, device=X_t.device, dtype=X_t.dtype)
        var = K_xx_diag - (K_xz_C * K_xz).sum(dim=1)
        var = torch.clamp(var, min=1e-6)
        return dict(mean=mean, var=var)

    def predict_mean(self, X_test: Any) -> torch.Tensor:
        return self.predict_mean_var(X_test)["mean"]

    def predict_var(self, X_test: Any) -> torch.Tensor:
        return self.predict_mean_var(X_test)["var"]

    def kl_to_prior(
        self,
        *,
        prior_kernel: Optional[str] = None,
        prior_lengthscale: float = 1.0,
        prior_variance: float = 1.0,
        prior_rq_alpha: float = 1.0,
    ) -> torch.Tensor:
        kernel_name = self.kernel if prior_kernel is None else str(prior_kernel).lower()
        K_zz = _kernel_matrix(
            self.Z,
            self.Z,
            kernel=kernel_name,
            lengthscale=prior_lengthscale,
            variance=prior_variance,
            rq_alpha=prior_rq_alpha,
        )
        K_zz = K_zz + self.jitter * torch.eye(self.Z.shape[0], device=self.Z.device, dtype=self.Z.dtype)
        L = torch.linalg.cholesky(K_zz)

        L_q = self.get_L()
        Sigma = L_q @ L_q.T
        K_zz_inv_Sigma = torch.cholesky_solve(Sigma, L)
        trace_term = torch.trace(K_zz_inv_Sigma)

        mu_t = self.mu[:, None]
        quad_term = (mu_t.T @ torch.cholesky_solve(mu_t, L)).squeeze()

        log_det_K = 2.0 * torch.sum(torch.log(torch.diagonal(L)))
        log_det_Sigma = 2.0 * torch.sum(torch.log(torch.diagonal(L_q)))
        M = self.Z.shape[0]
        kl = 0.5 * (trace_term + quad_term - M + log_det_K - log_det_Sigma)
        return kl


def cate_mse_loss(
    phi: torch.Tensor,
    mean: torch.Tensor,
    var: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    err = phi - mean
    if var is None:
        return 0.5 * torch.mean(err ** 2)
    return 0.5 * torch.mean(err ** 2 + var)


def fit_cate_gbi(
    *,
    X: Any,
    phi: Any,
    q_family: InducingPointGP,
    prior_kernel: Optional[str] = None,
    prior_lengthscale: float = 1.0,
    prior_variance: float = 1.0,
    prior_rq_alpha: float = 1.0,
    omega: Any = 1.0,
    n_epochs: int = 1000,
    lr: float = 0.01,
    progress: bool = False,
    plot: bool = False,
    device: Optional[str] = None,
    **_: Any,
) -> List[Dict[str, float]]:
    dev = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    q_family.to(dev)
    X_t = _as_tensor(X, device=dev, dtype=torch.float32)
    phi_t = _as_tensor(phi, device=dev, dtype=torch.float32).view(-1)
    n = int(X_t.shape[0])
    omega_t = torch.as_tensor(omega, device=dev, dtype=phi_t.dtype)

    optimizer = torch.optim.Adam(q_family.vi_parameters(), lr=lr)
    history: List[Dict[str, float]] = []

    iterator = range(n_epochs)
    if progress:
        iterator = tqdm(iterator, desc="Fitting steps for CATE GBI")

    for step in iterator:
        optimizer.zero_grad()

        preds = q_family.predict_mean_var(X_t)
        data_loss = cate_mse_loss(phi_t, preds["mean"], preds["var"])
        kl_loss = q_family.kl_to_prior(
            prior_kernel=prior_kernel,
            prior_lengthscale=prior_lengthscale,
            prior_variance=prior_variance,
            prior_rq_alpha=prior_rq_alpha,
        )
        objective = omega_t * n * data_loss + kl_loss
        objective.backward()
        optimizer.step()

        history.append(
            {
                "step": float(step),
                "data_term": float(data_loss.detach().cpu().item()),
                "kl_term": float(kl_loss.detach().cpu().item()),
                "objective": float(objective.detach().cpu().item()),
            }
        )

    if plot:
        objective_cpu = [row["objective"] for row in history]
        plt.figure(figsize=(5, 3))
        plt.plot(objective_cpu)
        plt.title("CATE GBI objective")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.yscale("log")
        plt.tight_layout()
        plt.show()

    return history


# ----------------------------------------------
# Omega estimation

## Naive 

def estimate_omega_naive(
    phi: torch.Tensor,
    log: bool = True,
    **_: Any,
) -> torch.Tensor:
    """
    A naive estimate of omega based on the variance of pseudo-outcomes.
    """
    var_phi = torch.var(phi).item()
    if var_phi <= 0: 
        print(f"Warning: var(phi)={var_phi:.4f} <= 0 in omega estimation; defaulting omega to 1.0")
    omega_est = 1.0 / var_phi if var_phi > 0 else 1.0
    if log:
        print(f"Estimated omega (naive): {omega_est:.4f} based on var(phi)={var_phi:.4f}")
    return torch.tensor(omega_est, device=phi.device, dtype=phi.dtype)

## Non-parametric bootstrap


def estimate_omega_bootstrap(
    phi: torch.Tensor,
    log: bool = True,
    progress: bool = False,
    *,
    n_boot: int,
    n: int,
    seed: Optional[int],
    X: Array,
    A: Array,
    Y: Array,
    phi_fn: Callable[[Array, Array, Array, Array, Array, Array, Optional[str]], torch.Tensor],
    phi_n_splits: Optional[int],
    nuisance_fn: Callable[[Array, Array, Array], Dict[str, Any]],
    nuisance_params: Dict[str, Any],
) -> torch.Tensor:
    
    """
    Nonparametric bootstrap calibration of omega using Var(theta_hat).
    In omega_params pass n_boot (int): number of bootstrap samples.
    """

    if n_boot <= 1:
        raise ValueError("n_boot must be > 1 for bootstrap omega calibration")

    rng = np.random.default_rng(seed)
    theta_hats = np.empty(n_boot, dtype=float)
    boot_seeds = None
    if seed is not None:
        boot_seeds = np.random.SeedSequence(seed).spawn(n_boot)

    for b in tqdm(range(n_boot), disable=not progress, desc="Bootstrap omega estimation"):
        idx = rng.integers(0, n, size=n)
        X_b = X[idx]
        A_b = A[idx]
        Y_b = Y[idx]

        boot_seed = None
        if boot_seeds is not None:
            boot_seed = int(boot_seeds[b].generate_state(1, dtype=np.uint32)[0])
        phi_out = compute_phi_and_nuisances(
            X_b,
            A_b,
            Y_b,
            phi_fn=phi_fn,
            n_splits=phi_n_splits,
            seed=boot_seed,
            nuisance_fn=nuisance_fn,
            nuisance_params=nuisance_params,
            device=phi.device,
            verbose=False,
            return_models=False,
        )
        phi_b = phi_out["phi"]
        theta_hats[b] = float(np.var(phi_b.detach().cpu().numpy()))

    mean_hat = float(np.mean(theta_hats))
    if mean_hat <= 0:
        print(f"Warning: bootstrap mean(var(phi_b))={mean_hat:.6f} <= 0; defaulting omega to 1.0")
        omega_est = 1.0
    else:
        omega_est = 1.0 / mean_hat


    if log:
        print(
            "Estimated omega (bootstrap): "
            f"{omega_est:.6f} based on mean(var(phi_b))={mean_hat:.6f} with n_boot={n_boot}"
        )
    return torch.tensor(omega_est, device=phi.device, dtype=phi.dtype)


def estimate_omega_bootstrap_iv(
    phi: torch.Tensor,
    *,
    n_boot: int = 200,
    seed: Optional[int] = None,
    n: Optional[int] = None,
    X: Array,
    Z: Array,
    A: Array,
    Y: Array,
    phi_fn: Callable[[Array, Array, Array, Array, Array, Array, Array, Array, Optional[str]], torch.Tensor],
    phi_n_splits: Optional[int],
    nuisance_fn: Callable[[Array, Array, Array, Array], Dict[str, Any]],
    nuisance_params: Dict[str, Any],
    log: bool = False,
    progress: bool = False,
) -> torch.Tensor:
    """
    Bootstrap estimate of omega for IV pseudo-outcomes.
    """
    rng = np.random.default_rng(seed)
    n = len(Y)
    theta_hats = np.zeros(n_boot)
    iterator = range(n_boot)
    if progress:
        iterator = tqdm(iterator, desc="Bootstrap omega")
    for b in iterator:
        idx = rng.integers(0, n, size=n)
        Xb = X[idx]
        Zb = Z[idx]
        Ab = A[idx]
        Yb = Y[idx]
        phi_out = compute_phi_and_nuisances_iv(
            Xb,
            Zb,
            Ab,
            Yb,
            phi_fn=phi_fn,
            n_splits=phi_n_splits,
            seed=seed,
            nuisance_seed=seed,
            nuisance_fn=nuisance_fn,
            nuisance_params=nuisance_params,
            device=phi.device,
            verbose=False,
        )
        phi_b = phi_out["phi"]
        theta_hats[b] = float(np.var(phi_b.detach().cpu().numpy()))

    mean_hat = float(np.mean(theta_hats))
    if mean_hat <= 0:
        if log:
            print(f"Warning: bootstrap mean(var(phi_b))={mean_hat:.6f} <= 0; defaulting omega to 1.0")
        omega_est = 1.0
    else:
        omega_est = 1.0 / mean_hat
        if log:
            print(
                f"{omega_est:.6f} based on mean(var(phi_b))={mean_hat:.6f} with n_boot={n_boot}"
            )
    return torch.tensor(omega_est, device=phi.device, dtype=phi.dtype)


# ----------------------------------------------
# Variational Inference

## VI families (classes)

## === VARIATIONAL FAMILIES =================================================

class GaussianFamily(torch.nn.Module):
    def __init__(self, mu_init: float = 0.0, log_sigma_init: float = 0.0, device: Optional[str] = None) -> None:
        super().__init__()
        dev = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mu = torch.nn.Parameter(torch.tensor(mu_init, dtype=torch.float32, device=dev))
        self.log_sigma = torch.nn.Parameter(torch.tensor(log_sigma_init, dtype=torch.float32, device=dev))

    def sample(self, n_samples: int = 1) -> torch.Tensor:
        eps = torch.randn(n_samples, device=self.mu.device)
        return self.mu + torch.exp(self.log_sigma) * eps

    def log_prob(self, theta: torch.Tensor) -> torch.Tensor:
        sigma = torch.exp(self.log_sigma)
        log_norm = torch.log(torch.tensor(2.0 * np.pi, device=theta.device, dtype=theta.dtype))
        return -0.5 * ((theta - self.mu) ** 2 / sigma ** 2 + 2 * self.log_sigma + log_norm)

    def kl_to_prior(self, prior_mu: float = 0.0, prior_sigma: float = 1.0) -> torch.Tensor:
        prior_mu_t = torch.as_tensor(prior_mu, device=self.mu.device, dtype=self.mu.dtype)
        prior_sigma_t = torch.as_tensor(prior_sigma, device=self.mu.device, dtype=self.mu.dtype)
        sigma = torch.exp(self.log_sigma)
        term = (sigma ** 2 + (self.mu - prior_mu_t) ** 2) / (prior_sigma_t ** 2) - 1 + 2 * (
            torch.log(prior_sigma_t) - self.log_sigma
        )
        return 0.5 * term.sum()

    # def parameters(self):
    #     return [self.mu, self.log_sigma]

    # def to(self, device):
    #     self.mu = self.mu.to(device)
    #     self.log_sigma = self.log_sigma.to(device)
    #     return self

    def vi_parameters(self) -> List[torch.nn.Parameter]:
        # Explicit whitelist of trainable params for VI
        return [self.mu, self.log_sigma]

    def to(self, *args, **kwargs):
        # Preserve Parameter-ness; do not reassign self.mu/self.log_sigma
        return super().to(*args, **kwargs)


class MixtureGaussianFamily(torch.nn.Module):
    def __init__(self, K: int = 2, device: Optional[str] = None) -> None:
        super().__init__()
        dev = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.K = K
        self.logits = torch.nn.Parameter(torch.zeros(K, device=dev))
        self.mu = torch.nn.Parameter(torch.randn(K, device=dev))
        self.log_sigma = torch.nn.Parameter(torch.zeros(K, device=dev))

    def sample(self, n_samples: int = 1) -> torch.Tensor:
        probs = torch.softmax(self.logits, dim=0)
        comp_idx = torch.multinomial(probs, n_samples, replacement=True)
        eps = torch.randn(n_samples, device=self.mu.device)
        sigmas = torch.exp(self.log_sigma[comp_idx])
        mus = self.mu[comp_idx]
        return mus + sigmas * eps

    def log_prob(self, theta: torch.Tensor) -> torch.Tensor:
        sigma = torch.exp(self.log_sigma)
        log_norm = torch.log(torch.tensor(2.0 * np.pi, device=theta.device, dtype=theta.dtype))
        comp_logprob = -0.5 * (
            (theta.unsqueeze(1) - self.mu) ** 2 / sigma ** 2 + 2 * self.log_sigma + log_norm
        )
        log_pi = torch.log_softmax(self.logits, dim=0)
        return torch.logsumexp(comp_logprob + log_pi, dim=1)  # shape: (n,)

    def kl_to_prior(self, prior_mu: float = 0.0, prior_sigma: float = 1.0) -> torch.Tensor:
        prior_mu_t = torch.as_tensor(prior_mu, device=self.mu.device, dtype=self.mu.dtype)
        prior_sigma_t = torch.as_tensor(prior_sigma, device=self.mu.device, dtype=self.mu.dtype)
        probs = torch.softmax(self.logits, dim=0)
        sigma = torch.exp(self.log_sigma)
        kl_components = 0.5 * (
            (sigma ** 2 + (self.mu - prior_mu_t) ** 2) / prior_sigma_t ** 2
            - 1
            + 2 * (torch.log(prior_sigma_t) - self.log_sigma)
        )
        return torch.sum(probs * kl_components)

    # def parameters(self):
    #     return [self.logits, self.mu, self.log_sigma]

    # def to(self, device):
    #     self.logits = self.logits.to(device)
    #     self.mu = self.mu.to(device)
    #     self.log_sigma = self.log_sigma.to(device)
    #     return self
    
    def vi_parameters(self) -> List[torch.nn.Parameter]:
        # Explicit whitelist of trainable params for VI
        return [self.logits, self.mu, self.log_sigma]

    def to(self, *args, **kwargs):
        return super().to(*args, **kwargs)
    
def _assert_only_whitelisted_trainables(module: torch.nn.Module, whitelist: Iterable[torch.nn.Parameter]) -> None:
    wl = set(whitelist)

    # (A) Ensure every whitelisted object is actually a Parameter owned by the module
    owned = {p for _, p in module.named_parameters(recurse=True)}
    missing = [p for p in wl if p not in owned]
    if missing:
        raise RuntimeError("Whitelist contains parameters that are not registered on the module.")

    # (B) Ensure there are no other trainable parameters besides the whitelist
    extra_trainables = [(name, p) for name, p in module.named_parameters(recurse=True) if p.requires_grad and p not in wl]
    if extra_trainables:
        names = ", ".join([n for n, _ in extra_trainables])
        raise RuntimeError(f"Found trainable parameters not in whitelist: {names}")


## VI fitting routine


def _debug_check_cuda_fit_state(dev: torch.device, q_family: torch.nn.Module, prior_mu, prior_sigma, omega, loss_fn):
    # Normalize "cuda" (no index) to a concrete cuda:<idx> so equality checks behave as expected
    expected = dev
    if dev.type == "cuda" and dev.index is None:
        expected = torch.device("cuda", torch.cuda.current_device())

    print(f"[debug] dev={dev} (normalized expected={expected}), cuda_available={torch.cuda.is_available()}")
    for name, p in q_family.named_parameters():
        print(f"[debug] param {name}: device={p.device}, requires_grad={p.requires_grad}, shape={tuple(p.shape)}")
        assert p.device == expected, f"Parameter {name} is on {p.device} but expected {expected}"

    assert prior_mu.device == expected
    assert prior_sigma.device == expected
    assert omega.device == expected

    theta0 = q_family.sample(1)[0]
    l0 = loss_fn(theta0)
    print(f"[debug] sample theta device={theta0.device}, loss device={l0.device}")
    assert theta0.device == expected
    assert l0.device == expected

def _normalize_cuda_device(dev: torch.device) -> torch.device:
    """
    Normalize `cuda` (no index) -> `cuda:<current_device>` so equality checks work.
    """
    if dev.type == "cuda" and dev.index is None:
        return torch.device("cuda", torch.cuda.current_device())
    return dev


def _assert_tensor_on_dev(name: str, t: torch.Tensor, expected: torch.device) -> None:
    if not torch.is_tensor(t):
        raise TypeError(f"{name} is not a torch.Tensor (got {type(t)})")
    if t.device != expected:
        raise RuntimeError(f"{name} is on {t.device}, expected {expected}")


def _assert_module_params_on_dev(module: torch.nn.Module, expected: torch.device) -> None:
    for n, p in module.named_parameters(recurse=True):
        if p.device != expected:
            raise RuntimeError(f"param {n} is on {p.device}, expected {expected}")


def _assert_grads_on_dev(module: torch.nn.Module, expected: torch.device) -> None:
    for n, p in module.named_parameters(recurse=True):
        if not p.requires_grad:
            continue
        if p.grad is None:
            raise RuntimeError(f"param {n} has grad=None after backward()")
        if p.grad.device != expected:
            raise RuntimeError(f"grad for {n} is on {p.grad.device}, expected {expected}")


def fit_gbi(
    loss_fn: Callable[[torch.Tensor], torch.Tensor],
    n: int,
    q_family: Any,
    prior_mu: float = 0.0,
    prior_sigma: float = 1.0,
    omega: Any = 1.0,
    batch_size: int = 25,
    n_epochs: int = 1000,
    lr: float = 0.05,
    progress: bool = False,
    track_grads: bool = False,
    plot: bool = True,
    device: Optional[str] = None,
    debug: bool = False, # Not part of json args -- for temporary debugging
) -> Optional[List[Dict[str, float]]]:
    """
    Fit a variational Generalized Bayes posterior q_family for a given loss function.

    Args:
        loss_fn: callable taking a torch scalar theta and returning torch scalar loss.
        n: sample size.
        q_family: variational family with sample, kl_to_prior, parameters, and to.
        prior_mu, prior_sigma: prior parameters.
        omega: GBI weight (temperature/calibration).
        batch_size: number of theta samples per update.
        n_epochs: optimization iterations.
        lr: learning rate for Adam.
        progress: show tqdm progress bar.
        track_grads: record gradients and updates (GaussianFamily expected).
        plot: plot objective loss curve at the end of training.
        device: torch device string or None.
    """
    if device is None:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(device)

    if debug:
        expected = _normalize_cuda_device(dev)

    # Move to device
    q_family.to(dev)
    prior_mu = torch.tensor(prior_mu, dtype=torch.float32, device=dev)
    prior_sigma = torch.tensor(prior_sigma, dtype=torch.float32, device=dev)
    omega = torch.as_tensor(omega, device=dev, dtype=prior_mu.dtype)

    # Debug: confirm everything is on GPU when requested 
    if debug and dev.type == "cuda":
        _debug_check_cuda_fit_state(dev, q_family, prior_mu, prior_sigma, omega, loss_fn)

    # Initialize
    # optimizer = torch.optim.Adam(q_family.parameters(), lr=lr)

    vi_params = list(q_family.vi_parameters())
    _assert_only_whitelisted_trainables(q_family, vi_params)
    optimizer = torch.optim.Adam(vi_params, lr=lr)

    # --- GPU/device correctness check (point 1) ---
    if debug and expected.type == "cuda":
        # Check persistent state
        _assert_module_params_on_dev(q_family, expected)
        _assert_tensor_on_dev("prior_mu", prior_mu, expected)
        _assert_tensor_on_dev("prior_sigma", prior_sigma, expected)
        _assert_tensor_on_dev("omega", omega, expected)

        # One “probe” step that mirrors the loop and asserts devices
        thetas_probe = q_family.sample(max(1, min(batch_size, 4)))
        _assert_tensor_on_dev("thetas_probe", thetas_probe, expected)

        losses_probe = torch.stack([loss_fn(theta) for theta in thetas_probe])
        _assert_tensor_on_dev("losses_probe", losses_probe, expected)

        data_term_probe = losses_probe.mean()
        kl_term_probe = q_family.kl_to_prior(prior_mu=prior_mu, prior_sigma=prior_sigma)
        _assert_tensor_on_dev("data_term_probe", data_term_probe, expected)
        _assert_tensor_on_dev("kl_term_probe", kl_term_probe, expected)

        objective_probe = omega * data_term_probe * n + kl_term_probe
        _assert_tensor_on_dev("objective_probe", objective_probe, expected)

        optimizer.zero_grad(set_to_none=True)
        objective_probe.backward()
        _assert_grads_on_dev(q_family, expected)
        optimizer.zero_grad(set_to_none=True)


    objective_losses: Optional[List[torch.Tensor]] = [] if plot else None
    history: List[Dict[str, torch.Tensor]] = []

    # Training loop
    for step in tqdm(range(n_epochs), disable=not progress, desc="Fitting steps for GBI VI"):
        thetas = q_family.sample(batch_size)
        losses = loss_fn(thetas)
        data_term = losses.mean()
        kl_term = q_family.kl_to_prior(prior_mu=prior_mu, prior_sigma=prior_sigma)
        objective = omega * data_term * n + kl_term

        optimizer.zero_grad()
        objective.backward()

        if track_grads:
            grad_mu = q_family.mu.grad.detach().mean()
            grad_log_sigma = q_family.log_sigma.grad.detach().mean()
            mu_before = q_family.mu.detach().mean()
            log_sigma_before = q_family.log_sigma.detach().mean()

        optimizer.step()

        if track_grads:
            mu_after = q_family.mu.detach().mean()
            log_sigma_after = q_family.log_sigma.detach().mean()
            delta_mu = mu_after - mu_before
            delta_log_sigma = log_sigma_after - log_sigma_before

        if objective_losses is not None:
            objective_losses.append(objective.detach())

        if track_grads:
            history.append(
                {
                    "step": step,
                    "mu": mu_after,
                    "sigma": torch.exp(q_family.log_sigma).detach().mean(),
                    "log_sigma": log_sigma_after,
                    "grad_mu": grad_mu,
                    "grad_log_sigma": grad_log_sigma,
                    "delta_mu": delta_mu,
                    "delta_log_sigma": delta_log_sigma,
                    "data_term": data_term.detach(),
                    "kl_term": kl_term.detach(),
                    "objective": objective.detach(),
                }
            )

    if plot and objective_losses is not None:
        objective_cpu = [float(x.cpu()) for x in objective_losses]
        plt.figure(figsize=(5, 3))
        plt.plot(objective_cpu)
        plt.title("Objective loss over training")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.yscale("log")
        plt.tight_layout()
        plt.show()

    if track_grads:
        history_cpu: List[Dict[str, float]] = []
        for h in history:
            row = {}
            for k, v in h.items():
                if isinstance(v, torch.Tensor):
                    row[k] = float(v.detach().cpu().item())
                else: # applies to <step> (int)
                    row[k] = float(v)
            history_cpu.append(row)
        return history_cpu
    return None


# ----------------------------------------------
# Experiment runner(s)

# For the ATE
def compute_reference_quantities(
    phi: torch.Tensor,
    prior_mu: float,
    prior_sigma: float,
    omega: Any,
) -> Dict[str, float]:
    """
    Compute reference quantities from pseudo-outcomes. NOTE: Only valid for Gaussian prior and MSE loss.
    """
    ate_phi = float(phi.mean().item())
    n = int(phi.shape[0])
    omega_val = float(omega.detach().cpu().item()) if torch.is_tensor(omega) else float(omega)
    post_var = (prior_sigma ** -2 + n * omega_val) ** -1
    post_mean = post_var * (prior_mu / prior_sigma ** 2 + n * omega_val * ate_phi)
    return dict(
        ate_phi=ate_phi,
        post_mean=post_mean,
        post_var=post_var,
        prior_mu=prior_mu,
        prior_sigma=prior_sigma,
    )


def _resolve_cate_eval_X(data: Dict[str, Any], cate_eval: Optional[Dict[str, Any]]) -> Array:
    X = data["X"]
    if not cate_eval:
        return X
    if "X_eval" in cate_eval:
        return np.asarray(cate_eval["X_eval"], dtype=float)
    grid_cfg = cate_eval.get("grid")
    if not grid_cfg:
        return X

    X = np.asarray(X, dtype=float)
    n = int(grid_cfg.get("n", 100))
    dim = int(grid_cfg.get("dim", 0))
    if X.shape[1] == 0:
        return X

    x_min = float(grid_cfg.get("min", np.min(X[:, dim])))
    x_max = float(grid_cfg.get("max", np.max(X[:, dim])))
    x_line = np.linspace(x_min, x_max, n)

    fixed = grid_cfg.get("fixed")
    if fixed is None:
        fixed_vals = X.mean(axis=0)
    else:
        fixed_vals = np.asarray(fixed, dtype=float)
        if fixed_vals.shape[0] != X.shape[1]:
            raise ValueError("cate_eval.grid.fixed must have length d")

    X_eval = np.tile(fixed_vals, (n, 1))
    X_eval[:, dim] = x_line
    return X_eval


def _compute_cate_mse(mean: Optional[Any], cate_true: Optional[Any]) -> float:
    if mean is None or cate_true is None:
        return float(np.nan)
    mean_arr = np.asarray(mean, dtype=float)
    true_arr = np.asarray(cate_true, dtype=float)
    return float(np.mean((mean_arr - true_arr) ** 2))




## Single-run experiment


_DGP_REGISTRY = {
    "dgp_1": simulate_dgp_1,
    "dgp_2": simulate_dgp_2,
    "dgp_3": simulate_dgp_3,
    "dgp_4": simulate_dgp_4,
    "dgp_5": simulate_dgp_5,
    "dgp_6": simulate_dgp_6,
    "dgp_7": simulate_dgp_7,
    "dgp_8": simulate_dgp_8,
    "dgp_9": simulate_dgp_9,
}

_DGP_IV_REGISTRY = {
    "dgp_iv_0": simulate_dgp_iv_0,
    "dgp_iv_1": simulate_dgp_iv_1,
    "dgp_iv_2": simulate_dgp_iv_2,
}




_PHI_REGISTRY = {
    "aipw": compute_phi_aipw,
    "ipw": compute_phi_ipw,
    "ra": compute_phi_ra,
}

_PHI_IV_REGISTRY = {
    "wald_iv": compute_phi_wald_iv,
    "dr_iv": compute_phi_dr_iv,
}

_VAR_FAMILY_REGISTRY = {
    "gaussian": GaussianFamily,
    "mixture_gaussian": MixtureGaussianFamily,
}

_VAR_FAMILY_CATE_REGISTRY = {
    "inducing_gp": InducingPointGP,
}

_OMEGA_REGISTRY = {
    "naive": estimate_omega_naive,
    "bootstrap": estimate_omega_bootstrap,
}

_OMEGA_IV_REGISTRY = {
    "naive": estimate_omega_naive,
    "bootstrap": estimate_omega_bootstrap_iv,
}

_NUISANCE_REGISTRY = {
    "default": fit_nuisance_models,
    "noisy": fit_nuisance_models_noisy,
}

_NUISANCE_IV_REGISTRY = {
    "default_iv": fit_nuisance_models_iv,
    "noisy_iv": fit_nuisance_models_iv_noisy,
    "fancy_iv": fit_nuisance_models_iv_fancy, 
}


def _resolve_from_registry(name_or_fn: Any, registry: Dict[str, Any], label: str):
    if callable(name_or_fn):
        return name_or_fn
    if isinstance(name_or_fn, str):
        if name_or_fn in registry:
            return registry[name_or_fn]
        raise ValueError(f"Unknown {label} '{name_or_fn}'. Available: {sorted(registry.keys())}")
    raise TypeError(f"{label} must be a callable or registry string key")


def _seed_sequence_to_int(seed_seq: np.random.SeedSequence) -> int:
    return int(seed_seq.generate_state(1, dtype=np.uint32)[0])


def build_run_seeds(run_seed: int) -> Dict[str, int]:
    seq = np.random.SeedSequence(run_seed)
    dgp_seq, nuisance_seq, split_seq, omega_seq, vi_seq = seq.spawn(5)
    return {
        "dgp": _seed_sequence_to_int(dgp_seq),
        "nuisance": _seed_sequence_to_int(nuisance_seq),
        "split": _seed_sequence_to_int(split_seq),
        "omega": _seed_sequence_to_int(omega_seq),
        "vi": _seed_sequence_to_int(vi_seq),
    }



def single_run(
    *,
    dgp_fn_REGISTRATION: str, n: int, run_seed: int, dgp_kwargs: Dict[str, Any],
    nuisance_fn_REGISTRATION: str,
    nuisance_params: Dict[str, Any],
    phi_fn_REGISTRATION: str, phi_n_splits: Optional[int],
    q_family_cls_REGISTRATION: str, q_family_kwargs: Dict[str, Any],
    omega_est_REGISTRATION: str,
    omega_params: Optional[Dict[str, Any]] = {},
    prior_mu: float, prior_sigma: float,
    learn_cfg: Dict[str, Any],
    analytic_flag: bool,
    device: str, log: bool, plot: bool, progress: bool, diagnostics: bool,
) -> Dict[str, Any]:
    """
    Run a single GBI fit with local variational family construction.
    The variational family instance is created inside this function to avoid sharing across runs.
    """
    # (0). Resolve functions from registries
    dgp_fn = _resolve_from_registry(dgp_fn_REGISTRATION, _DGP_REGISTRY, "dgp_fn_REGISTRATION")
    phi_fn = _resolve_from_registry(phi_fn_REGISTRATION, _PHI_REGISTRY, "phi_fn_REGISTRATION")
    nuisance_fn = _resolve_from_registry(nuisance_fn_REGISTRATION, _NUISANCE_REGISTRY, "nuisance_fn_REGISTRATION")
    omega_est = _resolve_from_registry(omega_est_REGISTRATION, _OMEGA_REGISTRY, "omega_est_REGISTRATION")
    q_family_cls = _resolve_from_registry(q_family_cls_REGISTRATION, _VAR_FAMILY_REGISTRY, "q_family_cls_REGISTRATION")

    # (0.5). Build run-specific seeds
    run_seeds = build_run_seeds(run_seed)
    dgp_seed = run_seeds["dgp"]
    nuisance_seed = run_seeds["nuisance"]
    split_seed = run_seeds["split"]
    omega_seed = run_seeds["omega"]
    vi_seed = run_seeds["vi"]

    # (1-2). Simulate data
    data = dgp_fn(n=n, seed=dgp_seed, **dgp_kwargs)
    X, A, Y = data["X"], data["A"], data["Y"]
    ate_true = data.get("ate_true", None)

    # (3). Estimate nuisances and compute pseudo-outcomes with cross-fitting
    phi_out = compute_phi_and_nuisances(
        X,
        A,
        Y,
        phi_fn=phi_fn,
        n_splits=phi_n_splits,
        seed=split_seed,
        nuisance_seed=nuisance_seed,
        nuisance_fn=nuisance_fn,
        nuisance_params=nuisance_params,
        device=device,
        verbose=log,
        return_models=True,
    )
    phi = phi_out["phi"]
    nuisances = phi_out["nuisances"]
    nuisance_models = phi_out["nuisance_models"]

    # (4). Construct loss
    loss = make_mse_from_phi(phi)

    # (5). Estimate omega
    bootstrap_kwargs = {
        "n": n,
        "seed": omega_seed,
        "X": X,
        "A": A,
        "Y": Y,
        "phi_fn": phi_fn,
        "phi_n_splits": phi_n_splits,
        "nuisance_fn": nuisance_fn,
        "nuisance_params": nuisance_params,
    }

    omega = omega_est(phi, log=log, progress=progress, **omega_params, **bootstrap_kwargs)

    # # # Testing bootstrap:
    # progress = False
    # bootstrap_omega = omega.detach().cpu().item()
    # comparison_omega = estimate_omega_naive(phi, log=False).detach().cpu().item()
    # print(f"DGP: {dgp_fn_REGISTRATION}, n={n}")
    # print(f"Boostsrap Estimated omega       : {bootstrap_omega:.6f}")
    # print(f"Naive Estimated omega           : {comparison_omega:.6f}")
    # print(f"Relative Difference (abs)       : {abs(bootstrap_omega - comparison_omega)/comparison_omega:.6f}")
    # print(f"bootstrap/comparison            : {bootstrap_omega/comparison_omega:.6f} \n")

    # (7). Instantiate variational family 
    torch.manual_seed(vi_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(vi_seed)
    q_family = q_family_cls(device=device, **q_family_kwargs)

    # (6,8). Fit GBI
    learn_cfg_local = dict(learn_cfg)
    learn_cfg_local.pop("track_grads", None)
    history = fit_gbi(
        loss_fn=loss,
        n=n,
        q_family=q_family,
        prior_mu=prior_mu,
        prior_sigma=prior_sigma,
        omega=omega,
        **learn_cfg_local,
        progress=progress,
        plot=plot,
        track_grads=bool(diagnostics),
        device=device,
    )

    # (9). Compute reference analytical solution
    reference = None
    if analytic_flag:
        reference = compute_reference_quantities(
            phi,
            prior_mu=prior_mu,
            prior_sigma=prior_sigma,
            omega=omega,
        )

    return dict(
        run_seed=run_seed,
        data=data,
        q_family=q_family,
        omega=omega,
        history=history,
        reference=reference,
        phi=phi,
        nuisances=nuisances,
        nuisance_models=nuisance_models,
        ate_true=ate_true,
    )


def single_run_iv(
    *,
    dgp_fn_REGISTRATION: str,
    n: int,
    run_seed: int,
    dgp_kwargs: Dict[str, Any],
    nuisance_fn_REGISTRATION: str,
    nuisance_params: Dict[str, Any],
    phi_fn_REGISTRATION: str,
    phi_n_splits: Optional[int],
    q_family_cls_REGISTRATION: str,
    q_family_kwargs: Dict[str, Any],
    omega_est_REGISTRATION: str,
    omega_params: Optional[Dict[str, Any]] = {},
    prior_mu: float,
    prior_sigma: float,
    learn_cfg: Dict[str, Any],
    analytic_flag: bool,
    device: str,
    log: bool,
    plot: bool,
    progress: bool,
    diagnostics: bool,
) -> Dict[str, Any]:
    """
    Run a single IV GBI fit with local variational family construction.
    """
    dgp_fn = _resolve_from_registry(dgp_fn_REGISTRATION, _DGP_IV_REGISTRY, "dgp_fn_REGISTRATION")
    phi_fn = _resolve_from_registry(phi_fn_REGISTRATION, _PHI_IV_REGISTRY, "phi_fn_REGISTRATION")
    nuisance_fn = _resolve_from_registry(nuisance_fn_REGISTRATION, _NUISANCE_IV_REGISTRY, "nuisance_fn_REGISTRATION")
    omega_est = _resolve_from_registry(omega_est_REGISTRATION, _OMEGA_IV_REGISTRY, "omega_est_REGISTRATION")
    q_family_cls = _resolve_from_registry(q_family_cls_REGISTRATION, _VAR_FAMILY_REGISTRY, "q_family_cls_REGISTRATION")

    run_seeds = build_run_seeds(run_seed)
    dgp_seed = run_seeds["dgp"]
    nuisance_seed = run_seeds["nuisance"]
    split_seed = run_seeds["split"]
    omega_seed = run_seeds["omega"]
    vi_seed = run_seeds["vi"]

    data = dgp_fn(n=n, seed=dgp_seed, **dgp_kwargs)
    X, Z, A, Y = data["X"], data["Z"], data["A"], data["Y"]
    ate_true = data.get("ate_true", None)

    phi_out = compute_phi_and_nuisances_iv(
        X,
        Z,
        A,
        Y,
        phi_fn=phi_fn,
        n_splits=phi_n_splits,
        seed=split_seed,
        nuisance_seed=nuisance_seed,
        nuisance_fn=nuisance_fn,
        nuisance_params=nuisance_params,
        device=device,
        verbose=log,
        return_models=True,
    )
    phi = phi_out["phi"]
    nuisances = phi_out["nuisances"]
    nuisance_models = phi_out["nuisance_models"]

    loss = make_mse_from_phi(phi)

    bootstrap_kwargs = {
        "n": n,
        "seed": omega_seed,
        "X": X,
        "Z": Z,
        "A": A,
        "Y": Y,
        "phi_fn": phi_fn,
        "phi_n_splits": phi_n_splits,
        "nuisance_fn": nuisance_fn,
        "nuisance_params": nuisance_params,
    }

    omega = omega_est(phi, log=log, progress=progress, **omega_params, **bootstrap_kwargs)

    torch.manual_seed(vi_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(vi_seed)
    q_family = q_family_cls(device=device, **q_family_kwargs)

    learn_cfg_local = dict(learn_cfg)
    learn_cfg_local.pop("track_grads", None)
    history = fit_gbi(
        loss_fn=loss,
        n=n,
        q_family=q_family,
        prior_mu=prior_mu,
        prior_sigma=prior_sigma,
        omega=omega,
        **learn_cfg_local,
        progress=progress,
        plot=plot,
        track_grads=bool(diagnostics),
        device=device,
    )

    reference = None
    if analytic_flag:
        reference = compute_reference_quantities(
            phi,
            prior_mu=prior_mu,
            prior_sigma=prior_sigma,
            omega=omega,
        )

    return dict(
        run_seed=run_seed,
        data=data,
        q_family=q_family,
        omega=omega,
        history=history,
        reference=reference,
        phi=phi,
        nuisances=nuisances,
        nuisance_models=nuisance_models,
        ate_true=ate_true,
    )


def single_run_cate(
    *,
    dgp_fn_REGISTRATION: str,
    n: int,
    run_seed: int,
    dgp_kwargs: Dict[str, Any],
    nuisance_fn_REGISTRATION: str,
    nuisance_params: Dict[str, Any],
    phi_fn_REGISTRATION: str,
    phi_n_splits: Optional[int],
    q_family_cls_REGISTRATION: str,
    q_family_kwargs: Dict[str, Any],
    omega_est_REGISTRATION: str,
    omega_params: Optional[Dict[str, Any]] = {},
    prior_mu: float = 0.0,
    prior_sigma: float = 1.0,
    prior_lengthscale: Optional[float] = None,
    prior_variance: Optional[float] = None,
    learn_cfg: Dict[str, Any],
    analytic_flag: bool,
    device: str,
    log: bool,
    plot: bool,
    progress: bool,
    diagnostics: bool,
    cate_eval: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Run a single CATE GBI fit using a sparse GP variational family.
    """
    dgp_fn = _resolve_from_registry(dgp_fn_REGISTRATION, _DGP_REGISTRY, "dgp_fn_REGISTRATION")
    phi_fn = _resolve_from_registry(phi_fn_REGISTRATION, _PHI_REGISTRY, "phi_fn_REGISTRATION")
    nuisance_fn = _resolve_from_registry(nuisance_fn_REGISTRATION, _NUISANCE_REGISTRY, "nuisance_fn_REGISTRATION")
    omega_est = _resolve_from_registry(omega_est_REGISTRATION, _OMEGA_REGISTRY, "omega_est_REGISTRATION")
    q_family_cls = _resolve_from_registry(q_family_cls_REGISTRATION, _VAR_FAMILY_CATE_REGISTRY, "q_family_cls_REGISTRATION")

    run_seeds = build_run_seeds(run_seed)
    dgp_seed = run_seeds["dgp"]
    nuisance_seed = run_seeds["nuisance"]
    split_seed = run_seeds["split"]
    omega_seed = run_seeds["omega"]
    vi_seed = run_seeds["vi"]

    data = dgp_fn(n=n, seed=dgp_seed, **dgp_kwargs)
    X, A, Y = data["X"], data["A"], data["Y"]
    ate_true = data.get("ate_true", None)

    phi_out = compute_phi_and_nuisances(
        X,
        A,
        Y,
        phi_fn=phi_fn,
        n_splits=phi_n_splits,
        seed=split_seed,
        nuisance_seed=nuisance_seed,
        nuisance_fn=nuisance_fn,
        nuisance_params=nuisance_params,
        device=device,
        verbose=log,
        return_models=True,
    )
    phi = phi_out["phi"]
    nuisances = phi_out["nuisances"]
    nuisance_models = phi_out["nuisance_models"]

    mean_type = q_family_kwargs.pop("mean_type", "none")
    mean_alpha = q_family_kwargs.pop("mean_alpha", 1e-3)
    mean_value = q_family_kwargs.pop("mean_value", None)

    phi_np = phi.detach().cpu().numpy()
    mean_info = _fit_mean_function(
        X,
        phi_np,
        mean_type=mean_type,
        mean_alpha=mean_alpha,
        mean_value=mean_value,
    )
    mean_fn = mean_info["mean_fn"]
    phi_centered = phi_np - mean_fn(X)

    bootstrap_kwargs = {
        "n": n,
        "seed": omega_seed,
        "X": X,
        "A": A,
        "Y": Y,
        "phi_fn": phi_fn,
        "phi_n_splits": phi_n_splits,
        "nuisance_fn": nuisance_fn,
        "nuisance_params": nuisance_params,
    }
    omega = omega_est(phi, log=log, progress=progress, **omega_params, **bootstrap_kwargs)

    torch.manual_seed(vi_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(vi_seed)

    q_family = q_family_cls(X_train=X, device=device, **q_family_kwargs)

    learn_cfg_local = dict(learn_cfg)
    learn_cfg_local.pop("track_grads", None)

    if prior_lengthscale is None:
        prior_lengthscale = float(q_family_kwargs.get("lengthscale", 1.0))
    if prior_variance is None:
        prior_variance = float(q_family_kwargs.get("variance", prior_sigma ** 2))

    history = fit_cate_gbi(
        X=X,
        phi=phi_centered,
        q_family=q_family,
        prior_lengthscale=prior_lengthscale,
        prior_variance=prior_variance,
        prior_kernel=q_family_kwargs.get("kernel", None),
        prior_rq_alpha=q_family_kwargs.get("rq_alpha", 1.0),
        omega=omega,
        **learn_cfg_local,
        progress=progress,
        plot=plot,
        device=device,
    )

    reference = None
    reference_gp = None
    if analytic_flag:
        kernel = str(q_family_kwargs.get("kernel", "rbf"))
        lengthscale = float(q_family_kwargs.get("lengthscale", 1.0))
        variance = float(q_family_kwargs.get("variance", prior_variance))
        rq_alpha = float(q_family_kwargs.get("rq_alpha", 1.0))
        jitter = float(q_family_kwargs.get("jitter", 1e-6))
        omega_val = float(omega.detach().cpu().item()) if torch.is_tensor(omega) else float(omega)
        noise_var = 1.0 / omega_val
        reference_gp = GPPosterior(
            X,
            phi_centered,
            kernel=kernel,
            lengthscale=lengthscale,
            variance=variance,
            rq_alpha=rq_alpha,
            noise_var=noise_var,
            jitter=jitter,
            device=device,
        )

    X_eval = _resolve_cate_eval_X(data, cate_eval)
    cate_true_fn = data.get("cate_true", None)
    cate_true = cate_true_fn(X_eval) if callable(cate_true_fn) else None

    X_eval_t = _as_tensor(X_eval, device=torch.device(device), dtype=torch.float32)
    with torch.no_grad():
        preds_vi = q_family.predict_mean_var(X_eval_t)
        mean_vi = preds_vi["mean"].detach().cpu().numpy() + mean_fn(X_eval)
        var_vi = preds_vi["var"].detach().cpu().numpy()

    mean_analytic = None
    var_analytic = None
    if reference_gp is not None:
        with torch.no_grad():
            mean_ref, var_ref = reference_gp.predict(X_eval, return_var=True)
            mean_analytic = mean_ref.detach().cpu().numpy() + mean_fn(X_eval)
            var_analytic = var_ref.detach().cpu().numpy()

    cate_eval_out = dict(
        X_eval=X_eval,
        cate_true=cate_true,
        mean_vi=mean_vi,
        var_vi=var_vi,
        mean_analytic=mean_analytic,
        var_analytic=var_analytic,
        mean_info={k: v for k, v in mean_info.items() if k != "mean_fn"},
    )

    return dict(
        run_seed=run_seed,
        data=data,
        q_family=q_family,
        omega=omega,
        history=history,
        reference=reference,
        phi=phi,
        nuisances=nuisances,
        nuisance_models=nuisance_models,
        ate_true=ate_true,
        cate_eval=cate_eval_out,
    )



def flatten_one_level(cfg: dict) -> dict:
    out = {}
    for _, inner in cfg.items():
        if not isinstance(inner, dict):
            continue  # or: out[...] = inner if you want to keep non-dicts
        for k, v in inner.items():
            if k in out:
                raise KeyError(f"Duplicate key after flattening: {k!r}")
            out[k] = v
    return out


def single_run_from_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run a single experiment from a hierarchical config dict.
    """
    data_kind = str(config.get("data", {}).get("data_kind", "backdoor")).lower()
    flat = flatten_one_level(config)
    flat.pop("data_kind", None)
    if data_kind == "iv":
        return single_run_iv(**flat)
    if data_kind in {"backdoor_cate", "cate"}:
        return single_run_cate(**flat)
    return single_run(**flat)


## Single-run diagnostics

def diagnose_single_run(result: Dict[str, Any]) -> None:
    """
    Plot diagnostics for a single_run() result dict.
    """
    cate_eval = result.get("cate_eval")
    if cate_eval is not None:
        cate_true = cate_eval.get("cate_true")
        mean_vi = cate_eval.get("mean_vi")
        mean_analytic = cate_eval.get("mean_analytic")
        if cate_true is not None:
            if mean_vi is not None:
                mse_vi = _compute_cate_mse(mean_vi, cate_true)
                print(f"CATE MSE (VI): {mse_vi:.4f}")
            if mean_analytic is not None:
                mse_analytic = _compute_cate_mse(mean_analytic, cate_true)
                print(f"CATE MSE (Analytic): {mse_analytic:.4f}")
        X_eval = cate_eval.get("X_eval")
        if X_eval is not None and cate_true is not None:
            X_eval = np.asarray(X_eval)
            if X_eval.ndim == 2 and X_eval.shape[1] >= 1:
                order = np.argsort(X_eval[:, 0])
                plt.figure(figsize=(10, 4))
                plt.plot(X_eval[order, 0], np.asarray(cate_true)[order], "k-", lw=2, label="True CATE")
                if mean_analytic is not None:
                    plt.plot(X_eval[order, 0], np.asarray(mean_analytic)[order], "C3-", lw=2, label="Analytic")
                if mean_vi is not None:
                    plt.plot(X_eval[order, 0], np.asarray(mean_vi)[order], "C1-", lw=2, label="VI")
                plt.xlabel("X[:, 0]")
                plt.ylabel("CATE")
                plt.title("CATE fit diagnostics")
                plt.legend()
                plt.tight_layout()
                plt.show()
        return

    q = result["q_family"]
    history = result["history"]
    reference = result.get("reference", None)
    ate_true = result.get("ate_true", None)
    phi = result.get("phi", None)
    omega = result.get("omega", None)

    mu_post = q.mu.detach().item()
    sigma_post = torch.exp(q.log_sigma).detach().item()

    if reference is not None and ate_true is not None:
        mu_prior = reference.get("prior_mu", None)
        sigma_prior = reference.get("prior_sigma", None)
        post_mean_analytic = reference.get("post_mean", None)
        post_var_analytic = reference.get("post_var", None)

        if mu_prior is None or sigma_prior is None:
            mu_prior = 0.0
            sigma_prior = 1.0

        grid = np.linspace(
            min(mu_post - 4 * sigma_post, mu_prior - 4 * sigma_prior, ate_true - 1),
            max(mu_post + 4 * sigma_post, mu_prior + 4 * sigma_prior, ate_true + 1),
            400,
        )

        def normal_pdf(x, mu, sigma):
            return 1 / (np.sqrt(2 * np.pi) * sigma) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

        prior_pdf = normal_pdf(grid, mu_prior, sigma_prior)
        post_pdf = normal_pdf(grid, mu_post, sigma_post)
        post_pdf_analytic = normal_pdf(grid, post_mean_analytic, np.sqrt(post_var_analytic))

        plt.figure(figsize=(18, 9))
        plt.plot(grid, prior_pdf, "k--", label="Prior")
        plt.plot(grid, post_pdf, "C1-", lw=2, label="GBI Posterior")
        plt.plot(grid, post_pdf_analytic, "C3-", lw=2, label="Analytic Posterior")

        plt.axvline(ate_true, color="C2", linestyle="-", lw=2, label="True ATE")
        if phi is not None:
            ate_phi = float(phi.mean().item())
            plt.axvline(ate_phi, color="C1", linestyle="--", lw=2, label="Phi mean")
        plt.axvline(post_mean_analytic, color="C3", linestyle="-.", lw=2, label="Analytic Posterior Mean")

        plt.title("Posterior vs Prior vs True ATE")
        plt.xlabel(r"$\theta$")
        plt.ylabel("Density")
        plt.legend()
        plt.tight_layout()
        plt.show()

        print(
            f"True ATE: {ate_true:.3f}, Phi mean: {ate_phi:.3f},\n"
            f" COMPUTED: Posterior mean: {mu_post:.3f}, Posterior sd: {sigma_post:.3f},\n"
            f" ANALYTIC: Posterior mean: {post_mean_analytic:.3f}, Posterior sd: {np.sqrt(post_var_analytic):.3f}"
        )

    if history:
        plt.plot([abs(h["grad_mu"]) for h in history], label="abs grad mu")
        plt.plot([abs(h["grad_log_sigma"]) for h in history], label="abs grad log sigma")
        plt.legend()
        plt.title("Gradient updates over training")
        plt.xlabel("Iteration")
        plt.ylabel("Value")
        plt.tight_layout()
        plt.yscale("log")
        plt.show()

        plt.plot([h["delta_mu"] for h in history], label="delta mu")
        plt.plot([h["delta_log_sigma"] for h in history], label="delta log sigma")
        plt.legend()
        plt.title("Parameter updates over training")
        plt.xlabel("Iteration")
        plt.ylabel("Value")
        plt.tight_layout()
        plt.show()

        omega_val = float(omega.detach().cpu().item()) if torch.is_tensor(omega) else float(omega)
        n = int(result["data"]["X"].shape[0])
        plt.plot([h["data_term"] * omega_val * n for h in history], label="data term")
        plt.plot([h["kl_term"] for h in history], label="KL term")
        plt.plot([h["objective"] for h in history], label="objective")
        plt.legend()
        plt.title("Objective components over training")
        plt.xlabel("Iteration")
        plt.ylabel("Value")
        plt.tight_layout()
        plt.yscale("log")
        plt.show()



## Multi-run composer

## Storing and retrieving

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIGS_DIR = os.path.join(BASE_DIR, "configs")
SINGLE_RUN_CONFIG_DIR = os.path.join(CONFIGS_DIR, "single_runs")
EXPERIMENT_CONFIG_DIR = os.path.join(CONFIGS_DIR, "experiments")
RESULTS_DIR = os.path.join(BASE_DIR, "results")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def deep_merge_dicts(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge updates into base without mutating base.
    """
    out = dict(base)
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge_dicts(out[k], v)
        else:
            out[k] = v
    return out


def resolve_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(BASE_DIR, path)


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def save_json(path: str, data: Dict[str, Any]) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")

def load_config(filename: str) -> Dict[str, Any]:
    """
    Load a JSON config from the same directory as this script.
    """
    path = os.path.join(BASE_DIR, filename + ".json")
    return load_json(path)


def load_config_by_path(path: str) -> Dict[str, Any]:
    return load_json(resolve_path(path))


def derive_run_seed(seed0: int, dgp: str, method: str, rep: int, n: int) -> int:
    payload = _canonical_json({"seed0": seed0, "dgp": dgp, "method": method, "rep": rep, "n": n})
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def compute_run_id(experiment_id: str, run_config: Dict[str, Any]) -> str:
    payload = _canonical_json({"experiment_id": experiment_id, "run_config": run_config})
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()
    return "run_" + digest[:12]


def compute_ci(mean: float, sd: float, z: float = 1.96) -> Dict[str, float]:
    if np.isnan(mean) or np.isnan(sd):
        return dict(ci_lo=np.nan, ci_hi=np.nan, ci_len=np.nan)
    ci_lo = mean - z * sd
    ci_hi = mean + z * sd
    return dict(ci_lo=ci_lo, ci_hi=ci_hi, ci_len=ci_hi - ci_lo)


def extract_variational_summary(q_family: Any) -> Dict[str, float]:
    if isinstance(q_family, GaussianFamily):
        mu = float(q_family.mu.detach().cpu().item())
        sd = float(torch.exp(q_family.log_sigma).detach().cpu().item())
        return dict(post_mean=mu, post_sd=sd, post_var=sd ** 2)
    if isinstance(q_family, MixtureGaussianFamily):
        logits = q_family.logits.detach().cpu()
        mu = q_family.mu.detach().cpu()
        log_sigma = q_family.log_sigma.detach().cpu()
        weights = torch.softmax(logits, dim=0)
        mean = float(torch.sum(weights * mu).item())
        second_moment = torch.sum(weights * (torch.exp(2 * log_sigma) + mu ** 2))
        var = float(second_moment.item() - mean ** 2)
        sd = float(np.sqrt(max(var, 0.0)))
        return dict(post_mean=mean, post_sd=sd, post_var=var)
    return dict(post_mean=np.nan, post_sd=np.nan, post_var=np.nan) 


def compute_nuisance_diagnostics(
    A: Array,
    Y: Array,
    e_hat: Array,
    mu0_hat: Array,
    mu1_hat: Array,
) -> Dict[str, float]:
    if e_hat is None or mu0_hat is None or mu1_hat is None:
        return dict(
            nuisance_mse0=np.nan,
            nuisance_mse1=np.nan,
            propensity_mean=np.nan,
            propensity_std=np.nan,
            propensity_auc_like=np.nan,
            propensity_overlap=np.nan,
        )
    if np.any(A == 0):
        mse0 = float(np.mean((Y[A == 0] - mu0_hat[A == 0]) ** 2))
    else:
        mse0 = np.nan
    if np.any(A == 1):
        mse1 = float(np.mean((Y[A == 1] - mu1_hat[A == 1]) ** 2))
    else:
        mse1 = np.nan
    a1 = e_hat[A == 1]
    a0 = e_hat[A == 0]
    if len(a1) > 0 and len(a0) > 0:
        auc_like = float(np.mean(a1[:, None] > a0[None, :]))
    else:
        auc_like = np.nan
    overlap = float(np.mean((e_hat > 0.05) & (e_hat < 0.95)))
    return dict(
        nuisance_mse0=mse0,
        nuisance_mse1=mse1,
        propensity_mean=float(np.mean(e_hat)),
        propensity_std=float(np.std(e_hat)),
        propensity_auc_like=auc_like,
        propensity_overlap=overlap,
    )


def compute_nuisance_diagnostics_iv(
    Z: Array,
    A: Array,
    Y: Array,
    pi_hat: Array,
    m_y0_hat: Array,
    m_y1_hat: Array,
    m_a0_hat: Array,
    m_a1_hat: Array,
) -> Dict[str, float]:
    if (
        pi_hat is None
        or m_y0_hat is None
        or m_y1_hat is None
        or m_a0_hat is None
        or m_a1_hat is None
    ):
        return dict(
            nuisance_mse_z0=np.nan,
            nuisance_mse_z1=np.nan,
            instrument_mean=np.nan,
            instrument_std=np.nan,
            instrument_overlap=np.nan,
            first_stage_mean=np.nan,
        )
    mse_z0 = float(np.mean((Y[Z == 0] - m_y0_hat[Z == 0]) ** 2)) if np.any(Z == 0) else np.nan
    mse_z1 = float(np.mean((Y[Z == 1] - m_y1_hat[Z == 1]) ** 2)) if np.any(Z == 1) else np.nan
    instrument_overlap = float(np.mean((pi_hat > 0.05) & (pi_hat < 0.95)))
    first_stage = float(np.mean(m_a1_hat - m_a0_hat))
    return dict(
        nuisance_mse_z0=mse_z0,
        nuisance_mse_z1=mse_z1,
        instrument_mean=float(np.mean(pi_hat)),
        instrument_std=float(np.std(pi_hat)),
        instrument_overlap=instrument_overlap,
        first_stage_mean=first_stage,
    )


def compute_vi_diagnostics(history: Optional[List[Dict[str, Any]]]) -> Dict[str, float]:
    if not history:
        return dict(
            vi_last_objective=np.nan,
            vi_min_objective=np.nan,
            vi_last_grad_mu=np.nan,
            vi_last_grad_log_sigma=np.nan,
            vi_last_delta_mu=np.nan,
            vi_last_delta_log_sigma=np.nan,
        )
    last = history[-1]
    objectives = [h.get("objective") for h in history if h.get("objective") is not None]
    vi_min_objective = float(np.min(objectives)) if objectives else np.nan
    return dict(
        vi_last_objective=float(last.get("objective", np.nan)),
        vi_min_objective=vi_min_objective,
        vi_last_grad_mu=float(last.get("grad_mu", np.nan)),
        vi_last_grad_log_sigma=float(last.get("grad_log_sigma", np.nan)),
        vi_last_delta_mu=float(last.get("delta_mu", np.nan)),
        vi_last_delta_log_sigma=float(last.get("delta_log_sigma", np.nan)),
    )


def summarize_run(
    *,
    experiment_id: str,
    run_id: str,
    run_config: Dict[str, Any],
    result: Dict[str, Any],
    runtime_sec: float,
    artifacts_dir: str,
    run_config_path: str,
    rep: int,
) -> Dict[str, Any]:
    q_summary = extract_variational_summary(result["q_family"])
    ci_vi = compute_ci(q_summary["post_mean"], q_summary["post_sd"])

    reference = result.get("reference")
    if reference is not None:
        analytic_mean = float(reference.get("post_mean", np.nan))
        analytic_var = float(reference.get("post_var", np.nan))
        analytic_sd = float(np.sqrt(max(analytic_var, 0.0)))
    else:
        analytic_mean = np.nan
        analytic_var = np.nan
        analytic_sd = np.nan
    ci_analytic = compute_ci(analytic_mean, analytic_sd)

    phi = result["phi"]
    phi_mean = float(phi.mean().item())
    phi_var = float(torch.var(phi).item())
    omega = result["omega"]
    omega_val = float(omega.detach().cpu().item()) if torch.is_tensor(omega) else float(omega)

    data_kind = str(run_config.get("data", {}).get("data_kind", "backdoor")).lower()

    X = result["data"]["X"]
    A = result["data"]["A"]
    Y = result["data"]["Y"]
    nuisances = result.get("nuisances", {})
    if data_kind == "iv":
        Z = result["data"].get("Z")
        nuisance_diag = compute_nuisance_diagnostics_iv(
            Z=Z,
            A=A,
            Y=Y,
            pi_hat=nuisances.get("pi_hat"),
            m_y0_hat=nuisances.get("m_y0_hat"),
            m_y1_hat=nuisances.get("m_y1_hat"),
            m_a0_hat=nuisances.get("m_a0_hat"),
            m_a1_hat=nuisances.get("m_a1_hat"),
        )
    else:
        nuisance_diag = compute_nuisance_diagnostics(
            A=A,
            Y=Y,
            e_hat=nuisances.get("e_hat"),
            mu0_hat=nuisances.get("mu0_hat"),
            mu1_hat=nuisances.get("mu1_hat"),
        )
    vi_diag = compute_vi_diagnostics(result.get("history"))

    cate_eval = result.get("cate_eval", None)
    cate_eval_n = np.nan
    cate_mse_vi = np.nan
    cate_mse_analytic = np.nan
    if cate_eval is not None:
        cate_true = cate_eval.get("cate_true")
        if cate_true is not None:
            cate_eval_n = int(np.asarray(cate_true).shape[0])
            cate_mse_vi = _compute_cate_mse(cate_eval.get("mean_vi"), cate_true)
            cate_mse_analytic = _compute_cate_mse(cate_eval.get("mean_analytic"), cate_true)

    return dict(
        run_id=run_id,
        experiment_id=experiment_id,
        dgp=run_config["data"]["dgp_fn_REGISTRATION"],
        method=run_config["loss"]["phi_fn_REGISTRATION"],
        q_family_cls=run_config["var_family"]["q_family_cls_REGISTRATION"],
        omega_est=run_config["omega"]["omega_est_REGISTRATION"],
        phi_n_splits=run_config["loss"].get("phi_n_splits", None),
        analytic_flag=bool(run_config["analytic"].get("analytic_flag", False)),
        device=run_config["misc"].get("device", ""),
        rep=int(rep),
        run_seed=int(run_config["misc"]["run_seed"]),
        n=int(run_config["data"]["n"]),
        ate_true=float(result.get("ate_true", np.nan)) if result.get("ate_true", None) is not None else np.nan,
        post_mean_vi=float(q_summary["post_mean"]),
        post_var_vi=float(q_summary["post_var"]),
        post_sd_vi=float(q_summary["post_sd"]),
        ci95_lo_vi=float(ci_vi["ci_lo"]),
        ci95_hi_vi=float(ci_vi["ci_hi"]),
        ci95_len_vi=float(ci_vi["ci_len"]),
        post_mean_analytic=float(analytic_mean),
        post_var_analytic=float(analytic_var),
        post_sd_analytic=float(analytic_sd),
        ci95_lo_analytic=float(ci_analytic["ci_lo"]),
        ci95_hi_analytic=float(ci_analytic["ci_hi"]),
        ci95_len_analytic=float(ci_analytic["ci_len"]),
        phi_mean=phi_mean,
        phi_var=phi_var,
        omega=omega_val,
        cate_eval_n=cate_eval_n,
        cate_mse_vi=float(cate_mse_vi),
        cate_mse_analytic=float(cate_mse_analytic),
        runtime_sec=float(runtime_sec),
        artifacts_dir=artifacts_dir,
        run_config_path=run_config_path,
        status="ok",
        error_message="",
        **nuisance_diag,
        **vi_diag,
    )


def summarize_failed_run(
    *,
    experiment_id: str,
    run_id: str,
    run_config: Dict[str, Any],
    runtime_sec: float,
    artifacts_dir: str,
    run_config_path: str,
    rep: int,
    error_message: str,
) -> Dict[str, Any]:
    return dict(
        run_id=run_id,
        experiment_id=experiment_id,
        dgp=run_config["data"]["dgp_fn_REGISTRATION"],
        method=run_config["loss"]["phi_fn_REGISTRATION"],
        q_family_cls=run_config["var_family"]["q_family_cls_REGISTRATION"],
        omega_est=run_config["omega"]["omega_est_REGISTRATION"],
        phi_n_splits=run_config["loss"].get("phi_n_splits", None),
        analytic_flag=bool(run_config["analytic"].get("analytic_flag", False)),
        device=run_config["misc"].get("device", ""),
        rep=int(rep),
        run_seed=int(run_config["misc"]["run_seed"]),
        n=int(run_config["data"]["n"]),
        ate_true=np.nan,
        post_mean_vi=np.nan,
        post_var_vi=np.nan,
        post_sd_vi=np.nan,
        ci95_lo_vi=np.nan,
        ci95_hi_vi=np.nan,
        ci95_len_vi=np.nan,
        post_mean_analytic=np.nan,
        post_var_analytic=np.nan,
        post_sd_analytic=np.nan,
        ci95_lo_analytic=np.nan,
        ci95_hi_analytic=np.nan,
        ci95_len_analytic=np.nan,
        phi_mean=np.nan,
        phi_var=np.nan,
        omega=np.nan,
        cate_eval_n=np.nan,
        cate_mse_vi=np.nan,
        cate_mse_analytic=np.nan,
        nuisance_mse0=np.nan,
        nuisance_mse1=np.nan,
        propensity_mean=np.nan,
        propensity_std=np.nan,
        propensity_auc_like=np.nan,
        propensity_overlap=np.nan,
        nuisance_mse_z0=np.nan,
        nuisance_mse_z1=np.nan,
        instrument_mean=np.nan,
        instrument_std=np.nan,
        instrument_overlap=np.nan,
        first_stage_mean=np.nan,
        vi_last_objective=np.nan,
        vi_min_objective=np.nan,
        vi_last_grad_mu=np.nan,
        vi_last_grad_log_sigma=np.nan,
        vi_last_delta_mu=np.nan,
        vi_last_delta_log_sigma=np.nan,
        runtime_sec=float(runtime_sec),
        artifacts_dir=artifacts_dir,
        run_config_path=run_config_path,
        status="error",
        error_message=error_message,
    )


SUMMARY_COLUMNS = [
    "run_id",
    "experiment_id",
    "dgp",
    "method",
    "q_family_cls",
    "omega_est",
    "phi_n_splits",
    "analytic_flag",
    "device",
    "rep",
    "run_seed",
    "n",
    "ate_true",
    "post_mean_vi",
    "post_var_vi",
    "post_sd_vi",
    "ci95_lo_vi",
    "ci95_hi_vi",
    "ci95_len_vi",
    "post_mean_analytic",
    "post_var_analytic",
    "post_sd_analytic",
    "ci95_lo_analytic",
    "ci95_hi_analytic",
    "ci95_len_analytic",
    "phi_mean",
    "phi_var",
    "omega",
    "cate_eval_n",
    "cate_mse_vi",
    "cate_mse_analytic",
    "nuisance_mse0",
    "nuisance_mse1",
    "propensity_mean",
    "propensity_std",
    "propensity_auc_like",
    "propensity_overlap",
    "nuisance_mse_z0",
    "nuisance_mse_z1",
    "instrument_mean",
    "instrument_std",
    "instrument_overlap",
    "first_stage_mean",
    "vi_last_objective",
    "vi_min_objective",
    "vi_last_grad_mu",
    "vi_last_grad_log_sigma",
    "vi_last_delta_mu",
    "vi_last_delta_log_sigma",
    "runtime_sec",
    "artifacts_dir",
    "run_config_path",
    "status",
    "error_message",
]


def write_summary_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in SUMMARY_COLUMNS})


def append_failure_jsonl(path: str, entry: Dict[str, Any]) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=True))
        f.write("\n")


def save_artifacts(
    artifacts_dir: str,
    run_config: Dict[str, Any],
    result: Dict[str, Any],
) -> None:
    ensure_dir(artifacts_dir)
    save_json(os.path.join(artifacts_dir, "run_config.json"), run_config)

    arrays = {}
    phi = result.get("phi")
    if phi is not None:
        arrays["phi"] = phi.detach().cpu().numpy()

    nuisances = result.get("nuisances", {})
    for key in [
        "e_hat",
        "mu0_hat",
        "mu1_hat",
        "pi_hat",
        "m_y0_hat",
        "m_y1_hat",
        "m_a0_hat",
        "m_a1_hat",
    ]:
        if key in nuisances and nuisances[key] is not None:
            arrays[key] = np.asarray(nuisances[key])

    cate_eval = result.get("cate_eval")
    if cate_eval:
        for key, out_key in [
            ("X_eval", "cate_X_eval"),
            ("cate_true", "cate_true"),
            ("mean_vi", "cate_mean_vi"),
            ("var_vi", "cate_var_vi"),
            ("mean_analytic", "cate_mean_analytic"),
            ("var_analytic", "cate_var_analytic"),
        ]:
            if key in cate_eval and cate_eval[key] is not None:
                arrays[out_key] = np.asarray(cate_eval[key])

    q_family = result.get("q_family")
    if isinstance(q_family, GaussianFamily):
        arrays["q_mu"] = q_family.mu.detach().cpu().numpy()
        arrays["q_log_sigma"] = q_family.log_sigma.detach().cpu().numpy()
    elif isinstance(q_family, MixtureGaussianFamily):
        arrays["q_logits"] = q_family.logits.detach().cpu().numpy()
        arrays["q_mu"] = q_family.mu.detach().cpu().numpy()
        arrays["q_log_sigma"] = q_family.log_sigma.detach().cpu().numpy()

    if arrays:
        np.savez(os.path.join(artifacts_dir, "arrays.npz"), **arrays)

    history = result.get("history")
    if history:
        save_json(os.path.join(artifacts_dir, "history.json"), history)

    nuisance_models = result.get("nuisance_models")
    if nuisance_models:
        models_path = os.path.join(artifacts_dir, "nuisance_models.npz")
        if isinstance(nuisance_models, dict):
            np.savez(models_path, **{k: np.asarray(v) for k, v in nuisance_models.items()})
        elif isinstance(nuisance_models, list):
            bundle = {}
            for i, m in enumerate(nuisance_models):
                for k, v in m.items():
                    bundle[f"{k}_fold{i}"] = np.asarray(v)
            np.savez(models_path, **bundle)

    if q_family is not None:
        torch.save(q_family.state_dict(), os.path.join(artifacts_dir, "q_family_state.pt"))


def run_experiment_from_config(exp_cfg: Dict[str, Any]) -> Dict[str, Any]:
    t0_total = time.perf_counter()
    exp = exp_cfg.get("experiment", {})
    experiment_id = exp.get("experiment_id")
    if not experiment_id:
        raise ValueError("experiment.experiment_id is required")
    save_artifacts_flag = bool(exp.get("save_artifacts", False))
    overwrite = bool(exp.get("overwrite", False))
    exp_plot = bool(exp.get("plot", False))
    exp_log = bool(exp.get("log", False))
    exp_progress = bool(exp.get("progress", True))
    exp_diagnostics = bool(exp.get("diagnostics", True))
    failure_mode = str(exp.get("failure_mode", "fail_fast")).lower()
    if failure_mode not in {"continue", "fail_fast"}:
        raise ValueError("experiment.failure_mode must be 'continue' or 'fail_fast'")

    base_cfg_info = exp_cfg.get("base_single_run", {})
    base_cfg_path = base_cfg_info.get("config_path")
    if not base_cfg_path:
        raise ValueError("base_single_run.config_path is required")
    base_cfg = load_config_by_path(base_cfg_path)

    sweep = exp_cfg.get("sweep", {})
    dgp_list = sweep.get("dgp_list", [base_cfg["data"]["dgp_fn_REGISTRATION"]])
    method_list = sweep.get("method_list", [base_cfg["loss"]["phi_fn_REGISTRATION"]])
    n_list = sweep.get("n_list", [base_cfg["data"]["n"]])
    repetitions = int(sweep.get("repetitions", 1))
    seed0 = int(sweep.get("run_seed0", base_cfg["misc"].get("run_seed", 0)))

    exp_dir = os.path.join(RESULTS_DIR, experiment_id)
    if os.path.exists(exp_dir):
        if overwrite:
            shutil.rmtree(exp_dir)
        else:
            raise FileExistsError(
                f"Experiment folder already exists: {exp_dir}. Delete it or choose a new experiment_id."
            )

    ensure_dir(exp_dir)
    ensure_dir(os.path.join(exp_dir, "runs"))
    if save_artifacts_flag:
        ensure_dir(os.path.join(exp_dir, "artifacts"))

    save_json(os.path.join(exp_dir, "experiment_config.json"), exp_cfg)
    save_json(os.path.join(exp_dir, "base_single_run_config.json"), base_cfg)

    summary_rows = []
    failures_path = os.path.join(exp_dir, "failures.jsonl")
    run_specs = list(itertools.product(dgp_list, method_list, n_list, range(repetitions)))

    for dgp, method, n, rep in tqdm(run_specs, disable=not exp_progress, desc="Experiment runs"):
        run_seed = derive_run_seed(seed0, dgp=dgp, method=method, rep=rep, n=n)
        overrides = {
            "data": {
                "dgp_fn_REGISTRATION": dgp,
                "n": n,
            },
            "loss": {
                "phi_fn_REGISTRATION": method,
            },
            "misc": {
                "run_seed": run_seed,
                "plot": exp_plot,
                "log": exp_log,
                "progress": False,
                "diagnostics": exp_diagnostics,
            },
        }
        run_config = deep_merge_dicts(base_cfg, overrides)
        run_id = compute_run_id(experiment_id, run_config)

        run_cfg_path = os.path.join(exp_dir, "runs", f"{run_id}.json")
        save_json(run_cfg_path, run_config)

        save_this_artifact = bool(save_artifacts_flag and rep == 0)
        artifacts_dir = os.path.join(exp_dir, "artifacts", run_id) if save_this_artifact else ""
        t0 = time.perf_counter()
        try:
            result = single_run_from_config(run_config)
            runtime_sec = time.perf_counter() - t0

            summary = summarize_run(
                experiment_id=experiment_id,
                run_id=run_id,
                run_config=run_config,
                result=result,
                runtime_sec=runtime_sec,
                artifacts_dir=artifacts_dir,
                run_config_path=run_cfg_path,
                rep=rep,
            )
            summary_rows.append(summary)

            if save_this_artifact:
                save_artifacts(artifacts_dir, run_config, result)
        except Exception as exc:
            runtime_sec = time.perf_counter() - t0
            error_message = f"{type(exc).__name__}: {exc}"
            summary = summarize_failed_run(
                experiment_id=experiment_id,
                run_id=run_id,
                run_config=run_config,
                runtime_sec=runtime_sec,
                artifacts_dir=artifacts_dir,
                run_config_path=run_cfg_path,
                rep=rep,
                error_message=error_message,
            )
            summary_rows.append(summary)
            append_failure_jsonl(
                failures_path,
                dict(
                    experiment_id=experiment_id,
                    run_id=run_id,
                    run_config_path=run_cfg_path,
                    rep=int(rep),
                    error_message=error_message,
                    traceback=traceback.format_exc(),
                ),
            )
            if failure_mode == "fail_fast":
                raise

    summary_path = os.path.join(exp_dir, "summary.csv")
    write_summary_csv(summary_path, summary_rows)
    n_failures = sum(1 for row in summary_rows if row.get("status") != "ok")

    total_runtime_sec = time.perf_counter() - t0_total
    return dict(
        experiment_id=experiment_id,
        exp_dir=exp_dir,
        n_runs=len(summary_rows),
        summary_path=summary_path,
        failures_path=failures_path if n_failures else "",
        n_failures=n_failures,
        total_runtime_sec=total_runtime_sec,
    )


# ----------------------------------------------
# Post-processing / Output analysis


## Plots


## Tables


## Statistical tests


# ----------------------------------------------

if __name__ == "__main__":

    # Command-line interface
    # Example: python run1.py --experiment configs/experiments/experiment1_backdoor_ate.json
    if len(sys.argv) >= 3 and sys.argv[1] == "--experiment":
        exp_cfg = load_config_by_path(sys.argv[2])
        outcome = run_experiment_from_config(exp_cfg)
        print(f"Experiment complete: {outcome['experiment_id']} ({outcome['n_runs']} runs)")
        print(f"Summary: {outcome['summary_path']}")
        if outcome.get("n_failures", 0) > 0:
            print(f"Failures: {outcome['n_failures']} (see {outcome['failures_path']})")
        print(f"Total runtime (sec): {outcome['total_runtime_sec']:.2f}")
        raise SystemExit(0)

    # Example: python run1.py --single configs/single_runs/single_run_config_example.json
    if len(sys.argv) >= 3 and sys.argv[1] == "--single":
        cfg = load_config_by_path(sys.argv[2])
        result = single_run_from_config(cfg)
        diagnose_single_run(result)
        raise SystemExit(0)




    # # Example config run
    # cfg = load_config("single_run_config_example")
    # result = single_run_from_config(cfg)
    # diagnose_single_run(result)
