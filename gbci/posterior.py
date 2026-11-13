"""Gaussian generalized-Bayes posteriors for ATE and CATE.

The squared-error loss is scaled by omega / 2, equivalent to Gaussian
observations with variance 1 / omega. CATE predictions return latent-function
uncertainty, not the variance of a future pseudo-outcome.
"""

from __future__ import annotations

import math

import numpy as np
import torch


_Z = 1.959963984540054
_DTYPE = torch.float64


def _values(phi, omega, prior_mean, prior_sd):
    phi = np.asarray(phi, dtype=np.float64)
    assert phi.ndim == 1 and phi.size > 0 and np.all(np.isfinite(phi))
    assert np.isfinite(omega) and omega > 0
    assert np.isfinite(prior_mean) and np.isfinite(prior_sd) and prior_sd > 0
    return phi


def _summary(mean, variance):
    mean, variance = float(mean), float(variance)
    assert np.isfinite(mean) and np.isfinite(variance) and variance > 0
    sd = math.sqrt(variance)
    return {
        "post_mean": mean,
        "post_var": variance,
        "post_sd": sd,
        "ci_lo": mean - _Z * sd,
        "ci_hi": mean + _Z * sd,
        "ci_length": 2 * _Z * sd,
    }


def gaussian_ate(phi, omega, *, prior_mean=0.0, prior_sd=100.0):
    """Exact posterior for a Gaussian prior and quadratic pseudo-outcome loss."""
    phi = _values(phi, omega, prior_mean, prior_sd)
    precision = 1.0 / prior_sd**2 + len(phi) * omega
    mean = (prior_mean / prior_sd**2 + omega * phi.sum()) / precision
    return _summary(mean, 1.0 / precision)


def _x_array(X, dimension=None):
    X = np.asarray(X, dtype=np.float64)
    assert X.ndim == 2 and X.shape[0] > 0 and X.shape[1] > 0
    assert np.all(np.isfinite(X))
    assert dimension is None or X.shape[1] == dimension
    return X


def _kernel(X, Y, kernel, lengthscale, variance):
    d2 = torch.clamp(((X[:, None, :] - Y[None, :, :]) ** 2).sum(dim=2), min=0.0)
    if kernel == "rbf":
        return variance * torch.exp(-0.5 * d2 / lengthscale**2)
    # The tiny offset keeps derivatives finite at coincident inducing points.
    r = torch.sqrt(d2 + 1e-30) / lengthscale
    s = math.sqrt(5.0) * r
    return variance * (1.0 + s + s**2 / 3.0) * torch.exp(-s)


class _ExactGP:
    def __init__(self, X, y, omega, mean, kernel, lengthscale, variance, jitter):
        self.X = X
        self.mean = mean
        self.kernel, self.lengthscale, self.variance = kernel, lengthscale, variance
        K = _kernel(X, X, kernel, lengthscale, variance)
        self.L = torch.linalg.cholesky(K + (1.0 / omega + jitter) * torch.eye(len(X), dtype=_DTYPE))
        self.alpha = torch.cholesky_solve((y - mean)[:, None], self.L)

    def predict(self, X_new):
        X_new = torch.as_tensor(_x_array(X_new, self.X.shape[1]), dtype=_DTYPE)
        with torch.no_grad():
            K = _kernel(X_new, self.X, self.kernel, self.lengthscale, self.variance)
            mean = self.mean + (K @ self.alpha).squeeze(1)
            v = torch.linalg.solve_triangular(self.L, K.T, upper=False)
            var = self.variance - (v**2).sum(dim=0)
            assert bool(torch.all(var > -1e-8))
            var = torch.clamp(var, min=0.0)
        return mean.numpy(), var.numpy()


class _VariationalGP:
    def __init__(self, Z, q_mean, q_cholesky, mean, kernel, lengthscale, variance, jitter):
        self.Z, self.q_mean, self.q_cholesky = Z, q_mean, q_cholesky
        self.mean, self.kernel, self.lengthscale = mean, kernel, lengthscale
        self.variance, self.jitter = variance, jitter
        Kzz = _kernel(Z, Z, kernel, lengthscale, variance)
        self.L = torch.linalg.cholesky(Kzz + jitter * torch.eye(len(Z), dtype=_DTYPE))

    def predict(self, X_new):
        X_new = torch.as_tensor(_x_array(X_new, self.Z.shape[1]), dtype=_DTYPE)
        with torch.no_grad():
            k = _kernel(self.Z, X_new, self.kernel, self.lengthscale, self.variance)
            A = torch.linalg.solve_triangular(self.L, k, upper=False)
            mean = self.mean + A.T @ self.q_mean
            var = self.variance - (A**2).sum(dim=0) + ((self.q_cholesky.T @ A) ** 2).sum(dim=0)
            assert bool(torch.all(var > -1e-8))
            var = torch.clamp(var, min=0.0)
        return mean.numpy(), var.numpy()


def fit_cate_gp(
    X,
    phi,
    omega,
    *,
    inducing_points=20,
    kernel="matern52",
    lengthscale=2.0,
    variance=2.0,
    jitter=1e-4,
    epochs=2000,
    learning_rate=0.03,
    seed=0,
):
    """Fit exact and inducing-point variational GPs with a shared constant mean.

    The variational GP optimizes inducing locations and a full Gaussian q in
    whitened inducing coordinates using the full-data expected squared loss.
    """
    X = _x_array(X)
    phi = _values(phi, omega, 0.0, 1.0)
    assert len(X) == len(phi)
    assert kernel in ("rbf", "matern52")
    assert all(np.isfinite(v) and v > 0 for v in (lengthscale, variance, jitter, learning_rate))
    assert isinstance(inducing_points, int) and inducing_points > 0
    assert isinstance(epochs, int) and epochs > 0
    x = torch.as_tensor(X, dtype=_DTYPE)
    y = torch.as_tensor(phi, dtype=_DTYPE)
    mean = float(phi.mean())
    exact = _ExactGP(x, y, omega, mean, kernel, lengthscale, variance, jitter)
    m = min(inducing_points, len(X))
    rng = np.random.default_rng(seed)
    Z = torch.nn.Parameter(x[rng.choice(len(X), m, replace=False)].clone())
    q_mean = torch.nn.Parameter(torch.zeros(m, dtype=_DTYPE))
    # log diagonal zero initializes the whitened covariance to identity.
    raw = torch.nn.Parameter(torch.zeros((m, m), dtype=_DTYPE))
    optimizer = torch.optim.Adam([Z, q_mean, raw], lr=learning_rate)
    eye = torch.eye(m, dtype=_DTYPE)
    for _ in range(epochs):
        Lq = torch.tril(raw, diagonal=-1) + torch.diag(torch.exp(torch.diagonal(raw)))
        L = torch.linalg.cholesky(_kernel(Z, Z, kernel, lengthscale, variance) + jitter * eye)
        A = torch.linalg.solve_triangular(
            L, _kernel(Z, x, kernel, lengthscale, variance), upper=False
        )
        pred = mean + A.T @ q_mean
        pred_var = variance - (A**2).sum(dim=0) + ((Lq.T @ A) ** 2).sum(dim=0)
        kl = 0.5 * ((Lq**2).sum() + (q_mean**2).sum() - m - 2 * torch.diagonal(raw).sum())
        objective = 0.5 * omega * (((y - pred) ** 2 + pred_var).sum()) + kl
        assert bool(torch.isfinite(objective))
        optimizer.zero_grad()
        objective.backward()
        optimizer.step()
    with torch.no_grad():
        Lq = torch.tril(raw, diagonal=-1) + torch.diag(torch.exp(torch.diagonal(raw)))
        approximate = _VariationalGP(
            Z.detach().clone(),
            q_mean.detach().clone(),
            Lq.detach().clone(),
            mean,
            kernel,
            lengthscale,
            variance,
            jitter,
        )
    return {"analytic": exact, "variational": approximate}
