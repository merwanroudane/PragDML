"""Outcome-regression learners (the ``g`` head).

A learner is an object that, once fitted on a training fold, is **callable** as
``learner(d, X) -> (n,)``. That is the only contract the functionals in
:mod:`pragdml.functionals` need, because they must evaluate ``g`` at
counterfactual treatment values (``h(1, Z)``, ``h(0, Z)``, ``h(D + eps, Z)``).
Learners that are linear in the treatment also expose an exact
``deriv_d(d, X)``, which the average-derivative functional of Eq. (67) picks up
automatically.

The population object being targeted is ``g_0`` of Definition 3: the ``L2(P)``
projection of ``Y`` onto the information space ``H``. For
:class:`LinearOutcomeLearner` over a :class:`~pragdml.features.FeatureMap`, that
projection is computed **exactly** by least squares.

Compatibility map
-----------------
========================================  =========================================
Paper                                      Here
========================================  =========================================
Definition 3, ``g_0 = argmin_H E(Y-g)^2``   :class:`LinearOutcomeLearner`
Sec. 3.1, population risk ``R(g)``          :func:`square_risk`
Eq. (28), excess risk identity on ``H``     :func:`square_risk` + Lemma 2 tests
Eq. (68), ``g = D g' b(X) + d' b(X)``       ``LinearOutcomeLearner(VaryingCoefficientFeatures())``
Sec. 3.2, regularised ERM over a sieve      ``ridge=`` / ``basis=``
========================================  =========================================

Author
------
Dr Merwan Roudane <merwanroudane920@gmail.com>
https://github.com/merwanroudane
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from sklearn.base import clone

from .features import FeatureMap, VaryingCoefficientFeatures
from .functionals import Functional

__all__ = [
    "OutcomeLearner",
    "LinearOutcomeLearner",
    "SklearnOutcomeLearner",
    "square_risk",
    "solve_scaled_normal_equations",
]


def solve_scaled_normal_equations(
    gram: np.ndarray, rhs: np.ndarray, ridge: float = 0.0, rcond: float = 1e-12
) -> np.ndarray:
    """Solve ``(Q + ridge I) c = r`` with column rescaling for conditioning.

    The rescaling is an exact reparameterisation, not an approximation. Writing
    ``Phi_tilde = Phi S`` for a diagonal ``S`` of column RMS values gives
    ``Q_tilde = S Q S`` and ``r_tilde = S r`` (the latter because ``m`` is linear in
    its second argument), and the solutions satisfy ``c = S c_tilde``. Only the
    conditioning of the linear system changes, which matters when features mix
    scales such as dollars of income with binary indicators.

    Parameters
    ----------
    gram : ndarray of shape (p, p)
        ``Q``, a symmetric positive-semidefinite Gram matrix.
    rhs : ndarray of shape (p,)
        ``r``.
    ridge : float, default 0.0
        Penalty applied **in the rescaled space** (the standard convention, and
        the reason rescaling is done before penalising).
    rcond : float, default 1e-12
        Cutoff passed to the pseudo-inverse fallback.

    Returns
    -------
    ndarray of shape (p,)
        The coefficient vector ``c``.

    Examples
    --------
    >>> import numpy as np
    >>> from pragdml.learners import solve_scaled_normal_equations
    >>> Q = np.array([[2.0, 0.0], [0.0, 8.0]]); r = np.array([2.0, 4.0])
    >>> solve_scaled_normal_equations(Q, r)
    array([1. , 0.5])
    """
    gram = np.asarray(gram, dtype=np.float64)
    rhs = np.asarray(rhs, dtype=np.float64).reshape(-1)
    p = rhs.shape[0]
    if gram.shape != (p, p):
        raise ValueError(f"gram must be ({p}, {p}), got {gram.shape}")
    scale = np.sqrt(np.clip(np.diag(gram), 0.0, None))
    scale[~np.isfinite(scale) | (scale <= 0.0)] = 1.0
    inv = 1.0 / scale
    q_t = gram * np.outer(inv, inv)
    r_t = rhs * inv
    if ridge > 0.0:
        q_t = q_t + float(ridge) * np.eye(p)
    try:
        c_t = np.linalg.solve(q_t, r_t)
        if not np.all(np.isfinite(c_t)):
            raise np.linalg.LinAlgError("non-finite solution")
    except np.linalg.LinAlgError:
        c_t = np.linalg.pinv(q_t, rcond=rcond) @ r_t
    return c_t * inv


def square_risk(learner, Y: np.ndarray, D: np.ndarray, X: np.ndarray) -> float:
    """Empirical square risk ``P_n (Y - g(W))^2`` (Sec. 3.1, ``R(g)``).

    On the space ``H`` the excess of this risk over its minimum equals the squared
    ``L2(P)`` prediction error (Lemma 2, Eq. 28), which is why the paper can place
    its product-rate condition directly on risks.

    Examples
    --------
    >>> import numpy as np
    >>> from pragdml.learners import square_risk
    >>> g = lambda d, X: d
    >>> square_risk(g, np.array([1.0, 2.0]), np.array([1.0, 1.0]), np.zeros((2, 1)))
    0.5
    """
    Y = np.asarray(Y, dtype=np.float64).reshape(-1)
    pred = np.asarray(learner(np.asarray(D, dtype=np.float64), X), dtype=np.float64)
    return float(np.mean((Y - pred.reshape(-1)) ** 2))


class OutcomeLearner(ABC):
    """Base class for the ``g`` head.

    Subclasses must be callable as ``self(d, X)`` after :meth:`fit`.
    """

    @abstractmethod
    def fit(
        self,
        Y: np.ndarray,
        D: np.ndarray,
        X: np.ndarray,
        functional: Optional[Functional] = None,
    ) -> "OutcomeLearner":
        """Fit on a training fold. ``functional`` is accepted for API symmetry."""

    @abstractmethod
    def __call__(self, d: np.ndarray, X: np.ndarray) -> np.ndarray:
        """Evaluate the fitted regression at ``(d, X)``."""

    def clone(self) -> "OutcomeLearner":
        """An unfitted copy, so each fold trains from scratch."""
        raise NotImplementedError

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{type(self).__name__}()"


class LinearOutcomeLearner(OutcomeLearner):
    """Least squares over a :class:`~pragdml.features.FeatureMap`.

    Computes the **exact** projection ``g_0`` of Definition 3 onto
    ``H = span(phi)``, which is what makes the excess-risk identity of Eq. (28)
    hold to machine precision. With
    :class:`~pragdml.features.VaryingCoefficientFeatures` this is the paper's own
    outcome architecture, Eq. (68).

    Parameters
    ----------
    features : FeatureMap, optional
        Defaults to ``VaryingCoefficientFeatures()``.
    ridge : float, default 0.0
        Ridge penalty in the column-rescaled space. ``0.0`` gives plain OLS (via
        a pseudo-inverse fallback, so a rank-deficient design is handled rather
        than raising). Use a small positive value for a high-dimensional
        embedding basis; note that any ``ridge > 0`` makes the fit a
        *regularised* ERM over a shrunken subset of ``H`` (Sec. 3.2), so the
        fitted object converges to a penalised projection, not to ``g_0`` itself.

    Attributes
    ----------
    coef_ : ndarray of shape (n_features,)
        Fitted coefficients.
    features_ : FeatureMap
        The fitted feature map.

    References
    ----------
    Aradillas Fernandez et al. (2026), Definition 3 and Eq. (68),
    arXiv:2610.01935.

    Examples
    --------
    >>> import numpy as np
    >>> from pragdml.learners import LinearOutcomeLearner
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(200, 2)); D = rng.normal(size=200)
    >>> Y = 2.0 * D + X[:, 0]
    >>> g = LinearOutcomeLearner().fit(Y, D, X)
    >>> float(np.round(np.mean(g.deriv_d(D, X)), 6))
    2.0
    """

    def __init__(self, features: Optional[FeatureMap] = None, ridge: float = 0.0) -> None:
        self.features = features
        self.ridge = float(ridge)
        if self.ridge < 0.0:
            raise ValueError(f"ridge must be >= 0, got {ridge}")

    def clone(self) -> "LinearOutcomeLearner":
        return LinearOutcomeLearner(features=self.features, ridge=self.ridge)

    def fit(
        self,
        Y: np.ndarray,
        D: np.ndarray,
        X: np.ndarray,
        functional: Optional[Functional] = None,
    ) -> "LinearOutcomeLearner":
        Y = np.asarray(Y, dtype=np.float64).reshape(-1)
        D = np.asarray(D, dtype=np.float64).reshape(-1)
        fm = self.features if self.features is not None else VaryingCoefficientFeatures()
        self.features_ = fm.fit(D, X)
        phi = self.features_.transform(D, X)
        if phi.shape[0] != Y.shape[0]:
            raise ValueError(
                f"feature map produced {phi.shape[0]} rows for {Y.shape[0]} outcomes"
            )
        n = phi.shape[0]
        gram = phi.T @ phi / n
        rhs = phi.T @ Y / n
        self.coef_ = solve_scaled_normal_equations(gram, rhs, ridge=self.ridge)
        return self

    def __call__(self, d: np.ndarray, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "coef_"):
            raise RuntimeError("LinearOutcomeLearner is not fitted; call .fit() first.")
        return self.features_.transform(d, X) @ self.coef_

    def deriv_d(self, d: np.ndarray, X: np.ndarray) -> np.ndarray:
        """Exact ``d/dd g(d, X)``; picked up automatically by ``AverageDerivative``."""
        if not hasattr(self, "coef_"):
            raise RuntimeError("LinearOutcomeLearner is not fitted; call .fit() first.")
        return self.features_.deriv_d(d, X) @ self.coef_

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"LinearOutcomeLearner(features={self.features!r}, ridge={self.ridge!r})"


class SklearnOutcomeLearner(OutcomeLearner):
    """Wrap any scikit-learn regressor, fitted on ``[d, X]``.

    This is the route to random forests, gradient boosting and neural nets for the
    outcome head, which is what makes the first stage "modern ML" in the sense of
    the paper. Note the two caveats below; they are real and the class warns about
    the second.

    Parameters
    ----------
    estimator : sklearn regressor
        Anything with ``fit(Z, y)`` and ``predict(Z)``.

    Warnings
    --------
    * The fitted function is **not** in the span of an explicit feature map, so it
      targets the projection of ``Y`` onto whatever space the learner can express.
      That space is still the ``H`` of Definition 3 for the purposes of the theory,
      but it is implicit, so the exact excess-risk identity of Eq. (28) can no
      longer be asserted numerically -- only the product-rate condition of
      Eq. (32) is checkable, via held-out loss.
    * A piecewise-constant learner (tree, forest, boosting) has **zero derivative
      almost everywhere**, so it is inadmissible for
      :class:`~pragdml.functionals.AverageDerivative`. A warning is raised at fit
      time when that combination is requested. For a continuous treatment prefer
      :class:`LinearOutcomeLearner` with a spline or embedding basis, which is
      what the paper's Eq. (68) does.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.linear_model import LinearRegression
    >>> from pragdml.learners import SklearnOutcomeLearner
    >>> rng = np.random.default_rng(1)
    >>> X = rng.normal(size=(100, 2)); D = rng.integers(0, 2, 100).astype(float)
    >>> Y = 1.5 * D + X[:, 1]
    >>> g = SklearnOutcomeLearner(LinearRegression()).fit(Y, D, X)
    >>> float(np.round(np.mean(g(np.ones(100), X) - g(np.zeros(100), X)), 6))
    1.5
    """

    _TREE_HINTS = (
        "Forest", "Tree", "Boosting", "GBM", "LGBM", "XGB", "Bagging", "ExtraTrees",
    )

    def __init__(self, estimator) -> None:
        if not (hasattr(estimator, "fit") and hasattr(estimator, "predict")):
            raise TypeError(
                "`estimator` must be a scikit-learn style regressor with "
                "fit/predict methods."
            )
        self.estimator = estimator

    def clone(self) -> "SklearnOutcomeLearner":
        return SklearnOutcomeLearner(clone(self.estimator))

    def fit(
        self,
        Y: np.ndarray,
        D: np.ndarray,
        X: np.ndarray,
        functional: Optional[Functional] = None,
    ) -> "SklearnOutcomeLearner":
        Y = np.asarray(Y, dtype=np.float64).reshape(-1)
        D = np.asarray(D, dtype=np.float64).reshape(-1)
        if functional is not None and getattr(functional, "requires_smooth_in_d", False):
            label = type(self.estimator).__name__
            if any(hint in label for hint in self._TREE_HINTS):
                warnings.warn(
                    f"{label} is piecewise constant in D, so its derivative with "
                    f"respect to D is zero almost everywhere and "
                    f"{type(functional).__name__} cannot be estimated with it. Use "
                    "pragdml.learners.LinearOutcomeLearner with a spline or embedding "
                    "basis (the paper's Eq. 68 architecture) instead.",
                    RuntimeWarning,
                    stacklevel=2,
                )
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(self._design(D, X), Y)
        return self

    @staticmethod
    def _design(d: np.ndarray, X: np.ndarray) -> np.ndarray:
        d = np.asarray(d, dtype=np.float64).reshape(-1, 1)
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X[:, None]
        return np.hstack([d, X])

    def __call__(self, d: np.ndarray, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "estimator_"):
            raise RuntimeError("SklearnOutcomeLearner is not fitted; call .fit() first.")
        return np.asarray(
            self.estimator_.predict(self._design(d, X)), dtype=np.float64
        ).reshape(-1)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"SklearnOutcomeLearner(estimator={self.estimator!r})"
