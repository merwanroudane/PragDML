"""Feature maps: the information space ``H`` made concrete.

Definition 3 of the paper defines the pipeline targets as **projections onto a
common closed linear space** ``H``:

.. math::

    g_0 = \\arg\\min_{g \\in H} E(Y - g)^2, \\qquad
    \\alpha_0 = \\arg\\min_{\\alpha \\in H} E[\\alpha^2 - 2 m(O, \\alpha)].

When ``H`` is the span of an explicit feature map ``phi(d, x)``, both projections
have **closed forms** -- ordinary least squares for ``g_0`` and the Riesz normal
equations for ``alpha_0``. That is the single most useful structural fact in the
paper for a software implementation: it makes ``g_0`` and ``alpha_0`` exactly
computable rather than merely approximable, so Lemma 2's excess-risk identities
(Eqs. 28-29) hold to machine precision and can be asserted in tests.

The paper's own application uses the varying-coefficient architecture of
Eqs. (68)-(69),

.. math::

    g_j(X, D) = D \\cdot \\gamma_j' b_j(X) + \\delta_j' b_j(X), \\qquad
    \\alpha_j(X, D) = D \\cdot \\tilde\\gamma_j' b_j(X) + \\tilde\\delta_j' b_j(X),

which is :class:`VaryingCoefficientFeatures` with ``phi(d, x) = [d * b(x), b(x)]``.
The coefficients are **not** constrained to be equal across the two heads: Sec. 5.2
says so explicitly, and Sec. 5.3 shows them selecting different candidates.

Compatibility map
-----------------
=========================================  ========================================
Paper                                      Here
=========================================  ========================================
Definition 3, the space ``H``               :class:`FeatureMap` span
Eqs. (68)-(69), varying coefficients        :class:`VaryingCoefficientFeatures`
Eq. (67)/(70), derivative in ``d``          :meth:`FeatureMap.deriv_d`
Sec. 3.2, finite dictionary / sieve         ``basis=`` any sklearn transformer
=========================================  ========================================

Author
------
Dr Merwan Roudane <merwanroudane920@gmail.com>
https://github.com/merwanroudane
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from sklearn.base import clone

__all__ = [
    "FeatureMap",
    "VaryingCoefficientFeatures",
    "PolynomialTreatmentFeatures",
    "AdditiveFeatures",
]


def _check_dt(d: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    d = np.asarray(d, dtype=np.float64).reshape(-1)
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X[:, None]
    if d.shape[0] != X.shape[0]:
        raise ValueError(
            f"d has {d.shape[0]} rows but X has {X.shape[0]}; they must match."
        )
    return d, X


class FeatureMap(ABC):
    """Base class for a feature map ``phi(d, x)`` spanning the space ``H``.

    A feature map is fitted **once per training fold** (never on the evaluation
    fold), then evaluated at arbitrary treatment values -- which is what the
    functionals need in order to apply ``m`` to counterfactual arguments.

    Attributes
    ----------
    n_features_ : int
        Set by :meth:`fit`.
    """

    n_features_: int

    @abstractmethod
    def fit(self, D: np.ndarray, X: np.ndarray) -> "FeatureMap":
        """Learn any basis parameters from the **training fold only**."""

    @abstractmethod
    def transform(self, d: np.ndarray, X: np.ndarray) -> np.ndarray:
        """Evaluate ``phi(d, X)``, returning ``(n, n_features_)``."""

    @abstractmethod
    def deriv_d(self, d: np.ndarray, X: np.ndarray) -> np.ndarray:
        """Evaluate ``d/dd phi(d, X)`` exactly, returning ``(n, n_features_)``."""

    def _check_fitted(self) -> None:
        if not hasattr(self, "n_features_"):
            raise RuntimeError(
                f"{type(self).__name__} is not fitted; call .fit(D, X) first."
            )

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{type(self).__name__}()"


class _BasisHolder:
    """Mixin handling the optional sklearn transformer used as ``b(x)``."""

    def _init_basis(self, basis, include_intercept: bool) -> None:
        self.basis = basis
        self.include_intercept = bool(include_intercept)

    def _fit_basis(self, X: np.ndarray) -> int:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X[:, None]
        if self.basis is None:
            self.basis_ = None
            width = X.shape[1]
        else:
            # A "frozen" basis is one that is already fitted and must not be
            # cloned: sklearn's clone() recursively clones estimator-valued
            # params, which would hand us back an *unfitted* copy of whatever
            # model the basis wraps. pragdml.riesz._LeafOneHot is the case in
            # point -- it carries an already-grown forest.
            if getattr(self.basis, "_pragdml_frozen", False):
                self.basis_ = self.basis
            else:
                self.basis_ = clone(self.basis)
                self.basis_.fit(X)
            width = np.asarray(self.basis_.transform(X[:1])).shape[1]
        return width + (1 if self.include_intercept else 0)

    def _b(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X[:, None]
        b = X if self.basis_ is None else np.asarray(
            self.basis_.transform(X), dtype=np.float64
        )
        if b.ndim == 1:
            b = b[:, None]
        if self.include_intercept:
            b = np.hstack([np.ones((b.shape[0], 1), dtype=np.float64), b])
        return np.ascontiguousarray(b, dtype=np.float64)


class VaryingCoefficientFeatures(FeatureMap, _BasisHolder):
    """``phi(d, x) = [d * b(x), b(x)]`` -- the paper's Eqs. (68)-(69) architecture.

    Linear in the treatment, so the average-derivative functional of Eq. (67) has
    an **exact** derivative (no numerical differencing), and the ATE contrast is
    exact as well. This is the architecture the paper's own application uses, and
    it is the default throughout :mod:`pragdml`.

    Parameters
    ----------
    basis : sklearn transformer, optional
        Any transformer implementing ``fit``/``transform`` to build ``b(x)`` --
        e.g. ``PCA(n_components=20)`` for an embedding, ``SplineTransformer(...)``
        for a sieve, ``PolynomialFeatures(2)``, or a ``Pipeline`` of them.
        ``None`` (default) uses ``X`` itself.
    include_intercept : bool, default True
        Prepend a constant to ``b(x)``. Keep it on: without it the model cannot
        represent a constant treatment effect, and Sec. 5 of the paper's
        specification includes one. A missing constant silently biases
        everything (see the faithfulness traps in the methods page).

    Notes
    -----
    The span of this map is the ``H`` of Definition 3, so for a fitted
    :class:`~pragdml.riesz.LinearRieszLearner` and an OLS outcome head the fitted
    nuisances are the **exact** ``L2`` projections ``(g_0, alpha_0)`` restricted to
    the training fold's empirical measure.

    References
    ----------
    Aradillas Fernandez et al. (2026), Eqs. (68)-(69), arXiv:2610.01935.

    Examples
    --------
    >>> import numpy as np
    >>> from pragdml.features import VaryingCoefficientFeatures
    >>> X = np.array([[2.0], [3.0]]); D = np.array([1.0, 0.0])
    >>> fm = VaryingCoefficientFeatures().fit(D, X)
    >>> fm.transform(D, X)
    array([[1., 2., 1., 2.],
           [0., 0., 1., 3.]])
    >>> fm.deriv_d(D, X)
    array([[1., 2., 0., 0.],
           [1., 3., 0., 0.]])
    """

    def __init__(self, basis=None, include_intercept: bool = True) -> None:
        self._init_basis(basis, include_intercept)

    def fit(self, D: np.ndarray, X: np.ndarray) -> "VaryingCoefficientFeatures":
        width = self._fit_basis(X)
        self.n_features_ = 2 * width
        self._width_ = width
        return self

    def transform(self, d: np.ndarray, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        d, X = _check_dt(d, X)
        b = self._b(X)
        return np.hstack([d[:, None] * b, b])

    def deriv_d(self, d: np.ndarray, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        d, X = _check_dt(d, X)
        b = self._b(X)
        return np.hstack([b, np.zeros_like(b)])

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"VaryingCoefficientFeatures(basis={self.basis!r}, "
            f"include_intercept={self.include_intercept!r})"
        )


class PolynomialTreatmentFeatures(FeatureMap, _BasisHolder):
    """``phi(d, x) = [b(x), d b(x), ..., d^degree b(x)]``.

    Generalises :class:`VaryingCoefficientFeatures` (which is ``degree=1``) to a
    treatment response that is polynomial rather than linear in ``d``, while
    keeping the exact derivative that Eq. (67) needs. Useful when the
    average-derivative target should not be forced to be constant in ``d``.

    Parameters
    ----------
    degree : int, default 2
        Highest power of ``d``. ``degree=1`` reproduces the paper's architecture.
    basis : sklearn transformer, optional
        As in :class:`VaryingCoefficientFeatures`.
    include_intercept : bool, default True
        As in :class:`VaryingCoefficientFeatures`.

    Examples
    --------
    >>> import numpy as np
    >>> from pragdml.features import PolynomialTreatmentFeatures
    >>> X = np.ones((1, 1)); D = np.array([2.0])
    >>> fm = PolynomialTreatmentFeatures(degree=2, include_intercept=False).fit(D, X)
    >>> fm.transform(D, X)
    array([[1., 2., 4.]])
    >>> fm.deriv_d(D, X)
    array([[0., 1., 4.]])
    """

    def __init__(
        self, degree: int = 2, basis=None, include_intercept: bool = True
    ) -> None:
        if int(degree) < 1:
            raise ValueError(f"degree must be >= 1, got {degree}")
        self.degree = int(degree)
        self._init_basis(basis, include_intercept)

    def fit(self, D: np.ndarray, X: np.ndarray) -> "PolynomialTreatmentFeatures":
        width = self._fit_basis(X)
        self._width_ = width
        self.n_features_ = (self.degree + 1) * width
        return self

    def transform(self, d: np.ndarray, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        d, X = _check_dt(d, X)
        b = self._b(X)
        return np.hstack([(d[:, None] ** k) * b for k in range(self.degree + 1)])

    def deriv_d(self, d: np.ndarray, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        d, X = _check_dt(d, X)
        b = self._b(X)
        blocks = [np.zeros_like(b)]
        for k in range(1, self.degree + 1):
            blocks.append(k * (d[:, None] ** (k - 1)) * b)
        return np.hstack(blocks)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"PolynomialTreatmentFeatures(degree={self.degree!r}, "
            f"basis={self.basis!r}, include_intercept={self.include_intercept!r})"
        )


class AdditiveFeatures(FeatureMap, _BasisHolder):
    """``phi(d, x) = [d, b(x)]`` -- the partially linear specification.

    The treatment enters with a single constant coefficient, so the ATE and the
    average derivative coincide and equal that coefficient. This is the
    "partially linear model" of [9] Sec. 2 and is useful as a benchmark against
    the nonparametric specification, exactly as [9] Table 1 reports both.

    References
    ----------
    Chernozhukov, Cinelli, Newey, Sharma & Syrgkanis (2026), Sec. 2,
    doi:10.1162/REST.a.1705.

    Examples
    --------
    >>> import numpy as np
    >>> from pragdml.features import AdditiveFeatures
    >>> X = np.array([[5.0]]); D = np.array([1.0])
    >>> fm = AdditiveFeatures(include_intercept=False).fit(D, X)
    >>> fm.transform(D, X)
    array([[1., 5.]])
    >>> fm.deriv_d(D, X)
    array([[1., 0.]])
    """

    def __init__(self, basis=None, include_intercept: bool = True) -> None:
        self._init_basis(basis, include_intercept)

    def fit(self, D: np.ndarray, X: np.ndarray) -> "AdditiveFeatures":
        width = self._fit_basis(X)
        self._width_ = width
        self.n_features_ = 1 + width
        return self

    def transform(self, d: np.ndarray, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        d, X = _check_dt(d, X)
        return np.hstack([d[:, None], self._b(X)])

    def deriv_d(self, d: np.ndarray, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        d, X = _check_dt(d, X)
        b = self._b(X)
        return np.hstack([np.ones((b.shape[0], 1)), np.zeros_like(b)])

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"AdditiveFeatures(basis={self.basis!r}, "
            f"include_intercept={self.include_intercept!r})"
        )
