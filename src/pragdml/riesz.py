"""Riesz representer learners (the ``alpha`` head) and the Riesz loss.

The Riesz representer -- the paper's "balancing weight" -- is the unique
``alpha`` with ``E[m(O, h)] = E[h(W) alpha(W)]`` for all ``h`` in the relevant
space (Eqs. 4-5). The paper, following [8], learns it **variationally** rather
than by inverting a propensity score, by minimising the *Riesz loss*

.. math::

    L(\\alpha) = E\\big[\\alpha(W)^2 - 2\\, m(O, \\alpha)\\big],

whose minimiser over a closed linear space ``H`` is the projection ``alpha_0`` of
Definition 3. Two facts make this loss the right object and are used throughout
:mod:`pragdml`:

1. **It is quadratic.** Over the span of a feature map it has a closed-form
   minimiser -- the *Riesz normal equations* ``Q c = r`` with ``Q = P_n[phi phi']``
   and ``r = P_n[m(O, phi)]``. No propensity score is inverted, so no
   near-zero denominator can blow up.
2. **Its minimum identifies a second moment.** The normal equation gives
   ``inf_alpha L(alpha) = L(alpha_0) = -E[alpha_0^2]``, which is exactly why
   Eq. (26) can read off representer information loss from a profiled loss, and
   why Eq. (56) estimates ``nu_0^2 = E[alpha_0^2]`` by the Neyman-orthogonal
   ``P_n[2 m(O, alpha) - alpha^2]`` rather than by ``P_n[alpha^2]``.

Compatibility map
-----------------
==========================================  =======================================
Paper                                       Here
==========================================  =======================================
Eqs. (4)-(5), Riesz representation            the contract every learner satisfies
Sec. 3.1, Riesz loss ``L(alpha)``             :func:`riesz_loss`
Eq. (26), ``inf L = -E[alpha^2]``             :func:`representer_second_moment`
Definition 3, ``alpha_0 = argmin_H L``        :class:`LinearRieszLearner`
Eq. (69)/(70), varying-coefficient ``alpha``  ``LinearRieszLearner(VaryingCoefficientFeatures())``
Sec. 2.1.2, analytic IPW weights              :class:`PlugInRieszLearner`
Eq. (29), excess risk identity                :func:`riesz_loss` + Lemma 2 tests
==========================================  =======================================

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

from .features import FeatureMap, VaryingCoefficientFeatures
from .functionals import Functional
from .learners import solve_scaled_normal_equations

__all__ = [
    "RieszLearner",
    "LinearRieszLearner",
    "PlugInRieszLearner",
    "PartitionRieszLearner",
    "riesz_loss",
    "representer_second_moment",
    "FeatureEvaluable",
]


class FeatureEvaluable:
    """Adapter making a :class:`~pragdml.features.FeatureMap` act like ``h``.

    A :class:`~pragdml.functionals.Functional` applies ``m`` to anything callable
    as ``h(d, X)``. Wrapping a feature map in this adapter therefore applies ``m``
    to **all** basis functions at once, giving the ``(n, p)`` matrix
    ``M[i, j] = m(O_i, phi_j)`` whose column means form the right-hand side ``r``
    of the Riesz normal equations. This is where linearity of ``m`` in its second
    argument does the real work.

    Examples
    --------
    >>> import numpy as np
    >>> from pragdml.features import VaryingCoefficientFeatures
    >>> from pragdml.functionals import ATE
    >>> from pragdml.riesz import FeatureEvaluable
    >>> X = np.array([[1.0], [2.0]]); D = np.array([1.0, 0.0])
    >>> fm = VaryingCoefficientFeatures(include_intercept=False).fit(D, X)
    >>> ATE().apply(FeatureEvaluable(fm), D, X)
    array([[1., 0.],
           [2., 0.]])
    """

    __slots__ = ("features",)

    def __init__(self, features: FeatureMap) -> None:
        self.features = features

    def __call__(self, d: np.ndarray, X: np.ndarray) -> np.ndarray:
        return self.features.transform(d, X)

    def deriv_d(self, d: np.ndarray, X: np.ndarray) -> np.ndarray:
        return self.features.deriv_d(d, X)


def riesz_loss(
    alpha, functional: Functional, D: np.ndarray, X: np.ndarray
) -> float:
    """Empirical Riesz loss ``P_n[alpha(W)^2 - 2 m(O, alpha)]`` (Sec. 3.1).

    Smaller is better. On the space ``H`` the excess of this loss over its minimum
    equals the squared ``L2(P)`` error ``||alpha - alpha_0||^2`` (Lemma 2,
    Eq. 29), which is what lets the paper place its product-rate condition
    directly on losses.

    Parameters
    ----------
    alpha : callable
        ``alpha(d, X) -> (n,)``, optionally with ``deriv_d``.
    functional : Functional
        The ``m`` being represented.
    D, X : ndarray
        Observed treatment and controls.

    Returns
    -------
    float

    Examples
    --------
    >>> import numpy as np
    >>> from pragdml.functionals import ATE
    >>> from pragdml.riesz import riesz_loss
    >>> X = np.zeros((2, 1)); D = np.array([1.0, 0.0])
    >>> a = lambda d, X: 2.0 * d - 1.0
    >>> riesz_loss(a, ATE(), D, X)
    -3.0
    """
    D = np.asarray(D, dtype=np.float64).reshape(-1)
    vals = np.asarray(alpha(D, X), dtype=np.float64).reshape(-1)
    m_vals = np.asarray(functional.apply(alpha, D, X), dtype=np.float64).reshape(-1)
    return float(np.mean(vals**2 - 2.0 * m_vals))


def representer_second_moment(
    alpha, functional: Functional, D: np.ndarray, X: np.ndarray
) -> float:
    """Orthogonal estimate of ``E[alpha_0^2]``: ``P_n[2 m(O, alpha) - alpha^2]``.

    This is the ``nu_0^2`` of Eq. (56) and of [9] Sec. 4, and equals minus the
    Riesz loss. It is preferred over ``P_n[alpha^2]`` because it is
    Neyman-orthogonal with respect to perturbations of ``alpha`` -- a consequence
    of the restricted Riesz identity of Proposition 2(i) and the linearity of
    ``m``. The two agree at the truth and differ at a fitted ``alpha``, and only
    this one supports root-n inference on the sensitivity bounds.

    Examples
    --------
    >>> import numpy as np
    >>> from pragdml.functionals import ATE
    >>> from pragdml.riesz import representer_second_moment
    >>> X = np.zeros((2, 1)); D = np.array([1.0, 0.0])
    >>> representer_second_moment(lambda d, X: 2.0 * d - 1.0, ATE(), D, X)
    3.0
    """
    return -riesz_loss(alpha, functional, D, X)


class RieszLearner(ABC):
    """Base class for the ``alpha`` head.

    Subclasses must be callable as ``self(d, X)`` after :meth:`fit`, and should
    expose ``deriv_d`` whenever the representer is differentiable in ``d`` (needed
    by :class:`~pragdml.functionals.AverageDerivative`, which must evaluate
    ``m(O, alpha)``).
    """

    @abstractmethod
    def fit(
        self,
        Y: Optional[np.ndarray],
        D: np.ndarray,
        X: np.ndarray,
        functional: Functional,
    ) -> "RieszLearner":
        """Fit on a training fold. ``Y`` is unused by most Riesz learners."""

    @abstractmethod
    def __call__(self, d: np.ndarray, X: np.ndarray) -> np.ndarray:
        """Evaluate the fitted representer at ``(d, X)``."""

    def clone(self) -> "RieszLearner":
        """An unfitted copy."""
        raise NotImplementedError

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{type(self).__name__}()"


class LinearRieszLearner(RieszLearner):
    """Riesz regression in closed form over a :class:`~pragdml.features.FeatureMap`.

    Solves the Riesz normal equations

    .. math::

        Q c = r, \\qquad Q = P_n[\\phi \\phi'], \\qquad r_j = P_n[m(O, \\phi_j)],

    which is the exact minimiser of the quadratic Riesz loss over
    ``H = span(phi)``, i.e. the ``alpha_0`` of Definition 3. With
    :class:`~pragdml.features.VaryingCoefficientFeatures` this is the paper's own
    representer architecture, Eqs. (69)-(70).

    No propensity score is inverted, so there is no overlap-driven blow-up; the
    price is that ``alpha_0`` is only the projection of ``alpha^*`` onto ``H``, and
    the gap is exactly the ``alpha^* - alpha_0`` term of the mixed-bias identity.

    Parameters
    ----------
    features : FeatureMap, optional
        Defaults to ``VaryingCoefficientFeatures()``.
    ridge : float, default 0.0
        Penalty in the column-rescaled space. A small positive value is advisable
        for a wide embedding basis; ``0.0`` solves the exact normal equations,
        falling back to a pseudo-inverse if ``Q`` is singular.

    Attributes
    ----------
    coef_ : ndarray of shape (n_features,)
    features_ : FeatureMap
    gram_ : ndarray of shape (n_features, n_features)
        ``Q``, kept for diagnostics.
    moment_ : ndarray of shape (n_features,)
        ``r``, kept for diagnostics.

    References
    ----------
    Chernozhukov, Newey & Singh (2022), *Econometrica* 90(3):967-1027,
    doi:10.3982/ECTA18515. Aradillas Fernandez et al. (2026), Definition 3 and
    Eqs. (69)-(70), arXiv:2610.01935.

    Examples
    --------
    In a saturated binary-covariate design the exact IPW weight lies in the span
    of the varying-coefficient map, so Riesz regression recovers it up to
    sampling error (which falls at the ``n^{-1/2}`` rate --
    ``tests/test_riesz.py`` asserts the rate itself):

    >>> import numpy as np
    >>> from pragdml.functionals import ATE
    >>> from pragdml.riesz import LinearRieszLearner
    >>> rng = np.random.default_rng(0)
    >>> x = rng.integers(0, 2, 4000).astype(float)
    >>> e = np.where(x > 0.5, 0.7, 0.3)
    >>> D = (rng.uniform(size=4000) < e).astype(float)
    >>> X = x[:, None]
    >>> a = LinearRieszLearner().fit(None, D, X, ATE())
    >>> ipw = D / e - (1 - D) / (1 - e)
    >>> bool(np.corrcoef(a(D, X), ipw)[0, 1] > 0.999)
    True
    """

    def __init__(self, features: Optional[FeatureMap] = None, ridge: float = 0.0) -> None:
        self.features = features
        self.ridge = float(ridge)
        if self.ridge < 0.0:
            raise ValueError(f"ridge must be >= 0, got {ridge}")

    def clone(self) -> "LinearRieszLearner":
        return LinearRieszLearner(features=self.features, ridge=self.ridge)

    def fit(
        self,
        Y: Optional[np.ndarray],
        D: np.ndarray,
        X: np.ndarray,
        functional: Functional,
    ) -> "LinearRieszLearner":
        D = np.asarray(D, dtype=np.float64).reshape(-1)
        fm = self.features if self.features is not None else VaryingCoefficientFeatures()
        self.features_ = fm.fit(D, X)
        phi = self.features_.transform(D, X)          # (n, p)
        n = phi.shape[0]
        # r_j = P_n m(O, phi_j): apply m to every basis function at once (Eqs. 4-5).
        m_phi = np.asarray(
            functional.apply(FeatureEvaluable(self.features_), D, X), dtype=np.float64
        )
        if m_phi.ndim == 1:
            m_phi = m_phi[:, None]
        if m_phi.shape != phi.shape:
            raise ValueError(
                f"m applied to the basis returned shape {m_phi.shape}, expected "
                f"{phi.shape}; the functional and the feature map disagree."
            )
        self.gram_ = phi.T @ phi / n
        self.moment_ = m_phi.mean(axis=0)
        self.coef_ = solve_scaled_normal_equations(
            self.gram_, self.moment_, ridge=self.ridge
        )
        return self

    def __call__(self, d: np.ndarray, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "coef_"):
            raise RuntimeError("LinearRieszLearner is not fitted; call .fit() first.")
        return self.features_.transform(d, X) @ self.coef_

    def deriv_d(self, d: np.ndarray, X: np.ndarray) -> np.ndarray:
        """Exact ``d/dd alpha(d, X)``."""
        if not hasattr(self, "coef_"):
            raise RuntimeError("LinearRieszLearner is not fitted; call .fit() first.")
        return self.features_.deriv_d(d, X) @ self.coef_

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"LinearRieszLearner(features={self.features!r}, ridge={self.ridge!r})"


class PlugInRieszLearner(RieszLearner):
    """The analytic inverse-propensity representer (binary treatment only).

    For the ATE, ``alpha(D, X) = D/e(X) - (1-D)/(1-e(X))`` with
    ``e(X) = P(D = 1 | X)``; for the ATT numerator,
    ``alpha(D, X) = D - (1-D) e(X)/(1-e(X))`` (Sec. 2.1.2 and
    :meth:`~pragdml.functionals.ATE.analytic_representer`).

    Shipped for two reasons: it is the estimator everyone already knows, and it is
    the bridge that lets :mod:`pragdml` be cross-checked against ``DoubleML``'s
    ``DoubleMLIRM`` on the ATE. For anything else prefer
    :class:`LinearRieszLearner`, which never divides by a fitted probability.

    Parameters
    ----------
    propensity : sklearn classifier
        Anything with ``fit`` and ``predict_proba``.
    trim : float, default 0.01
        Fitted propensities are clipped into ``[trim, 1 - trim]``. Clipping is
        reported, not silent: :attr:`n_trimmed_` counts the affected observations.
        Set ``trim=0.0`` to disable and let an overlap failure raise instead.

    Attributes
    ----------
    n_trimmed_ : int
        How many fitted propensities were clipped.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.linear_model import LogisticRegression
    >>> from pragdml.functionals import ATE
    >>> from pragdml.riesz import PlugInRieszLearner
    >>> rng = np.random.default_rng(2)
    >>> X = rng.normal(size=(300, 2))
    >>> D = (rng.uniform(size=300) < 0.5).astype(float)
    >>> a = PlugInRieszLearner(LogisticRegression()).fit(None, D, X, ATE())
    >>> a.n_trimmed_
    0
    """

    def __init__(self, propensity, trim: float = 0.01) -> None:
        if not (hasattr(propensity, "fit") and hasattr(propensity, "predict_proba")):
            raise TypeError(
                "`propensity` must be a scikit-learn style classifier with "
                "fit/predict_proba."
            )
        self.propensity = propensity
        self.trim = float(trim)
        if not 0.0 <= self.trim < 0.5:
            raise ValueError(f"trim must be in [0, 0.5), got {trim}")

    def clone(self) -> "PlugInRieszLearner":
        return PlugInRieszLearner(clone(self.propensity), trim=self.trim)

    def fit(
        self,
        Y: Optional[np.ndarray],
        D: np.ndarray,
        X: np.ndarray,
        functional: Functional,
    ) -> "PlugInRieszLearner":
        D = np.asarray(D, dtype=np.float64).reshape(-1)
        functional.validate(D, X)
        if not hasattr(functional, "analytic_representer"):
            raise TypeError(
                f"{type(functional).__name__} has no analytic representer; use "
                "LinearRieszLearner."
            )
        try:
            functional.analytic_representer(D[:1], np.asarray(X)[:1], np.array([0.5]))
        except NotImplementedError as exc:
            raise TypeError(
                f"{type(functional).__name__} has no closed-form Riesz representer, so "
                "PlugInRieszLearner cannot be used with it. Use LinearRieszLearner, "
                "which learns the representer variationally (this is the paper's own "
                "route; see Sec. 3.1)."
            ) from exc
        self.functional_ = functional
        self.propensity_ = clone(self.propensity)
        self.propensity_.fit(np.asarray(X, dtype=np.float64), D.astype(int))
        # Evaluate on the training fold so n_trimmed_ reports overlap trouble at
        # fit time rather than only after the first prediction call.
        self._e(X)
        return self

    def _e(self, X: np.ndarray) -> np.ndarray:
        proba = np.asarray(
            self.propensity_.predict_proba(np.asarray(X, dtype=np.float64)),
            dtype=np.float64,
        )
        classes = np.asarray(getattr(self.propensity_, "classes_", [0, 1]))
        idx = int(np.argmax(classes == 1)) if 1 in set(classes.tolist()) else -1
        e = proba[:, idx]
        if self.trim > 0.0:
            clipped = np.clip(e, self.trim, 1.0 - self.trim)
            self.n_trimmed_ = int(np.sum(clipped != e))
            return clipped
        self.n_trimmed_ = 0
        return e

    def __call__(self, d: np.ndarray, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "propensity_"):
            raise RuntimeError("PlugInRieszLearner is not fitted; call .fit() first.")
        d = np.asarray(d, dtype=np.float64).reshape(-1)
        return self.functional_.analytic_representer(d, X, self._e(X))

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"PlugInRieszLearner(propensity={self.propensity!r}, trim={self.trim!r})"


class PartitionRieszLearner(RieszLearner):
    """Riesz regression on a tree-ensemble partition: a nonlinear menu member.

    A forest is grown on an auxiliary target to obtain a **data-adaptive
    partition** of the covariate space; the leaf indicators of that partition
    (crossed with ``{1, D}``, as in the varying-coefficient architecture) then form
    the basis in which the closed-form Riesz normal equations are solved. The
    result is a piecewise-linear-in-``D`` representer that is nonlinear in ``X``.

    Why it exists: Sec. 3.3 of the paper is written for a *nonconvex menu* of
    fitted pipelines, and star aggregation is the tool for combining one. A menu
    of only linear-in-basis learners is convex, so the library needs at least one
    genuinely nonlinear, data-adaptive member for that machinery to be exercised
    on anything real.

    Notes
    -----
    **This is not the ForestRiesz algorithm of Chernozhukov, Newey,
    Quintas-Martinez & Syrgkanis (2022).** That method drives the Riesz loss into
    the *splitting criterion* of the forest. Here the partition is chosen by an
    ordinary supervised fit and only the within-partition representer is solved
    for exactly. The paper-faithful consequence is the one that matters: the
    fitted function still lies in a linear span, so it is a legitimate member of
    the common space ``H`` of Definition 3 and keeps the mixed-bias algebra of
    Proposition 2 intact.

    **One tree, not an intersection of many.** The partition comes from a *single*
    tree. Intersecting the leaves of several trees looks like a richer basis but
    is a trap: crossing three trees of a 100-tree forest on 1,600 observations
    produces over 500 cells with a median of 2 observations each, so the Riesz
    normal equations become hopelessly ill-conditioned and the fitted
    ``nu_hat^2 = E[alpha^2]`` comes out *negative*. One tree with a leaf-size
    floor gives cells large enough to estimate within.

    Parameters
    ----------
    tree : sklearn tree or ensemble, optional
        Defaults to ``DecisionTreeRegressor(min_samples_leaf=40)``. A forest is
        accepted and its **first** tree supplies the partition; pass a single
        tree if you want that choice to be explicit.
    target : {"outcome", "treatment"}, default "outcome"
        What the partition is grown on. ``"treatment"`` looks like the natural
        choice -- the representer does depend on the propensity -- but it is the
        worse one: a tree grown on ``D`` *deliberately* isolates cells with
        extreme treated shares (0.96 against 0.92 for an outcome-grown tree on the
        same data), and those are exactly the cells where the within-cell
        inverse-propensity weight diverges. ``"outcome"`` is the default for that
        reason. The overlap guard below protects either choice.
    ridge : float, default 1e-4
        Penalty in the column-rescaled space. Leaf dummies crossed with ``D`` are
        collinear by construction, so a non-trivial default is deliberate.
    max_leaves : int, default 64
        Cap on retained leaf columns (largest cells first).
    min_leaf_count : int, default 25
        Cells smaller than this are pooled into the intercept rather than given
        their own column. Without such a floor a handful of near-empty cells can
        dominate the solve. For a binary treatment an **overlap guard** also
        applies: a cell is kept only if it holds at least
        ``max(2, min_leaf_count // 5)`` units in *each* arm, which keeps the
        within-cell representer bounded. The number of cells dropped by that
        guard is recorded in ``n_cells_dropped_overlap_``.

    Examples
    --------
    With ``target="outcome"`` (the default) the outcome is needed to grow the
    partition; the cross-fitting engine always supplies it.

    >>> import numpy as np
    >>> from pragdml.functionals import ATE
    >>> from pragdml.riesz import PartitionRieszLearner, representer_second_moment
    >>> rng = np.random.default_rng(3)
    >>> X = rng.normal(size=(1200, 2))
    >>> D = (rng.uniform(size=1200) < 0.5).astype(float)
    >>> Y = 2.0 * D + X[:, 0] + rng.normal(size=1200)
    >>> a = PartitionRieszLearner().fit(Y, D, X, ATE())
    >>> a(D, X).shape
    (1200,)
    >>> bool(representer_second_moment(a, ATE(), D, X) > 0)   # a sane fit
    True
    >>> a.n_cells_ > 0
    True
    """

    def __init__(
        self,
        tree=None,
        target: str = "outcome",
        ridge: float = 1e-4,
        max_leaves: int = 64,
        min_leaf_count: int = 25,
    ) -> None:
        if target not in ("outcome", "treatment"):
            raise ValueError(
                f"target must be 'outcome' or 'treatment', got {target!r}"
            )
        self.tree = tree
        self.target = target
        self.ridge = float(ridge)
        self.max_leaves = int(max_leaves)
        self.min_leaf_count = int(min_leaf_count)

    def clone(self) -> "PartitionRieszLearner":
        return PartitionRieszLearner(
            tree=None if self.tree is None else clone(self.tree),
            target=self.target,
            ridge=self.ridge,
            max_leaves=self.max_leaves,
            min_leaf_count=self.min_leaf_count,
        )

    def _default_tree(self):
        from sklearn.tree import DecisionTreeRegressor

        return DecisionTreeRegressor(min_samples_leaf=40, random_state=0)

    def fit(
        self,
        Y: Optional[np.ndarray],
        D: np.ndarray,
        X: np.ndarray,
        functional: Functional,
    ) -> "PartitionRieszLearner":
        D = np.asarray(D, dtype=np.float64).reshape(-1)
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X[:, None]
        if self.target == "outcome":
            if Y is None:
                raise ValueError("target='outcome' needs Y; pass target='treatment'.")
            aux = np.asarray(Y, dtype=np.float64).reshape(-1)
        else:
            aux = D
        self.tree_ = self._default_tree() if self.tree is None else clone(self.tree)
        self.tree_.fit(X, aux)
        if not hasattr(self.tree_, "apply"):
            raise TypeError(
                f"{type(self.tree_).__name__} has no .apply(X); "
                "PartitionRieszLearner needs a decision tree or tree ensemble."
            )
        codes = _leaf_codes(self.tree_, X)
        uniq, counts = np.unique(codes, return_counts=True)

        # Keep only cells large enough to estimate within, largest first; the
        # rest are absorbed by the intercept.
        eligible = counts >= self.min_leaf_count

        # Overlap guard. Within a cell this basis is saturated in a binary D, so
        # the fitted representer reproduces the cell's own inverse-propensity
        # weight -- which diverges as the cell's treated share approaches 0 or 1.
        # A cell with (almost) no units in one arm therefore contributes an
        # arbitrarily large weight and can drive the estimated E[alpha^2]
        # negative. Requiring both arms to be present makes the within-cell
        # weight bounded by construction.
        if getattr(functional, "requires_binary_treatment", False):
            floor = max(2, self.min_leaf_count // 5)
            treated = np.array(
                [int(np.sum(D[codes == k] == 1.0)) for k in uniq], dtype=np.int64
            )
            control = counts - treated
            eligible &= (treated >= floor) & (control >= floor)
            self.n_cells_dropped_overlap_ = int(
                np.sum((counts >= self.min_leaf_count) & ~eligible)
            )
        else:
            self.n_cells_dropped_overlap_ = 0

        keep = uniq[eligible][np.argsort(-counts[eligible])][: self.max_leaves]
        self.leaves_ = np.sort(keep)
        self.n_cells_ = int(self.leaves_.size)
        from .features import VaryingCoefficientFeatures

        self._inner = LinearRieszLearner(
            features=VaryingCoefficientFeatures(
                basis=_LeafOneHot(self.tree_, self.leaves_), include_intercept=True
            ),
            ridge=self.ridge,
        )
        self._inner.fit(None, D, X, functional)
        return self

    def __call__(self, d: np.ndarray, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "_inner"):
            raise RuntimeError("PartitionRieszLearner is not fitted; call .fit() first.")
        return self._inner(d, X)

    def deriv_d(self, d: np.ndarray, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "_inner"):
            raise RuntimeError("PartitionRieszLearner is not fitted; call .fit() first.")
        return self._inner.deriv_d(d, X)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"PartitionRieszLearner(target={self.target!r}, ridge={self.ridge!r}, "
            f"max_leaves={self.max_leaves!r}, "
            f"min_leaf_count={self.min_leaf_count!r})"
        )


def _leaf_codes(tree, X: np.ndarray) -> np.ndarray:
    """Leaf index of each row under a **single** tree's partition.

    A forest is accepted and its first tree is used. Intersecting several trees'
    leaves is deliberately not done; see :class:`PartitionRieszLearner`.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X[:, None]
    leaf = np.asarray(tree.apply(X))
    if leaf.ndim == 1:
        return leaf.astype(np.int64)
    return leaf[:, 0].astype(np.int64)


class _LeafOneHot:
    """Fitted-tree leaf indicators as a frozen sklearn-style transformer.

    Marked ``_pragdml_frozen`` so :class:`~pragdml.features.FeatureMap` does not
    clone it: ``sklearn.base.clone`` recursively clones estimator-valued params,
    which would replace the already-grown tree with an unfitted copy.
    """

    _pragdml_frozen = True

    def __init__(self, tree, leaves: np.ndarray) -> None:
        self.tree = tree
        self.leaves = np.asarray(leaves)

    def get_params(self, deep: bool = True) -> dict:
        return {"tree": self.tree, "leaves": self.leaves}

    def set_params(self, **params) -> "_LeafOneHot":
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def fit(self, X, y=None) -> "_LeafOneHot":
        return self

    def transform(self, X) -> np.ndarray:
        code = _leaf_codes(self.tree, X)
        return (code[:, None] == self.leaves[None, :]).astype(np.float64)
