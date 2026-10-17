"""Cross-fitting engine: the debiased score and the ``K``-fold DML estimator.

This module implements Sec. 2.1.3 and Sec. 2.5 of the paper:

* the debiased (Neyman-orthogonal) score of Eq. (8)

  .. math:: \\psi(O; g, \\alpha) = m(O, g) + \\alpha(W)\\{Y - g(W)\\},

* the ``K``-fold cross-fitted estimator of Eq. (11) and its variance Eq. (12),
* the "nothing leaks across folds" discipline that Theorems 1 and 3 rest on: for
  each fold ``k`` the nuisances are fitted on ``I_k^c`` only, and the score is
  evaluated on ``I_k``. Conditional on the training fold the fitted functions are
  fixed, which is exactly the independence structure the proofs use.

Because Theorem 3 places its conditions on the **fitted functions** rather than on
the tuning algorithm, the same engine serves a frozen embedding, a fold-wise
fine-tuned pipeline and an aggregated menu without modification. That is the
paper's main practical claim, and it is why this file has no special case for
"fine-tuned".

Compatibility map
-----------------
=========================================  ========================================
Paper                                      Here
=========================================  ========================================
Eq. (8), short score ``psi_lambda``         :func:`debiased_score`
Eq. (10), long score ``psi^*``              same function, long nuisances
Eq. (11), ``theta_hat``                      :attr:`CrossFitOutput.theta`
Eq. (12), ``sigma_hat^2``                    :attr:`CrossFitOutput.sigma2`
Eq. (16), Wald interval                      :meth:`CrossFitOutput.conf_int`
Sec. 2.1.3, fold construction                :func:`make_folds`
Eq. (55), ``sigma_0^2``                      :attr:`CrossFitOutput.sigma2_outcome`
Eq. (56), ``nu_0^2``                         :attr:`CrossFitOutput.nu2_representer`
Sec. 5.1, within-product dependence          ``cluster=`` (extension E1)
[9] fn. 11, repeated splits + median         ``n_rep=`` (extension E4)
=========================================  ========================================

Author
------
Dr Merwan Roudane <merwanroudane920@gmail.com>
https://github.com/merwanroudane
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
from scipy import stats

from .functionals import Functional

__all__ = [
    "debiased_score",
    "make_folds",
    "CrossFitOutput",
    "cross_fit",
    "combine_repetitions",
]


def debiased_score(
    Y: np.ndarray,
    D: np.ndarray,
    X: np.ndarray,
    g,
    alpha,
    functional: Functional,
) -> np.ndarray:
    """The Neyman-orthogonal score of Eq. (8), evaluated pointwise.

    .. math:: \\psi_i = m(O_i, g) + \\alpha(W_i)\\{Y_i - g(W_i)\\}

    Orthogonality is what makes first-stage estimation error enter only through
    the product ``||g - g_0|| * ||alpha - alpha_0||`` (Eq. 9), rather than
    linearly.

    Parameters
    ----------
    Y, D, X : ndarray
        Outcome, treatment, controls.
    g, alpha : callable
        Fitted nuisances, callable as ``f(d, X)``.
    functional : Functional
        The ``m`` defining the target.

    Returns
    -------
    ndarray of shape (n,)
        For a ratio target (:class:`~pragdml.functionals.ATT`) this is the
        **numerator** score; the denominator is handled by :func:`cross_fit`.

    Examples
    --------
    >>> import numpy as np
    >>> from pragdml.functionals import ATE
    >>> from pragdml._core import debiased_score
    >>> Y = np.array([1.0, 0.0]); D = np.array([1.0, 0.0]); X = np.zeros((2, 1))
    >>> g = lambda d, X: d
    >>> a = lambda d, X: 2.0 * d - 1.0
    >>> debiased_score(Y, D, X, g, a, ATE())
    array([1., 1.])
    """
    Y = np.asarray(Y, dtype=np.float64).reshape(-1)
    D = np.asarray(D, dtype=np.float64).reshape(-1)
    m_g = np.asarray(functional.apply(g, D, X), dtype=np.float64).reshape(-1)
    g_w = np.asarray(g(D, X), dtype=np.float64).reshape(-1)
    a_w = np.asarray(alpha(D, X), dtype=np.float64).reshape(-1)
    return m_g + a_w * (Y - g_w)


def make_folds(
    n: int,
    n_folds: int = 5,
    cluster: Optional[np.ndarray] = None,
    rng: Optional[np.random.Generator] = None,
) -> list[np.ndarray]:
    """Partition ``{0, ..., n-1}`` into ``n_folds`` evaluation folds.

    With ``cluster`` given, whole clusters are assigned to folds, never split.
    Sec. 5.1 of the paper requires exactly this for its panel application ("if a
    product contributes several adjacent changes, all of its observations should
    remain in the same training or validation fold"); splitting a cluster would
    leak information across folds and break the conditional-independence argument
    behind Theorems 1 and 3.

    Unlike ``sklearn.model_selection.GroupKFold``, clusters here are **randomly
    permuted** before being dealt into folds, so the split is honestly random and
    reproducible from ``rng``.

    Parameters
    ----------
    n : int
        Sample size.
    n_folds : int, default 5
        Number of folds, ``K >= 2`` (the paper fixes ``K >= 2``).
    cluster : array-like of shape (n,), optional
        Cluster labels.
    rng : numpy.random.Generator, optional
        Source of randomness.

    Returns
    -------
    list of ndarray
        Index arrays, one per fold, disjoint and covering every observation.

    Examples
    --------
    >>> import numpy as np
    >>> from pragdml._core import make_folds
    >>> folds = make_folds(10, 2, rng=np.random.default_rng(0))
    >>> sorted(np.concatenate(folds).tolist())
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    >>> cl = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    >>> f = make_folds(8, 2, cluster=cl, rng=np.random.default_rng(1))
    >>> all(len(set(cl[idx])) * 2 == len(idx) for idx in f)
    True
    """
    n = int(n)
    n_folds = int(n_folds)
    if n_folds < 2:
        raise ValueError(
            f"n_folds must be >= 2 (the paper fixes K >= 2), got {n_folds}"
        )
    if n < n_folds:
        raise ValueError(f"n = {n} is smaller than n_folds = {n_folds}")
    rng = np.random.default_rng() if rng is None else rng

    if cluster is None:
        perm = rng.permutation(n)
        return [np.sort(part) for part in np.array_split(perm, n_folds)]

    cluster = np.asarray(cluster).reshape(-1)
    if cluster.shape[0] != n:
        raise ValueError(f"cluster has {cluster.shape[0]} labels, expected {n}")
    groups = np.unique(cluster)
    if groups.size < n_folds:
        raise ValueError(
            f"only {groups.size} distinct clusters for {n_folds} folds; reduce "
            "n_folds or drop the clustering."
        )
    order = rng.permutation(groups.size)
    chunks = np.array_split(groups[order], n_folds)
    return [
        np.sort(np.flatnonzero(np.isin(cluster, chunk))) for chunk in chunks
    ]


@dataclass
class CrossFitOutput:
    """Everything a cross-fitted DML pass produces, kept for reuse downstream.

    The sensitivity machinery of Sec. 4 needs the same fitted quantities as the
    point estimate, so they are stored once rather than recomputed.

    Attributes
    ----------
    psi : ndarray of shape (n,)
        Cross-fitted score values, Eq. (8). The numerator score for a ratio target.
    den : ndarray of shape (n,)
        Denominator contributions; all ones unless the target is a ratio.
    g_hat, alpha_hat : ndarray of shape (n,)
        Out-of-fold nuisance predictions at the observed ``(D_i, X_i)``.
    m_g, m_alpha : ndarray of shape (n,)
        ``m(O_i, g_hat)`` and ``m(O_i, alpha_hat)``.
    y : ndarray of shape (n,)
        The outcome, aligned with the above.
    fold_id : ndarray of shape (n,)
        Which evaluation fold each observation belonged to.
    cluster : ndarray or None
        Cluster labels, if clustering was requested.
    functional_name : str
    n_folds : int
    """

    psi: np.ndarray
    den: np.ndarray
    g_hat: np.ndarray
    alpha_hat: np.ndarray
    m_g: np.ndarray
    m_alpha: np.ndarray
    y: np.ndarray
    fold_id: np.ndarray
    cluster: Optional[np.ndarray] = None
    functional_name: str = "functional"
    n_folds: int = 5
    is_ratio: bool = False
    diagnostics: dict = field(default_factory=dict)

    # ---- point estimate and variance (Eqs. 11-12) ----------------------------

    @property
    def n(self) -> int:
        """Sample size."""
        return int(self.psi.shape[0])

    @property
    def theta(self) -> float:
        """The cross-fitted estimate, Eq. (11) (ratio-corrected if needed)."""
        if self.is_ratio:
            den = float(np.mean(self.den))
            if den == 0.0:
                raise ZeroDivisionError(
                    "the ratio target has an estimated denominator of exactly zero "
                    "(no treated units?); the parameter is not identified here."
                )
            return float(np.mean(self.psi) / den)
        return float(np.mean(self.psi))

    @property
    def influence(self) -> np.ndarray:
        """Centred influence values ``psi_i - theta`` (delta-corrected if ratio)."""
        theta = self.theta
        if self.is_ratio:
            den = float(np.mean(self.den))
            return (self.psi - theta * self.den) / den
        return self.psi - theta

    @property
    def sigma2(self) -> float:
        """``sigma_hat^2`` of Eq. (12): the empirical variance of the scores."""
        return float(np.mean(self.influence**2))

    @property
    def se(self) -> float:
        """Standard error. Cluster-robust when ``cluster`` was supplied (E1)."""
        infl = self.influence
        n = self.n
        if self.cluster is None:
            return float(np.sqrt(self.sigma2 / n))
        labels = np.asarray(self.cluster).reshape(-1)
        _, inverse = np.unique(labels, return_inverse=True)
        sums = np.bincount(inverse, weights=infl)
        return float(np.sqrt(np.sum(sums**2)) / n)

    @property
    def n_clusters(self) -> Optional[int]:
        """Number of clusters, or ``None`` when not clustering."""
        if self.cluster is None:
            return None
        return int(np.unique(np.asarray(self.cluster)).size)

    def conf_int(self, level: float = 0.95) -> tuple[float, float]:
        """The Wald interval of Eq. (16).

        Parameters
        ----------
        level : float, default 0.95
            Nominal coverage.

        Returns
        -------
        (float, float)
        """
        if not 0.0 < level < 1.0:
            raise ValueError(f"level must be in (0, 1), got {level}")
        z = float(stats.norm.ppf(0.5 + level / 2.0))
        theta, se = self.theta, self.se
        return (theta - z * se, theta + z * se)

    @property
    def tvalue(self) -> float:
        """``theta_hat / se``."""
        se = self.se
        return float(self.theta / se) if se > 0 else float("nan")

    @property
    def pvalue(self) -> float:
        """Two-sided normal p-value against zero."""
        return float(2.0 * stats.norm.sf(abs(self.tvalue)))

    # ---- sensitivity components (Eqs. 55-57) --------------------------------

    @property
    def sigma2_outcome(self) -> float:
        """``sigma_hat_0^2 = P_n (Y - g_hat)^2``, Eq. (55)."""
        return float(np.mean((self.y - self.g_hat) ** 2))

    @property
    def nu2_representer(self) -> float:
        """``nu_hat_0^2 = P_n[2 m(O, alpha_hat) - alpha_hat^2]``, Eq. (56).

        The orthogonal form, not ``P_n[alpha^2]``; see
        :func:`pragdml.riesz.representer_second_moment` for why.
        """
        return float(np.mean(2.0 * self.m_alpha - self.alpha_hat**2))

    @property
    def S(self) -> float:
        """``S_hat_0 = (sigma_hat_0^2 nu_hat_0^2)^{1/2}``, Eq. (57)."""
        prod = self.sigma2_outcome * self.nu2_representer
        if prod <= 0.0:
            return 0.0
        return float(np.sqrt(prod))

    @property
    def riesz_loss(self) -> float:
        """Out-of-fold Riesz loss ``P_n[alpha^2 - 2 m(O, alpha)]`` (Sec. 3.1)."""
        return -self.nu2_representer

    @property
    def outcome_r2(self) -> float:
        """Out-of-fold uncentred-residual ``R^2`` of the outcome head.

        ``1 - P_n(Y - g_hat)^2 / Var(Y)``. Used by the benchmarking calibration of
        Sec. 5.4 and [9] Appendix E.
        """
        var = float(np.var(self.y))
        if var <= 0.0:
            return float("nan")
        return float(1.0 - self.sigma2_outcome / var)


def cross_fit(
    Y: np.ndarray,
    D: np.ndarray,
    X: np.ndarray,
    functional: Functional,
    outcome_learner,
    riesz_learner,
    n_folds: int = 5,
    cluster: Optional[np.ndarray] = None,
    rng: Optional[np.random.Generator] = None,
) -> CrossFitOutput:
    """One cross-fitted DML pass (Sec. 2.1.3 / Sec. 2.5).

    For each fold ``k``: clone both learners, fit them on ``I_k^c`` only, then
    evaluate the Eq. (8) score on ``I_k``. Nothing fitted on a fold ever touches
    that fold.

    Parameters
    ----------
    Y, D, X : ndarray
        Outcome ``(n,)``, treatment ``(n,)``, controls ``(n, k)``.
    functional : Functional
        The target's ``m``.
    outcome_learner, riesz_learner
        Learner objects; cloned per fold so no state leaks between folds.
    n_folds : int, default 5
        ``K``.
    cluster : array-like, optional
        Cluster labels; whole clusters are kept within a fold.
    rng : numpy.random.Generator, optional

    Returns
    -------
    CrossFitOutput
    """
    Y = np.asarray(Y, dtype=np.float64).reshape(-1)
    D = np.asarray(D, dtype=np.float64).reshape(-1)
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X[:, None]
    n = Y.shape[0]
    if not (D.shape[0] == n == X.shape[0]):
        raise ValueError(
            f"Y, D and X must have the same number of rows; got {n}, {D.shape[0]}, "
            f"{X.shape[0]}."
        )
    if n == 0:
        raise ValueError("empty sample")
    functional.validate(D, X)

    folds = make_folds(n, n_folds=n_folds, cluster=cluster, rng=rng)

    psi = np.empty(n, dtype=np.float64)
    g_hat = np.empty(n, dtype=np.float64)
    a_hat = np.empty(n, dtype=np.float64)
    m_g = np.empty(n, dtype=np.float64)
    m_a = np.empty(n, dtype=np.float64)
    fold_id = np.empty(n, dtype=np.int64)
    fold_losses: list[tuple[float, float]] = []

    all_idx = np.arange(n)
    for k, test in enumerate(folds):
        train = np.setdiff1d(all_idx, test, assume_unique=False)
        if train.size == 0:
            raise ValueError(
                f"fold {k} leaves no training observations; reduce n_folds."
            )
        g_k = outcome_learner.clone().fit(Y[train], D[train], X[train], functional)
        a_k = riesz_learner.clone().fit(Y[train], D[train], X[train], functional)

        Dt, Xt = D[test], X[test]
        g_vals = np.asarray(g_k(Dt, Xt), dtype=np.float64).reshape(-1)
        a_vals = np.asarray(a_k(Dt, Xt), dtype=np.float64).reshape(-1)
        mg = np.asarray(functional.apply(g_k, Dt, Xt), dtype=np.float64).reshape(-1)
        ma = np.asarray(functional.apply(a_k, Dt, Xt), dtype=np.float64).reshape(-1)

        g_hat[test] = g_vals
        a_hat[test] = a_vals
        m_g[test] = mg
        m_a[test] = ma
        psi[test] = mg + a_vals * (Y[test] - g_vals)
        fold_id[test] = k
        fold_losses.append(
            (
                float(np.mean((Y[test] - g_vals) ** 2)),
                float(np.mean(a_vals**2 - 2.0 * ma)),
            )
        )

    den = functional.denominator(D, X) if functional.is_ratio else np.ones(n)

    # nu_hat^2 estimates E[alpha_0^2] > 0. A non-positive value is not a quirk of
    # the estimator: it means the fitted representer is worse than the zero
    # function under its own loss, so the point estimate is not trustworthy and
    # no sensitivity region exists. The usual cause is an over-wide basis making
    # the Riesz normal equations ill-conditioned. Warn here rather than only when
    # the user later reaches SensitivityAnalysis.
    nu2 = float(np.mean(2.0 * m_a - a_hat**2))
    if nu2 <= 0.0:
        warnings.warn(
            f"the cross-fitted representer has nu_hat^2 = {nu2:.4g} <= 0, which "
            "estimates E[alpha_0^2] > 0. The Riesz fit has failed, so this point "
            "estimate should not be reported. Add regularisation (ridge=...), "
            "narrow the basis, or use AdditiveFeatures instead of "
            "VaryingCoefficientFeatures for a wide basis.",
            RuntimeWarning,
            stacklevel=2,
        )

    return CrossFitOutput(
        psi=psi,
        den=np.asarray(den, dtype=np.float64).reshape(-1),
        g_hat=g_hat,
        alpha_hat=a_hat,
        m_g=m_g,
        m_alpha=m_a,
        y=Y,
        fold_id=fold_id,
        cluster=None if cluster is None else np.asarray(cluster).reshape(-1),
        functional_name=getattr(functional, "name", type(functional).__name__),
        n_folds=n_folds,
        is_ratio=bool(functional.is_ratio),
        diagnostics={
            "fold_square_loss": [a for a, _ in fold_losses],
            "fold_riesz_loss": [b for _, b in fold_losses],
        },
    )


def combine_repetitions(
    thetas: Sequence[float], sigma2s: Sequence[float], n: int
) -> tuple[float, float]:
    """Median-aggregate repeated cross-fitting splits (extension E4).

    Follows the rule of Chernozhukov et al. (2018) used by [9] footnote 11: the
    point estimate is the median across repetitions, and the variance is the
    median of ``sigma_r^2 + (theta_r - theta_median)^2``, so that between-split
    variation is carried into the standard error rather than discarded.

    Parameters
    ----------
    thetas : sequence of float
        One estimate per repetition.
    sigma2s : sequence of float
        The matching ``sigma_hat^2`` values.
    n : int
        Sample size.

    Returns
    -------
    (float, float)
        ``(theta, se)``.

    Examples
    --------
    >>> from pragdml._core import combine_repetitions
    >>> theta, se = combine_repetitions([1.0, 1.2, 0.8], [1.0, 1.0, 1.0], 100)
    >>> round(theta, 6)
    1.0
    >>> round(se, 6)
    0.10198
    """
    thetas = np.asarray(list(thetas), dtype=np.float64)
    sigma2s = np.asarray(list(sigma2s), dtype=np.float64)
    if thetas.shape != sigma2s.shape:
        raise ValueError("thetas and sigma2s must have the same length")
    if thetas.size == 0:
        raise ValueError("no repetitions to combine")
    theta = float(np.median(thetas))
    sigma2 = float(np.median(sigma2s + (thetas - theta) ** 2))
    return theta, float(np.sqrt(max(sigma2, 0.0) / int(n)))
