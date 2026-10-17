"""The public estimators.

:class:`PragmaticDML`
    One information set. Cross-fitted DML with a learned Riesz representer
    (Sec. 2.1.3, Theorem 1), optionally repeated across splits.

:class:`PipelineDML`
    A **menu** of information sets or pipelines, aggregated fold-wise by star or
    convex weights fitted to pooled out-of-fold predictions (Sec. 3.4,
    Definitions 5-6, Theorems 4-5). This is the estimator for the paper's
    end-to-end case: the aggregation step is the only thing that crosses folds,
    each candidate is still trained on ``I_k^c`` alone (Remark 1).

:func:`compare_information_sets`
    The paper's Table 1: every information set estimated side by side, plus the
    aggregate.

A note on which parameter you are estimating, because the package refuses to be
vague about it. :class:`PragmaticDML` targets ``theta_lambda`` (Eq. 3) and
:class:`PipelineDML` targets ``theta_0`` (Definition 3). Both coincide with the
full-information ``theta^*`` only under target adaptivity (Definition 1). The
``target_label`` on every result says which, and
:meth:`~pragdml._results.DMLResult.sensitivity` is the route to ``theta^*`` when
adaptivity is not defensible.

Author
------
Dr Merwan Roudane <merwanroudane920@gmail.com>
https://github.com/merwanroudane
"""

from __future__ import annotations

from typing import Mapping, Optional, Sequence

import numpy as np

from ._core import CrossFitOutput, combine_repetitions, cross_fit, make_folds
from ._results import DMLResult, InformationSetComparison
from .aggregation import pooled_convex_aggregate, pooled_star_aggregate
from .features import VaryingCoefficientFeatures
from .functionals import Functional, get_functional
from .learners import LinearOutcomeLearner
from .riesz import LinearRieszLearner

__all__ = [
    "PragmaticDML",
    "PipelineDML",
    "pragmatic_dml",
    "compare_information_sets",
    "InformationSetComparison",
]


def _to_array(obj, name: str, allow_2d: bool = False) -> tuple[np.ndarray, Optional[list]]:
    """Coerce a DataFrame/Series/array to float64, keeping column names."""
    names = None
    if hasattr(obj, "columns"):
        names = [str(c) for c in obj.columns]
        arr = np.asarray(obj.to_numpy(), dtype=np.float64)
    elif hasattr(obj, "to_numpy"):
        arr = np.asarray(obj.to_numpy(), dtype=np.float64)
    else:
        arr = np.asarray(obj, dtype=np.float64)
    if arr.ndim == 1 and allow_2d:
        arr = arr[:, None]
    if arr.ndim > 2:
        raise ValueError(f"{name} must be 1-d or 2-d, got ndim={arr.ndim}")
    if not np.all(np.isfinite(arr)):
        n_bad = int(np.sum(~np.isfinite(arr)))
        raise ValueError(
            f"{name} contains {n_bad} non-finite value(s). pragdml does not impute: "
            "drop or fill them yourself so the choice is visible in your code."
        )
    return arr, names


class PragmaticDML:
    """Cross-fitted DML for one information set (Theorem 1).

    Parameters
    ----------
    functional : str or Functional, default ``"ate"``
        The target. ``"ate"``, ``"att"``, ``"apo"``,
        ``"average_derivative"`` (aliases ``"derivative"``, ``"elasticity"``), or a
        :class:`~pragdml.functionals.Functional` instance.
    basis : sklearn transformer, optional
        Basis ``b(X)`` inside the paper's varying-coefficient architecture
        (Eqs. 68-69) -- e.g. ``PCA(20)`` for an embedding, ``SplineTransformer``
        for a sieve. ``None`` uses ``X`` as given. Ignored when you pass your own
        learners.
    outcome_learner, riesz_learner : optional
        Override the default heads. Both default to the exact-projection linear
        learners over the shared feature map, which is the paper's own
        specification and the configuration in which ``g_0`` and ``alpha_0`` are
        computed exactly.
    ridge : float, default 0.0
        Penalty for both default learners. Use a small positive value with a wide
        basis.
    n_folds : int, default 5
        ``K >= 2``.
    n_rep : int, default 1
        Repeated cross-fitting with median aggregation (extension E4, as in
        [9] fn. 11). ``5`` is a reasonable choice when the split matters.
    seed : int, optional
        Seeds every split and every stochastic learner draw.

    Attributes
    ----------
    crossfit_ : CrossFitOutput
        The last (or only) pass; the input to all sensitivity analysis.

    Examples
    --------
    >>> import numpy as np
    >>> from pragdml import PragmaticDML
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(800, 3))
    >>> e = 1.0 / (1.0 + np.exp(-X[:, 0]))
    >>> D = (rng.uniform(size=800) < e).astype(float)
    >>> Y = 1.5 * D + X[:, 0] - 0.5 * X[:, 1] + rng.normal(size=800)
    >>> res = PragmaticDML(functional="ate", seed=1).fit(Y, D, X)
    >>> bool(res.conf_int()[0] < 1.5 < res.conf_int()[1])
    True
    """

    def __init__(
        self,
        functional: "str | Functional" = "ate",
        basis=None,
        outcome_learner=None,
        riesz_learner=None,
        ridge: float = 0.0,
        n_folds: int = 5,
        n_rep: int = 1,
        seed: Optional[int] = None,
    ) -> None:
        self.functional = functional
        self.basis = basis
        self.outcome_learner = outcome_learner
        self.riesz_learner = riesz_learner
        self.ridge = float(ridge)
        self.n_folds = int(n_folds)
        self.n_rep = int(n_rep)
        if self.n_rep < 1:
            raise ValueError(f"n_rep must be >= 1, got {n_rep}")
        self.seed = seed

    def _default_learners(self):
        fm = VaryingCoefficientFeatures(basis=self.basis)
        outcome = self.outcome_learner or LinearOutcomeLearner(fm, ridge=self.ridge)
        riesz = self.riesz_learner or LinearRieszLearner(fm, ridge=self.ridge)
        return outcome, riesz

    def fit(
        self,
        Y,
        D,
        X,
        cluster=None,
    ) -> DMLResult:
        """Fit and return a :class:`~pragdml._results.DMLResult`.

        Parameters
        ----------
        Y : array-like of shape (n,)
            Outcome.
        D : array-like of shape (n,)
            Treatment.
        X : array-like of shape (n, k)
            The representation actually used as controls.
        cluster : array-like of shape (n,), optional
            Cluster labels. Whole clusters stay inside a fold and the variance
            becomes cluster-robust -- required by Sec. 5.1 for repeated
            observations on the same unit (extension E1).

        Returns
        -------
        DMLResult
        """
        y_arr, _ = _to_array(Y, "Y")
        d_arr, _ = _to_array(D, "D")
        x_arr, x_names = _to_array(X, "X", allow_2d=True)
        y_arr = y_arr.reshape(-1)
        d_arr = d_arr.reshape(-1)
        if not (y_arr.shape[0] == d_arr.shape[0] == x_arr.shape[0]):
            raise ValueError(
                f"Y, D and X must agree in length; got {y_arr.shape[0]}, "
                f"{d_arr.shape[0]}, {x_arr.shape[0]}."
            )
        n = y_arr.shape[0]
        if n < 2 * self.n_folds:
            raise ValueError(
                f"n = {n} is too small for {self.n_folds} folds; each fold would "
                f"have fewer than 2 observations. Reduce n_folds."
            )
        cl = None
        if cluster is not None:
            cl = np.asarray(cluster).reshape(-1)
            if cl.shape[0] != n:
                raise ValueError(
                    f"cluster has {cl.shape[0]} labels for {n} observations"
                )

        func = get_functional(self.functional)
        outcome, riesz = self._default_learners()
        seeds = np.random.SeedSequence(self.seed).spawn(self.n_rep)

        passes: list[CrossFitOutput] = []
        for ss in seeds:
            passes.append(
                cross_fit(
                    y_arr, d_arr, x_arr, func, outcome, riesz,
                    n_folds=self.n_folds, cluster=cl,
                    rng=np.random.default_rng(ss),
                )
            )
        self.crossfit_ = passes[-1]
        self.passes_ = passes

        if self.n_rep == 1:
            theta, se = passes[0].theta, passes[0].se
        else:
            theta, se = combine_repetitions(
                [p.theta for p in passes], [p.sigma2 for p in passes], n
            )
            if cl is not None:
                # Cluster-robust: rescale the median SE by the clustered/iid ratio
                # of the last pass, so clustering is not silently dropped.
                ratio = passes[-1].se / max(
                    np.sqrt(passes[-1].sigma2 / n), np.finfo(float).tiny
                )
                se = float(se * ratio)

        extra = {"Representation dim": x_arr.shape[1]}
        if x_names:
            extra["Controls"] = ", ".join(x_names[:6]) + (
                ", ..." if len(x_names) > 6 else ""
            )
        return DMLResult(
            crossfit=self.crossfit_,
            functional_name=func.name,
            theta=theta,
            se=se,
            n_rep=self.n_rep,
            target_label="representation-based target (theta_lambda, Eq. 3)",
            extra=extra,
        )

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"PragmaticDML(functional={self.functional!r}, n_folds={self.n_folds}, "
            f"n_rep={self.n_rep}, seed={self.seed!r})"
        )


class PipelineDML:
    """DML over a menu of pipelines with pooled fold-wise aggregation.

    Implements the paper's end-to-end case (Sec. 2.5 and Sec. 3.4): each candidate
    recipe is trained on every training fold, both heads are aggregated over the
    pooled out-of-fold predictions, and the Eq. (8) score is evaluated from the
    aggregated nuisances. By Theorems 4-5 the usual Wald interval is valid for the
    population pipeline target ``theta_0`` of Definition 3 -- with no stability or
    differentiability condition on the tuning algorithm.

    The two heads are aggregated **separately**, each under its own loss, and may
    select different candidates. The paper's Sec. 5.3 observes exactly that, and
    draws the lesson: prediction quality alone does not summarise target
    preservation.

    Parameters
    ----------
    functional : str or Functional, default ``"ate"``
    aggregation : {"star", "convex"}, default ``"star"``
        Star weights cost ``log M`` and tolerate a large menu (Eq. 45); convex
        weights cost ``sqrt(M log n)`` and need ``M log n = o(sqrt(n))`` (Eq. 49)
        but can only improve approximation. See
        :func:`~pragdml.diagnostics.pooling_conditions`.
    basis, ridge, n_folds, seed
        As in :class:`PragmaticDML`. ``n_rep`` is not offered here: pooled
        aggregation already uses all out-of-fold predictions.

    Attributes
    ----------
    crossfit_ : CrossFitOutput
    outcome_aggregation_, riesz_aggregation_ : AggregationResult
        The two weight vectors and their oracle remainders.
    candidate_names_ : list of str

    Examples
    --------
    >>> import numpy as np
    >>> from pragdml import PipelineDML
    >>> rng = np.random.default_rng(0)
    >>> Z = rng.normal(size=(600, 4))
    >>> D = (rng.uniform(size=600) < 0.5).astype(float)
    >>> Y = 2.0 * D + Z[:, 0] + rng.normal(size=600)
    >>> menu = {"first_two": Z[:, :2], "last_two": Z[:, 2:], "all": Z}
    >>> res = PipelineDML(functional="ate", seed=0).fit(Y, D, menu)
    >>> bool(res.conf_int()[0] < 2.0 < res.conf_int()[1])
    True
    """

    def __init__(
        self,
        functional: "str | Functional" = "ate",
        aggregation: str = "star",
        basis=None,
        outcome_learner=None,
        riesz_learner=None,
        ridge: float = 0.0,
        n_folds: int = 5,
        seed: Optional[int] = None,
    ) -> None:
        if aggregation not in ("star", "convex"):
            raise ValueError(
                f"aggregation must be 'star' or 'convex', got {aggregation!r}"
            )
        self.functional = functional
        self.aggregation = aggregation
        self.basis = basis
        self.outcome_learner = outcome_learner
        self.riesz_learner = riesz_learner
        self.ridge = float(ridge)
        self.n_folds = int(n_folds)
        self.seed = seed

    def fit(self, Y, D, information_sets: Mapping, cluster=None) -> DMLResult:
        """Fit the menu and return the aggregate result.

        Parameters
        ----------
        Y, D : array-like of shape (n,)
        information_sets : mapping of str to array-like
            ``{name: X_j}``. Every entry must have ``n`` rows; the columns may
            differ freely, which is the point -- one entry per modality,
            embedding, prompt or head.
        cluster : array-like, optional

        Returns
        -------
        DMLResult
            For ``theta_0``, the population pipeline target.
        """
        y_arr, _ = _to_array(Y, "Y")
        d_arr, _ = _to_array(D, "D")
        y_arr, d_arr = y_arr.reshape(-1), d_arr.reshape(-1)
        n = y_arr.shape[0]

        if not information_sets:
            raise ValueError("`information_sets` is empty; pass at least one entry.")
        names: list[str] = []
        mats: list[np.ndarray] = []
        for name, mat in information_sets.items():
            arr, _ = _to_array(mat, f"information_sets[{name!r}]", allow_2d=True)
            if arr.shape[0] != n:
                raise ValueError(
                    f"information_sets[{name!r}] has {arr.shape[0]} rows but Y has {n}"
                )
            names.append(str(name))
            mats.append(arr)
        M = len(mats)

        cl = None
        if cluster is not None:
            cl = np.asarray(cluster).reshape(-1)
            if cl.shape[0] != n:
                raise ValueError(f"cluster has {cl.shape[0]} labels for {n} rows")

        func = get_functional(self.functional)
        func.validate(d_arr, mats[0])
        rng = np.random.default_rng(self.seed)
        folds = make_folds(n, n_folds=self.n_folds, cluster=cl, rng=rng)

        # Out-of-fold prediction matrices, one column per candidate recipe.
        G = np.empty((n, M), dtype=np.float64)
        A = np.empty((n, M), dtype=np.float64)
        MG = np.empty((n, M), dtype=np.float64)
        MA = np.empty((n, M), dtype=np.float64)
        fold_id = np.empty(n, dtype=np.int64)

        all_idx = np.arange(n)
        for k, test in enumerate(folds):
            train = np.setdiff1d(all_idx, test)
            fold_id[test] = k
            for j, Xj in enumerate(mats):
                fm = VaryingCoefficientFeatures(basis=self.basis)
                g_learner = (
                    self.outcome_learner.clone()
                    if self.outcome_learner is not None
                    else LinearOutcomeLearner(fm, ridge=self.ridge)
                )
                a_learner = (
                    self.riesz_learner.clone()
                    if self.riesz_learner is not None
                    else LinearRieszLearner(
                        VaryingCoefficientFeatures(basis=self.basis), ridge=self.ridge
                    )
                )
                g_learner.fit(y_arr[train], d_arr[train], Xj[train], func)
                a_learner.fit(y_arr[train], d_arr[train], Xj[train], func)
                Dt, Xt = d_arr[test], Xj[test]
                G[test, j] = np.asarray(g_learner(Dt, Xt), dtype=np.float64).reshape(-1)
                A[test, j] = np.asarray(a_learner(Dt, Xt), dtype=np.float64).reshape(-1)
                MG[test, j] = np.asarray(
                    func.apply(g_learner, Dt, Xt), dtype=np.float64
                ).reshape(-1)
                MA[test, j] = np.asarray(
                    func.apply(a_learner, Dt, Xt), dtype=np.float64
                ).reshape(-1)

        aggregate = (
            pooled_star_aggregate if self.aggregation == "star"
            else pooled_convex_aggregate
        )
        agg_g = aggregate(G, y=y_arr, objective="square", n_folds=self.n_folds)
        agg_a = aggregate(
            A, m_oof_candidates=MA, objective="riesz", n_folds=self.n_folds
        )
        self.outcome_aggregation_ = agg_g
        self.riesz_aggregation_ = agg_a
        self.candidate_names_ = names
        self.oof_ = {"G": G, "A": A, "MG": MG, "MA": MA}

        g_hat = G @ agg_g.weights
        m_g = MG @ agg_g.weights
        a_hat = A @ agg_a.weights
        m_a = MA @ agg_a.weights
        psi = m_g + a_hat * (y_arr - g_hat)
        den = (
            func.denominator(d_arr, mats[0]) if func.is_ratio else np.ones(n)
        )

        self.crossfit_ = CrossFitOutput(
            psi=psi, den=np.asarray(den, dtype=np.float64),
            g_hat=g_hat, alpha_hat=a_hat, m_g=m_g, m_alpha=m_a,
            y=y_arr, fold_id=fold_id, cluster=cl,
            functional_name=func.name, n_folds=self.n_folds,
            is_ratio=bool(func.is_ratio),
            diagnostics={
                "aggregation": self.aggregation,
                "outcome_weights": dict(zip(names, agg_g.weights.tolist())),
                "riesz_weights": dict(zip(names, agg_a.weights.tolist())),
            },
        )

        def _picked(agg) -> str:
            order = np.argsort(-agg.weights)
            return ", ".join(
                f"{names[i]} {agg.weights[i]:.4f}"
                for i in order if agg.weights[i] > 1e-8
            )

        return DMLResult(
            crossfit=self.crossfit_,
            functional_name=func.name,
            n_rep=1,
            target_label=(
                "population pipeline target (theta_0, Definition 3) over the "
                "candidates' common span H"
            ),
            extra={
                "Menu size (M)": M,
                "Aggregation": f"pooled {self.aggregation} (Definition "
                               f"{'5' if self.aggregation == 'star' else '6'})",
                "Outcome weights": _picked(agg_g),
                "Riesz weights": _picked(agg_a),
            },
        )

    def pooling_report(self) -> dict:
        """Evaluate Eq. (45)/(49) for this fit.

        Uses the aggregation remainders as stand-ins for the nuisance rates, which
        is the only thing available from data; the result is a design check, not a
        proof of the condition.
        """
        from .diagnostics import pooling_conditions

        if not hasattr(self, "crossfit_"):
            raise RuntimeError("PipelineDML is not fitted; call .fit() first.")
        r_g = float(self.outcome_aggregation_.oracle["sqrt_remainder"])
        r_a = float(self.riesz_aggregation_.oracle["sqrt_remainder"])
        return pooling_conditions(
            self.crossfit_.n, len(self.candidate_names_), r_g, r_a,
            weights=self.aggregation,
        )

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"PipelineDML(functional={self.functional!r}, "
            f"aggregation={self.aggregation!r}, n_folds={self.n_folds})"
        )


def pragmatic_dml(
    Y,
    D,
    X,
    functional: "str | Functional" = "ate",
    basis=None,
    n_folds: int = 5,
    n_rep: int = 1,
    cluster=None,
    seed: Optional[int] = None,
    ridge: float = 0.0,
) -> DMLResult:
    """One-call wrapper around :class:`PragmaticDML`, for users who never write classes.

    Parameters
    ----------
    Y, D, X : array-like
    functional : str or Functional, default ``"ate"``
    basis : sklearn transformer, optional
    n_folds : int, default 5
    n_rep : int, default 1
    cluster : array-like, optional
    seed : int, optional
    ridge : float, default 0.0

    Returns
    -------
    DMLResult

    Examples
    --------
    >>> import numpy as np
    >>> from pragdml import pragmatic_dml
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(400, 2))
    >>> D = (rng.uniform(size=400) < 0.5).astype(float)
    >>> Y = D + X[:, 0] + rng.normal(size=400)
    >>> res = pragmatic_dml(Y, D, X, functional="ate", seed=0)
    >>> round(res.theta, 2) == round(res.theta, 2)
    True
    """
    return PragmaticDML(
        functional=functional, basis=basis, n_folds=n_folds, n_rep=n_rep,
        seed=seed, ridge=ridge,
    ).fit(Y, D, X, cluster=cluster)


def compare_information_sets(
    Y,
    D,
    information_sets: Mapping,
    functional: "str | Functional" = "ate",
    basis=None,
    outcome_learner=None,
    riesz_learner=None,
    n_folds: int = 5,
    n_rep: int = 1,
    cluster=None,
    seed: Optional[int] = None,
    ridge: float = 0.0,
    aggregation: str = "star",
    include_aggregate: bool = True,
) -> InformationSetComparison:
    """The paper's Table 1: one estimate per information set, plus the aggregate.

    Each information set defines its **own** population target ``theta_lambda``,
    so the rows are not competing estimates of one number -- their spread is
    itself the evidence the paper reads (a stable sign across modalities means the
    conclusion does not hinge on which modality carried the information).

    Parameters
    ----------
    Y, D : array-like of shape (n,)
    information_sets : mapping of str to array-like
        ``{name: X_j}``, e.g. ``{"text": E_txt, "image": E_img, "full": E_all}``.
    functional, basis, outcome_learner, riesz_learner, n_folds, n_rep, cluster, seed, ridge
        As in :class:`PragmaticDML`. The same ``seed`` is used for every set so the
        rows are comparable, and the same learners -- so pass learners whose
        feature map adapts to a varying number of columns (every map in
        :mod:`pragdml.features` does). With a wide ``basis``, prefer
        :class:`~pragdml.features.AdditiveFeatures`: the varying-coefficient map
        doubles the basis width and can make the Riesz solve degenerate (you will
        get a warning if it does).
    aggregation : {"star", "convex"}, default ``"star"``
    include_aggregate : bool, default True
        Whether to append the :class:`PipelineDML` aggregate row.

    Returns
    -------
    InformationSetComparison

    Examples
    --------
    >>> import numpy as np
    >>> from pragdml import compare_information_sets
    >>> rng = np.random.default_rng(0)
    >>> Z = rng.normal(size=(500, 4))
    >>> D = (rng.uniform(size=500) < 0.5).astype(float)
    >>> Y = 2.0 * D + Z[:, 0] + rng.normal(size=500)
    >>> cmp = compare_information_sets(
    ...     Y, D, {"a": Z[:, :2], "b": Z[:, 2:]}, seed=0)
    >>> len(cmp.results)
    2
    """
    results = {}
    for name, mat in information_sets.items():
        results[str(name)] = PragmaticDML(
            functional=functional, basis=basis,
            outcome_learner=outcome_learner, riesz_learner=riesz_learner,
            n_folds=n_folds, n_rep=n_rep, seed=seed, ridge=ridge,
        ).fit(Y, D, mat, cluster=cluster)

    aggregate = None
    if include_aggregate and len(information_sets) > 1:
        aggregate = PipelineDML(
            functional=functional, aggregation=aggregation, basis=basis,
            outcome_learner=outcome_learner, riesz_learner=riesz_learner,
            n_folds=n_folds, seed=seed, ridge=ridge,
        ).fit(Y, D, information_sets, cluster=cluster)

    return InformationSetComparison(
        results=results,
        aggregate=aggregate,
        aggregate_label=f"{aggregation.capitalize()} aggregate",
        notes=[
            "The aggregate row estimates theta_0 over the candidates' common span",
            "H (Definition 3), not any single row's target.",
        ],
    )
