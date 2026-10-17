"""Star and convex aggregation over a menu of fitted pipelines (Sec. 3).

Applied pipelines produce a **menu**, not a convex class: varying a prompt, a
checkpoint, a feature modality or a head changes the fitted function rather than
a coefficient in a common parametric family. Sec. 3.3 of the paper handles that
with *star aggregation* (Audibert's star hull, sharpened by Vijaykumar 2021):
take the empirical winner as a centre, then refit one scalar along each segment
joining it to a competitor.

Two facts make this the right tool, and both are implemented here exactly:

1. **The segment step is closed form.** The objective is quadratic in the mixing
   weight, so the optimum is a ratio of two empirical means -- Eq. (36) for the
   square loss, Eq. (37) for the Riesz loss -- projected onto ``[0, 1]``. A finite
   menu is therefore aggregated by one scan over ``j = 1, ..., M``, with no
   iterative optimiser anywhere.
2. **It stays inside ``H``.** Every segment lies in the span of the candidates, so
   aggregation never enlarges the information space and the mixed-bias algebra of
   Proposition 2 survives intact. This is why the two heads may legitimately
   select *different* candidates (which they do in the paper's own Sec. 5.3), as
   long as both live in the common span.

Pooling across folds
--------------------
Sec. 3.4 is the paper's most practical result. Honest star aggregation *inside* a
training fold needs a second split (train the candidates, then fit the weights on
held-out data), which multiplies the number of expensive pipeline fits -- ``K K'``
of them under nested cross-fitting. The cheaper design everyone actually uses is
to train each candidate once per fold and fit **one** set of weights to the pooled
out-of-fold predictions, as in stacking and the super learner. The weights then
see the evaluation fold, so cross-fitting holds for the candidates but not for the
aggregate.

The paper shows the breach is cheap: it costs one extra remainder term, of order
``sqrt(log M)(r_g + r_alpha)`` for star weights (Eq. 45) and
``sqrt(M log n)(r_g + r_alpha)`` for convex weights (Eq. 49). So a star menu may
grow faster than any power of ``n`` while a convex menu must satisfy
``M log n = o(sqrt(n))``. :func:`pragdml.diagnostics.pooling_conditions` evaluates
both.

Definition 5 notes that the pooled star estimator *is* the ordinary star estimator
computed from the augmented observations -- which is exactly why
:func:`pooled_star_aggregate` is a thin, documented wrapper over
:func:`star_aggregate` applied to out-of-fold predictions, rather than a second
algorithm.

Compatibility map
-----------------
=========================================  =======================================
Paper                                      Here
=========================================  =======================================
Definition 4, star estimator                 :func:`star_aggregate`
Eq. (36), ``beta`` for the square loss       :func:`segment_weight_square`
Eq. (37), ``beta`` for the Riesz loss        :func:`segment_weight_riesz`
Eqs. (38)-(39), oracle inequalities          :func:`star_oracle_bound`
Eqs. (40)-(41), pooled objectives            :func:`pooled_star_aggregate`
Definition 5, pooled star estimator          :func:`pooled_star_aggregate`
Eq. (42), best single recipe benchmark       ``AggregationResult.best_single``
Eqs. (43)-(44), Proposition 7                :func:`star_oracle_bound`
Definition 6, pooled convex estimator        :func:`pooled_convex_aggregate`
Eqs. (47)-(48), Proposition 8                :func:`convex_oracle_bound`
=========================================  =======================================

Author
------
Dr Merwan Roudane <merwanroudane920@gmail.com>
https://github.com/merwanroudane
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy import optimize

__all__ = [
    "AggregationResult",
    "segment_weight_square",
    "segment_weight_riesz",
    "star_aggregate",
    "convex_aggregate",
    "pooled_star_aggregate",
    "pooled_convex_aggregate",
    "star_oracle_bound",
    "convex_oracle_bound",
]


def _as_matrix(candidates, name: str) -> np.ndarray:
    arr = np.asarray(candidates, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(
            f"{name} must be a 2-d array of shape (n, M); got ndim={arr.ndim}"
        )
    if arr.shape[1] < 1:
        raise ValueError(f"{name} has no candidate columns")
    if not np.all(np.isfinite(arr)):
        raise ValueError(
            f"{name} contains non-finite values; a candidate learner produced NaN or "
            "inf predictions. Inspect the menu before aggregating."
        )
    return arr


# ---------------------------------------------------------------------------
# The two closed-form segment steps
# ---------------------------------------------------------------------------

def segment_weight_square(
    y: np.ndarray, centre: np.ndarray, competitor: np.ndarray
) -> float:
    """Eq. (36): the optimal mixing weight along a square-loss segment.

    .. math::

        \\beta(f) = \\frac{P_n[(Y - f)(\\tilde f - f)]}{P_n[(\\tilde f - f)^2]},
        \\qquad \\text{projected onto } [0, 1],

    where ``f`` is the competitor and ``tilde f`` the star centre. The aggregate is
    ``beta * centre + (1 - beta) * competitor``.

    Returns ``1.0`` (keep the centre) when the denominator vanishes, i.e. when the
    two candidates coincide, which is the paper's "any beta is optimal" convention
    resolved in favour of the centre.

    Examples
    --------
    >>> import numpy as np
    >>> from pragdml.aggregation import segment_weight_square
    >>> y = np.array([1.0, 1.0])
    >>> segment_weight_square(y, np.ones(2), np.zeros(2))
    1.0
    >>> segment_weight_square(y, np.full(2, 2.0), np.zeros(2))
    0.5
    """
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    centre = np.asarray(centre, dtype=np.float64).reshape(-1)
    competitor = np.asarray(competitor, dtype=np.float64).reshape(-1)
    delta = centre - competitor
    denom = float(np.mean(delta**2))
    if denom <= 0.0:
        return 1.0
    beta = float(np.mean((y - competitor) * delta) / denom)
    return float(np.clip(beta, 0.0, 1.0))


def segment_weight_riesz(
    centre: np.ndarray,
    competitor: np.ndarray,
    m_centre: np.ndarray,
    m_competitor: np.ndarray,
) -> float:
    """Eq. (37): the optimal mixing weight along a Riesz-loss segment.

    .. math::

        \\beta(\\alpha) = -\\frac{P_n\\big[\\alpha(\\tilde\\alpha - \\alpha)
                            - m(O, \\tilde\\alpha - \\alpha)\\big]}
                           {P_n[(\\tilde\\alpha - \\alpha)^2]},
        \\qquad \\text{projected onto } [0, 1],

    with ``alpha`` the competitor and ``tilde alpha`` the centre. Linearity of
    ``m`` in its second argument is what makes ``m(O, tilde alpha - alpha)`` simply
    the difference of the two candidates' ``m`` values -- which is why this step
    needs no re-evaluation of the functional.

    Parameters
    ----------
    centre, competitor : ndarray of shape (n,)
        Representer values at the observed points.
    m_centre, m_competitor : ndarray of shape (n,)
        ``m(O_i, .)`` for the same two candidates.

    Examples
    --------
    >>> import numpy as np
    >>> from pragdml.aggregation import segment_weight_riesz
    >>> c = np.ones(2); k = np.zeros(2)
    >>> segment_weight_riesz(c, k, np.ones(2), np.zeros(2))
    1.0
    """
    centre = np.asarray(centre, dtype=np.float64).reshape(-1)
    competitor = np.asarray(competitor, dtype=np.float64).reshape(-1)
    delta = centre - competitor
    denom = float(np.mean(delta**2))
    if denom <= 0.0:
        return 1.0
    m_delta = np.asarray(m_centre, dtype=np.float64).reshape(-1) - np.asarray(
        m_competitor, dtype=np.float64
    ).reshape(-1)
    beta = float((np.mean(m_delta) - np.mean(competitor * delta)) / denom)
    return float(np.clip(beta, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Objectives
# ---------------------------------------------------------------------------

def _square_objective(pred: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((np.asarray(y).reshape(-1) - pred.reshape(-1)) ** 2))


def _riesz_objective(pred: np.ndarray, m_pred: np.ndarray) -> float:
    return float(np.mean(pred.reshape(-1) ** 2 - 2.0 * m_pred.reshape(-1)))


@dataclass
class AggregationResult:
    """What an aggregation step produced.

    Attributes
    ----------
    weights : ndarray of shape (M,)
        Weights on the candidate menu. Sum to one for every method here.
    fitted : ndarray of shape (n,)
        Aggregate predictions.
    m_fitted : ndarray or None
        ``m(O_i, aggregate)`` -- present for the Riesz objective, ``None`` for the
        square loss.
    objective : float
        The aggregate's empirical objective (square risk or Riesz loss).
    candidate_objectives : ndarray of shape (M,)
        Each candidate's own objective, so the gain from aggregating is visible.
    centre : int or None
        Index of the star centre; ``None`` for convex aggregation.
    segment : int or None
        Index of the selected competitor.
    beta : float or None
        The selected mixing weight on the centre.
    method : str
        ``"star"``, ``"convex"``, ``"pooled_star"`` or ``"pooled_convex"``.
    oracle : dict
        The relevant oracle remainder; see :func:`star_oracle_bound`.
    """

    weights: np.ndarray
    fitted: np.ndarray
    objective: float
    candidate_objectives: np.ndarray
    method: str
    m_fitted: Optional[np.ndarray] = None
    centre: Optional[int] = None
    segment: Optional[int] = None
    beta: Optional[float] = None
    oracle: dict = field(default_factory=dict)

    @property
    def best_single(self) -> int:
        """Index of the best single candidate -- the Eq. (42) benchmark."""
        return int(np.argmin(self.candidate_objectives))

    @property
    def best_single_objective(self) -> float:
        """The benchmark's objective value."""
        return float(np.min(self.candidate_objectives))

    @property
    def gain_over_best_single(self) -> float:
        """How much the aggregate improved on the best single recipe.

        Positive means the aggregate is better. On the common space ``H`` this is
        a difference of risks, which by Lemma 2 (Eqs. 28-29) equals a difference
        of squared ``L2`` errors -- so it is directly interpretable as the
        reduction in nuisance error that aggregation bought.
        """
        return float(self.best_single_objective - self.objective)

    @property
    def is_degenerate(self) -> bool:
        """``True`` when all weight sits on one candidate.

        The paper's own application hits this case for the outcome head: no
        competitor improved on the centre, so every fitted star weight equalled
        one and the aggregate reduced to the centre. That is a legitimate outcome,
        not a failure.
        """
        return bool(np.max(self.weights) > 1.0 - 1e-10)

    @property
    def effective_candidates(self) -> int:
        """How many candidates carry non-negligible weight."""
        return int(np.sum(self.weights > 1e-8))

    def summary_frame(self):
        """Per-candidate weights and objectives as a :class:`pandas.DataFrame`."""
        import pandas as pd

        return pd.DataFrame(
            {
                "candidate": np.arange(self.weights.shape[0]),
                "weight": self.weights,
                "objective": self.candidate_objectives,
            }
        )


# ---------------------------------------------------------------------------
# Star aggregation (Definition 4)
# ---------------------------------------------------------------------------

def star_aggregate(
    candidates: np.ndarray,
    y: Optional[np.ndarray] = None,
    m_candidates: Optional[np.ndarray] = None,
    objective: str = "square",
    oracle_envelope: Optional[float] = None,
    method_label: str = "star",
) -> AggregationResult:
    """Definition 4: the star estimator over a finite menu.

    Step 1 picks the empirical-best candidate as the centre ``tilde c``. Step 2
    scans every competitor, computes the closed-form segment weight (Eq. 36 or
    Eq. 37), and keeps the best segment fit. The whole procedure is one pass over
    the menu.

    Parameters
    ----------
    candidates : ndarray of shape (n, M)
        Candidate predictions. For the pooled variant these must be **out-of-fold**
        predictions; see :func:`pooled_star_aggregate`.
    y : ndarray of shape (n,), optional
        Required for ``objective="square"``.
    m_candidates : ndarray of shape (n, M), optional
        ``m(O_i, a_j)`` for each candidate. Required for ``objective="riesz"``.
    objective : {"square", "riesz"}, default "square"
        Which loss to aggregate under. The two heads should be aggregated
        **separately** -- the paper's Sec. 5.3 finds them selecting different
        candidates, which is the expected behaviour, not an inconsistency.
    oracle_envelope : float, optional
        Envelope constant for the reported oracle remainder. Defaults to a
        data-driven ``max|.|`` over the menu (and over ``Y`` for the square loss).
    method_label : str, default "star"
        Recorded in the result.

    Returns
    -------
    AggregationResult

    References
    ----------
    Aradillas Fernandez et al. (2026), Definition 4, Eqs. (36)-(39),
    arXiv:2610.01935. Vijaykumar (2021), *NeurIPS* 34:4570-4581. Audibert (2007),
    *NeurIPS* 20.

    Examples
    --------
    A deliberately bad candidate should receive no weight, and mixing two
    imperfect candidates should beat either:

    >>> import numpy as np
    >>> from pragdml.aggregation import star_aggregate
    >>> y = np.array([1.0, 2.0, 3.0, 4.0])
    >>> F = np.column_stack([y + 0.5, y - 0.5, np.zeros(4)])
    >>> res = star_aggregate(F, y=y)
    >>> float(round(res.weights[2], 10))
    0.0
    >>> bool(res.objective < res.best_single_objective)
    True
    """
    if objective not in ("square", "riesz"):
        raise ValueError(f"objective must be 'square' or 'riesz', got {objective!r}")
    cand = _as_matrix(candidates, "candidates")
    n, M = cand.shape

    if objective == "square":
        if y is None:
            raise ValueError("objective='square' requires `y`")
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        if y.shape[0] != n:
            raise ValueError(f"y has {y.shape[0]} rows, candidates have {n}")
        obj_of = lambda p, mp=None: _square_objective(p, y)  # noqa: E731
        m_cand = None
    else:
        if m_candidates is None:
            raise ValueError("objective='riesz' requires `m_candidates`")
        m_cand = _as_matrix(m_candidates, "m_candidates")
        if m_cand.shape != cand.shape:
            raise ValueError(
                f"m_candidates has shape {m_cand.shape} but candidates {cand.shape}"
            )
        obj_of = lambda p, mp: _riesz_objective(p, mp)  # noqa: E731

    cand_obj = np.array(
        [
            obj_of(cand[:, j], None if m_cand is None else m_cand[:, j])
            for j in range(M)
        ],
        dtype=np.float64,
    )
    centre = int(np.argmin(cand_obj))

    best = {
        "obj": float(cand_obj[centre]),
        "beta": 1.0,
        "segment": centre,
        "pred": cand[:, centre].copy(),
        "m_pred": None if m_cand is None else m_cand[:, centre].copy(),
    }
    for j in range(M):
        if j == centre:
            continue
        if objective == "square":
            beta = segment_weight_square(y, cand[:, centre], cand[:, j])
        else:
            beta = segment_weight_riesz(
                cand[:, centre], cand[:, j], m_cand[:, centre], m_cand[:, j]
            )
        pred = beta * cand[:, centre] + (1.0 - beta) * cand[:, j]
        m_pred = (
            None
            if m_cand is None
            else beta * m_cand[:, centre] + (1.0 - beta) * m_cand[:, j]
        )
        value = obj_of(pred, m_pred)
        if value < best["obj"] - 1e-15:
            best = {
                "obj": float(value),
                "beta": float(beta),
                "segment": j,
                "pred": pred,
                "m_pred": m_pred,
            }

    weights = np.zeros(M, dtype=np.float64)
    weights[centre] += best["beta"]
    weights[best["segment"]] += 1.0 - best["beta"]

    if oracle_envelope is None:
        env = float(np.max(np.abs(cand)))
        if objective == "square":
            env = max(env, float(np.max(np.abs(y))))
        oracle_envelope = env

    return AggregationResult(
        weights=weights,
        fitted=best["pred"],
        m_fitted=best["m_pred"],
        objective=best["obj"],
        candidate_objectives=cand_obj,
        method=method_label,
        centre=centre,
        segment=int(best["segment"]),
        beta=float(best["beta"]),
        oracle=star_oracle_bound(n, M, envelope=oracle_envelope),
    )


# ---------------------------------------------------------------------------
# Convex aggregation (Definition 6)
# ---------------------------------------------------------------------------

def _simplex_qp(gram: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Minimise ``w'Qw - 2 w'r`` over the probability simplex.

    Both aggregation objectives reduce to this one convex quadratic program --
    with ``Q = A'A/n`` throughout, ``r = A'Y/n`` for the square loss and
    ``r = P_n[m(O, a_j)]`` for the Riesz loss -- so one solver serves both.
    """
    M = rhs.shape[0]
    if M == 1:
        return np.ones(1, dtype=np.float64)

    def fun(w: np.ndarray) -> float:
        return float(w @ gram @ w - 2.0 * w @ rhs)

    def jac(w: np.ndarray) -> np.ndarray:
        return 2.0 * (gram @ w - rhs)

    constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0,
                    "jac": lambda w: np.ones_like(w)},)
    bounds = [(0.0, 1.0)] * M

    starts = [np.full(M, 1.0 / M)]
    diag = np.diag(gram)
    single = diag - 2.0 * rhs
    best_single = np.zeros(M)
    best_single[int(np.argmin(single))] = 1.0
    starts.append(best_single)

    best_w, best_v = None, np.inf
    for w0 in starts:
        res = optimize.minimize(
            fun, w0, jac=jac, bounds=bounds, constraints=constraints,
            method="SLSQP", options={"maxiter": 500, "ftol": 1e-12},
        )
        w = np.clip(np.asarray(res.x, dtype=np.float64), 0.0, None)
        total = w.sum()
        w = w / total if total > 0 else np.full(M, 1.0 / M)
        value = fun(w)
        if value < best_v:
            best_w, best_v = w, value
    return best_w


def convex_aggregate(
    candidates: np.ndarray,
    y: Optional[np.ndarray] = None,
    m_candidates: Optional[np.ndarray] = None,
    objective: str = "square",
    method_label: str = "convex",
) -> AggregationResult:
    """Convex aggregation over the simplex (the Definition 6 weight problem).

    Richer than star aggregation -- it can mix every candidate at once, so its
    approximation benchmark satisfies ``e_bar <= e`` (Eq. 46 versus Eq. 42) -- but
    it pays twice: its estimation term grows like ``sqrt(M log n / n)`` instead of
    ``sqrt(log M / n)``, and pooling costs ``sqrt(M log n)`` instead of
    ``sqrt(log M)``. Use it for a small menu, and star aggregation for a large one;
    :func:`pragdml.diagnostics.pooling_conditions` makes the comparison concrete.

    Parameters
    ----------
    candidates, y, m_candidates, objective
        As in :func:`star_aggregate`.
    method_label : str, default "convex"

    Returns
    -------
    AggregationResult

    Examples
    --------
    >>> import numpy as np
    >>> from pragdml.aggregation import convex_aggregate
    >>> y = np.array([1.0, 2.0, 3.0, 4.0])
    >>> F = np.column_stack([y + 0.5, y - 0.5])
    >>> res = convex_aggregate(F, y=y)
    >>> bool(abs(res.weights[0] - 0.5) < 1e-4)
    True
    """
    if objective not in ("square", "riesz"):
        raise ValueError(f"objective must be 'square' or 'riesz', got {objective!r}")
    cand = _as_matrix(candidates, "candidates")
    n, M = cand.shape
    gram = cand.T @ cand / n

    if objective == "square":
        if y is None:
            raise ValueError("objective='square' requires `y`")
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        if y.shape[0] != n:
            raise ValueError(f"y has {y.shape[0]} rows, candidates have {n}")
        rhs = cand.T @ y / n
        m_cand = None
        cand_obj = np.array(
            [_square_objective(cand[:, j], y) for j in range(M)], dtype=np.float64
        )
    else:
        if m_candidates is None:
            raise ValueError("objective='riesz' requires `m_candidates`")
        m_cand = _as_matrix(m_candidates, "m_candidates")
        if m_cand.shape != cand.shape:
            raise ValueError(
                f"m_candidates has shape {m_cand.shape} but candidates {cand.shape}"
            )
        rhs = m_cand.mean(axis=0)
        cand_obj = np.array(
            [_riesz_objective(cand[:, j], m_cand[:, j]) for j in range(M)],
            dtype=np.float64,
        )

    weights = _simplex_qp(gram, rhs)
    fitted = cand @ weights
    m_fitted = None if m_cand is None else m_cand @ weights
    value = (
        _square_objective(fitted, y)
        if objective == "square"
        else _riesz_objective(fitted, m_fitted)
    )
    env = float(np.max(np.abs(cand)))
    if objective == "square":
        env = max(env, float(np.max(np.abs(y))))
    return AggregationResult(
        weights=weights,
        fitted=fitted,
        m_fitted=m_fitted,
        objective=float(value),
        candidate_objectives=cand_obj,
        method=method_label,
        oracle=convex_oracle_bound(n, M, envelope=env),
    )


# ---------------------------------------------------------------------------
# Pooled variants (Sec. 3.4)
# ---------------------------------------------------------------------------

def pooled_star_aggregate(
    oof_candidates: np.ndarray,
    y: Optional[np.ndarray] = None,
    m_oof_candidates: Optional[np.ndarray] = None,
    objective: str = "square",
    n_folds: int = 5,
) -> AggregationResult:
    """Definition 5: star weights fitted to pooled out-of-fold predictions.

    Each column of ``oof_candidates`` holds, for every observation ``i``, the
    prediction of one candidate **recipe** fitted on the training folds that
    exclude ``i``. The pooled objectives of Eqs. (40)-(41) are then ordinary
    averages over the whole sample, and Definition 5 observes that the resulting
    estimator *is* the Definition 4 star estimator computed from the augmented
    observations -- so this function delegates to :func:`star_aggregate` and only
    the bookkeeping and the reported remainder differ.

    What this buys: ``K`` fits of each candidate, the fewest cross-fitting allows,
    instead of the ``K K'`` that nested splitting would need (Remark 2).

    What must still hold (Remark 1): **each candidate must be trained on
    ``I_k^c`` alone.** Only the low-dimensional aggregation step may cross folds.
    If a candidate itself saw the evaluation fold, the candidates -- not merely a
    choice among them -- depend on that fold, and no result here applies.

    Parameters
    ----------
    oof_candidates : ndarray of shape (n, M)
        Out-of-fold predictions, one column per recipe.
    y, m_oof_candidates, objective
        As in :func:`star_aggregate`.
    n_folds : int, default 5
        ``K``, used only in the reported remainder (Prop. 7 carries a factor ``K``).

    Returns
    -------
    AggregationResult

    Examples
    --------
    >>> import numpy as np
    >>> from pragdml.aggregation import pooled_star_aggregate
    >>> y = np.arange(6.0)
    >>> F = np.column_stack([y + 0.3, y - 0.7])
    >>> res = pooled_star_aggregate(F, y=y, n_folds=3)
    >>> res.method
    'pooled_star'
    >>> bool(res.objective <= res.best_single_objective)
    True
    """
    res = star_aggregate(
        oof_candidates,
        y=y,
        m_candidates=m_oof_candidates,
        objective=objective,
        method_label="pooled_star",
    )
    n, M = np.asarray(oof_candidates).shape
    env = float(np.max(np.abs(np.asarray(oof_candidates, dtype=np.float64))))
    if objective == "square" and y is not None:
        env = max(env, float(np.max(np.abs(np.asarray(y, dtype=np.float64)))))
    res.oracle = star_oracle_bound(n, M, envelope=env, n_folds=n_folds, pooled=True)
    return res


def pooled_convex_aggregate(
    oof_candidates: np.ndarray,
    y: Optional[np.ndarray] = None,
    m_oof_candidates: Optional[np.ndarray] = None,
    objective: str = "square",
    n_folds: int = 5,
) -> AggregationResult:
    """Definition 6: convex weights fitted to pooled out-of-fold predictions.

    Same construction and same Remark 1 caveat as :func:`pooled_star_aggregate`,
    with the simplex weight problem in place of the star segment scan, and the
    Eq. (47)-(48) remainder in place of Eq. (43)-(44).

    Examples
    --------
    >>> import numpy as np
    >>> from pragdml.aggregation import pooled_convex_aggregate
    >>> y = np.arange(6.0)
    >>> F = np.column_stack([y + 0.3, y - 0.3])
    >>> res = pooled_convex_aggregate(F, y=y, n_folds=3)
    >>> res.method
    'pooled_convex'
    """
    res = convex_aggregate(
        oof_candidates,
        y=y,
        m_candidates=m_oof_candidates,
        objective=objective,
        method_label="pooled_convex",
    )
    n, M = np.asarray(oof_candidates).shape
    env = float(np.max(np.abs(np.asarray(oof_candidates, dtype=np.float64))))
    if objective == "square" and y is not None:
        env = max(env, float(np.max(np.abs(np.asarray(y, dtype=np.float64)))))
    res.oracle = convex_oracle_bound(
        n, M, envelope=env, n_folds=n_folds, pooled=True
    )
    return res


# ---------------------------------------------------------------------------
# Oracle remainders
# ---------------------------------------------------------------------------

def star_oracle_bound(
    n: int,
    n_candidates: int,
    envelope: float = 1.0,
    delta: float = 0.05,
    n_folds: int = 1,
    pooled: bool = False,
    constant: float = 1.0,
) -> dict:
    """The ``log(M)/n`` remainder of Eqs. (38)-(39), or (43)-(44) when pooled.

    Within a fold (Props. 5-6):
    ``C (B_Y + B_F)^2 log(M / delta) / n``.
    Pooled across folds (Prop. 7) the same shape carries a factor ``K``:
    ``C K (B_Y + B_F)^2 log(K M / delta) / n``.

    .. warning::
       The constant ``C`` in the paper is *universal but unspecified*. The number
       returned here therefore uses ``constant=1.0`` and is a **rate shape**, not a
       numerical guarantee: it tells you how the remainder scales in ``n``, ``M``
       and the envelope, and lets you compare two menu designs, but it is not a
       valid finite-sample bound. Treat the ratio of two such numbers as
       meaningful and the level as arbitrary.

    Parameters
    ----------
    n : int
        Effective sample size used for aggregation.
    n_candidates : int
        ``M``.
    envelope : float, default 1.0
        ``B_Y + B_F`` (square loss) or the matching Riesz envelope.
    delta : float, default 0.05
        The oracle inequality holds with probability at least ``1 - delta``.
    n_folds : int, default 1
        ``K``; only used when ``pooled=True``.
    pooled : bool, default False
        Whether to report the Proposition 7 form.
    constant : float, default 1.0
        The unspecified universal constant, exposed so it is never hidden.

    Returns
    -------
    dict

    Examples
    --------
    >>> from pragdml.aggregation import star_oracle_bound
    >>> out = star_oracle_bound(1000, 7, envelope=2.0)
    >>> out["scales_as"]
    'log(M/delta)/n'
    >>> bool(out["remainder"] > 0)
    True
    """
    n, M = int(n), int(n_candidates)
    if n < 1 or M < 1:
        raise ValueError("n and n_candidates must be >= 1")
    K = max(1, int(n_folds))
    if pooled:
        rem = constant * K * envelope**2 * np.log(K * M / delta) / n
        label = "K log(K M/delta)/n"
    else:
        rem = constant * envelope**2 * np.log(M / delta) / n
        label = "log(M/delta)/n"
    return {
        "remainder": float(rem),
        "sqrt_remainder": float(np.sqrt(max(rem, 0.0))),
        "n": n,
        "n_candidates": M,
        "n_folds": K,
        "envelope": float(envelope),
        "delta": float(delta),
        "pooled": bool(pooled),
        "constant": float(constant),
        "scales_as": label,
        "caveat": (
            "the universal constant is unspecified in the paper; this is a rate "
            "shape for comparing designs, not a finite-sample guarantee"
        ),
    }


def convex_oracle_bound(
    n: int,
    n_candidates: int,
    envelope: float = 1.0,
    delta: float = 0.05,
    n_folds: int = 1,
    pooled: bool = False,
    constant: float = 1.0,
) -> dict:
    """The ``M log(n)/n`` remainder of Eqs. (47)-(48) (Proposition 8).

    ``C K (B_Y + B)^2 (M log(3n) + log(K/delta)) / n``. The contrast with
    :func:`star_oracle_bound` is the whole design argument of Sec. 3.4: this term
    is **linear in ``M``**, so a convex menu cannot grow much, while the star
    remainder is logarithmic in ``M``.

    The same warning about the unspecified universal constant applies.

    Examples
    --------
    >>> from pragdml.aggregation import convex_oracle_bound
    >>> a = convex_oracle_bound(1000, 5)["remainder"]
    >>> b = convex_oracle_bound(1000, 50)["remainder"]
    >>> bool(b > 5 * a)
    True
    """
    n, M = int(n), int(n_candidates)
    if n < 1 or M < 1:
        raise ValueError("n and n_candidates must be >= 1")
    K = max(1, int(n_folds))
    factor = K if pooled else 1
    rem = (
        constant * factor * envelope**2
        * (M * np.log(3.0 * n) + np.log(K / delta)) / n
    )
    return {
        "remainder": float(rem),
        "sqrt_remainder": float(np.sqrt(max(rem, 0.0))),
        "n": n,
        "n_candidates": M,
        "n_folds": K,
        "envelope": float(envelope),
        "delta": float(delta),
        "pooled": bool(pooled),
        "constant": float(constant),
        "scales_as": "(M log(3n) + log(K/delta))/n",
        "caveat": (
            "the universal constant is unspecified in the paper; this is a rate "
            "shape for comparing designs, not a finite-sample guarantee"
        ),
    }
