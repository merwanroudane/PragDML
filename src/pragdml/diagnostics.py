"""The paper's identities and conditions, made computable.

The paper's conceptual core is a set of *identities* and a set of *conditions*.
The identities are exact algebra and can be checked to machine precision; the
conditions are assumptions, and the honest service a library can provide is to
compute their left-hand sides from fitted objects so the user can see whether
they are plausible rather than merely assumed.

The identities
--------------
**Proposition 1 (omitted-information identity), Eqs. (6)-(7).** The gap between
the full-information and representation-based targets is the inner product of two
representation errors:

.. math::

    \\theta^* - \\theta_\\lambda
      = E\\big[(g^* - g_\\lambda)(\\alpha^* - \\alpha_\\lambda)\\big],
    \\qquad
    |\\theta^* - \\theta_\\lambda| \\le \\|g^* - g_\\lambda\\|_{P,2}\\,
                                    \\|\\alpha^* - \\alpha_\\lambda\\|_{P,2}.

A representation preserves the target if it preserves **either** the outcome
regression **or** the balancing weight -- only the *product* matters. This is
:func:`omitted_information_gap`.

**Proposition 2(ii)-(iii), Eqs. (23)-(24).** The same algebra governs the drift of
the orthogonal score and the distance from the pipeline target ``theta_0`` to
``theta^*``: :func:`mixed_bias_drift` and :func:`pipeline_target_gap`.

**Lemma 2, Eqs. (28)-(29).** On the common space ``H`` excess risk *equals*
squared ``L2`` error, for both heads: :func:`excess_risk_identity`.

The conditions
--------------
**Definition 1, target adaptivity, Eqs. (17)-(18)** and **Definition 2, adaptive
efficiency, Eq. (19)** -- :func:`target_adaptivity` and
:func:`adaptive_efficiency`. **Eq. (14)/(32), the product-rate condition** --
:func:`product_rate`. **Eqs. (45)/(49), the extra price of pooling weights across
folds** -- :func:`pooling_conditions`. **Eq. (35), the critical radius of a signed
``l1`` hull** -- :func:`critical_radius_l1_hull`.

A caution that applies to all of them: these diagnostics are computed by treating
a *richer* fitted pipeline as a stand-in for the unknown full-information objects
``(g^*, alpha^*)``. That is exactly the paper's own Sec. 5.4 deletion exercise, and
it bounds only the information the richer pipeline has. Information outside every
observed modality is what the sensitivity analysis of :mod:`pragdml.sensitivity`
is for.

Author
------
Dr Merwan Roudane <merwanroudane920@gmail.com>
https://github.com/merwanroudane
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

__all__ = [
    "omitted_information_gap",
    "OmittedInformation",
    "mixed_bias_drift",
    "pipeline_target_gap",
    "excess_risk_identity",
    "target_adaptivity",
    "adaptive_efficiency",
    "product_rate",
    "pooling_conditions",
    "critical_radius_l1_hull",
]


def _l2(a: np.ndarray) -> float:
    """``L2(P_n)`` norm, i.e. root mean square."""
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    return float(np.sqrt(np.mean(a**2)))


@dataclass(frozen=True)
class OmittedInformation:
    """Result of :func:`omitted_information_gap` (Proposition 1).

    Attributes
    ----------
    gap : float
        ``E[(g_long - g_short)(alpha_long - alpha_short)]`` -- the **exact** target
        gap of Eq. (6), signed.
    bound : float
        The Cauchy-Schwarz bound of Eq. (7), ``>= |gap|`` always.
    g_error : float
        ``||g_long - g_short||_{P,2}``, the regression information loss.
    alpha_error : float
        ``||alpha_long - alpha_short||_{P,2}``, the representer information loss.
    rho : float
        Normalised alignment ``gap / (g_error * alpha_error)`` in ``[-1, 1]``,
        ``0.0`` when either norm vanishes. **Uncentred**, matching Eq. (50) of the
        paper and Theorem 2 of [9]; this is not the Pearson correlation unless both
        errors are mean-zero.
    """

    gap: float
    bound: float
    g_error: float
    alpha_error: float
    rho: float

    @property
    def is_tight(self) -> bool:
        """Whether the Cauchy-Schwarz bound is attained (``|rho| = 1``)."""
        return bool(abs(abs(self.rho) - 1.0) < 1e-10)

    def scaled_gap(self, n: int) -> float:
        """``sqrt(n) * |gap|`` -- the quantity Definition 1 requires to vanish."""
        return float(np.sqrt(int(n)) * abs(self.gap))


def omitted_information_gap(
    g_long: np.ndarray,
    g_short: np.ndarray,
    alpha_long: np.ndarray,
    alpha_short: np.ndarray,
) -> OmittedInformation:
    """Proposition 1: the target gap as a product of two representation errors.

    Computes Eqs. (6)-(7) from four evaluated nuisance vectors. In practice
    ``*_long`` come from a richer pipeline (more modalities, a bigger basis) and
    ``*_short`` from the pipeline actually used -- the paper's Sec. 5.4
    full-versus-text comparison.

    Parameters
    ----------
    g_long, g_short : array-like of shape (n,)
        Outcome regressions evaluated at the observed ``(D_i, X_i)``.
    alpha_long, alpha_short : array-like of shape (n,)
        Riesz representers evaluated at the same points.

    Returns
    -------
    OmittedInformation

    Notes
    -----
    The identity is exact, so ``bound >= abs(gap)`` holds to machine precision and
    is asserted in ``tests/test_diagnostics.py``. Equality happens exactly when
    the two error vectors are proportional.

    References
    ----------
    Aradillas Fernandez et al. (2026), Proposition 1, Eqs. (6)-(7),
    arXiv:2610.01935. Chernozhukov, Cinelli, Newey, Sharma & Syrgkanis (2026),
    Theorem 2, doi:10.1162/REST.a.1705.

    Examples
    --------
    >>> import numpy as np
    >>> from pragdml.diagnostics import omitted_information_gap
    >>> gl = np.array([1.0, 2.0]); gs = np.array([0.0, 0.0])
    >>> al = np.array([1.0, 1.0]); a_s = np.array([0.0, 0.0])
    >>> res = omitted_information_gap(gl, gs, al, a_s)
    >>> round(res.gap, 6), round(res.bound, 6)
    (1.5, 1.581139)
    >>> res.is_tight
    False
    """
    dg = np.asarray(g_long, dtype=np.float64).reshape(-1) - np.asarray(
        g_short, dtype=np.float64
    ).reshape(-1)
    da = np.asarray(alpha_long, dtype=np.float64).reshape(-1) - np.asarray(
        alpha_short, dtype=np.float64
    ).reshape(-1)
    if dg.shape != da.shape:
        raise ValueError(
            f"regression errors have length {dg.shape[0]} but representer errors "
            f"{da.shape[0]}; they must match."
        )
    gap = float(np.mean(dg * da))
    gn, an = _l2(dg), _l2(da)
    bound = float(gn * an)
    rho = 0.0 if (gn == 0.0 or an == 0.0) else float(gap / bound)
    return OmittedInformation(
        gap=gap, bound=bound, g_error=gn, alpha_error=an, rho=float(np.clip(rho, -1, 1))
    )


def mixed_bias_drift(
    g_fitted: np.ndarray,
    g_target: np.ndarray,
    alpha_fitted: np.ndarray,
    alpha_target: np.ndarray,
) -> float:
    """Proposition 2(ii) / Eq. (9): the population drift of the orthogonal score.

    .. math::
        E[\\psi(O; g, \\alpha)] - \\theta_0 = -E[(g - g_0)(\\alpha - \\alpha_0)]

    The sign matters: a *positive* product of errors biases the score **down**.
    Returning the drift itself (not its absolute value) lets a user see the
    direction, which is what makes the diagnostic actionable.

    Examples
    --------
    >>> import numpy as np
    >>> from pragdml.diagnostics import mixed_bias_drift
    >>> mixed_bias_drift(np.array([1.0]), np.array([0.0]),
    ...                  np.array([2.0]), np.array([0.0]))
    -2.0
    """
    return -omitted_information_gap(
        g_fitted, g_target, alpha_fitted, alpha_target
    ).gap


def pipeline_target_gap(
    g_long: np.ndarray,
    g_pipeline: np.ndarray,
    alpha_long: np.ndarray,
    alpha_pipeline: np.ndarray,
) -> OmittedInformation:
    """Proposition 2(iii) / Eq. (24): how far ``theta_0`` can lie from ``theta^*``.

    Algebraically identical to :func:`omitted_information_gap`; kept as a separate
    name because it answers a different question -- there the short object is a
    fixed representation, here it is the population target of an end-to-end
    fold-wise tuned pipeline (Definition 3).
    """
    return omitted_information_gap(g_long, g_pipeline, alpha_long, alpha_pipeline)


def excess_risk_identity(
    risk: float, risk_min: float, l2_error: float, tol: float = 1e-8
) -> dict:
    """Lemma 2, Eqs. (28)-(29): excess risk equals squared ``L2`` error on ``H``.

    Holds for the square loss and for the Riesz loss alike, which is what lets the
    paper state its rate requirement on either. Use this to verify that a learner
    really is minimising over the claimed linear space: a large ``discrepancy``
    means the fitted function left ``H`` (a penalty, an early stop, a non-linear
    head), and then the clean identity no longer applies.

    Parameters
    ----------
    risk : float
        ``R(g_hat)`` or ``L(alpha_hat)``.
    risk_min : float
        ``R(g_0)`` or ``L(alpha_0)``.
    l2_error : float
        ``||g_hat - g_0||_{P,2}`` or ``||alpha_hat - alpha_0||_{P,2}``.
    tol : float, default 1e-8
        Tolerance for the ``holds`` flag.

    Returns
    -------
    dict
        ``excess_risk``, ``squared_error``, ``discrepancy``, ``holds``.

    Examples
    --------
    >>> from pragdml.diagnostics import excess_risk_identity
    >>> excess_risk_identity(1.25, 1.0, 0.5)["holds"]
    True
    """
    excess = float(risk) - float(risk_min)
    sq = float(l2_error) ** 2
    disc = abs(excess - sq)
    return {
        "excess_risk": excess,
        "squared_error": sq,
        "discrepancy": disc,
        "holds": bool(disc <= float(tol) * max(1.0, abs(sq))),
    }


def target_adaptivity(
    n: int,
    g_error: float,
    alpha_error: float,
    gap: Optional[float] = None,
    threshold: float = 0.1,
) -> dict:
    """Definition 1, Eqs. (17)-(18): is the target gap negligible at ``sqrt(n)``?

    Reports both forms of the condition:

    * the **exact** requirement ``sqrt(n)|theta^* - theta_lambda| = o(1)``, when
      ``gap`` is supplied;
    * the **sufficient** product form ``sqrt(n)||g^*-g_l|| ||alpha^*-alpha_l||
      = o(1)`` of Eq. (18), which is conservative because it ignores alignment.

    The product form can declare trouble where the exact form is fine: the two
    errors may be large yet nearly orthogonal, in which case the target barely
    moves. That is the practical content of Proposition 1, so both are shown.

    Parameters
    ----------
    n : int
        Sample size.
    g_error, alpha_error : float
        The two ``L2`` representation errors.
    gap : float, optional
        The exact gap from :func:`omitted_information_gap`.
    threshold : float, default 0.1
        What counts as "small" relative to one standard error. A scaled gap of
        ``0.1`` shifts a 95% interval by about 5% of its half-width; it is a
        reporting convention, not a test, and no asymptotic statement follows from
        it.

    Returns
    -------
    dict
        ``scaled_product``, ``scaled_gap`` (or ``None``), ``threshold``,
        ``product_ok``, ``gap_ok``, ``verdict``.

    Examples
    --------
    >>> from pragdml.diagnostics import target_adaptivity
    >>> out = target_adaptivity(10000, 0.01, 0.01, gap=1e-5)
    >>> out["product_ok"], out["gap_ok"]
    (True, True)
    >>> target_adaptivity(10000, 1.0, 1.0)["verdict"]
    'product condition fails; check the exact gap and report a sensitivity region'
    """
    n = int(n)
    root = float(np.sqrt(n))
    scaled_product = float(root * float(g_error) * float(alpha_error))
    scaled_gap = None if gap is None else float(root * abs(float(gap)))
    product_ok = bool(scaled_product <= threshold)
    gap_ok = None if scaled_gap is None else bool(scaled_gap <= threshold)

    if gap_ok is True:
        verdict = "target-adaptive at this sample size (exact gap is negligible)"
    elif gap_ok is False:
        verdict = (
            "exact gap is NOT negligible; report a sensitivity region for theta*"
        )
    elif product_ok:
        verdict = "sufficient product condition holds"
    else:
        verdict = (
            "product condition fails; check the exact gap and report a "
            "sensitivity region"
        )
    return {
        "scaled_product": scaled_product,
        "scaled_gap": scaled_gap,
        "threshold": float(threshold),
        "product_ok": product_ok,
        "gap_ok": gap_ok,
        "verdict": verdict,
    }


def adaptive_efficiency(
    g_error: float,
    alpha_error: float,
    alpha_norm: float,
    tol: float = 0.05,
) -> dict:
    """Definition 2, Eq. (19): does the short score approach the efficient one?

    Requires ``||g^*-g_l|| -> 0``, ``||alpha^*-alpha_l|| -> 0`` and
    ``sup_n ||alpha_l||_{P,2} < inf``. When this holds **together with** target
    adaptivity the representation is *information-lossless* for the target, and by
    Theorem 2 the estimator attains the semiparametric efficiency bound for
    ``theta^*``.

    The distinction that matters in practice: target adaptivity decides **which
    parameter** the interval covers, adaptive efficiency decides **how wide** it
    has to be. A representation can be adaptive but inefficient.

    Parameters
    ----------
    g_error, alpha_error : float
        The two ``L2`` representation errors.
    alpha_norm : float
        ``||alpha_lambda||_{P,2}``, from
        :attr:`~pragdml._core.CrossFitOutput.nu2_representer` as its square root.
    tol : float, default 0.05
        Relative size below which an error counts as "approaching zero". A
        reporting convention only.

    Returns
    -------
    dict

    Examples
    --------
    >>> from pragdml.diagnostics import adaptive_efficiency
    >>> adaptive_efficiency(0.01, 0.01, 2.0)["plausible"]
    True
    """
    g_error, alpha_error, alpha_norm = (
        float(g_error), float(alpha_error), float(alpha_norm)
    )
    rel_alpha = alpha_error / alpha_norm if alpha_norm > 0 else float("inf")
    g_ok = bool(g_error <= tol)
    a_ok = bool(rel_alpha <= tol)
    bounded = bool(np.isfinite(alpha_norm) and alpha_norm > 0)
    return {
        "g_error": g_error,
        "alpha_error": alpha_error,
        "alpha_norm": alpha_norm,
        "alpha_error_relative": rel_alpha,
        "regression_ok": g_ok,
        "representer_ok": a_ok,
        "representer_bounded": bounded,
        "plausible": bool(g_ok and a_ok and bounded),
        "note": (
            "with target adaptivity this is the information-lossless case of "
            "Theorem 2, and the estimator attains the efficiency bound for theta*"
        ),
    }


def product_rate(n: int, g_rate: float, alpha_rate: float) -> dict:
    """Eq. (14) / Eq. (32): the DML product-rate condition.

    Requires ``sqrt(n) * r_g * r_alpha -> 0``. Each nuisance converging at
    ``o(n^{-1/4})`` suffices, and faster learning of one compensates for slower
    learning of the other -- which is the practical reason the two heads can be
    learned by completely different methods.

    Parameters
    ----------
    n : int
    g_rate, alpha_rate : float
        ``L2`` errors (or rate bounds) of the two fitted nuisances.

    Returns
    -------
    dict
        Includes ``n_quarter_root``, the ``n^{-1/4}`` benchmark each error is
        compared against.

    Examples
    --------
    >>> from pragdml.diagnostics import product_rate
    >>> product_rate(10000, 0.02, 0.02)["satisfied"]
    True
    """
    n = int(n)
    scaled = float(np.sqrt(n) * float(g_rate) * float(alpha_rate))
    bench = float(n ** (-0.25))
    return {
        "n": n,
        "g_rate": float(g_rate),
        "alpha_rate": float(alpha_rate),
        "scaled_product": scaled,
        "n_quarter_root": bench,
        "g_beats_quarter_root": bool(float(g_rate) <= bench),
        "alpha_beats_quarter_root": bool(float(alpha_rate) <= bench),
        "satisfied": bool(scaled <= 0.1),
    }


def pooling_conditions(
    n: int,
    n_candidates: int,
    g_rate: float,
    alpha_rate: float,
    weights: str = "star",
) -> dict:
    """Eqs. (45) and (49): the extra condition paid for pooling weights.

    Fitting aggregation weights on pooled out-of-fold predictions breaks strict
    cross-fitting for the *aggregate* (the weights see the evaluation fold), so
    Theorems 4 and 5 need a second condition beyond the product rate:

    * star weights (Eq. 45): ``sqrt(log M) (r_g + r_alpha) -> 0``, reducing to
      ``log M = o(sqrt(n))`` at the usual rates -- so ``M`` may grow faster than
      any power of ``n``;
    * convex weights (Eq. 49): ``sqrt(M log n) (r_g + r_alpha) -> 0``, reducing to
      ``M log n = o(sqrt(n))`` -- so ``M`` must stay small.

    This is the quantitative statement of the paper's "the breach is cheap"
    claim, and it is the single calculation that should decide between the two
    aggregation schemes.

    Parameters
    ----------
    n : int
    n_candidates : int
        ``M``, the menu size.
    g_rate, alpha_rate : float
    weights : {"star", "convex"}, default "star"

    Returns
    -------
    dict

    Examples
    --------
    >>> from pragdml.diagnostics import pooling_conditions
    >>> pooling_conditions(10000, 7, 0.02, 0.02)["satisfied"]
    True
    >>> pooling_conditions(400, 50, 0.3, 0.3, weights="convex")["satisfied"]
    False
    """
    if weights not in ("star", "convex"):
        raise ValueError(f"weights must be 'star' or 'convex', got {weights!r}")
    n, M = int(n), int(n_candidates)
    if M < 1:
        raise ValueError(f"n_candidates must be >= 1, got {n_candidates}")
    rates = float(g_rate) + float(alpha_rate)
    prod = product_rate(n, g_rate, alpha_rate)
    if weights == "star":
        complexity = float(np.sqrt(np.log(max(M, 2))))
        reduced = float(np.log(max(M, 2)) / np.sqrt(n))
        label = "sqrt(log M) (r_g + r_alpha)"
        reduced_label = "log M / sqrt(n)"
    else:
        complexity = float(np.sqrt(M * np.log(max(n, 2))))
        reduced = float(M * np.log(max(n, 2)) / np.sqrt(n))
        label = "sqrt(M log n) (r_g + r_alpha)"
        reduced_label = "M log n / sqrt(n)"
    pooling_term = complexity * rates
    return {
        "weights": weights,
        "n": n,
        "n_candidates": M,
        "product_rate": prod["scaled_product"],
        "product_rate_satisfied": prod["satisfied"],
        "pooling_term": pooling_term,
        "pooling_term_label": label,
        "reduced_condition": reduced,
        "reduced_condition_label": reduced_label,
        "pooling_satisfied": bool(pooling_term <= 0.1),
        "satisfied": bool(prod["satisfied"] and pooling_term <= 0.1),
        "note": (
            "star aggregation costs log M and tolerates a large menu; convex "
            "aggregation costs sqrt(M log n) and needs M = o(sqrt(n))"
        ),
    }


def critical_radius_l1_hull(
    n: int, n_candidates: int, radius: float, sup_norm: float
) -> dict:
    """Eq. (35): the critical radius of a signed ``l1`` hull of ``M`` candidates.

    .. math::

        r_n \\lesssim \\min\\Big\\{\\sqrt{M/n},\\;
          \\big(B \\max_j \\|f_j\\|_\\infty\\big)
          \\big(\\log(2M)/n\\big)^{1/4}\\Big\\}

    The two regimes are the point of the formula: the first behaves like
    ``sqrt(M/n)``, so reaching an ``n^{-1/4}`` nuisance rate by *proper* convex
    aggregation typically needs ``M = o(sqrt(n))``; the second grows only
    logarithmically in ``M`` but pays the envelope constant ``B * max_j
    ||f_j||_inf``. Which regime binds is exactly what decides whether to enlarge
    the dictionary or switch to star aggregation.

    Parameters
    ----------
    n : int
        Effective training size (``n_tr``, not the full sample -- see the paper's
        cross-fitting convention at the start of Sec. 3).
    n_candidates : int
        ``M``.
    radius : float
        The ``l1`` radius ``B``.
    sup_norm : float
        ``max_j ||f_j||_inf``, an envelope bound on the candidates.

    Returns
    -------
    dict
        ``radius`` (the bound ``r_n``), both regimes, and which one binds.

    Examples
    --------
    >>> from pragdml.diagnostics import critical_radius_l1_hull
    >>> out = critical_radius_l1_hull(1000, 7, radius=2.0, sup_norm=1.0)
    >>> out["binding_regime"]
    'sqrt(M/n)'
    """
    n, M = int(n), int(n_candidates)
    if n < 1 or M < 1:
        raise ValueError("n and n_candidates must be >= 1")
    dim_regime = float(np.sqrt(M / n))
    ent_regime = float(
        float(radius) * float(sup_norm) * (np.log(2.0 * M) / n) ** 0.25
    )
    binding = "sqrt(M/n)" if dim_regime <= ent_regime else "B*sup*(log 2M/n)^(1/4)"
    return {
        "n": n,
        "n_candidates": M,
        "dimension_regime": dim_regime,
        "entropy_regime": ent_regime,
        "radius": float(min(dim_regime, ent_regime)),
        "binding_regime": binding,
        "n_quarter_root": float(n ** (-0.25)),
        "beats_quarter_root": bool(min(dim_regime, ent_regime) <= n ** (-0.25)),
    }
