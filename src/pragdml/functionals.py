"""Riesz-linear functionals ``m(O, h)``.

The paper studies targets that are continuous linear functionals of a regression,

.. math::

    \\theta^* = E[m(O, g^*)]   \\quad\\text{(Eq. 2)},\\qquad
    \\theta_\\lambda = E[m(O, g_\\lambda)] \\quad\\text{(Eq. 3)},

where ``m`` is measurable and **linear in its second argument**. Every object in
this module is one such ``m``. Because ``m`` is linear in ``h``, a single
implementation serves three jobs at once:

* evaluating ``m(O_i, g)`` for a fitted outcome regression (the first term of the
  debiased score, Eq. 8);
* building the right-hand side of the Riesz normal equations, ``r_j = P_n m(O, phi_j)``
  (:mod:`pragdml.riesz`);
* evaluating ``m(O_i, alpha)`` inside the Riesz loss ``E[alpha^2 - 2 m(O, alpha)]``.

All three reduce to "apply ``m`` to a callable that returns either an ``(n,)``
vector or an ``(n, p)`` matrix of basis values", which is what
:meth:`Functional.apply` does.

Compatibility map
-----------------
========================================  =========================================
Paper                                     Here
========================================  =========================================
Eq. (2)-(3), class of targets             :class:`Functional`
Footnote 1, p. 6: ATE ``m = h(1,z)-h(0,z)``  :class:`ATE`
Sec. 2.1.1, average potential outcome     :class:`AveragePotentialOutcome`
Sec. 2.1.1, weighted ATE                  :class:`WeightedATE`
Eq. (67), average partial derivative       :class:`AverageDerivative`
Sec. 2.1.1, policy effects                :class:`PolicyEffect`
Extension E7 (ratio target)               :class:`ATT`
Sec. 2.1.2, analytic representers          :meth:`Functional.analytic_representer`
========================================  =========================================

Author
------
Dr Merwan Roudane <merwanroudane920@gmail.com>
https://github.com/merwanroudane
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import Callable, Optional

import numpy as np

__all__ = [
    "Functional",
    "ATE",
    "ATT",
    "AveragePotentialOutcome",
    "WeightedATE",
    "AverageDerivative",
    "PolicyEffect",
    "get_functional",
]

#: Callable signature accepted by :meth:`Functional.apply`.
#: ``h(d, X)`` must broadcast over ``d`` and return ``(n,)`` or ``(n, p)``.
Evaluable = Callable[[np.ndarray, np.ndarray], np.ndarray]


def _as2d(a: np.ndarray) -> tuple[np.ndarray, bool]:
    """Return ``(a_2d, was_1d)`` so one code path serves vectors and matrices."""
    arr = np.asarray(a, dtype=np.float64)
    if arr.ndim == 1:
        return arr[:, None], True
    if arr.ndim != 2:
        raise ValueError(
            f"an evaluable must return a 1-d or 2-d array, got ndim={arr.ndim}"
        )
    return arr, False


def _restore(arr: np.ndarray, was_1d: bool) -> np.ndarray:
    return arr[:, 0] if was_1d else arr


class Functional(ABC):
    """Base class for a Riesz-linear functional ``m(O, h)``.

    Subclasses implement :meth:`apply`. Linearity in ``h`` is what makes the whole
    package work, so :meth:`apply` must satisfy
    ``apply(a*h1 + b*h2) == a*apply(h1) + b*apply(h2)`` exactly; this is asserted
    for every shipped functional in ``tests/test_functionals.py``.

    Attributes
    ----------
    name : str
        Short label used in tables and figures.
    is_ratio : bool
        ``True`` for targets that are a ratio of two means (only :class:`ATT`).
        Ratio targets get the delta-method correction in :mod:`pragdml._core`;
        they are an extension beyond the paper's linear class (E7).
    requires_binary_treatment : bool
        ``True`` when ``D`` must be in ``{0, 1}``.
    requires_smooth_in_d : bool
        ``True`` when ``m`` differentiates ``h`` in ``d``, so a step-function
        learner (a tree or a forest) is not admissible.
    """

    name: str = "functional"
    is_ratio: bool = False
    requires_binary_treatment: bool = False
    requires_smooth_in_d: bool = False

    @abstractmethod
    def apply(self, h: Evaluable, D: np.ndarray, X: np.ndarray) -> np.ndarray:
        """Evaluate ``m(O_i, h)`` for every observation.

        Parameters
        ----------
        h : callable
            ``h(d, X)`` returning ``(n,)`` for a single function or ``(n, p)`` for
            ``p`` basis functions evaluated at the same arguments. ``d`` is an
            ``(n,)`` array of treatment values at which to evaluate.
        D : ndarray of shape (n,)
            Observed treatment.
        X : ndarray of shape (n, k)
            Representation (the controls actually used).

        Returns
        -------
        ndarray
            ``(n,)`` or ``(n, p)``, matching the shape ``h`` returned.
        """

    def denominator(self, D: np.ndarray, X: np.ndarray) -> np.ndarray:
        """Per-observation denominator for a ratio target. Non-ratio: all ones."""
        return np.ones(np.asarray(D).shape[0], dtype=np.float64)

    def analytic_representer(
        self, D: np.ndarray, X: np.ndarray, propensity: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """The closed-form Riesz representer, when one exists.

        Only :class:`ATE` and :class:`ATT` override this (Sec. 2.1.2 gives the
        inverse-propensity form). Everything else learns the representer by Riesz
        regression instead, which is the whole point of the Riesz calculus of [8].
        """
        raise NotImplementedError(
            f"{type(self).__name__} has no closed-form Riesz representer; learn it "
            "with pragdml.riesz instead (this is the normal case)."
        )

    def validate(self, D: np.ndarray, X: np.ndarray) -> None:
        """Check the treatment against the functional's requirements."""
        D = np.asarray(D, dtype=np.float64)
        if self.requires_binary_treatment:
            uniq = np.unique(D[np.isfinite(D)])
            if not np.all(np.isin(uniq, (0.0, 1.0))):
                raise ValueError(
                    f"{type(self).__name__} requires a binary treatment D in {{0, 1}}; "
                    f"got {len(uniq)} distinct values, e.g. {uniq[:5]}. "
                    "For a continuous treatment use AverageDerivative."
                )
            if uniq.size < 2:
                raise ValueError(
                    f"{type(self).__name__} needs both treatment arms present; D is "
                    "constant."
                )

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{type(self).__name__}()"


class ATE(Functional):
    """Average treatment effect: ``m(O, h) = h(1, Z) - h(0, Z)``.

    This is the paper's leading example (footnote 1, p. 6). With the full
    information set the target is ``theta* = E[g*(1,Z) - g*(0,Z)]``; with a learned
    representation it is the **average predictive effect (APE)**,
    ``theta_lambda = E[g_lambda(1,X) - g_lambda(0,X)]``.

    References
    ----------
    Aradillas Fernandez et al. (2026), Sec. 2.1.1 and footnote 1, arXiv:2610.01935.
    Chernozhukov, Newey & Singh (2022), *Econometrica* 90(3):967-1027,
    doi:10.3982/ECTA18515.

    Examples
    --------
    >>> import numpy as np
    >>> from pragdml.functionals import ATE
    >>> X = np.array([[0.0], [1.0]]); D = np.array([0.0, 1.0])
    >>> h = lambda d, X: 2.0 * d + X[:, 0]
    >>> ATE().apply(h, D, X)
    array([2., 2.])
    """

    name = "ATE"
    requires_binary_treatment = True

    def apply(self, h: Evaluable, D: np.ndarray, X: np.ndarray) -> np.ndarray:
        n = np.asarray(D).shape[0]
        one = np.ones(n, dtype=np.float64)
        hi, was_1d = _as2d(h(one, X))
        lo, _ = _as2d(h(np.zeros(n, dtype=np.float64), X))
        return _restore(hi - lo, was_1d)

    def analytic_representer(
        self, D: np.ndarray, X: np.ndarray, propensity: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """``alpha(D, X) = D/e(X) - (1-D)/(1-e(X))`` (Sec. 2.1.2).

        Parameters
        ----------
        propensity : ndarray of shape (n,)
            Fitted ``e(X) = P(D = 1 | X)``. Required.
        """
        if propensity is None:
            raise ValueError("ATE.analytic_representer needs `propensity`, e(X).")
        D = np.asarray(D, dtype=np.float64)
        e = np.asarray(propensity, dtype=np.float64)
        if np.any((e <= 0.0) | (e >= 1.0)):
            raise ValueError(
                "fitted propensity scores must lie strictly in (0, 1); "
                f"range seen: [{e.min():.3g}, {e.max():.3g}]. Trim or regularise the "
                "propensity learner."
            )
        return D / e - (1.0 - D) / (1.0 - e)


class ATT(Functional):
    """Average treatment effect on the treated (a **ratio** target; extension E7).

    ``theta = E[D (g(1,Z) - g(0,Z))] / E[D]``. The numerator is Riesz-linear with
    the known weight ``D``; the denominator is a sample mean. :mod:`pragdml._core`
    combines the two with the delta method, so the reported standard error accounts
    for estimating ``P(D = 1)`` rather than treating it as known.

    This target is **outside** the strictly linear class of Eqs. (2)-(3); it is
    offered because Sec. 2.1.1 lists ATT-type parameters as covered by the Riesz
    calculus, and because applied users expect it. The mixed-bias identity of
    Proposition 1 applies to the numerator.

    References
    ----------
    Chernozhukov, Newey & Singh (2022), *Econometrica* 90(3):967-1027.

    Examples
    --------
    >>> import numpy as np
    >>> from pragdml.functionals import ATT
    >>> X = np.zeros((4, 1)); D = np.array([0.0, 1.0, 1.0, 0.0])
    >>> ATT().apply(lambda d, X: 3.0 * d, D, X)
    array([0., 3., 3., 0.])
    """

    name = "ATT"
    is_ratio = True
    requires_binary_treatment = True

    def apply(self, h: Evaluable, D: np.ndarray, X: np.ndarray) -> np.ndarray:
        D = np.asarray(D, dtype=np.float64)
        n = D.shape[0]
        hi, was_1d = _as2d(h(np.ones(n), X))
        lo, _ = _as2d(h(np.zeros(n), X))
        return _restore(D[:, None] * (hi - lo), was_1d)

    def denominator(self, D: np.ndarray, X: np.ndarray) -> np.ndarray:
        return np.asarray(D, dtype=np.float64).copy()

    def analytic_representer(
        self, D: np.ndarray, X: np.ndarray, propensity: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """``alpha(D,X) = D - (1-D) e(X)/(1-e(X))`` (numerator representer)."""
        if propensity is None:
            raise ValueError("ATT.analytic_representer needs `propensity`, e(X).")
        D = np.asarray(D, dtype=np.float64)
        e = np.asarray(propensity, dtype=np.float64)
        if np.any((e <= 0.0) | (e >= 1.0)):
            raise ValueError("fitted propensity scores must lie strictly in (0, 1).")
        return D - (1.0 - D) * e / (1.0 - e)


class AveragePotentialOutcome(Functional):
    """Average potential outcome at a fixed treatment: ``m(O, h) = h(d0, Z)``.

    Parameters
    ----------
    d0 : float
        The treatment value to evaluate at.

    Examples
    --------
    >>> import numpy as np
    >>> from pragdml.functionals import AveragePotentialOutcome
    >>> X = np.ones((2, 1)); D = np.array([0.0, 1.0])
    >>> AveragePotentialOutcome(1.0).apply(lambda d, X: d + X[:, 0], D, X)
    array([2., 2.])
    """

    name = "APO"

    def __init__(self, d0: float = 1.0) -> None:
        self.d0 = float(d0)
        self.name = f"APO(d={self.d0:g})"

    def apply(self, h: Evaluable, D: np.ndarray, X: np.ndarray) -> np.ndarray:
        n = np.asarray(D).shape[0]
        out, was_1d = _as2d(h(np.full(n, self.d0, dtype=np.float64), X))
        return _restore(out, was_1d)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"AveragePotentialOutcome(d0={self.d0!r})"


class WeightedATE(Functional):
    """Weighted ATE: ``m(O, h) = w(Z) [h(1,Z) - h(0,Z)]`` with ``w`` known.

    Parameters
    ----------
    weight : callable
        ``weight(X) -> (n,)``. Must be a **known** function of the controls, as
        Sec. 2.1.1 requires ("averages of ``g(d,z)`` against a known weight").

    Examples
    --------
    >>> import numpy as np
    >>> from pragdml.functionals import WeightedATE
    >>> X = np.array([[1.0], [2.0]]); D = np.array([0.0, 1.0])
    >>> f = WeightedATE(lambda X: X[:, 0])
    >>> f.apply(lambda d, X: 2.0 * d, D, X)
    array([2., 4.])
    """

    name = "Weighted ATE"
    requires_binary_treatment = True

    def __init__(self, weight: Callable[[np.ndarray], np.ndarray]) -> None:
        if not callable(weight):
            raise TypeError("`weight` must be a callable weight(X) -> (n,)")
        self.weight = weight

    def apply(self, h: Evaluable, D: np.ndarray, X: np.ndarray) -> np.ndarray:
        n = np.asarray(D).shape[0]
        hi, was_1d = _as2d(h(np.ones(n), X))
        lo, _ = _as2d(h(np.zeros(n), X))
        w = np.asarray(self.weight(X), dtype=np.float64).reshape(-1)
        if w.shape[0] != n:
            raise ValueError(f"weight(X) returned {w.shape[0]} values, expected {n}")
        return _restore(w[:, None] * (hi - lo), was_1d)


class AverageDerivative(Functional):
    """Average partial derivative: ``m(O, h) = d/dd h(d, Z) | d = D`` (Eq. 67).

    This is the functional of the paper's multimodal demand application: with the
    outcome and the price both in logs, the target is an elasticity-like average
    response (Sec. 5.1).

    The derivative is taken **exactly** when the evaluable exposes a ``deriv_d``
    attribute (every learner in :mod:`pragdml.learners` does, because the paper's
    Eqs. (68)-(69) architecture is linear in ``D``), and by a central difference
    otherwise.

    Parameters
    ----------
    step : float, optional
        Central-difference step. ``None`` (default) picks
        ``eps**(1/3) * max(sd(D), 1)``, the standard choice that balances
        truncation against cancellation error. Ignored when an exact derivative is
        available.

    Warnings
    --------
    A piecewise-constant learner (decision tree, random forest, gradient boosting)
    has zero derivative almost everywhere, so a numerical derivative of it is
    meaningless. Use a learner that is smooth in ``d`` -- the paper's own
    varying-coefficient architecture (:class:`pragdml.learners.VaryingCoefficientRegressor`)
    is linear in ``D`` for exactly this reason. A warning is raised when the
    numerical derivative is identically zero.

    References
    ----------
    Aradillas Fernandez et al. (2026), Eq. (67) and Sec. 5.1-5.2.

    Examples
    --------
    >>> import numpy as np
    >>> from pragdml.functionals import AverageDerivative
    >>> X = np.zeros((3, 1)); D = np.array([0.5, 1.0, 2.0])
    >>> np.round(AverageDerivative().apply(lambda d, X: 3.0 * d, D, X), 6)
    array([3., 3., 3.])
    """

    name = "Average derivative"
    requires_smooth_in_d = True

    def __init__(self, step: Optional[float] = None) -> None:
        self.step = None if step is None else float(step)

    def _resolve_step(self, D: np.ndarray) -> float:
        if self.step is not None:
            return self.step
        scale = float(np.std(np.asarray(D, dtype=np.float64)))
        return float(np.finfo(np.float64).eps ** (1.0 / 3.0)) * max(scale, 1.0)

    def apply(self, h: Evaluable, D: np.ndarray, X: np.ndarray) -> np.ndarray:
        D = np.asarray(D, dtype=np.float64)
        exact = getattr(h, "deriv_d", None)
        if callable(exact):
            out, was_1d = _as2d(exact(D, X))
            return _restore(out, was_1d)
        step = self._resolve_step(D)
        up, was_1d = _as2d(h(D + step, X))
        dn, _ = _as2d(h(D - step, X))
        out = (up - dn) / (2.0 * step)
        if np.all(out == 0.0) and up.size:
            warnings.warn(
                "the numerical derivative of this learner is identically zero, which "
                "is what a piecewise-constant learner (tree/forest/boosting) returns. "
                "AverageDerivative needs a learner that is smooth in D; see "
                "pragdml.learners.VaryingCoefficientRegressor.",
                RuntimeWarning,
                stacklevel=2,
            )
        return _restore(out, was_1d)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"AverageDerivative(step={self.step!r})"


class PolicyEffect(Functional):
    """Average policy effect: ``m(O, h) = h(pi(D, Z), Z) - h(D, Z)``.

    The effect of moving every unit from its observed treatment to the value a
    known policy assigns it. Covered by Sec. 2.1.1 ("many policy effects defined as
    averages of ``g(d,z)`` against a known weight").

    Parameters
    ----------
    policy : callable
        ``policy(D, X) -> (n,)``, the counterfactual treatment. Must be known, not
        estimated.

    Examples
    --------
    >>> import numpy as np
    >>> from pragdml.functionals import PolicyEffect
    >>> X = np.zeros((2, 1)); D = np.array([1.0, 2.0])
    >>> f = PolicyEffect(lambda D, X: D + 1.0)
    >>> f.apply(lambda d, X: 5.0 * d, D, X)
    array([5., 5.])
    """

    name = "Policy effect"

    def __init__(self, policy: Callable[[np.ndarray, np.ndarray], np.ndarray]) -> None:
        if not callable(policy):
            raise TypeError("`policy` must be a callable policy(D, X) -> (n,)")
        self.policy = policy

    def apply(self, h: Evaluable, D: np.ndarray, X: np.ndarray) -> np.ndarray:
        D = np.asarray(D, dtype=np.float64)
        d_new = np.asarray(self.policy(D, X), dtype=np.float64).reshape(-1)
        if d_new.shape[0] != D.shape[0]:
            raise ValueError(
                f"policy(D, X) returned {d_new.shape[0]} values, expected {D.shape[0]}"
            )
        new, was_1d = _as2d(h(d_new, X))
        old, _ = _as2d(h(D, X))
        return _restore(new - old, was_1d)


_REGISTRY: dict[str, type[Functional]] = {
    "ate": ATE,
    "ape": ATE,  # the representation-based ATE is the APE; same m
    "att": ATT,
    "apo": AveragePotentialOutcome,
    "average_derivative": AverageDerivative,
    "avg_derivative": AverageDerivative,
    "derivative": AverageDerivative,
    "elasticity": AverageDerivative,
}


def get_functional(spec: "str | Functional", **kwargs) -> Functional:
    """Resolve a functional from a name or pass a :class:`Functional` through.

    Parameters
    ----------
    spec : str or Functional
        ``"ate"``, ``"ape"``, ``"att"``, ``"apo"``, ``"average_derivative"``
        (aliases: ``"derivative"``, ``"elasticity"``), or an instance.
    **kwargs
        Forwarded to the constructor when ``spec`` is a name.

    Returns
    -------
    Functional

    Examples
    --------
    >>> from pragdml.functionals import get_functional
    >>> get_functional("ate").name
    'ATE'
    >>> get_functional("elasticity").name
    'Average derivative'
    """
    if isinstance(spec, Functional):
        return spec
    key = str(spec).strip().lower()
    if key not in _REGISTRY:
        raise ValueError(
            f"unknown functional {spec!r}; choose one of "
            f"{sorted(set(_REGISTRY))} or pass a Functional instance."
        )
    return _REGISTRY[key](**kwargs)
