"""Sensitivity inference for the full-information target (Sec. 4).

When target adaptivity is not defensible -- when the representation plausibly
discards information that matters -- the honest report is not a point but a
region. Sec. 4 builds it from the exact identity of Proposition 2(iii) by
factoring the target gap into an **identified scale** and **interpretable
sensitivity parameters** (Eq. 50, following [9] Theorem 2):

.. math::

    \\theta^* - \\theta_0 = \\rho \\cdot C_Y \\cdot C_D \\cdot S,

* ``rho in [-1, 1]`` -- the normalised **alignment** of the two omitted
  components. Note what it is not: a measure of strength. Confounding that is
  strong in both equations but *differently shaped* produces no bias at all, which
  is why ``rho`` multiplies the other two rather than being implied by them
  ([9] Appendix E.4 gives the ``D = U^2``, ``Y = D + U`` example where ``U``
  explains everything and biases nothing).
* ``C_Y^2 = E[(g^*-g_0)^2] / E[(Y-g_0)^2]`` -- the share of residual outcome
  variation the omitted information could explain.
* ``C_D^2 = E[(alpha^*-alpha_0)^2] / E[alpha_0^2]`` -- the matching quantity for
  the Riesz representer.
* ``S^2 = E[(Y-g_0)^2] \\, E[alpha_0^2]`` -- **identified**, and estimated by DML.

All moments are **uncentred**, matching Eq. (50) and [9] Theorem 2.

Given user-chosen bounds the region is Eq. (51), and Theorem 6 (= [9] Theorem 4)
supplies root-n inference for its endpoints via the delta method applied to
``S_0 = (sigma_0^2 nu_0^2)^{1/2}``.

Two parameterizations, deliberately not conflated
-------------------------------------------------
The two source papers grid the same region differently, and reporting one under
the other's label would misstate every robustness number:

* ``parameterization="cy_cd"`` -- the target paper's Table 3: the user sets
  ``C_Y^2`` and ``C_D^2`` directly (its grid uses ``C_Y^2 = C_D^2 = r^2``).
* ``parameterization="rv"`` -- [9] Definition 2: a single bound ``RV`` constrains
  ``C_Y^2 <= RV`` **and** ``1 - R^2_{alpha_s} <= RV``, which implies
  ``C_D^2 = RV / (1 - RV)``, *not* ``C_D^2 = RV``.

Compatibility map
-----------------
==========================================  ======================================
Paper                                       Here
==========================================  ======================================
Eq. (50), the factorization                   :class:`SensitivityAnalysis`
Eq. (51), identified region                   :meth:`SensitivityAnalysis.interval`
Eqs. (52)-(54), auxiliary scores              :meth:`SensitivityAnalysis.scores`
Eqs. (55)-(57), estimated components          :attr:`SensitivityAnalysis.components`
Eq. (58), endpoint estimates                  :meth:`SensitivityAnalysis.endpoints`
Eq. (64), endpoint influence function         :meth:`SensitivityAnalysis.scores`
Eq. (65), Bonferroni envelope                 ``envelope(method="bonferroni")``
[9] Thm. 4, one-sided bounds                  ``envelope(method="one_sided")``
after Eq. (65), shorter simultaneous          ``envelope(method="joint")``
[9] Def. 2, robustness value                  :func:`robustness_value`
[9] Sec. 5.2.3, contour plot                  :meth:`SensitivityAnalysis.contour_data`
[9] App. E, covariate benchmarking            :func:`benchmark_covariate`
Sec. 5.4, deletion calibration                :func:`calibrate_deletion`
==========================================  ======================================

Author
------
Dr Merwan Roudane <merwanroudane920@gmail.com>
https://github.com/merwanroudane
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np
from scipy import optimize, stats

__all__ = [
    "SensitivityAnalysis",
    "SensitivityResult",
    "BenchmarkResult",
    "robustness_value",
    "benchmark_covariate",
    "calibrate_deletion",
    "bias_factor",
]


def bias_factor(
    cy2: float, cd2: float = None, rho: float = 1.0, rv: float = None
) -> float:
    """The non-identified multiplier ``c = |rho| C_Y C_D`` of Eqs. (50)-(51).

    Supply either ``(cy2, cd2)`` -- the target paper's parameterization -- or
    ``rv`` alone, which is [9] Definition 2's and implies ``C_Y^2 = RV`` and
    ``C_D^2 = RV/(1-RV)``.

    Parameters
    ----------
    cy2 : float
        ``C_Y^2``, in ``[0, 1]``. Ignored when ``rv`` is given.
    cd2 : float, optional
        ``C_D^2``, non-negative. Required unless ``rv`` is given.
    rho : float, default 1.0
        Alignment bound; only ``|rho|`` matters and it is clipped to ``[0, 1]``.
        The default is the paper's conservative worst case.
    rv : float, optional
        A single robustness-value bound in ``[0, 1)``.

    Returns
    -------
    float
        ``c >= 0``.

    Examples
    --------
    >>> from pragdml.sensitivity import bias_factor
    >>> round(bias_factor(0.04, 0.04), 6)
    0.04
    >>> round(bias_factor(None, rv=0.05), 6)   # C_D^2 = 0.05/0.95
    0.051299
    >>> round(bias_factor(0.04, 0.04, rho=0.5), 6)
    0.02
    """
    rho_abs = float(np.clip(abs(float(rho)), 0.0, 1.0))
    if rv is not None:
        rv = float(rv)
        if not 0.0 <= rv < 1.0:
            raise ValueError(f"rv must be in [0, 1), got {rv}")
        cy2_eff, cd2_eff = rv, rv / (1.0 - rv)
    else:
        if cd2 is None:
            raise ValueError("supply either (cy2, cd2) or rv")
        cy2_eff, cd2_eff = float(cy2), float(cd2)
        if cy2_eff < 0.0 or cd2_eff < 0.0:
            raise ValueError(
                f"cy2 and cd2 must be non-negative, got {cy2_eff}, {cd2_eff}"
            )
        if cy2_eff > 1.0:
            raise ValueError(
                f"cy2 = {cy2_eff} exceeds 1; C_Y^2 is a share of residual outcome "
                "variation and cannot exceed one."
            )
    return float(rho_abs * np.sqrt(cy2_eff * cd2_eff))


@dataclass(frozen=True)
class SensitivityResult:
    """One sensitivity scenario, evaluated.

    Attributes
    ----------
    rho, cy2, cd2 : float
        The scenario, as applied (after any ``rv`` translation).
    c : float
        ``|rho| C_Y C_D``.
    theta : float
        The pipeline estimate ``theta_hat_0`` -- the centre of the region.
    S : float
        ``S_hat_0``, the identified scale.
    lower, upper : float
        The Eq. (51) endpoints ``theta_hat_0 -/+ c S_hat_0``.
    se_lower, se_upper : float
        Endpoint standard errors from the Eq. (64) influence functions.
    ci_lower, ci_upper : float
        The confidence envelope.
    level : float
    method : str
        Which envelope rule was used.
    rv_input : float or None
        The ``RV`` that generated this row, when ``parameterization="rv"``.
    """

    rho: float
    cy2: float
    cd2: float
    c: float
    theta: float
    S: float
    lower: float
    upper: float
    se_lower: float
    se_upper: float
    ci_lower: float
    ci_upper: float
    level: float
    method: str
    rv_input: Optional[float] = None

    @property
    def bias_bound(self) -> float:
        """``c * S`` -- the half-width of the identified region."""
        return float(self.c * self.S)

    @property
    def excludes_zero(self) -> bool:
        """Whether the whole **confidence envelope** stays on one side of zero."""
        return bool(self.ci_lower > 0.0 or self.ci_upper < 0.0)

    @property
    def region_excludes_zero(self) -> bool:
        """Whether the identified region alone excludes zero (no sampling noise)."""
        return bool(self.lower > 0.0 or self.upper < 0.0)

    def as_row(self) -> dict:
        """A flat dict, for assembling a table."""
        return {
            "rho": self.rho,
            "cy2": self.cy2,
            "cd2": self.cd2,
            "c": self.c,
            "theta": self.theta,
            "lower": self.lower,
            "upper": self.upper,
            "ci_lower": self.ci_lower,
            "ci_upper": self.ci_upper,
            "se_lower": self.se_lower,
            "se_upper": self.se_upper,
            "excludes_zero": self.excludes_zero,
        }


class SensitivityAnalysis:
    """Sensitivity region and endpoint inference for ``theta^*`` (Sec. 4).

    Built from a fitted :class:`~pragdml._core.CrossFitOutput`, which already
    holds every quantity the three auxiliary scores need -- so nothing is
    re-estimated and the point estimate and the region are guaranteed mutually
    consistent.

    Parameters
    ----------
    fit : CrossFitOutput
        A cross-fitted pass, from :func:`~pragdml._core.cross_fit` or
        :attr:`~pragdml.dml.PragmaticDML.crossfit_`.

    Attributes
    ----------
    theta : float
        ``theta_hat_0``.
    sigma2, nu2, S : float
        Eqs. (55), (56), (57).

    Raises
    ------
    ValueError
        If ``S_hat_0 <= 0``, which makes the region degenerate and the delta-method
        expansion invalid (Theorem 6 requires ``S_0 > 0``). A non-positive
        ``nu_hat_0^2`` is the usual cause and signals a badly fitted representer.

    References
    ----------
    Aradillas Fernandez et al. (2026), Sec. 4, arXiv:2610.01935.
    Chernozhukov, Cinelli, Newey, Sharma & Syrgkanis (2026), Thms. 2 and 4,
    Def. 2, App. E, doi:10.1162/REST.a.1705.

    Examples
    --------
    >>> import numpy as np
    >>> from pragdml import PragmaticDML
    >>> from pragdml.sensitivity import SensitivityAnalysis
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(600, 2))
    >>> D = (rng.uniform(size=600) < 0.5).astype(float)
    >>> Y = 2.0 * D + X[:, 0] + rng.normal(size=600)
    >>> res = PragmaticDML(functional="ate", seed=0).fit(Y, D, X)
    >>> sa = SensitivityAnalysis(res.crossfit_)
    >>> row = sa.evaluate(cy2=0.02, cd2=0.02)
    >>> bool(row.lower < row.theta < row.upper)
    True
    """

    def __init__(self, fit) -> None:
        self.fit = fit
        self.theta = float(fit.theta)
        self.sigma2 = float(fit.sigma2_outcome)
        self.nu2 = float(fit.nu2_representer)
        if self.sigma2 <= 0.0:
            raise ValueError(
                "sigma_hat_0^2 = P_n(Y - g_hat)^2 is not positive; the outcome "
                "residuals are degenerate and no sensitivity region is defined."
            )
        if self.nu2 <= 0.0:
            raise ValueError(
                f"nu_hat_0^2 = P_n[2 m(O, alpha) - alpha^2] = {self.nu2:.6g} is not "
                "positive. It estimates E[alpha_0^2] > 0, so a non-positive value "
                "means the Riesz representer is badly fitted (too much "
                "regularisation, or a basis that cannot represent it). Theorem 6 "
                "requires S_0 > 0."
            )
        self.S = float(np.sqrt(self.sigma2 * self.nu2))
        self.n = int(fit.n)
        self._cluster = fit.cluster

    # ---- the three auxiliary scores, Eqs. (52)-(54) -------------------------

    def _base_scores(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Centred ``(psi_theta, psi_sigma2, psi_nu2)`` of Eqs. (52)-(54)."""
        f = self.fit
        psi_theta = f.influence
        resid2 = (f.y - f.g_hat) ** 2
        psi_sigma2 = resid2 - self.sigma2
        nu_vals = 2.0 * f.m_alpha - f.alpha_hat**2
        psi_nu2 = nu_vals - self.nu2
        return psi_theta, psi_sigma2, psi_nu2

    def scores(self, c: float) -> tuple[np.ndarray, np.ndarray]:
        """The endpoint influence functions of Eq. (64) / [9] Theorem 4.

        .. math::

            \\psi_{0,\\pm} = \\psi_{\\theta_0}
              \\pm \\frac{c}{2 S_0}\\Big(\\nu_0^2\\, \\psi_{\\sigma_0^2}
                                     + \\sigma_0^2\\, \\psi_{\\nu_0^2}\\Big)

        The factor comes from the delta method applied to
        ``(u, v) -> (u v)^{1/2}``: ``dS/du = v/(2S)`` and ``dS/dv = u/(2S)``.

        Returns
        -------
        (ndarray, ndarray)
            ``(psi_minus, psi_plus)``, each of shape ``(n,)``.
        """
        psi_theta, psi_s2, psi_n2 = self._base_scores()
        shift = (float(c) / (2.0 * self.S)) * (
            self.nu2 * psi_s2 + self.sigma2 * psi_n2
        )
        return psi_theta - shift, psi_theta + shift

    def _se(self, influence: np.ndarray) -> float:
        """Standard error of an endpoint, cluster-robust when applicable."""
        if self._cluster is None:
            return float(np.sqrt(np.mean(influence**2) / self.n))
        labels = np.asarray(self._cluster).reshape(-1)
        _, inverse = np.unique(labels, return_inverse=True)
        sums = np.bincount(inverse, weights=influence)
        return float(np.sqrt(np.sum(sums**2)) / self.n)

    # ---- region and endpoints ----------------------------------------------

    def interval(
        self, cy2: float = None, cd2: float = None, rho: float = 1.0, rv: float = None
    ) -> tuple[float, float]:
        """The identified region of Eq. (51), ignoring sampling uncertainty.

        Examples
        --------
        >>> import numpy as np
        >>> from pragdml import PragmaticDML
        >>> from pragdml.sensitivity import SensitivityAnalysis
        >>> rng = np.random.default_rng(1)
        >>> X = rng.normal(size=(400, 2))
        >>> D = (rng.uniform(size=400) < 0.5).astype(float)
        >>> Y = X[:, 0] + rng.normal(size=400)
        >>> sa = SensitivityAnalysis(
        ...     PragmaticDML(functional="ate", seed=0).fit(Y, D, X).crossfit_)
        >>> lo, hi = sa.interval(cy2=0.0, cd2=0.0)
        >>> bool(abs(hi - lo) < 1e-12)
        True
        """
        c = bias_factor(cy2, cd2, rho=rho, rv=rv)
        return (self.theta - c * self.S, self.theta + c * self.S)

    def evaluate(
        self,
        cy2: float = None,
        cd2: float = None,
        rho: float = 1.0,
        rv: float = None,
        level: float = 0.95,
        method: str = "bonferroni",
    ) -> SensitivityResult:
        """Evaluate one scenario: region, endpoint SEs and confidence envelope.

        Parameters
        ----------
        cy2, cd2 : float, optional
            ``C_Y^2`` and ``C_D^2`` (the target paper's Table 3 parameterization).
        rho : float, default 1.0
            Alignment bound.
        rv : float, optional
            A single ``RV`` bound instead ([9] Definition 2); implies
            ``C_Y^2 = RV``, ``C_D^2 = RV/(1-RV)``.
        level : float, default 0.95
            Nominal coverage of the envelope.
        method : {"bonferroni", "one_sided", "joint"}, default "bonferroni"
            See :meth:`envelope`.

        Returns
        -------
        SensitivityResult
        """
        c = bias_factor(cy2, cd2, rho=rho, rv=rv)
        if rv is not None:
            cy2_eff, cd2_eff = float(rv), float(rv) / (1.0 - float(rv))
        else:
            cy2_eff, cd2_eff = float(cy2), float(cd2)
        lower = self.theta - c * self.S
        upper = self.theta + c * self.S
        psi_minus, psi_plus = self.scores(c)
        se_minus, se_plus = self._se(psi_minus), self._se(psi_plus)
        ci_lo, ci_hi = self.envelope(
            c=c, level=level, method=method,
            lower=lower, upper=upper, se_lower=se_minus, se_upper=se_plus,
            psi_minus=psi_minus, psi_plus=psi_plus,
        )
        return SensitivityResult(
            rho=float(np.clip(abs(float(rho)), 0.0, 1.0)),
            cy2=cy2_eff, cd2=cd2_eff, c=c,
            theta=self.theta, S=self.S,
            lower=lower, upper=upper,
            se_lower=se_minus, se_upper=se_plus,
            ci_lower=ci_lo, ci_upper=ci_hi,
            level=float(level), method=str(method),
            rv_input=None if rv is None else float(rv),
        )

    def envelope(
        self,
        c: float,
        level: float = 0.95,
        method: str = "bonferroni",
        lower: Optional[float] = None,
        upper: Optional[float] = None,
        se_lower: Optional[float] = None,
        se_upper: Optional[float] = None,
        psi_minus: Optional[np.ndarray] = None,
        psi_plus: Optional[np.ndarray] = None,
    ) -> tuple[float, float]:
        """A confidence envelope for the sensitivity region.

        Three rules, because the two source papers state **different guarantees**
        and both are worth having:

        ``"bonferroni"`` (default, the target paper's Eq. 65)
            ``[theta^- - z_{1-a/2} se^-, theta^+ + z_{1-a/2} se^+]``. By
            Bonferroni over the two endpoint CLTs this covers the **whole region**
            ``[theta^-, theta^+]`` with probability at least ``level``, hence also
            ``theta^*`` whenever the maintained restrictions place it there.
        ``"one_sided"`` ([9] Theorem 4)
            the same shape with ``z_{1-a}``. The guarantee is per endpoint and
            one-sided: ``P(theta^- >= l) -> >= level`` and
            ``P(theta^+ <= u) -> >= level``. Narrower, but it is *not* a
            simultaneous statement about the region.
        ``"joint"`` (suggested one sentence after Eq. 65, implemented here)
            a single critical value read off the **estimated joint Gaussian law**
            of the two endpoint scores, giving simultaneous coverage like
            Bonferroni but shorter, because it uses the correlation between the
            endpoints instead of ignoring it.

        Returns
        -------
        (float, float)
        """
        if method not in ("bonferroni", "one_sided", "joint"):
            raise ValueError(
                f"method must be 'bonferroni', 'one_sided' or 'joint', got {method!r}"
            )
        if not 0.0 < level < 1.0:
            raise ValueError(f"level must be in (0, 1), got {level}")
        if lower is None or upper is None:
            lower = self.theta - c * self.S
            upper = self.theta + c * self.S
        if psi_minus is None or psi_plus is None:
            psi_minus, psi_plus = self.scores(c)
        if se_lower is None:
            se_lower = self._se(psi_minus)
        if se_upper is None:
            se_upper = self._se(psi_plus)

        alpha = 1.0 - level
        if method == "one_sided":
            z = float(stats.norm.ppf(level))
        elif method == "bonferroni":
            z = float(stats.norm.ppf(1.0 - alpha / 2.0))
        else:
            z = self._joint_critical_value(psi_minus, psi_plus, alpha)
        return (lower - z * se_lower, upper + z * se_upper)

    @staticmethod
    def _joint_critical_value(
        psi_minus: np.ndarray, psi_plus: np.ndarray, alpha: float
    ) -> float:
        """Simultaneous critical value from the estimated bivariate normal law.

        Coverage of the whole region needs ``Z^- <= k`` and ``Z^+ >= -k``
        simultaneously, i.e. ``P(W_1 <= k, W_2 <= k) >= 1 - alpha`` for
        ``W = (Z^-, -Z^+)``, whose correlation is minus that of the two endpoint
        scores. Solving that CDF equation gives a ``k`` no larger than the
        Bonferroni value, and strictly smaller whenever the endpoints are
        correlated.
        """
        sd_m = float(np.std(psi_minus))
        sd_p = float(np.std(psi_plus))
        if sd_m <= 0.0 or sd_p <= 0.0:
            return float(stats.norm.ppf(1.0 - alpha / 2.0))
        r = float(np.corrcoef(psi_minus, psi_plus)[0, 1])
        if not np.isfinite(r):
            return float(stats.norm.ppf(1.0 - alpha / 2.0))
        r = float(np.clip(-r, -0.999999, 0.999999))
        cov = np.array([[1.0, r], [r, 1.0]])
        mvn = stats.multivariate_normal(mean=[0.0, 0.0], cov=cov, allow_singular=True)

        def shortfall(k: float) -> float:
            return float(mvn.cdf([k, k]) - (1.0 - alpha))

        lo = float(stats.norm.ppf(1.0 - alpha))
        hi = float(stats.norm.ppf(1.0 - alpha / 2.0))
        if shortfall(lo) >= 0.0:
            return lo
        if shortfall(hi) <= 0.0:
            return hi
        try:
            return float(optimize.brentq(shortfall, lo, hi, xtol=1e-10))
        except ValueError:  # pragma: no cover - defensive
            return hi

    # ---- grids and contours -------------------------------------------------

    def grid(
        self,
        rhos: Sequence[float] = (1.0, 0.5, 0.2),
        r2s: Sequence[float] = (0.010, 0.025, 0.050),
        level: float = 0.95,
        method: str = "bonferroni",
        parameterization: str = "cy_cd",
        tail: Optional[float] = None,
    ):
        """A scenario grid in the shape of the paper's Table 3.

        Parameters
        ----------
        rhos : sequence of float, default ``(1.0, 0.5, 0.2)``
            Alignment bounds, as in Table 3.
        r2s : sequence of float, default ``(0.010, 0.025, 0.050)``
            The paper's "mild / medium / strong" strengths.
        level : float, default 0.95
        method : str, default ``"bonferroni"``
        parameterization : {"cy_cd", "rv"}, default ``"cy_cd"``
            ``"cy_cd"`` sets ``C_Y^2 = C_D^2 = r^2`` (Table 3's own convention);
            ``"rv"`` treats each ``r2`` as an ``RV`` ([9] Definition 2), so
            ``C_D^2 = RV/(1-RV)``.
        tail : float, optional
            When given (the paper's Table 3 uses ``0.10``), the extra columns
            ``p10`` and ``p90`` report the one-sided sampling limits of the two
            endpoints at that tail probability, reproducing Table 3's outer
            columns.

        Returns
        -------
        pandas.DataFrame
            One row per ``(rho, r2)`` pair.
        """
        import pandas as pd

        if parameterization not in ("cy_cd", "rv"):
            raise ValueError(
                f"parameterization must be 'cy_cd' or 'rv', got {parameterization!r}"
            )
        rows = []
        for rho in rhos:
            for r2 in r2s:
                if parameterization == "cy_cd":
                    res = self.evaluate(
                        cy2=r2, cd2=r2, rho=rho, level=level, method=method
                    )
                else:
                    res = self.evaluate(rv=r2, rho=rho, level=level, method=method)
                row = res.as_row()
                row["r2"] = float(r2)
                row["parameterization"] = parameterization
                if tail is not None:
                    z = float(stats.norm.ppf(1.0 - float(tail)))
                    row["p10"] = res.lower - z * res.se_lower
                    row["p90"] = res.upper + z * res.se_upper
                rows.append(row)
        cols = ["rho", "r2", "cy2", "cd2", "c", "theta"]
        if tail is not None:
            cols += ["p10", "lower", "upper", "p90"]
        else:
            cols += ["lower", "upper"]
        cols += ["ci_lower", "ci_upper", "se_lower", "se_upper", "excludes_zero",
                 "parameterization"]
        return pd.DataFrame(rows)[cols]

    def contour_data(
        self,
        cy2_max: float = 0.3,
        cd_share_max: float = 0.3,
        n_grid: int = 60,
        rho: float = 1.0,
        level: float = 0.95,
        bound: str = "lower",
        method: str = "one_sided",
    ) -> dict:
        """Grid for the [9] Sec. 5.2.3 sensitivity contour plot.

        The canonical display in this literature (Imbens 2003; Cinelli & Hazlett
        2020; [9] Fig. 3): the horizontal axis is the share of representer
        variation the omitted information could explain
        (``1 - R^2_{alpha_s}``), the vertical axis is ``C_Y^2``, and the contours
        are a confidence limit of ``theta^*``. The robustness value is where the
        45-degree line crosses the zero contour.

        Note the horizontal axis is ``1 - R^2_{alpha_s}``, not ``C_D^2``: that is
        [9]'s convention, and the two differ by ``C_D^2 = s/(1-s)``.

        Parameters
        ----------
        cy2_max, cd_share_max : float, default 0.3
            Upper limits of the two axes.
        n_grid : int, default 60
            Points per axis.
        rho : float, default 1.0
        level : float, default 0.95
        bound : {"lower", "upper", "point_lower", "point_upper"}, default "lower"
            Which surface to contour. ``"point_*"`` omits sampling uncertainty and
            contours the identified endpoint itself.
        method : str, default ``"one_sided"``
            Envelope rule; [9]'s figures use the one-sided bound.

        Returns
        -------
        dict
            ``cd_share`` ``(n_grid,)``, ``cy2`` ``(n_grid,)``, ``values``
            ``(n_grid, n_grid)`` with rows indexed by ``cy2``, plus ``bound``,
            ``rho`` and ``level``.
        """
        if bound not in ("lower", "upper", "point_lower", "point_upper"):
            raise ValueError(
                "bound must be 'lower', 'upper', 'point_lower' or 'point_upper', "
                f"got {bound!r}"
            )
        n_grid = max(2, int(n_grid))
        shares = np.linspace(0.0, float(cd_share_max), n_grid)
        cys = np.linspace(0.0, float(cy2_max), n_grid)
        values = np.empty((n_grid, n_grid), dtype=np.float64)
        for i, cy2 in enumerate(cys):
            for j, share in enumerate(shares):
                share = min(share, 0.999999)
                cd2 = share / (1.0 - share)
                if bound.startswith("point"):
                    lo, hi = self.interval(cy2=cy2, cd2=cd2, rho=rho)
                    values[i, j] = lo if bound.endswith("lower") else hi
                else:
                    res = self.evaluate(
                        cy2=cy2, cd2=cd2, rho=rho, level=level, method=method
                    )
                    values[i, j] = (
                        res.ci_lower if bound == "lower" else res.ci_upper
                    )
        return {
            "cd_share": shares,
            "cy2": cys,
            "values": values,
            "bound": bound,
            "rho": float(rho),
            "level": float(level),
            "theta": self.theta,
        }

    def robustness_value(
        self, target: float = 0.0, level: float = 0.95, rho: float = 1.0
    ) -> float:
        """[9] Definition 2: the robustness value ``RV_{theta, a}``.

        Convenience wrapper around :func:`robustness_value`.
        """
        return robustness_value(self, target=target, level=level, rho=rho)

    def summary(self, level: float = 0.95) -> str:
        """A compact text report of the identified components."""
        lines = [
            "Sensitivity components (Eqs. 55-57)",
            "-" * 52,
            f"  theta_hat_0  (pipeline target)   {self.theta: .6f}",
            f"  sigma_hat_0^2 = P_n(Y-g)^2       {self.sigma2: .6f}",
            f"  nu_hat_0^2    = P_n[2m(a)-a^2]   {self.nu2: .6f}",
            f"  S_hat_0       = (sigma^2 nu^2)^.5{self.S: .6f}",
            "-" * 52,
            f"  robustness value RV(0, {1 - level:.2f})       "
            f"{self.robustness_value(level=level):.4f}",
            "",
            "  Region half-width at a scenario = |rho| C_Y C_D * S_hat_0.",
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"SensitivityAnalysis(theta={self.theta:.4g}, S={self.S:.4g}, "
            f"n={self.n})"
        )


def robustness_value(
    analysis: SensitivityAnalysis,
    target: float = 0.0,
    level: float = 0.95,
    rho: float = 1.0,
) -> float:
    """[9] Definition 2: the minimum ``RV`` at which the bound reaches ``target``.

    ``RV_{theta, a}`` is the smallest common bound on the sensitivity parameters
    -- ``C_Y^2 <= RV`` **and** ``1 - R^2_{alpha_s} <= RV``, with ``|rho| <= 1`` --
    at which the confidence bound of Theorem 4 includes ``target``. It answers in
    one number what a whole contour plot answers in a picture: *how strong would
    the omitted information have to be to overturn this result?*

    Two details that matter and are easy to get wrong:

    * it binds ``1 - R^2_{alpha_s}``, not ``C_D^2``, so internally
      ``C_D^2 = RV/(1-RV)``;
    * it inverts the **confidence** bound, not the identified region, so it
      already accounts for sampling noise. Pass ``level=0.0`` for the
      point-bound version.

    Parameters
    ----------
    analysis : SensitivityAnalysis
    target : float, default 0.0
        The value the bound must reach -- zero for "could the sign flip?".
    level : float, default 0.95
        Confidence level of the one-sided bound ([9] uses the one-sided rule
        here). ``0.0`` inverts the identified region instead.
    rho : float, default 1.0
        Alignment bound; [9]'s reported values use the worst case ``1``.

    Returns
    -------
    float
        ``RV`` in ``[0, 1)``, or ``0.0`` when the bound already includes
        ``target`` with no omitted information at all, or ``nan`` when no ``RV < 1``
        suffices.

    Examples
    --------
    >>> import numpy as np
    >>> from pragdml import PragmaticDML
    >>> from pragdml.sensitivity import SensitivityAnalysis, robustness_value
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(800, 2))
    >>> D = (rng.uniform(size=800) < 0.5).astype(float)
    >>> Y = 3.0 * D + X[:, 0] + rng.normal(size=800)
    >>> sa = SensitivityAnalysis(
    ...     PragmaticDML(functional="ate", seed=0).fit(Y, D, X).crossfit_)
    >>> rv = robustness_value(sa)
    >>> bool(0.0 < rv < 1.0)
    True
    """
    theta = analysis.theta
    sign = 1.0 if theta >= target else -1.0

    def limit(rv: float) -> float:
        """The confidence (or point) limit facing ``target``, as a function of RV."""
        if level <= 0.0:
            lo, hi = analysis.interval(rv=rv, rho=rho)
            return lo if sign > 0 else hi
        res = analysis.evaluate(rv=rv, rho=rho, level=level, method="one_sided")
        return res.ci_lower if sign > 0 else res.ci_upper

    def gap(rv: float) -> float:
        """Signed distance from the limit to ``target``; decreasing in ``rv``."""
        return sign * (limit(rv) - target)

    if gap(0.0) <= 0.0:
        return 0.0
    hi = 1.0 - 1e-9
    if gap(hi) > 0.0:
        return float("nan")
    return float(optimize.brentq(gap, 0.0, hi, xtol=1e-10))


@dataclass(frozen=True)
class BenchmarkResult:
    """Result of :func:`benchmark_covariate` ([9] Appendix E).

    Attributes
    ----------
    name : str
        Label of the covariate (or block) used as the yardstick.
    g_y : float
        ``G_{Y,j}`` of Eq. (29): the observed outcome-side gain,
        ``R^2 / (1 - R^2)`` in partial-``R^2`` form.
    g_d : float
        ``G_{D,j}`` of Eq. (30): the observed representer-side gain.
    k_y, k_d : float
        The relative-strength multipliers applied ("U as strong as ``X_j``" is
        ``k = 1``; "a quarter as strong" is ``k = 0.25``).
    cy2 : float
        Implied ``C_Y^2 = k_Y G_{Y,j}``. **Reported uncapped**: a value above one
        means the scenario is infeasible, because ``X_j`` alone would explain more
        than all the residual outcome variation. That is information about the
        yardstick, so it is surfaced rather than clipped.
    cd_share : float
        Implied ``1 - R^2_{alpha_s} = k_D G_{D,j}``.
    cd2 : float
        Implied ``C_D^2 = cd_share / (1 - cd_share)``.
    rho_pearson : float
        ``rho_j``, the **Pearson** correlation of the two induced errors, which is
        the yardstick [9] App. E.4 proposes for calibrating ``rho``.
    rho_uncentred : float
        The uncentred alignment, i.e. the same functional form as the sensitivity
        parameter ``rho`` of Eq. (50). Reported alongside because the two are
        different objects and the source papers use different ones; see Notes on
        :func:`benchmark_covariate`.
    delta_theta : float
        ``Delta theta_{s,j}``: how much the estimate itself moved when ``X_j`` was
        dropped.
    """

    name: str
    g_y: float
    g_d: float
    k_y: float
    k_d: float
    cy2: float
    cd_share: float
    cd2: float
    rho_pearson: float
    rho_uncentred: float
    delta_theta: float

    def as_row(self) -> dict:
        """A flat dict for tabulation."""
        return {
            "benchmark": self.name,
            "k_y": self.k_y,
            "k_d": self.k_d,
            "G_Y": self.g_y,
            "G_D": self.g_d,
            "cy2": self.cy2,
            "cd_share": self.cd_share,
            "cd2": self.cd2,
            "rho_pearson": self.rho_pearson,
            "rho_uncentred": self.rho_uncentred,
            "delta_theta": self.delta_theta,
        }


def benchmark_covariate(
    fit_with,
    fit_without,
    name: str = "X_j",
    k_y: float = 1.0,
    k_d: float = 1.0,
) -> BenchmarkResult:
    """[9] Appendix E: calibrate the sensitivity parameters from observed covariates.

    Rather than asserting ``C_Y^2`` and ``C_D^2`` out of thin air, bound them by
    *relative* claims: "the omitted information is no stronger than the observed
    covariate ``X_j``" (``k = 1``), or "a quarter as strong" (``k = 0.25``). The
    observed gains are

    .. math::

        G_{Y,j} = \\frac{R^2_{Y \\sim X_j \\mid D, X_{-j}}}
                        {1 - R^2_{Y \\sim X_j \\mid D, X_{-j}}}, \\qquad
        G_{D,j} = \\frac{E[\\alpha_s^2] - E[\\alpha_{s,-j}^2]}{E[\\alpha_{s,-j}^2]},

    and then ``C_Y^2 = k_Y G_{Y,j}`` (Eq. 29) and
    ``1 - R^2_{alpha_s} = k_D G_{D,j}`` (Eq. 30).

    Parameters
    ----------
    fit_with : CrossFitOutput
        Fitted **with** ``X_j`` in the controls.
    fit_without : CrossFitOutput
        Fitted **without** it, everything else identical -- same folds, same
        learners, same seed. A different split would confound the comparison.
    name : str, default ``"X_j"``
        Label for reporting.
    k_y, k_d : float, default 1.0
        Relative-strength multipliers.

    Returns
    -------
    BenchmarkResult

    Notes
    -----
    **Two conventions for ``rho``, and they are not the same object.** The
    sensitivity parameter ``rho`` of Eq. (50) and [9] Theorem 2 is an *uncentred*
    normalised inner product; the benchmark ``rho_j`` of [9] App. E.4 is the
    *Pearson* (centred) correlation of the two induced error vectors. Both are
    returned, unmixed. Use ``rho_pearson`` when following [9]'s benchmarking
    recipe and ``rho_uncentred`` when you want the quantity that literally enters
    the bound.

    **Relation to the target paper's Sec. 5.4.** Its full-versus-text calibration
    is this function with ``k_y = k_d = 1`` and ``X_j`` taken to be a whole
    modality block: ``G_Y`` is its ``C_Y^2 = (R^2_full - R^2_text)/(1 - R^2_text)``
    and ``G_D`` is its ``C_D^2 = (L_text - L_full)/(-L_text)``, algebraically
    identical. The one difference is that Sec. 5.4 reports ``G_D`` directly as
    ``C_D^2``, whereas [9] maps it to ``1 - R^2_{alpha_s}`` and then to
    ``C_D^2 = G_D/(1 - G_D)``. The two agree to first order and diverge only when
    ``G_D`` is large; both are reported here (``g_d`` and ``cd2``) so neither
    convention is silently imposed. See also :func:`calibrate_deletion`.

    Examples
    --------
    >>> import numpy as np
    >>> from pragdml import PragmaticDML
    >>> from pragdml.sensitivity import benchmark_covariate
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(600, 3))
    >>> D = (rng.uniform(size=600) < 0.5).astype(float)
    >>> Y = 2.0 * D + 2.0 * X[:, 0] + rng.normal(size=600)
    >>> full = PragmaticDML(functional="ate", seed=7).fit(Y, D, X)
    >>> drop = PragmaticDML(functional="ate", seed=7).fit(Y, D, X[:, 1:])
    >>> bm = benchmark_covariate(full.crossfit_, drop.crossfit_, name="X0")
    >>> bool(bm.g_y > 0)
    True
    """
    s2_with = float(fit_with.sigma2_outcome)
    s2_without = float(fit_without.sigma2_outcome)
    nu2_with = float(fit_with.nu2_representer)
    nu2_without = float(fit_without.nu2_representer)

    if s2_without <= 0.0:
        raise ValueError("the reduced model has non-positive outcome residual variance")
    if nu2_without <= 0.0:
        raise ValueError(
            f"the reduced model has nu^2 = {nu2_without:.6g} <= 0; its Riesz "
            "representer is badly fitted, so no benchmark can be formed from it."
        )

    partial_r2 = max(0.0, (s2_without - s2_with) / s2_without)
    g_y = partial_r2 / (1.0 - partial_r2) if partial_r2 < 1.0 else float("inf")
    g_d = max(0.0, (nu2_with - nu2_without) / nu2_without)

    cy2 = float(k_y) * g_y
    cd_share = min(float(k_d) * g_d, 0.999999)
    cd2 = cd_share / (1.0 - cd_share)

    dg = np.asarray(fit_with.g_hat) - np.asarray(fit_without.g_hat)
    da = np.asarray(fit_with.alpha_hat) - np.asarray(fit_without.alpha_hat)
    if np.std(dg) > 0 and np.std(da) > 0:
        rho_p = float(np.corrcoef(dg, da)[0, 1])
    else:
        rho_p = 0.0
    denom = float(np.sqrt(np.mean(dg**2) * np.mean(da**2)))
    rho_u = float(np.mean(dg * da) / denom) if denom > 0 else 0.0

    # cy2 is reported uncapped on purpose. A value above one means the claim
    # "U is as strong as X_j" is infeasible -- X_j alone explains more than all
    # the residual outcome variation there is -- so X_j is too strong to serve as
    # a yardstick. Capping it silently would turn an infeasible scenario into a
    # plausible-looking one; instead it is surfaced here and rejected with an
    # explanatory message by bias_factor() if it is ever fed into a bound.
    return BenchmarkResult(
        name=str(name),
        g_y=float(g_y),
        g_d=float(g_d),
        k_y=float(k_y),
        k_d=float(k_d),
        cy2=float(cy2),
        cd_share=float(cd_share),
        cd2=float(cd2),
        rho_pearson=float(np.clip(rho_p, -1.0, 1.0)),
        rho_uncentred=float(np.clip(rho_u, -1.0, 1.0)),
        delta_theta=float(fit_with.theta - fit_without.theta),
    )


def calibrate_deletion(fit_long, fit_short, name: str = "deleted block") -> dict:
    """Sec. 5.4 of the paper: the full-versus-short deletion calibration.

    The paper's own calibration exercise. Treat a richer *observed* pipeline as
    "long" and the pipeline actually used as "short", delete the extra
    information, and read off the empirical scale of the loss:

    .. math::

        C_Y^2 = \\frac{R^2_{\\text{long}} - R^2_{\\text{short}}}
                      {1 - R^2_{\\text{short}}}, \\qquad
        C_D^2 = \\frac{L_{\\text{short}} - L_{\\text{long}}}{-L_{\\text{short}}},

    with ``rho`` the alignment of the two induced errors. These are then used to
    *anchor* the grid for information beyond every observed modality.

    The paper's reported values for its multimodal application were
    ``C_Y^2 = 0.0017``, ``C_D^2 = 0.0088`` and ``rho_hat = 0.8654``, with
    equal-strength benchmark ``r^2 = (C_Y^2 C_D^2)^{1/2} = 0.0039``; the
    ``r2_equal_strength`` key reproduces that last definition.

    The diagnostic lesson the paper draws from it is worth repeating: in its
    application the extra modalities barely moved the outcome ``R^2`` yet visibly
    moved the Riesz loss, so **prediction quality alone does not summarise target
    preservation**.

    Parameters
    ----------
    fit_long, fit_short : CrossFitOutput
        The richer and the poorer pipeline, fitted with the same folds and seed.
    name : str, default ``"deleted block"``

    Returns
    -------
    dict
        ``cy2``, ``cd2``, ``rho``, ``rho_pearson``, ``r2_equal_strength``,
        ``r2_long``, ``r2_short``, ``riesz_loss_long``, ``riesz_loss_short``,
        ``delta_theta``.

    Examples
    --------
    >>> import numpy as np
    >>> from pragdml import PragmaticDML
    >>> from pragdml.sensitivity import calibrate_deletion
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(600, 3))
    >>> D = (rng.uniform(size=600) < 0.5).astype(float)
    >>> Y = 2.0 * D + 2.0 * X[:, 0] + rng.normal(size=600)
    >>> lon = PragmaticDML(functional="ate", seed=5).fit(Y, D, X)
    >>> sho = PragmaticDML(functional="ate", seed=5).fit(Y, D, X[:, 1:])
    >>> cal = calibrate_deletion(lon.crossfit_, sho.crossfit_)
    >>> bool(cal["cy2"] >= 0 and cal["cd2"] >= 0)
    True
    """
    r2_long = float(fit_long.outcome_r2)
    r2_short = float(fit_short.outcome_r2)
    loss_long = float(fit_long.riesz_loss)
    loss_short = float(fit_short.riesz_loss)

    if not np.isfinite(r2_short) or r2_short >= 1.0:
        raise ValueError("the short model's outcome R^2 is not usable for calibration")
    if loss_short >= 0.0:
        raise ValueError(
            f"the short model's Riesz loss is {loss_short:.6g} >= 0, so "
            "-L_short <= 0 and the C_D^2 ratio is undefined. The representer is "
            "badly fitted."
        )
    cy2 = max(0.0, (r2_long - r2_short) / (1.0 - r2_short))
    cd2 = max(0.0, (loss_short - loss_long) / (-loss_short))

    dg = np.asarray(fit_long.g_hat) - np.asarray(fit_short.g_hat)
    da = np.asarray(fit_long.alpha_hat) - np.asarray(fit_short.alpha_hat)
    denom = float(np.sqrt(np.mean(dg**2) * np.mean(da**2)))
    rho = float(np.mean(dg * da) / denom) if denom > 0 else 0.0
    rho_p = (
        float(np.corrcoef(dg, da)[0, 1])
        if (np.std(dg) > 0 and np.std(da) > 0)
        else 0.0
    )
    return {
        "name": str(name),
        "cy2": float(cy2),
        "cd2": float(cd2),
        "rho": float(np.clip(rho, -1.0, 1.0)),
        "rho_pearson": float(np.clip(rho_p, -1.0, 1.0)),
        "r2_equal_strength": float(np.sqrt(cy2 * cd2)),
        "r2_long": r2_long,
        "r2_short": r2_short,
        "riesz_loss_long": loss_long,
        "riesz_loss_short": loss_short,
        "delta_theta": float(fit_long.theta - fit_short.theta),
    }
