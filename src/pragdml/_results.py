"""Results objects: what a fitted estimator hands back.

Follows the statsmodels/sklearn convention -- an estimator produces a result
object that carries the numbers, prints a journal-quality table, exports to LaTeX
and Markdown, draws its own figures, and can hand off to the sensitivity
machinery without refitting anything.

Author
------
Dr Merwan Roudane <merwanroudane920@gmail.com>
https://github.com/merwanroudane
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from . import _tables
from ._core import CrossFitOutput

__all__ = ["DMLResult", "InformationSetComparison"]


class DMLResult:
    """The result of a cross-fitted DML fit.

    Attributes
    ----------
    crossfit_ : CrossFitOutput
        The underlying pass, with every fitted quantity. Feed it to
        :class:`~pragdml.sensitivity.SensitivityAnalysis` or to
        :func:`~pragdml.sensitivity.benchmark_covariate`.
    theta, se, tvalue, pvalue : float
    target_label : str
        Which population object the interval refers to -- the honest answer
        depends on what the user is willing to assume, so it is recorded
        explicitly rather than left implicit.
    """

    def __init__(
        self,
        crossfit: CrossFitOutput,
        functional_name: str,
        theta: Optional[float] = None,
        se: Optional[float] = None,
        n_rep: int = 1,
        target_label: str = "representation-based target (theta_lambda)",
        extra: Optional[dict] = None,
    ) -> None:
        self.crossfit_ = crossfit
        self.functional_name = str(functional_name)
        self._theta = float(crossfit.theta if theta is None else theta)
        self._se = float(crossfit.se if se is None else se)
        self.n_rep = int(n_rep)
        self.target_label = str(target_label)
        self.extra = dict(extra or {})

    # ---- scalars -----------------------------------------------------------

    @property
    def theta(self) -> float:
        """The point estimate, Eq. (11)."""
        return self._theta

    @property
    def se(self) -> float:
        """The standard error, Eq. (12) (cluster-robust if requested)."""
        return self._se

    @property
    def n(self) -> int:
        """Sample size."""
        return self.crossfit_.n

    @property
    def tvalue(self) -> float:
        """``theta / se``."""
        return float(self._theta / self._se) if self._se > 0 else float("nan")

    @property
    def pvalue(self) -> float:
        """Two-sided normal p-value against zero."""
        from scipy import stats

        return float(2.0 * stats.norm.sf(abs(self.tvalue)))

    @property
    def sigma2(self) -> float:
        """Score variance."""
        return self.crossfit_.sigma2

    def conf_int(self, level: float = 0.95) -> tuple[float, float]:
        """The Wald interval of Eq. (16).

        What it covers depends on what you assume, and the paper is explicit about
        the hierarchy: always the representation-based target ``theta_lambda``
        (Theorem 1); also the full-information ``theta^*`` under target adaptivity;
        and efficiently so under adaptive efficiency (Theorem 2). Use
        :meth:`sensitivity` when neither is defensible.
        """
        from scipy import stats

        if not 0.0 < level < 1.0:
            raise ValueError(f"level must be in (0, 1), got {level}")
        z = float(stats.norm.ppf(0.5 + level / 2.0))
        return (self._theta - z * self._se, self._theta + z * self._se)

    def bootstrap_conf_int(
        self,
        level: float = 0.95,
        n_boot: int = 999,
        seed: Optional[int] = None,
    ) -> tuple[float, float]:
        """Gaussian-multiplier bootstrap interval (extension E3).

        Multiplies the cross-fitted influence values by standard normal weights.
        The paper reports only normal Wald intervals; this costs almost nothing
        because the influence values are already in hand, and behaves better in
        small samples. Respects clustering by drawing one multiplier per cluster.

        Parameters
        ----------
        level : float, default 0.95
        n_boot : int, default 999
        seed : int, optional

        Returns
        -------
        (float, float)
        """
        rng = np.random.default_rng(seed)
        infl = self.crossfit_.influence
        n = infl.shape[0]
        if self.crossfit_.cluster is None:
            draws = rng.normal(size=(int(n_boot), n))
            stat = draws @ infl / n
        else:
            labels = np.asarray(self.crossfit_.cluster).reshape(-1)
            _, inverse = np.unique(labels, return_inverse=True)
            sums = np.bincount(inverse, weights=infl)
            draws = rng.normal(size=(int(n_boot), sums.shape[0]))
            stat = draws @ sums / n
        lo, hi = np.quantile(stat, [(1 - level) / 2, 1 - (1 - level) / 2])
        return (float(self._theta - hi), float(self._theta - lo))

    # ---- hand-offs ---------------------------------------------------------

    def sensitivity(self):
        """A :class:`~pragdml.sensitivity.SensitivityAnalysis` for this fit."""
        from .sensitivity import SensitivityAnalysis

        return SensitivityAnalysis(self.crossfit_)

    def robustness_value(self, target: float = 0.0, level: float = 0.95) -> float:
        """[9] Definition 2 for this fit; see :func:`~pragdml.sensitivity.robustness_value`."""
        return self.sensitivity().robustness_value(target=target, level=level)

    # ---- rendering ---------------------------------------------------------

    def to_frame(self, level: float = 0.95):
        """The estimate as a one-row :class:`pandas.DataFrame`."""
        import pandas as pd

        lo, hi = self.conf_int(level)
        return pd.DataFrame(
            {
                "target": [self.functional_name],
                "estimate": [self.theta],
                "std_error": [self.se],
                "t": [self.tvalue],
                "p_value": [self.pvalue],
                "ci_lower": [lo],
                "ci_upper": [hi],
            }
        )

    def summary(self, level: float = 0.95, digits: int = 4) -> str:
        """An aligned journal-style table.

        Returns
        -------
        str
            Printable; ``print(res.summary())``.
        """
        cf = self.crossfit_
        lo, hi = self.conf_int(level)
        import pandas as pd

        body = pd.DataFrame(
            {
                "quantity": [self.functional_name],
                "estimate": [self.theta],
                "std_error": [self.se],
                "t": [self.tvalue],
                "p_value": [self.pvalue],
                f"[{level:.0%} CI lower": [lo],
                "upper]": [hi],
                "": [_tables.stars(self.pvalue)],
            }
        )
        var_kind = (
            f"cluster-robust ({cf.n_clusters} clusters)"
            if cf.cluster is not None
            else "cross-fitted score variance (Eq. 12)"
        )
        subtitle = [
            "",
            f"  Observations      {self.n}",
            f"  Folds (K)         {cf.n_folds}"
            + (f"   x {self.n_rep} repetitions (median-aggregated)"
               if self.n_rep > 1 else ""),
            f"  Variance          {var_kind}",
            f"  Interval refers to: {self.target_label}",
        ]
        for key, value in self.extra.items():
            subtitle.append(f"  {key:<17} {value}")
        footer = [
            _tables.STAR_LEGEND,
            "Score: psi = m(O, g) + alpha(W){Y - g(W)}  (Eq. 8), Neyman-orthogonal.",
            "Under target adaptivity (Def. 1) the same interval covers theta*;",
            "otherwise report a sensitivity region (Sec. 4): res.sensitivity().",
        ]
        return _tables.render_table(
            body,
            title=f"Pragmatic DML -- {self.functional_name}",
            subtitle_lines=subtitle,
            footer_lines=footer,
            digits=digits,
        )

    def to_latex(self, level: float = 0.95, digits: int = 4, label: str = "tab:dml") -> str:
        """LaTeX (booktabs) version of :meth:`to_frame`."""
        return _tables.to_latex(
            self.to_frame(level),
            caption=f"Pragmatic DML estimate: {self.functional_name}",
            label=label,
            digits=digits,
            notes=(
                f"n = {self.n}. {_tables.STAR_LEGEND}. Cross-fitted "
                f"orthogonal-score standard errors."
            ),
        )

    def to_markdown(self, level: float = 0.95, digits: int = 4) -> str:
        """Markdown version of :meth:`to_frame`."""
        return _tables.to_markdown(self.to_frame(level), digits=digits)

    # ---- figures -----------------------------------------------------------

    def plot_influence(self, ax=None, **kwargs):
        """Histogram of the cross-fitted influence values (a normality check)."""
        from ._plots import plot_influence

        return plot_influence(self, ax=ax, **kwargs)

    def plot_sensitivity_contour(self, **kwargs):
        """The [9] Sec. 5.2.3 contour plot for this fit."""
        from ._plots import plot_sensitivity_contour

        return plot_sensitivity_contour(self.sensitivity(), **kwargs)

    def __repr__(self) -> str:
        return (
            f"<DMLResult {self.functional_name}: {self.theta:.4f} "
            f"(se {self.se:.4f}), n={self.n}>"
        )


class InformationSetComparison:
    """Several information sets estimated side by side -- the paper's Table 1.

    The paper's central empirical display: one row per representation (text,
    image, tabular and their combinations), plus an aggregate row. Each row is a
    **different population target** ``theta_lambda``, which is why they need not
    agree and why their spread is itself informative.

    Attributes
    ----------
    results : dict[str, DMLResult]
    aggregate : DMLResult or None
    """

    def __init__(
        self,
        results: dict,
        aggregate=None,
        aggregate_label: str = "Star aggregate",
        notes: Sequence[str] = (),
    ) -> None:
        self.results = dict(results)
        self.aggregate = aggregate
        self.aggregate_label = str(aggregate_label)
        self.notes = list(notes)

    def to_frame(self, level: float = 0.95):
        """All rows as a :class:`pandas.DataFrame`."""
        import pandas as pd

        rows = []
        for name, res in self.results.items():
            lo, hi = res.conf_int(level)
            rows.append(
                {
                    "information_set": name,
                    "estimate": res.theta,
                    "std_error": res.se,
                    "ci_lower": lo,
                    "ci_upper": hi,
                }
            )
        if self.aggregate is not None:
            lo, hi = self.aggregate.conf_int(level)
            rows.append(
                {
                    "information_set": self.aggregate_label,
                    "estimate": self.aggregate.theta,
                    "std_error": self.aggregate.se,
                    "ci_lower": lo,
                    "ci_upper": hi,
                }
            )
        return pd.DataFrame(rows)

    @property
    def spread(self) -> float:
        """Max minus min estimate across information sets.

        The paper reads this as evidence in its own right: a narrow spread means
        the conclusion does not hinge on which modality carried the information.
        """
        vals = [r.theta for r in self.results.values()]
        return float(max(vals) - min(vals)) if vals else float("nan")

    @property
    def all_same_sign(self) -> bool:
        """Whether every confidence interval excludes zero on the same side."""
        signs = set()
        for res in self.results.values():
            lo, hi = res.conf_int()
            if lo > 0:
                signs.add(1)
            elif hi < 0:
                signs.add(-1)
            else:
                return False
        return len(signs) == 1

    def summary(self, level: float = 0.95, digits: int = 4) -> str:
        """Table 1 of the paper, rendered."""
        frame = self.to_frame(level)
        footer = [
            f"Each row is a distinct population target theta_lambda (Eq. 3):",
            "the average predictive effect of the information that row retains.",
            f"Spread across information sets: {self.spread:.4f}.",
        ]
        footer += list(self.notes)
        return _tables.render_table(
            frame,
            title="Estimates by information set",
            subtitle_lines=["", f"  Confidence level  {level:.0%}"],
            footer_lines=footer,
            digits=digits,
            headers={
                "information_set": "information set",
                "estimate": "estimate",
                "std_error": "std error",
                "ci_lower": "CI lower",
                "ci_upper": "CI upper",
            },
        )

    def to_latex(self, level: float = 0.95, digits: int = 4) -> str:
        """LaTeX version."""
        return _tables.to_latex(
            self.to_frame(level),
            caption="Estimates by information set",
            label="tab:infosets",
            digits=digits,
            notes="Each row is a distinct representation-based target.",
        )

    def to_markdown(self, level: float = 0.95, digits: int = 4) -> str:
        """Markdown version."""
        return _tables.to_markdown(self.to_frame(level), digits=digits)

    def plot(self, **kwargs):
        """Forest plot of the estimates and intervals."""
        from ._plots import plot_information_sets

        return plot_information_sets(self, **kwargs)

    def __repr__(self) -> str:
        return (
            f"<InformationSetComparison: {len(self.results)} sets, "
            f"spread {self.spread:.4f}>"
        )
