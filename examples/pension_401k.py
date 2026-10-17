"""Complete worked example: 401(k) eligibility and net financial assets.

Reproduces the empirical workflow of

    Chernozhukov, Cinelli, Newey, Sharma & Syrgkanis, "Long Story Short: Omitted
    Variable Bias in Causal Machine Learning", Review of Economics and Statistics
    (2026), doi:10.1162/REST.a.1705, Section 5,

on the public 1991 SIPP data shipped with ``pragdml``, using the pragmatic-DML
machinery of Aradillas Fernandez et al. (2026), arXiv:2610.01935.

The question: 401(k) eligibility is not randomised -- it depends on your employer
-- so the naive $19,559 gap in net financial assets between eligible and
ineligible households is confounded. Adjusting for observed covariates shrinks it
substantially. How strong would *unobserved firm characteristics* have to be to
overturn the result?

Run it:

    python examples/pension_401k.py

It prints every table and writes the figures to ``docs/assets/``. Takes under a
minute.

Author
------
Dr Merwan Roudane <merwanroudane920@gmail.com>
https://github.com/merwanroudane
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import SplineTransformer

from pragdml import (
    AdditiveFeatures,
    LinearOutcomeLearner,
    LinearRieszLearner,
    PragmaticDML,
    compare_information_sets,
)
from pragdml._plots import (
    plot_information_sets,
    plot_sensitivity_contour,
    plot_sensitivity_grid,
    save_figure,
)
from pragdml.datasets import PENSION_COVARIATES, load_pension
from pragdml.sensitivity import benchmark_covariate

SEED = 20261017
ASSETS = Path(__file__).resolve().parents[1] / "docs" / "assets"


def rule(title: str) -> None:
    print("\n" + "=" * 78)
    print(f"  {title}")
    print("=" * 78)


def flexible_basis():
    """Cubic splines on the continuous covariates; indicators passed through.

    The "partially linear model" of [9] is linear in the treatment and flexible
    in the covariates. That distinction is not cosmetic here: net financial
    assets against income is strongly nonlinear, and a specification linear in
    ``X`` returns about $4,900 -- essentially plain OLS -- instead of the ~$8,000
    the literature reports.
    """
    continuous = [
        PENSION_COVARIATES.index(c) for c in ("age", "inc", "fsize", "educ")
    ]
    return ColumnTransformer(
        [("spline", SplineTransformer(n_knots=6, degree=3), continuous)],
        remainder="passthrough",
    )


def partially_linear_model(**kwargs):
    basis = flexible_basis()
    return PragmaticDML(
        functional="ate",
        outcome_learner=LinearOutcomeLearner(
            AdditiveFeatures(basis=basis), ridge=1e-8
        ),
        riesz_learner=LinearRieszLearner(
            AdditiveFeatures(basis=basis), ridge=1e-8
        ),
        n_folds=5,
        **kwargs,
    )


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- data ---
    rule("1. The data")
    Y, D, X = load_pension(return_X_y=True)
    print(f"  Observations                 {Y.shape[0]:,}")
    print(f"  Controls                     {', '.join(PENSION_COVARIATES)}")
    naive = float(Y[D == 1].mean() - Y[D == 0].mean())
    print(f"  Naive eligible - ineligible  ${naive:,.0f}")
    print("  [9] Sec. 5.1 reports          $19,559")
    print(
        "\n  That gap is confounded: eligibility depends on the employer, and\n"
        "  employers that offer 401(k) plans differ from those that do not."
    )

    # ------------------------------------------------------ point estimate ---
    rule("2. The ATE under conditional ignorability")
    res = partially_linear_model(n_rep=5, seed=SEED).fit(Y, D, X)
    print(res.summary())
    print(
        f"\n  [9] reports about $9,000 (partially linear) and $8,000"
        f" (nonparametric).\n  Here: ${res.theta:,.0f}, i.e."
        f" {100 * res.theta / naive:.0f}% of the naive gap survives adjustment."
    )
    boot = res.bootstrap_conf_int(n_boot=999, seed=1)
    print(f"  Multiplier-bootstrap 95% CI: [{boot[0]:,.0f}, {boot[1]:,.0f}]")

    # ----------------------------------------------- identified components ---
    rule("3. What the sensitivity analysis can and cannot see")
    sa = res.sensitivity()
    print(sa.summary())

    rv = sa.robustness_value(target=0.0, level=0.95)
    print(
        f"\n  Robustness value RV(0, 0.05) = {rv:.4f}  ({100 * rv:.1f}%)\n"
        f"  [9] Table 1 reports 5.4% (partially linear) and 4.5%"
        f" (nonparametric).\n\n"
        f"  Reading: unobserved firm characteristics that explain less than\n"
        f"  {100 * rv:.1f}% of the residual variation -- both of the outcome and\n"
        f"  of the treatment -- cannot bring the lower confidence limit to zero."
    )

    # ------------------------------------------------- [9]'s own scenario ---
    rule("4. The confounding scenario of [9] Sec. 5.2.2")
    print(
        "  [9] builds a scenario from the employer 'match': unobserved firm\n"
        "  characteristics explaining an extra 4% of outcome variation and 3% of\n"
        "  treatment variation, in the worst-case alignment rho = 1."
    )
    share = 0.03
    row = sa.evaluate(
        cy2=0.04, cd2=share / (1 - share), rho=1.0, level=0.95,
        method="one_sided",
    )
    print(f"\n  Identified region      [{row.lower:,.0f}, {row.upper:,.0f}]")
    print(f"  95% confidence bound   [{row.ci_lower:,.0f}, {row.ci_upper:,.0f}]")
    print(f"  Implied bias bound     ${row.bias_bound:,.0f}")
    print(
        f"  Sign survives?         "
        f"{'yes' if row.ci_lower > 0 else 'no'}"
        f"   ([9] concludes yes, with a lower bound of $2,497)"
    )

    # ------------------------------------------------------- the full grid ---
    rule("5. Sensitivity grid (the paper's Table 3 layout)")
    grid = sa.grid(
        rhos=(1.0, 0.5, 0.2),
        r2s=(0.010, 0.025, 0.050),
        parameterization="rv",
        tail=0.10,
    )
    show = grid[["rho", "r2", "p10", "lower", "theta", "upper", "p90"]].copy()
    print(show.to_string(index=False, float_format=lambda v: f"{v:,.0f}"))
    print(
        "\n  Note the asymmetry the paper emphasises: the *sign* is far more\n"
        "  robust than the *magnitude*."
    )

    # ------------------------------------------------------- benchmarking ---
    rule("6. Benchmarking against observed covariates ([9] Appendix E)")
    print(
        "  Rather than asserting strengths out of thin air, calibrate them:\n"
        "  'the omitted information is no stronger than observed income'.\n"
    )
    benchmarks = []
    for name in ("inc", "pira", "twoearn", "educ"):
        j = PENSION_COVARIATES.index(name)
        short = partially_linear_model(seed=SEED).fit(
            Y, D, np.delete(X, j, axis=1)
        )
        for k in (1.0, 0.25):
            bm = benchmark_covariate(
                res.crossfit_, short.crossfit_, name=name, k_y=k, k_d=k
            )
            benchmarks.append(bm)
    header = f"  {'covariate':<10} {'k':>5} {'C_Y^2':>9} {'1-R2_a':>9} {'rho_j':>8}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for bm in benchmarks:
        print(
            f"  {bm.name:<10} {bm.k_y:>5.2f} {bm.cy2:>9.4f} "
            f"{bm.cd_share:>9.4f} {bm.rho_pearson:>8.3f}"
        )
    print(
        "\n  [9] singles out income as the dominant covariate and argues that a\n"
        "  latent firm characteristic as strong as income is implausible, so the\n"
        "  '1/4 x income' row is the realistic conservative scenario."
    )

    # ------------------------------------------- comparing information sets --
    rule("7. Estimates by information set (the paper's Table 1 layout)")
    sets = {
        "demographics": X[:, [PENSION_COVARIATES.index(c)
                              for c in ("age", "fsize", "marr", "twoearn")]],
        "economic": X[:, [PENSION_COVARIATES.index(c)
                          for c in ("inc", "educ")]],
        "assets": X[:, [PENSION_COVARIATES.index(c) for c in ("pira", "hown")]],
        "all": X,
    }
    # A width-agnostic flexible map, so the same specification serves every
    # subset. AdditiveFeatures rather than the varying-coefficient map: the
    # latter doubles the basis width and makes the Riesz solve degenerate here
    # (pragdml warns when that happens -- try it and see).
    def wide_map():
        return AdditiveFeatures(basis=SplineTransformer(n_knots=5, degree=3))

    comparison = compare_information_sets(
        Y, D, sets, functional="ate",
        outcome_learner=LinearOutcomeLearner(wide_map(), ridge=1e-8),
        riesz_learner=LinearRieszLearner(wide_map(), ridge=1e-8),
        n_folds=5, seed=SEED, aggregation="star",
    )
    print(comparison.summary())
    print(
        "\n  Each row is a different population target theta_lambda (Eq. 3):\n"
        "  the average predictive effect of the information that row retains.\n"
        "  They are not competing estimates of one number.\n\n"
        "  What the rows show here: adjusting only for demographics leaves the\n"
        "  estimate near the naive gap, while the economic block (income and\n"
        "  education) reproduces almost the whole adjustment. Income is doing\n"
        "  the work -- which is exactly why [9] benchmarks against it."
    )

    # ------------------------------------------------------------ figures ---
    rule("8. Figures")
    ax = plot_information_sets(comparison, title="401(k) ATE by information set")
    save_figure(ax.figure, str(ASSETS / "pension_information_sets"))

    ax = plot_sensitivity_contour(
        sa, cy2_max=0.12, cd_share_max=0.12, n_grid=45, rho=1.0, level=0.95,
    )
    save_figure(ax.figure, str(ASSETS / "pension_sensitivity_contour"))

    ax = plot_sensitivity_grid(
        sa, rhos=(1.0, 0.5, 0.2), r2s=(0.010, 0.025, 0.050),
        parameterization="rv",
    )
    save_figure(ax.figure, str(ASSETS / "pension_sensitivity_grid"))

    ax = res.plot_influence()
    save_figure(ax.figure, str(ASSETS / "pension_influence"))

    print(f"  Written to {ASSETS}:")
    for stem in (
        "pension_information_sets", "pension_sensitivity_contour",
        "pension_sensitivity_grid", "pension_influence",
    ):
        print(f"    {stem}.pdf / .png")

    # ------------------------------------------------------------ exports ---
    rule("9. Exports")
    print("  LaTeX (booktabs):\n")
    print(res.to_latex(label="tab:pension"))
    print("\n  Markdown:\n")
    print(comparison.to_markdown())

    rule("Summary")
    print(
        f"  Naive gap                 ${naive:,.0f}\n"
        f"  Adjusted ATE              ${res.theta:,.0f}  (se ${res.se:,.0f})\n"
        f"  Robustness value          {100 * rv:.1f}%\n"
        f"  [9]'s scenario            sign "
        f"{'survives' if row.ci_lower > 0 else 'does not survive'}\n\n"
        "  Conclusion, following [9]: the positive effect of 401(k) eligibility\n"
        "  on net financial assets is robust to the plausible confounding\n"
        "  scenarios considered, though its magnitude is sensitive to them."
    )


if __name__ == "__main__":
    main()
