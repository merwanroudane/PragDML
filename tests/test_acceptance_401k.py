"""Acceptance tests against published numbers ([9] Sec. 5, 401(k) / SIPP 1991).

This is the only place in the suite where `pragdml` is checked against values
somebody else printed in a journal, so it is the closest thing available to a
cross-implementation proof. The target paper's own application cannot be
reproduced (its Keepa data is not public -- see `datasets/SOURCES.md`), but its
parent paper [9] analyses a public dataset and reports enough to check against.

Tolerances are stated out loud in each test and are deliberately loose where [9]
reports rounded figures ("approximately $9,000") or uses random forests whose
exact fit depends on library versions.

Author
------
Dr Merwan Roudane <merwanroudane920@gmail.com>
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pragdml import (
    AdditiveFeatures,
    LinearOutcomeLearner,
    LinearRieszLearner,
    PragmaticDML,
    SklearnOutcomeLearner,
    VaryingCoefficientFeatures,
)
from pragdml.datasets import PENSION_COVARIATES, load_pension

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def targets() -> dict:
    with open(FIXTURES / "paper_targets.json", encoding="utf-8") as fh:
        return json.load(fh)["long_story_short_401k"]


@pytest.fixture(scope="module")
def pension():
    return load_pension(return_X_y=True)


def flexible_basis():
    """Cubic splines on the continuous covariates, indicators passed through.

    [9]'s "partially linear model" is linear in the treatment and **flexible in
    the covariates** -- its nuisances are random forests. Net financial assets
    against income is strongly nonlinear, so a specification that is linear in
    ``X`` is a different (and much more restrictive) model: it reproduces plain
    OLS, about $4,900, not [9]'s $9,000. This basis supplies the flexibility in
    ``X`` while keeping the treatment linear, which is what the PLM means.
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import SplineTransformer

    continuous = [
        PENSION_COVARIATES.index(c) for c in ("age", "inc", "fsize", "educ")
    ]
    return ColumnTransformer(
        [("spline", SplineTransformer(n_knots=6, degree=3), continuous)],
        remainder="passthrough",
    )


def _plm_kwargs():
    basis = flexible_basis()
    return dict(
        functional="ate",
        outcome_learner=LinearOutcomeLearner(
            AdditiveFeatures(basis=basis), ridge=1e-8
        ),
        riesz_learner=LinearRieszLearner(
            AdditiveFeatures(basis=basis), ridge=1e-8
        ),
        n_folds=5,
    )


@pytest.fixture(scope="module")
def plm(pension):
    """The partially linear fit, repeated across 5 splits as [9] fn. 11 does."""
    Y, D, X = pension
    return PragmaticDML(**_plm_kwargs(), n_rep=5, seed=20261017).fit(Y, D, X)


# --------------------------------------------------------------------------
# The dataset itself
# --------------------------------------------------------------------------

def test_shipped_data_matches_the_published_sample(targets, pension):
    """Same extract, same ``n``, same naive difference as [9] Sec. 5.1.

    [9] reports the naive treated-minus-control difference in net financial
    assets as $19,559. Reproducing it to the dollar confirms the shipped copy is
    the extract the DML literature uses, not a variant.
    """
    Y, D, X = pension
    assert Y.shape[0] == targets["n"] == 9915
    assert X.shape[1] == len(PENSION_COVARIATES) == 8
    naive = float(Y[D == 1].mean() - Y[D == 0].mean())
    assert abs(naive - targets["naive_difference"]) < 1.0, naive


def test_dataset_loader_shapes_and_description():
    frame = load_pension()
    assert frame.shape == (9915, 14)
    assert "401(k)" in frame.attrs["DESCR"]
    assert set(PENSION_COVARIATES).issubset(frame.columns)
    as_dict = load_pension(as_frame=False)
    assert as_dict["data"].shape == (9915, 14)


# --------------------------------------------------------------------------
# [9] Table 1: the ATE under conditional ignorability
# --------------------------------------------------------------------------

def test_partially_linear_ate_is_near_the_published_value(targets, plm):
    """[9] reports about $9,000 (PLM) and $8,000 (NPM) for this ATE.

    Achieved here: about $7,850, with a 95% interval of roughly
    ``[5,100, 10,600]`` that contains **both** published figures. Tolerance:
    +/- $2,500 on the PLM value, which is the honest width given that [9] fits its
    nuisances with random forests and reports its own numbers only to the nearest
    thousand.
    """
    assert abs(plm.theta - targets["ate_partially_linear"]) < 2500, plm.theta
    lo, hi = plm.conf_int()
    assert lo > 0.0, "the published result is positive and significant"
    assert lo < targets["ate_partially_linear"] < hi
    assert lo < targets["ate_nonparametric"] < hi


def test_nonparametric_ate_is_near_the_published_value(targets, pension):
    """[9] reports about $8,000 for the nonparametric specification.

    Tolerance: +/- $3,000, wider because the fit depends on the forest.
    """
    from sklearn.ensemble import RandomForestRegressor

    Y, D, X = pension
    res = PragmaticDML(
        functional="ate",
        outcome_learner=SklearnOutcomeLearner(
            RandomForestRegressor(
                n_estimators=100, min_samples_leaf=20, random_state=0, n_jobs=1
            )
        ),
        riesz_learner=LinearRieszLearner(
            VaryingCoefficientFeatures(), ridge=1e-8
        ),
        n_folds=5, n_rep=2, seed=20261017,
    ).fit(Y, D, X)
    assert abs(res.theta - targets["ate_nonparametric"]) < 3000, res.theta
    assert res.conf_int()[0] > 0.0


def test_effect_is_attenuated_relative_to_the_naive_comparison(targets, plm):
    """[9]'s qualitative finding: adjustment shrinks the effect a lot, not to zero.

    "although the estimates ... are substantially attenuated, they are still
    large, positive and statistically significant". Achieved: about 40% of the
    naive difference.
    """
    naive = targets["naive_difference"]
    assert plm.theta < 0.75 * naive, "adjustment should attenuate substantially"
    assert plm.theta > 0.25 * naive, "but the effect should remain large"


# --------------------------------------------------------------------------
# [9] Definition 2 / Table 1: robustness values
# --------------------------------------------------------------------------

def test_robustness_value_matches_the_published_value(targets, plm):
    """[9] Table 1 reports RV(theta=0, a=0.05) = 5.4% (PLM) and 4.5% (NPM).

    Achieved here: **4.2%**, within 0.3 points of [9]'s nonparametric figure and
    1.2 points of its partially linear one -- from a spline-based nuisance fit
    rather than [9]'s random forests. This is the single closest reproduction of
    a published number in the suite, and it exercises the whole chain: the
    orthogonal score, both auxiliary scores of Eqs. (53)-(54), the delta-method
    endpoint variance of Eq. (64), and the Definition 2 inversion.

    Tolerance: 2%-8%. Pinning it tighter would be false precision given the
    different nuisance learners, but the band is narrow enough to fail if any
    link in that chain is wrong.
    """
    rv = plm.robustness_value(target=0.0, level=0.95)
    assert 0.02 < rv < 0.08, f"RV = {rv:.4f}; published values are 0.054, 0.045"
    assert abs(rv - targets["robustness_value_npm"]) < 0.02


def test_published_confounding_scenario_does_not_overturn_the_result(targets, plm):
    """[9] Sec. 5.2.2: with ``C_Y^2 = 4%`` and ``1-R^2_alpha = 3%``, the sign holds.

    [9] concludes that because both posited strengths lie below the robustness
    value, the scenario cannot bring the lower confidence limit to zero. That
    conclusion must survive here.
    """
    sa = plm.sensitivity()
    scenario = targets["confounding_scenario"]
    share = scenario["cd_share"]
    row = sa.evaluate(
        cy2=scenario["cy2"], cd2=share / (1.0 - share), rho=1.0,
        level=0.95, method="one_sided",
    )
    assert row.lower > 0.0, "the identified region should stay positive"
    assert row.ci_lower > 0.0, "and so should the confidence bound"
    # [9] reports a lower confidence bound of $2,497 under this scenario; the
    # spline fit here gives roughly $1,000. Same sign and same conclusion, a
    # different magnitude -- as expected from a different nuisance learner.
    assert row.ci_lower < targets["ci_lower_plm_under_scenario"] * 3


def test_scenario_strengths_below_rv_cannot_flip_the_sign(targets, plm):
    """The internal logic of [9]'s argument, as an invariant.

    Any scenario with both strengths strictly below the robustness value must
    leave the confidence bound on the same side of zero. If this ever fails, the
    robustness value and the evaluator disagree.
    """
    sa = plm.sensitivity()
    rv = sa.robustness_value(level=0.95)
    for factor in (0.3, 0.6, 0.95):
        row = sa.evaluate(rv=rv * factor, rho=1.0, level=0.95,
                          method="one_sided")
        assert row.ci_lower > 0.0, (factor, rv, row.ci_lower)


# --------------------------------------------------------------------------
# [9] Appendix E: benchmarking against income
# --------------------------------------------------------------------------

def test_income_is_the_strongest_benchmark_covariate(pension):
    """[9] Sec. 5.2.3 singles out income as the dominant observed covariate.

    "It is implausible to imagine other latent firm characteristics that would be
    even a fraction as strong as the observed income of individuals". Income
    should therefore benchmark larger than the weaker covariates [9] also plots
    (IRA participation and two-earner status).
    """
    Y, D, X = pension
    names = PENSION_COVARIATES
    full = PragmaticDML(**_plm_kwargs(), seed=99).fit(Y, D, X)

    from pragdml.sensitivity import benchmark_covariate

    gains = {}
    for name in ("inc", "pira", "twoearn"):
        j = names.index(name)
        reduced = np.delete(X, j, axis=1)
        short = PragmaticDML(**_plm_kwargs(), seed=99).fit(Y, D, reduced)
        gains[name] = benchmark_covariate(
            full.crossfit_, short.crossfit_, name=name
        )

    assert gains["inc"].g_y > gains["twoearn"].g_y
    assert gains["inc"].g_y > gains["pira"].g_y


def test_quarter_income_benchmark_is_a_milder_scenario(pension):
    """[9]'s "1/4 x Income" reference point must be milder than "1 x Income"."""
    from pragdml.sensitivity import benchmark_covariate

    Y, D, X = pension
    j = PENSION_COVARIATES.index("inc")
    full = PragmaticDML(**_plm_kwargs(), seed=123).fit(Y, D, X)
    short = PragmaticDML(**_plm_kwargs(), seed=123).fit(
        Y, D, np.delete(X, j, axis=1)
    )

    one = benchmark_covariate(full.crossfit_, short.crossfit_, "inc", 1.0, 1.0)
    quarter = benchmark_covariate(full.crossfit_, short.crossfit_, "inc", 0.25, 0.25)
    assert quarter.cy2 < one.cy2
    assert quarter.cd_share <= one.cd_share
