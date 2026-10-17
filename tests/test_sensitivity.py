"""Sensitivity analysis: Eqs. (50)-(65), [9] Thm. 4 / Def. 2 / App. E.

Includes the strongest compatibility check available for the target paper without
its data: an exact reconstruction of all eighteen endpoints of its Table 3.

Author
------
Dr Merwan Roudane <merwanroudane920@gmail.com>
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pragdml import ATE, LinearOutcomeLearner, LinearRieszLearner, PragmaticDML
from pragdml._core import CrossFitOutput
from pragdml.sensitivity import (
    SensitivityAnalysis,
    benchmark_covariate,
    bias_factor,
    calibrate_deletion,
    robustness_value,
)

FIXTURES = Path(__file__).parent / "fixtures"


def _targets() -> dict:
    with open(FIXTURES / "paper_targets.json", encoding="utf-8") as fh:
        return json.load(fh)


def synthetic_fit(theta: float, sigma2: float, nu2: float, n: int = 2000):
    """A :class:`CrossFitOutput` with prescribed ``theta``, ``sigma_0^2``, ``nu_0^2``.

    Lets the Eq. (51)/(58) arithmetic be tested against published numbers without
    needing the data that produced them.
    """
    y = np.zeros(n)
    g_hat = -np.full(n, np.sqrt(sigma2))          # so (y - g)^2 == sigma2
    alpha = np.full(n, np.sqrt(nu2))
    m_alpha = alpha**2                            # so 2 m_alpha - alpha^2 == nu2
    psi = np.full(n, theta)
    return CrossFitOutput(
        psi=psi, den=np.ones(n), g_hat=g_hat, alpha_hat=alpha,
        m_g=psi.copy(), m_alpha=m_alpha, y=y,
        fold_id=np.zeros(n, dtype=np.int64), functional_name="test",
    )


def _data(n=800, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    e = 1.0 / (1.0 + np.exp(-0.8 * X[:, 0]))
    D = (rng.uniform(size=n) < e).astype(float)
    Y = 2.0 * D + X[:, 0] - 0.5 * X[:, 1] + rng.normal(size=n)
    return Y, D, X


# --------------------------------------------------------------------------
# Compatibility with the paper's published Table 3
# --------------------------------------------------------------------------

def test_reproduces_paper_table3_endpoints():
    """All 18 endpoints of Table 3, to the paper's printed precision.

    The paper's Table 3 reports the sensitivity region of its full multimodal
    estimate (centre -1.069) over nine ``(rho, r^2)`` scenarios. Given the centre
    and the single identified scale ``S``, Eq. (51) determines every endpoint, so
    the table is reproducible even though its underlying data is not public.

    **A documented discrepancy in the source.** Table 3's note states
    ``C_Y^2 = C_D^2 = r^2``, but its printed numbers are only reproducible under
    ``C_Y^2 = r^2`` with ``C_D^2 = r^2/(1-r^2)`` -- the [9] Definition 2
    convention, i.e. ``parameterization="rv"`` here. Fitting a single ``S`` by
    least squares gives max residual 0.00044 under that convention against
    0.00147 under the note's, and the note's version misses the printed third
    decimal in several rows. This is why ``pragdml`` keeps the two
    parameterizations separate instead of treating them as interchangeable.
    """
    table = _targets()["pragmatic_dml_table3"]
    rows = np.asarray(table["rows"], dtype=float)
    theta = float(table["_centre"])
    S = 6.7452                     # recovered by least squares from the table

    sa = SensitivityAnalysis(synthetic_fit(theta, S, S))
    assert abs(sa.S - S) < 1e-9
    assert abs(sa.theta - theta) < 1e-12

    for rho, r2, _p10, lower, centre, upper, _p90 in rows:
        assert abs(centre - theta) < 1e-12
        lo, hi = sa.interval(rv=r2, rho=rho)
        assert abs(lo - lower) < 5e-4, (rho, r2, lo, lower)
        assert abs(hi - upper) < 5e-4, (rho, r2, hi, upper)


def test_table3_note_convention_fits_worse():
    """Guard the finding above: the note's convention really does fit worse."""
    table = _targets()["pragmatic_dml_table3"]
    rows = np.asarray(table["rows"], dtype=float)
    theta = float(table["_centre"])
    rho, r2, lower = rows[:, 0], rows[:, 1], rows[:, 3]
    shift = np.abs(lower - theta)

    def best_resid(c):
        S = float(np.sum(c * shift) / np.sum(c * c))
        return float(np.max(np.abs(c * S - shift)))

    resid_note = best_resid(rho * r2)                       # C_Y^2 = C_D^2 = r^2
    resid_rv = best_resid(rho * r2 / np.sqrt(1.0 - r2))      # [9] Def. 2
    assert resid_rv < resid_note / 3.0


def test_paper_table2_calibration_is_internally_consistent():
    """Table 2's reported components, checked against its own stated formulas.

    ``C_D^2 = (L_text - L_full)/(-L_text)`` reproduces the printed 0.0088 exactly.

    **A second documented discrepancy in the source.** Table 2's printed
    ``C_Y^2 = 0.0017`` equals the *raw numerator* ``R^2_full - R^2_text``, not the
    partial-``R^2`` ratio ``(R^2_full - R^2_text)/(1 - R^2_text) = 0.001752`` that
    Sec. 5.4 writes. The two differ by 3% here and would differ far more at a
    higher baseline ``R^2``. ``pragdml`` implements the stated formula (the ratio),
    because that is the quantity Eq. (50) needs and the one [9] Eq. (24) derives;
    this test pins both numbers so the choice stays visible.
    """
    t2 = _targets()["pragmatic_dml_table2"]
    raw = t2["outcome_r2_full"] - t2["outcome_r2_text"]
    ratio = raw / (1.0 - t2["outcome_r2_text"])
    cd2 = (t2["riesz_loss_text"] - t2["riesz_loss_full"]) / (
        -t2["riesz_loss_text"]
    )

    assert abs(cd2 - t2["cd2"]) < 5e-5           # follows the paper's formula
    assert abs(raw - t2["cy2"]) < 5e-6           # what the paper actually printed
    assert abs(ratio - 0.001752) < 5e-6          # what its formula implies
    # Both round to the paper's reported equal-strength benchmark.
    assert abs(np.sqrt(raw * cd2) - t2["r2_equal_strength"]) < 5e-5
    assert abs(np.sqrt(ratio * cd2) - t2["r2_equal_strength"]) < 5e-5


# --------------------------------------------------------------------------
# The factorization and the two parameterizations
# --------------------------------------------------------------------------

def test_bias_factor_parameterizations_differ_as_documented():
    """``rv`` implies ``C_D^2 = RV/(1-RV)``, which exceeds ``C_D^2 = RV``."""
    rv = 0.2
    assert bias_factor(None, rv=rv) > bias_factor(rv, rv)
    assert abs(bias_factor(rv, rv / (1 - rv)) - bias_factor(None, rv=rv)) < 1e-12


def test_bias_factor_rejects_impossible_cy2():
    """``C_Y^2`` is a share of residual variance, so it cannot exceed one."""
    with pytest.raises(ValueError, match="exceeds 1"):
        bias_factor(1.5, 0.1)


def test_zero_scenario_collapses_the_region():
    """With no omitted information the region is the point estimate."""
    sa = SensitivityAnalysis(synthetic_fit(1.0, 4.0, 4.0))
    lo, hi = sa.interval(cy2=0.0, cd2=0.0)
    assert abs(hi - lo) < 1e-12 and abs(lo - 1.0) < 1e-12


def test_region_widens_monotonically_in_every_parameter():
    """More strength or more alignment can only widen the region."""
    sa = SensitivityAnalysis(synthetic_fit(1.0, 4.0, 4.0))
    widths = [
        sa.interval(cy2=r, cd2=r)[1] - sa.interval(cy2=r, cd2=r)[0]
        for r in (0.01, 0.02, 0.05, 0.10)
    ]
    assert all(b > a for a, b in zip(widths, widths[1:]))
    by_rho = [
        sa.interval(cy2=0.05, cd2=0.05, rho=r)[1] for r in (0.2, 0.5, 1.0)
    ]
    assert all(b > a for a, b in zip(by_rho, by_rho[1:]))


# --------------------------------------------------------------------------
# Eq. (64) / [9] Theorem 4: the endpoint influence function
# --------------------------------------------------------------------------

def test_delta_method_factors_match_numerical_derivatives():
    """``dS/d(sigma^2) = nu^2/(2S)`` and ``dS/d(nu^2) = sigma^2/(2S)``.

    Eq. (64)'s shape rests on exactly these two partials of
    ``(u, v) -> (u v)^{1/2}``. Checking them numerically validates the derivation
    independently of how it is coded.
    """
    s2, n2 = 1.7, 3.3
    S = np.sqrt(s2 * n2)
    h = 1e-6
    d_ds2 = (np.sqrt((s2 + h) * n2) - np.sqrt((s2 - h) * n2)) / (2 * h)
    d_dn2 = (np.sqrt(s2 * (n2 + h)) - np.sqrt(s2 * (n2 - h))) / (2 * h)
    assert abs(d_ds2 - n2 / (2 * S)) < 1e-7
    assert abs(d_dn2 - s2 / (2 * S)) < 1e-7


def test_endpoint_scores_are_symmetric_around_the_point_score():
    """``psi_minus + psi_plus == 2 psi_theta`` by construction of Eq. (64)."""
    Y, D, X = _data(600, seed=1)
    sa = PragmaticDML(functional="ate", seed=2).fit(Y, D, X).sensitivity()
    psi_m, psi_p = sa.scores(c=0.05)
    base = sa.fit.influence
    np.testing.assert_allclose(psi_m + psi_p, 2.0 * base, rtol=1e-12, atol=1e-12)


def test_zero_bias_factor_gives_the_point_estimate_scores():
    """At ``c = 0`` both endpoint scores reduce to the point-estimate score."""
    Y, D, X = _data(500, seed=3)
    sa = PragmaticDML(functional="ate", seed=4).fit(Y, D, X).sensitivity()
    psi_m, psi_p = sa.scores(c=0.0)
    np.testing.assert_allclose(psi_m, sa.fit.influence, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(psi_p, sa.fit.influence, rtol=1e-12, atol=1e-12)


# --------------------------------------------------------------------------
# Envelopes (Eq. 65 vs [9] Theorem 4 vs joint)
# --------------------------------------------------------------------------

def test_envelope_ordering_one_sided_inside_joint_inside_bonferroni():
    """The three rules nest as their guarantees imply.

    One-sided (per-endpoint) is narrowest, Bonferroni (simultaneous, ignoring
    correlation) is widest, and the joint rule sits between -- strictly inside
    Bonferroni whenever the endpoint scores are correlated.
    """
    Y, D, X = _data(1000, seed=5)
    sa = PragmaticDML(functional="ate", seed=6).fit(Y, D, X).sensitivity()
    one = sa.evaluate(cy2=0.03, cd2=0.03, method="one_sided")
    joint = sa.evaluate(cy2=0.03, cd2=0.03, method="joint")
    bonf = sa.evaluate(cy2=0.03, cd2=0.03, method="bonferroni")

    width = lambda r: r.ci_upper - r.ci_lower  # noqa: E731
    assert width(one) <= width(joint) + 1e-9
    assert width(joint) <= width(bonf) + 1e-9


def test_envelope_contains_the_identified_region():
    """Every rule's envelope must contain the region it is an envelope for."""
    Y, D, X = _data(600, seed=7)
    sa = PragmaticDML(functional="ate", seed=8).fit(Y, D, X).sensitivity()
    for method in ("bonferroni", "one_sided", "joint"):
        res = sa.evaluate(cy2=0.02, cd2=0.02, method=method)
        assert res.ci_lower <= res.lower
        assert res.ci_upper >= res.upper


# --------------------------------------------------------------------------
# [9] Definition 2: robustness values
# --------------------------------------------------------------------------

def test_robustness_value_is_the_breakdown_point():
    """At ``RV`` the bound reaches zero; just inside it does not, just outside it does."""
    Y, D, X = _data(1500, seed=9)
    sa = PragmaticDML(functional="ate", seed=10).fit(Y, D, X).sensitivity()
    rv = robustness_value(sa, target=0.0, level=0.95)
    assert 0.0 < rv < 1.0

    inside = sa.evaluate(rv=rv * 0.9, rho=1.0, level=0.95, method="one_sided")
    outside = sa.evaluate(rv=min(rv * 1.1, 0.999), rho=1.0, level=0.95,
                          method="one_sided")
    assert inside.ci_lower > 0.0
    assert outside.ci_lower < 0.0


def test_robustness_value_zero_when_estimate_is_already_insignificant():
    """If the interval already contains zero, no omitted information is needed."""
    rng = np.random.default_rng(11)
    X = rng.normal(size=(400, 2))
    D = (rng.uniform(size=400) < 0.5).astype(float)
    Y = X[:, 0] + rng.normal(size=400)          # no treatment effect
    sa = PragmaticDML(functional="ate", seed=12).fit(Y, D, X).sensitivity()
    assert sa.robustness_value() == 0.0


def test_point_robustness_value_exceeds_confidence_version():
    """Ignoring sampling noise (``level=0``) can only make the result look safer."""
    Y, D, X = _data(1500, seed=13)
    sa = PragmaticDML(functional="ate", seed=14).fit(Y, D, X).sensitivity()
    assert robustness_value(sa, level=0.0) >= robustness_value(sa, level=0.95)


# --------------------------------------------------------------------------
# [9] Appendix E and Sec. 5.4
# --------------------------------------------------------------------------

def test_benchmark_detects_a_strong_covariate_and_ignores_an_irrelevant_one():
    """A covariate that drives the outcome must benchmark larger than noise."""
    rng = np.random.default_rng(15)
    n = 1500
    X = rng.normal(size=(n, 3))
    e = 1.0 / (1.0 + np.exp(-1.2 * X[:, 0]))
    D = (rng.uniform(size=n) < e).astype(float)
    Y = 2.0 * D + 3.0 * X[:, 0] + rng.normal(size=n)   # X0 matters, X2 does not

    full = PragmaticDML(functional="ate", seed=16).fit(Y, D, X)
    no_x0 = PragmaticDML(functional="ate", seed=16).fit(Y, D, X[:, 1:])
    no_x2 = PragmaticDML(functional="ate", seed=16).fit(Y, D, X[:, :2])

    strong = benchmark_covariate(full.crossfit_, no_x0.crossfit_, name="X0")
    weak = benchmark_covariate(full.crossfit_, no_x2.crossfit_, name="X2")
    assert strong.g_y > weak.g_y
    assert strong.cd_share >= 0.0 and weak.cd_share >= 0.0


def test_benchmark_k_scales_linearly():
    """``k`` is a pure multiplier on the observed gain (Eqs. 26 and 22).

    Uses a *modest* benchmark covariate: a dominant one makes ``C_Y^2`` exceed
    one, which is a legitimate "this yardstick is infeasible" signal rather than
    something to clip -- see :func:`benchmark_covariate` and the next test.
    """
    rng = np.random.default_rng(17)
    n = 1200
    X = rng.normal(size=(n, 3))
    D = (rng.uniform(size=n) < 0.5).astype(float)
    Y = D + 0.3 * X[:, 0] + rng.normal(size=n)
    full = PragmaticDML(functional="ate", seed=18).fit(Y, D, X)
    short = PragmaticDML(functional="ate", seed=18).fit(Y, D, X[:, 1:])
    one = benchmark_covariate(full.crossfit_, short.crossfit_, k_y=1.0)
    quarter = benchmark_covariate(full.crossfit_, short.crossfit_, k_y=0.25)
    assert one.cy2 < 1.0, "pick a weaker benchmark covariate for this test"
    assert abs(quarter.cy2 - 0.25 * one.cy2) < 1e-12
    assert abs(quarter.cd_share - 0.25 * one.cd_share) < 1e-12 or one.cd_share == 0.0


def test_infeasible_benchmark_is_surfaced_not_clipped():
    """A dominant covariate yields ``C_Y^2 > 1``, and the bound then refuses it."""
    rng = np.random.default_rng(31)
    n = 1200
    X = rng.normal(size=(n, 3))
    D = (rng.uniform(size=n) < 0.5).astype(float)
    Y = D + 6.0 * X[:, 0] + 0.2 * rng.normal(size=n)   # X0 explains nearly all
    full = PragmaticDML(functional="ate", seed=32).fit(Y, D, X)
    short = PragmaticDML(functional="ate", seed=32).fit(Y, D, X[:, 1:])
    bm = benchmark_covariate(full.crossfit_, short.crossfit_, name="X0")
    assert bm.cy2 > 1.0
    with pytest.raises(ValueError, match="exceeds 1"):
        bias_factor(bm.cy2, bm.cd2)


def test_deletion_calibration_and_benchmark_relate_as_documented():
    """The two routines answer different questions, and the arithmetic shows it.

    * :func:`calibrate_deletion` is the target paper's Sec. 5.4 **deletion
      exercise**: the deleted block *is* the omitted information, so its
      ``C_Y^2`` is the partial ``R^2`` itself, ``(R^2_long - R^2_short)/(1 -
      R^2_short)``.
    * :func:`benchmark_covariate` is [9] App. E **relative benchmarking**: the
      omitted information is unobserved and ``X_j`` only sets the scale, so its
      ``G_Y`` is the *odds* form ``partial/(1 - partial)`` of Eq. (26), to be
      multiplied by ``k_Y``.

    Hence ``G_Y = C_Y^2_deletion / (1 - C_Y^2_deletion)`` exactly, and the
    representer side matches directly. Pinning this keeps the two conventions from
    being silently swapped.
    """
    rng = np.random.default_rng(19)
    n = 1200
    X = rng.normal(size=(n, 3))
    e = 1.0 / (1.0 + np.exp(-0.9 * X[:, 0]))
    D = (rng.uniform(size=n) < e).astype(float)
    Y = D + 2.0 * X[:, 0] + rng.normal(size=n)
    lon = PragmaticDML(functional="ate", seed=20).fit(Y, D, X)
    sho = PragmaticDML(functional="ate", seed=20).fit(Y, D, X[:, 1:])

    cal = calibrate_deletion(lon.crossfit_, sho.crossfit_)
    bm = benchmark_covariate(lon.crossfit_, sho.crossfit_)

    # Outcome side: deletion gives the partial R^2, benchmarking its odds form.
    partial = cal["cy2"]
    assert abs(bm.g_y - partial / (1.0 - partial)) < 1e-10
    # Representer side: the two agree directly (both are G_D).
    assert abs(cal["cd2"] - bm.g_d) < 1e-10
    # And [9] then maps G_D on to C_D^2.
    assert abs(bm.cd2 - bm.g_d / (1.0 - bm.g_d)) < 1e-10
    # The alignment reported by both is the same object, computed the same way.
    assert abs(cal["rho"] - bm.rho_uncentred) < 1e-10


# --------------------------------------------------------------------------
# Guards
# --------------------------------------------------------------------------

def test_rejects_nonpositive_nu2_with_an_actionable_message():
    """A degenerate representer must fail loudly, as Theorem 6 requires S_0 > 0."""
    bad = synthetic_fit(1.0, 2.0, 1.0)
    bad.alpha_hat = np.ones_like(bad.alpha_hat)
    bad.m_alpha = np.zeros_like(bad.m_alpha)      # 2*0 - 1 = -1 < 0
    with pytest.raises(ValueError, match="badly fitted"):
        SensitivityAnalysis(bad)


def test_grid_shape_and_columns():
    """The Table 3 grid comes back with the paper's columns, including tails."""
    Y, D, X = _data(500, seed=21)
    sa = PragmaticDML(functional="ate", seed=22).fit(Y, D, X).sensitivity()
    grid = sa.grid(tail=0.10)
    assert len(grid) == 9
    for col in ("rho", "r2", "p10", "lower", "theta", "upper", "p90"):
        assert col in grid.columns


def test_contour_grid_is_monotone_in_both_axes():
    """The lower bound falls as either omitted share grows."""
    Y, D, X = _data(600, seed=23)
    sa = PragmaticDML(functional="ate", seed=24).fit(Y, D, X).sensitivity()
    data = sa.contour_data(n_grid=8, bound="point_lower")
    vals = data["values"]
    assert np.all(np.diff(vals, axis=0) <= 1e-12)   # increasing C_Y^2
    assert np.all(np.diff(vals, axis=1) <= 1e-12)   # increasing representer share
