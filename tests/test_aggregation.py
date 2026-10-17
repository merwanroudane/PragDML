"""Aggregation: Definitions 4-6, Eqs. (36)-(49).

Author
------
Dr Merwan Roudane <merwanroudane920@gmail.com>
"""

from __future__ import annotations

import numpy as np
import pytest

from pragdml import (
    ATE,
    LinearRieszLearner,
    PipelineDML,
    convex_aggregate,
    pooled_convex_aggregate,
    pooled_star_aggregate,
    riesz_loss,
    star_aggregate,
)
from pragdml.aggregation import (
    convex_oracle_bound,
    segment_weight_riesz,
    segment_weight_square,
    star_oracle_bound,
)
from pragdml.diagnostics import critical_radius_l1_hull, pooling_conditions


# --------------------------------------------------------------------------
# Eq. (36): the closed-form square-loss segment weight
# --------------------------------------------------------------------------

def test_segment_weight_square_matches_brute_force():
    """The closed form must equal a grid search over ``beta`` in ``[0, 1]``."""
    rng = np.random.default_rng(0)
    for _ in range(25):
        n = 150
        y = rng.normal(size=n)
        centre = y + 0.4 * rng.normal(size=n)
        competitor = y + 0.9 * rng.normal(size=n)
        beta = segment_weight_square(y, centre, competitor)

        grid = np.linspace(0.0, 1.0, 4001)
        losses = [
            np.mean((y - (b * centre + (1 - b) * competitor)) ** 2) for b in grid
        ]
        assert abs(beta - grid[int(np.argmin(losses))]) < 1e-3


def test_segment_weight_riesz_matches_brute_force():
    """Eq. (37), the same check for the Riesz loss."""
    rng = np.random.default_rng(1)
    for _ in range(25):
        n = 150
        centre = rng.normal(size=n)
        competitor = rng.normal(size=n)
        m_centre = rng.normal(size=n)
        m_competitor = rng.normal(size=n)
        beta = segment_weight_riesz(centre, competitor, m_centre, m_competitor)

        grid = np.linspace(0.0, 1.0, 4001)
        losses = []
        for b in grid:
            a = b * centre + (1 - b) * competitor
            m = b * m_centre + (1 - b) * m_competitor
            losses.append(np.mean(a**2 - 2.0 * m))
        assert abs(beta - grid[int(np.argmin(losses))]) < 1e-3


def test_segment_weight_handles_identical_candidates():
    """A degenerate segment keeps the centre rather than dividing by zero."""
    v = np.ones(10)
    assert segment_weight_square(v, v, v) == 1.0
    assert segment_weight_riesz(v, v, v, v) == 1.0


# --------------------------------------------------------------------------
# Definition 4: star aggregation properties
# --------------------------------------------------------------------------

def test_star_never_does_worse_than_the_best_single_candidate():
    """The defining guarantee: the star hull contains every candidate."""
    rng = np.random.default_rng(2)
    for _ in range(30):
        n, M = 200, 6
        y = rng.normal(size=n)
        F = y[:, None] + rng.normal(size=(n, M)) * rng.uniform(0.2, 1.5, M)
        res = star_aggregate(F, y=y)
        assert res.objective <= res.best_single_objective + 1e-12
        assert abs(res.weights.sum() - 1.0) < 1e-12
        assert np.all(res.weights >= -1e-12)


def test_star_mixes_two_offsetting_candidates():
    """Two candidates biased in opposite directions should be blended, not chosen."""
    y = np.linspace(0.0, 10.0, 200)
    F = np.column_stack([y + 1.0, y - 1.0])
    res = star_aggregate(F, y=y)
    assert abs(res.beta - 0.5) < 1e-6
    assert res.effective_candidates == 2
    assert res.objective < 1e-10


def test_star_ignores_a_useless_candidate():
    """A pure-noise candidate must receive zero weight."""
    rng = np.random.default_rng(3)
    y = np.linspace(0.0, 5.0, 300)
    F = np.column_stack([y + 0.05 * rng.normal(size=300), rng.normal(size=300) * 9])
    res = star_aggregate(F, y=y)
    assert res.weights[1] < 1e-8
    assert res.is_degenerate


def test_star_on_the_riesz_objective_improves_the_loss():
    """Aggregating fitted representers lowers the out-of-sample Riesz loss."""
    rng = np.random.default_rng(4)
    n = 2000
    X = rng.normal(size=(n, 3))
    e = 1.0 / (1.0 + np.exp(-X[:, 0]))
    D = (rng.uniform(size=n) < e).astype(float)
    func = ATE()

    preds, m_preds, losses = [], [], []
    for cols in ([0], [1], [0, 1], [0, 1, 2]):
        learner = LinearRieszLearner().fit(None, D, X[:, cols], func)
        preds.append(learner(D, X[:, cols]))
        m_preds.append(func.apply(learner, D, X[:, cols]))
        losses.append(riesz_loss(learner, func, D, X[:, cols]))

    res = star_aggregate(
        np.column_stack(preds), m_candidates=np.column_stack(m_preds),
        objective="riesz",
    )
    assert res.objective <= min(losses) + 1e-10


# --------------------------------------------------------------------------
# Convex aggregation
# --------------------------------------------------------------------------

def test_convex_weights_live_on_the_simplex():
    rng = np.random.default_rng(5)
    y = rng.normal(size=250)
    F = y[:, None] + rng.normal(size=(250, 5))
    res = convex_aggregate(F, y=y)
    assert abs(res.weights.sum() - 1.0) < 1e-6
    assert np.all(res.weights >= -1e-9)


def test_convex_beats_or_matches_star_in_sample():
    """Eq. (46) vs Eq. (42): the convex benchmark is weakly better.

    The star hull of a menu is contained in its convex hull, so the convex
    optimum cannot be worse in sample. (Out of sample it pays for the larger
    class -- that is the whole Sec. 3.4 trade-off.)
    """
    rng = np.random.default_rng(6)
    n, M = 300, 5
    y = rng.normal(size=n)
    F = y[:, None] + rng.normal(size=(n, M)) * rng.uniform(0.3, 1.2, M)
    star = star_aggregate(F, y=y)
    convex = convex_aggregate(F, y=y)
    assert convex.objective <= star.objective + 1e-6


def test_convex_recovers_an_equal_blend():
    y = np.linspace(0.0, 10.0, 400)
    F = np.column_stack([y + 1.0, y - 1.0])
    res = convex_aggregate(F, y=y)
    assert abs(res.weights[0] - 0.5) < 1e-4


# --------------------------------------------------------------------------
# Pooled variants (Sec. 3.4)
# --------------------------------------------------------------------------

def test_pooled_star_equals_plain_star_on_the_same_matrix():
    """Definition 5's own observation: it *is* the Definition 4 estimator.

    Only the bookkeeping and the reported remainder differ, so the numbers must
    coincide exactly.
    """
    rng = np.random.default_rng(7)
    y = rng.normal(size=200)
    F = y[:, None] + rng.normal(size=(200, 4))
    plain = star_aggregate(F, y=y)
    pooled = pooled_star_aggregate(F, y=y, n_folds=5)
    np.testing.assert_allclose(plain.weights, pooled.weights, rtol=0, atol=0)
    assert plain.objective == pooled.objective
    assert pooled.method == "pooled_star"
    assert pooled.oracle["pooled"] is True


def test_pooled_convex_matches_plain_convex():
    rng = np.random.default_rng(8)
    y = rng.normal(size=200)
    F = y[:, None] + rng.normal(size=(200, 3))
    plain = convex_aggregate(F, y=y)
    pooled = pooled_convex_aggregate(F, y=y, n_folds=5)
    np.testing.assert_allclose(plain.weights, pooled.weights, rtol=1e-10, atol=1e-10)


# --------------------------------------------------------------------------
# Oracle remainders and the design conditions
# --------------------------------------------------------------------------

def test_star_remainder_is_logarithmic_and_convex_remainder_is_linear_in_M():
    """The whole Sec. 3.4 design argument, in one assertion.

    Going from ``M = 5`` to ``M = 500`` multiplies the star remainder by about
    ``log(500)/log(5)`` (a small factor) and the convex remainder by about 100.
    """
    small_star = star_oracle_bound(1000, 5)["remainder"]
    big_star = star_oracle_bound(1000, 500)["remainder"]
    small_cvx = convex_oracle_bound(1000, 5)["remainder"]
    big_cvx = convex_oracle_bound(1000, 500)["remainder"]
    assert big_star / small_star < 5.0
    assert big_cvx / small_cvx > 50.0


def test_pooling_conditions_prefer_star_for_a_large_menu():
    """Eq. (45) tolerates a big menu; Eq. (49) does not."""
    star = pooling_conditions(10000, 500, 0.02, 0.02, weights="star")
    convex = pooling_conditions(10000, 500, 0.02, 0.02, weights="convex")
    assert star["pooling_term"] < convex["pooling_term"]
    assert star["satisfied"] and not convex["satisfied"]


def test_critical_radius_regimes_switch_with_the_menu_size():
    """Eq. (35): ``sqrt(M/n)`` binds for a small menu, the entropy term for a big one."""
    small = critical_radius_l1_hull(1000, 3, radius=1.0, sup_norm=1.0)
    big = critical_radius_l1_hull(1000, 100000, radius=1.0, sup_norm=1.0)
    assert small["binding_regime"] == "sqrt(M/n)"
    assert big["binding_regime"] != "sqrt(M/n)"


def test_oracle_bounds_carry_their_caveat():
    """The unspecified universal constant must never be presented as a guarantee."""
    for out in (star_oracle_bound(100, 5), convex_oracle_bound(100, 5)):
        assert "not a finite-sample guarantee" in out["caveat"]
        assert out["constant"] == 1.0


# --------------------------------------------------------------------------
# PipelineDML (Theorems 4-5)
# --------------------------------------------------------------------------

def test_pipeline_dml_recovers_the_ate_over_a_menu():
    rng = np.random.default_rng(9)
    n = 3000
    Z = rng.normal(size=(n, 4))
    e = 1.0 / (1.0 + np.exp(-(0.9 * Z[:, 0] - 0.4 * Z[:, 1])))
    D = (rng.uniform(size=n) < e).astype(float)
    Y = 2.0 * D + 1.5 * Z[:, 0] - Z[:, 1] + rng.normal(size=n)

    menu = {
        "z0": Z[:, :1],
        "z01": Z[:, :2],
        "z23": Z[:, 2:],
        "all": Z,
    }
    res = PipelineDML(functional="ate", aggregation="star", seed=10).fit(Y, D, menu)
    lo, hi = res.conf_int()
    assert lo < 2.0 < hi
    assert "pipeline target" in res.target_label


def test_pipeline_heads_may_select_different_candidates():
    """Sec. 5.3's lesson: the two losses are different problems.

    Constructed so one block predicts the outcome well and another predicts the
    treatment well, so the two heads should not agree. The assertion is only that
    the machinery *permits* disagreement and records both weight vectors.
    """
    rng = np.random.default_rng(11)
    n = 4000
    Zy = rng.normal(size=(n, 2))      # drives Y
    Zd = rng.normal(size=(n, 2))      # drives D
    e = 1.0 / (1.0 + np.exp(-2.0 * Zd[:, 0]))
    D = (rng.uniform(size=n) < e).astype(float)
    Y = 1.0 * D + 3.0 * Zy[:, 0] + 0.2 * Zd[:, 0] + rng.normal(size=n)

    model = PipelineDML(functional="ate", aggregation="star", seed=12)
    model.fit(Y, D, {"outcome_block": Zy, "treatment_block": Zd})
    w_g = model.outcome_aggregation_.weights
    w_a = model.riesz_aggregation_.weights
    assert w_g.shape == w_a.shape == (2,)
    assert abs(w_g.sum() - 1.0) < 1e-10 and abs(w_a.sum() - 1.0) < 1e-10
    assert int(np.argmax(w_g)) == 0, "outcome head should prefer the outcome block"


def test_pipeline_convex_aggregation_runs_and_reports_conditions():
    rng = np.random.default_rng(13)
    n = 1500
    Z = rng.normal(size=(n, 3))
    D = (rng.uniform(size=n) < 0.5).astype(float)
    Y = 1.0 * D + Z[:, 0] + rng.normal(size=n)
    model = PipelineDML(functional="ate", aggregation="convex", seed=14)
    res = model.fit(Y, D, {"a": Z[:, :2], "b": Z})
    assert np.isfinite(res.theta)
    report = model.pooling_report()
    assert report["weights"] == "convex"
    assert "reduced_condition" in report


def test_pipeline_rejects_an_empty_menu():
    rng = np.random.default_rng(15)
    Y = rng.normal(size=100)
    D = (rng.uniform(size=100) < 0.5).astype(float)
    with pytest.raises(ValueError, match="empty"):
        PipelineDML().fit(Y, D, {})


def test_pipeline_rejects_mismatched_rows():
    rng = np.random.default_rng(16)
    Y = rng.normal(size=100)
    D = (rng.uniform(size=100) < 0.5).astype(float)
    with pytest.raises(ValueError, match="rows but Y has"):
        PipelineDML().fit(Y, D, {"ok": rng.normal(size=(100, 2)),
                                 "bad": rng.normal(size=(99, 2))})


def test_aggregation_rejects_non_finite_candidates():
    y = np.arange(10.0)
    F = np.column_stack([y, np.full(10, np.nan)])
    with pytest.raises(ValueError, match="non-finite"):
        star_aggregate(F, y=y)


def test_aggregation_summary_frame():
    rng = np.random.default_rng(17)
    y = rng.normal(size=100)
    F = y[:, None] + rng.normal(size=(100, 3))
    frame = star_aggregate(F, y=y).summary_frame()
    assert list(frame.columns) == ["candidate", "weight", "objective"]
    assert len(frame) == 3
