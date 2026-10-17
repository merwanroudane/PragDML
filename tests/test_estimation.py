"""Estimation: known-truth recovery, rates, reproducibility, input contracts.

Author
------
Dr Merwan Roudane <merwanroudane920@gmail.com>
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from pragdml import (
    ATE,
    AdditiveFeatures,
    AverageDerivative,
    LinearOutcomeLearner,
    LinearRieszLearner,
    PragmaticDML,
    SklearnOutcomeLearner,
    VaryingCoefficientFeatures,
    compare_information_sets,
    cross_fit,
    make_folds,
    pragmatic_dml,
)
from pragdml.riesz import PartitionRieszLearner, PlugInRieszLearner


def binary_dgp(n, seed, tau=2.0):
    """Confounded binary treatment with a known constant ATE."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    e = 1.0 / (1.0 + np.exp(-(0.9 * X[:, 0] - 0.4 * X[:, 1])))
    D = (rng.uniform(size=n) < e).astype(float)
    Y = tau * D + 1.5 * X[:, 0] - X[:, 1] + 0.5 * X[:, 2] + rng.normal(size=n)
    return Y, D, X


def continuous_dgp(n, seed, beta=1.7):
    """Confounded continuous treatment with a known average derivative."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    D = 0.7 * X[:, 0] - 0.3 * X[:, 1] + rng.normal(size=n)
    Y = beta * D + X[:, 0] + 0.5 * X[:, 2] + rng.normal(size=n)
    return Y, D, X


# --------------------------------------------------------------------------
# Known-truth recovery and the rate
# --------------------------------------------------------------------------

def test_recovers_the_ate():
    """The confidence interval covers the true ATE."""
    Y, D, X = binary_dgp(4000, seed=0, tau=2.0)
    res = PragmaticDML(functional="ate", seed=1).fit(Y, D, X)
    lo, hi = res.conf_int()
    assert lo < 2.0 < hi
    assert abs(res.theta - 2.0) < 4 * res.se


def test_recovers_the_average_derivative():
    """Same for a continuous treatment and Eq. (67)."""
    Y, D, X = continuous_dgp(4000, seed=2, beta=1.7)
    res = PragmaticDML(
        functional="average_derivative", seed=3,
        outcome_learner=LinearOutcomeLearner(AdditiveFeatures()),
        riesz_learner=LinearRieszLearner(AdditiveFeatures()),
    ).fit(Y, D, X)
    lo, hi = res.conf_int()
    assert lo < 1.7 < hi


def test_error_shrinks_at_the_root_n_rate():
    """Quadrupling ``n`` should roughly halve the error, averaged over seeds.

    Asserted loosely (the constant is not the point) but it is the check that
    distinguishes a consistent estimator from one that merely looks plausible at
    a single sample size.
    """
    def mean_abs_error(n):
        errs = []
        for seed in range(12):
            Y, D, X = binary_dgp(n, seed=100 + seed, tau=2.0)
            res = PragmaticDML(functional="ate", seed=seed).fit(Y, D, X)
            errs.append(abs(res.theta - 2.0))
        return float(np.mean(errs))

    small, large = mean_abs_error(500), mean_abs_error(2000)
    assert large < small
    ratio = small / large
    assert 1.3 < ratio < 3.2, f"error ratio {ratio:.2f} is not root-n-like"


def test_standard_error_shrinks_at_the_root_n_rate():
    """``se`` must fall like ``1/sqrt(n)``."""
    ses = []
    for n in (1000, 4000):
        Y, D, X = binary_dgp(n, seed=7, tau=2.0)
        ses.append(PragmaticDML(functional="ate", seed=8).fit(Y, D, X).se)
    assert 1.6 < ses[0] / ses[1] < 2.5


def test_naive_regression_is_biased_but_dml_is_not():
    """The confounding is real: an unadjusted comparison misses the truth.

    Without this, a recovery test proves nothing -- it could pass on data where
    adjustment was unnecessary.
    """
    Y, D, X = binary_dgp(4000, seed=9, tau=2.0)
    naive = float(Y[D == 1].mean() - Y[D == 0].mean())
    res = PragmaticDML(functional="ate", seed=10).fit(Y, D, X)
    assert abs(naive - 2.0) > 0.3, "the DGP is not actually confounded"
    assert abs(res.theta - 2.0) < abs(naive - 2.0) / 3.0


# --------------------------------------------------------------------------
# Analytical special cases
# --------------------------------------------------------------------------

def test_average_derivative_collapses_to_ols_in_sample():
    """With in-sample nuisances and a partially linear map, DML *is* OLS.

    Cross-fitting makes the two differ by ``O(1/n)``; fitting both nuisances on
    the whole sample removes that and the agreement becomes exact, which pins the
    score algebra rather than its asymptotics.
    """
    Y, D, X = continuous_dgp(2000, seed=11, beta=1.7)
    func = AverageDerivative()
    g = LinearOutcomeLearner(AdditiveFeatures()).fit(Y, D, X)
    a = LinearRieszLearner(AdditiveFeatures()).fit(None, D, X, func)
    psi = func.apply(g, D, X) + a(D, X) * (Y - g(D, X))
    theta = float(np.mean(psi))

    design = np.column_stack([np.ones(len(Y)), D, X])
    ols = np.linalg.lstsq(design, Y, rcond=None)[0][1]
    assert abs(theta - ols) < 1e-8


def test_crossfit_average_derivative_is_close_to_ols():
    """The cross-fitted version agrees with OLS up to the ``O(1/n)`` split cost."""
    Y, D, X = continuous_dgp(4000, seed=12, beta=1.7)
    res = PragmaticDML(
        functional="average_derivative", seed=13,
        outcome_learner=LinearOutcomeLearner(AdditiveFeatures()),
        riesz_learner=LinearRieszLearner(AdditiveFeatures()),
    ).fit(Y, D, X)
    design = np.column_stack([np.ones(len(Y)), D, X])
    ols = np.linalg.lstsq(design, Y, rcond=None)[0][1]
    assert abs(res.theta - ols) < 0.02


def test_learned_and_plugin_representers_agree_on_the_ate():
    """Riesz regression and inverse-propensity weighting target the same thing.

    They are entirely different computations -- a linear solve versus a fitted
    probability and a division -- so agreement is a genuine cross-check of the
    Riesz machinery.
    """
    from sklearn.linear_model import LogisticRegression

    Y, D, X = binary_dgp(4000, seed=14, tau=2.0)
    learned = PragmaticDML(functional="ate", seed=15).fit(Y, D, X)
    plugin = PragmaticDML(
        functional="ate", seed=15,
        riesz_learner=PlugInRieszLearner(LogisticRegression(max_iter=1000)),
    ).fit(Y, D, X)
    assert abs(learned.theta - plugin.theta) < 4 * max(learned.se, plugin.se)


# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------

def test_same_seed_gives_identical_results():
    """Bit-identical, not merely close."""
    Y, D, X = binary_dgp(800, seed=16)
    a = PragmaticDML(functional="ate", seed=42).fit(Y, D, X)
    b = PragmaticDML(functional="ate", seed=42).fit(Y, D, X)
    assert a.theta == b.theta and a.se == b.se


def test_different_seeds_give_different_splits():
    """Otherwise the seed is not threaded through at all."""
    Y, D, X = binary_dgp(800, seed=17)
    a = PragmaticDML(functional="ate", seed=1).fit(Y, D, X)
    b = PragmaticDML(functional="ate", seed=2).fit(Y, D, X)
    assert a.theta != b.theta


def test_repeated_crossfitting_is_reproducible_and_stabilises():
    """``n_rep`` is seeded, and median aggregation reduces split sensitivity."""
    Y, D, X = binary_dgp(1500, seed=18, tau=2.0)
    a = PragmaticDML(functional="ate", n_rep=5, seed=3).fit(Y, D, X)
    b = PragmaticDML(functional="ate", n_rep=5, seed=3).fit(Y, D, X)
    assert a.theta == b.theta
    assert len(a.extra) >= 1

    singles = [
        PragmaticDML(functional="ate", seed=s).fit(Y, D, X).theta for s in range(5)
    ]
    assert min(singles) <= a.theta <= max(singles)


def test_make_folds_partitions_exactly_once():
    """Folds are disjoint and cover everything -- the premise of cross-fitting."""
    for n, k in ((100, 5), (97, 4), (10, 2)):
        folds = make_folds(n, k, rng=np.random.default_rng(0))
        assert len(folds) == k
        joined = np.concatenate(folds)
        assert np.array_equal(np.sort(joined), np.arange(n))


# --------------------------------------------------------------------------
# Clustering (extension E1, required by Sec. 5.1)
# --------------------------------------------------------------------------

def test_clusters_are_never_split_across_folds():
    """Sec. 5.1's requirement, enforced."""
    cluster = np.repeat(np.arange(60), 5)
    folds = make_folds(300, 5, cluster=cluster, rng=np.random.default_rng(1))
    seen: set = set()
    for idx in folds:
        here = set(cluster[idx].tolist())
        assert not (here & seen), "a cluster appears in two folds"
        seen |= here


def test_cluster_robust_se_exceeds_iid_se_under_within_cluster_dependence():
    """With a shared cluster shock, ignoring clustering understates the SE."""
    rng = np.random.default_rng(19)
    n_cl, per = 150, 8
    n = n_cl * per
    cluster = np.repeat(np.arange(n_cl), per)
    shock = rng.normal(size=n_cl) * 2.0
    X = rng.normal(size=(n, 2))
    D = (rng.uniform(size=n) < 0.5).astype(float)
    Y = 1.0 * D + X[:, 0] + shock[cluster] + rng.normal(size=n) * 0.3

    iid = PragmaticDML(functional="ate", seed=20).fit(Y, D, X)
    clu = PragmaticDML(functional="ate", seed=20).fit(Y, D, X, cluster=cluster)
    assert clu.se > iid.se
    assert clu.crossfit_.n_clusters == n_cl


def test_clustered_repeated_crossfitting_keeps_the_cluster_adjustment():
    """``n_rep > 1`` must not quietly revert to iid standard errors."""
    rng = np.random.default_rng(21)
    n_cl, per = 120, 6
    n = n_cl * per
    cluster = np.repeat(np.arange(n_cl), per)
    shock = rng.normal(size=n_cl) * 2.0
    X = rng.normal(size=(n, 2))
    D = (rng.uniform(size=n) < 0.5).astype(float)
    Y = D + X[:, 0] + shock[cluster] + 0.3 * rng.normal(size=n)

    iid = PragmaticDML(functional="ate", n_rep=3, seed=22).fit(Y, D, X)
    clu = PragmaticDML(
        functional="ate", n_rep=3, seed=22
    ).fit(Y, D, X, cluster=cluster)
    assert clu.se > iid.se


# --------------------------------------------------------------------------
# Input contracts
# --------------------------------------------------------------------------

def test_dataframe_and_ndarray_give_identical_numbers():
    """Variable names must not change the arithmetic."""
    Y, D, X = binary_dgp(600, seed=23)
    frame = pd.DataFrame(X, columns=["a", "b", "c"])
    arr = PragmaticDML(functional="ate", seed=5).fit(Y, D, X)
    dfr = PragmaticDML(functional="ate", seed=5).fit(
        pd.Series(Y, name="y"), pd.Series(D, name="d"), frame
    )
    assert arr.theta == dfr.theta and arr.se == dfr.se
    assert "Controls" in dfr.extra


def test_one_dimensional_x_is_accepted():
    """A single control may be passed as a 1-d array."""
    rng = np.random.default_rng(24)
    x = rng.normal(size=600)
    D = (rng.uniform(size=600) < 0.5).astype(float)
    Y = D + x + rng.normal(size=600)
    res = PragmaticDML(functional="ate", seed=6).fit(Y, D, x)
    assert np.isfinite(res.theta)


def test_nan_input_raises_rather_than_imputing():
    """Silent imputation would hide a data problem; the message says so."""
    Y, D, X = binary_dgp(300, seed=25)
    X = X.copy()
    X[3, 1] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        PragmaticDML(functional="ate", seed=7).fit(Y, D, X)


def test_mismatched_lengths_raise():
    Y, D, X = binary_dgp(300, seed=26)
    with pytest.raises(ValueError, match="agree in length"):
        PragmaticDML(functional="ate").fit(Y[:-1], D, X)


def test_too_few_observations_for_the_fold_count_raises():
    Y, D, X = binary_dgp(8, seed=27)
    with pytest.raises(ValueError, match="too small"):
        PragmaticDML(functional="ate", n_folds=5).fit(Y, D, X)


def test_continuous_treatment_rejected_for_the_ate():
    """The ATE needs a binary treatment, and the message points to the fix."""
    Y, D, X = continuous_dgp(300, seed=28)
    with pytest.raises(ValueError, match="binary treatment"):
        PragmaticDML(functional="ate").fit(Y, D, X)


def test_single_arm_treatment_rejected():
    rng = np.random.default_rng(29)
    X = rng.normal(size=(200, 2))
    D = np.ones(200)
    Y = rng.normal(size=200)
    with pytest.raises(ValueError, match="both treatment arms"):
        PragmaticDML(functional="ate").fit(Y, D, X)


def test_tree_learner_warns_for_the_average_derivative():
    """A piecewise-constant learner has zero derivative; warn, do not pretend."""
    from sklearn.ensemble import RandomForestRegressor

    Y, D, X = continuous_dgp(400, seed=30)
    learner = SklearnOutcomeLearner(
        RandomForestRegressor(n_estimators=10, random_state=0)
    )
    with pytest.warns(RuntimeWarning, match="piecewise constant"):
        learner.fit(Y, D, X, AverageDerivative())


def test_n_folds_below_two_rejected():
    with pytest.raises(ValueError, match="n_folds must be >= 2"):
        make_folds(100, 1)


# --------------------------------------------------------------------------
# Other learners and the comparison helper
# --------------------------------------------------------------------------

@pytest.mark.parametrize("target", ["outcome", "treatment"])
def test_partition_riesz_learner_runs_and_recovers_the_ate(target):
    """The nonlinear menu member must work end to end, under both partitions."""
    Y, D, X = binary_dgp(2000, seed=33, tau=2.0)
    res = PragmaticDML(
        functional="ate", seed=34,
        riesz_learner=PartitionRieszLearner(target=target),
    ).fit(Y, D, X)
    assert abs(res.theta - 2.0) < 6 * res.se


@pytest.mark.parametrize("target", ["outcome", "treatment"])
def test_partition_riesz_learner_keeps_nu2_positive(target):
    """Regression guard: this learner used to return a negative ``E[alpha^2]``.

    Two defects caused it, both now fixed and both worth keeping tested:

    1. the partition intersected the leaves of three forest trees, producing
       hundreds of cells with a median of two observations each;
    2. a tree grown on the treatment isolates cells with near-zero control
       counts, where the within-cell inverse-propensity weight diverges.

    ``nu_hat^2`` estimates ``E[alpha_0^2] > 0``, so a negative value is proof the
    Riesz fit has failed. :func:`~pragdml._core.cross_fit` now warns when it
    happens; this test asserts it does not.
    """
    Y, D, X = binary_dgp(2000, seed=33, tau=2.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        res = PragmaticDML(
            functional="ate", seed=34,
            riesz_learner=PartitionRieszLearner(target=target),
        ).fit(Y, D, X)
    assert res.crossfit_.nu2_representer > 0.0


def test_degenerate_representer_is_warned_about():
    """A deliberately over-wide basis must warn rather than report quietly."""
    Y, D, X = binary_dgp(400, seed=50, tau=2.0)
    from sklearn.preprocessing import SplineTransformer

    wide = VaryingCoefficientFeatures(
        basis=SplineTransformer(n_knots=25, degree=3)
    )
    with pytest.warns(RuntimeWarning, match="nu_hat"):
        PragmaticDML(
            functional="ate", seed=51,
            outcome_learner=LinearOutcomeLearner(wide),
            riesz_learner=LinearRieszLearner(wide),
        ).fit(Y, D, X)


def test_sklearn_outcome_learner_runs_for_the_ate():
    from sklearn.ensemble import RandomForestRegressor

    Y, D, X = binary_dgp(1500, seed=35, tau=2.0)
    res = PragmaticDML(
        functional="ate", seed=36,
        outcome_learner=SklearnOutcomeLearner(
            RandomForestRegressor(n_estimators=60, min_samples_leaf=10,
                                  random_state=0)
        ),
    ).fit(Y, D, X)
    assert abs(res.theta - 2.0) < 6 * res.se


def test_compare_information_sets_shapes_and_aggregate():
    """The Table 1 helper returns one row per set plus an aggregate."""
    Y, D, X = binary_dgp(1200, seed=37, tau=2.0)
    cmp = compare_information_sets(
        Y, D,
        {"first": X[:, :1], "second": X[:, 1:2], "all": X},
        functional="ate", seed=38,
    )
    assert len(cmp.results) == 3
    assert cmp.aggregate is not None
    frame = cmp.to_frame()
    assert len(frame) == 4
    assert np.isfinite(cmp.spread)
    assert isinstance(cmp.summary(), str)


def test_functional_wrapper_matches_the_class():
    Y, D, X = binary_dgp(600, seed=39)
    a = pragmatic_dml(Y, D, X, functional="ate", seed=9)
    b = PragmaticDML(functional="ate", seed=9).fit(Y, D, X)
    assert a.theta == b.theta


def test_bootstrap_interval_is_close_to_the_wald_interval():
    """Extension E3 should agree with the normal interval in a well-behaved case."""
    Y, D, X = binary_dgp(2000, seed=40, tau=2.0)
    res = PragmaticDML(functional="ate", seed=41).fit(Y, D, X)
    wald = res.conf_int()
    boot = res.bootstrap_conf_int(n_boot=499, seed=1)
    assert abs(boot[0] - wald[0]) < 0.6 * res.se
    assert abs(boot[1] - wald[1]) < 0.6 * res.se


def test_summary_and_exports_render():
    """Every rendering path must produce something, for every target."""
    Y, D, X = binary_dgp(600, seed=43)
    res = PragmaticDML(functional="ate", seed=44).fit(Y, D, X)
    text = res.summary()
    assert "Pragmatic DML" in text and "Eq. 8" in text
    assert r"\begin{table}" in res.to_latex()
    assert "|" in res.to_markdown()
    assert len(res.to_frame()) == 1


def test_att_differs_from_ate_under_heterogeneity():
    """The ratio target is not silently the same number as the ATE."""
    rng = np.random.default_rng(45)
    n = 3000
    X = rng.normal(size=(n, 2))
    e = 1.0 / (1.0 + np.exp(-1.5 * X[:, 0]))
    D = (rng.uniform(size=n) < e).astype(float)
    Y = (1.0 + 2.0 * X[:, 0]) * D + X[:, 0] + rng.normal(size=n)
    ate = PragmaticDML(functional="ate", seed=46).fit(Y, D, X)
    att = PragmaticDML(functional="att", seed=46).fit(Y, D, X)
    assert att.theta > ate.theta        # treated have higher X0, hence larger effect
    assert np.isfinite(att.se) and att.se > 0
