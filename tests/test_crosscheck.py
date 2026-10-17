"""Cross-implementation checks against `DoubleML` and `statsmodels`.

There is no reference implementation of the target paper (searched: no code link
in arXiv:2610.01935v2, nothing on PyPI as of 17 October 2026), so the paper's own
estimators cannot be checked against someone else's code. What *can* be checked
is the special case where this package's score provably coincides with an
established one:

* for the binary ATE with an **analytic** (inverse-propensity) representer and a
  saturated outcome model, the Eq. (8) score is the AIPW/IRM score that
  ``DoubleML`` implements -- so with the same folds and the same learners the two
  must agree numerically;
* for the average derivative with a partially linear map, the estimator collapses
  to the OLS slope.

Those are the strongest compatibility proofs available, and their tolerances are
stated explicitly.

Author
------
Dr Merwan Roudane <merwanroudane920@gmail.com>
"""

from __future__ import annotations

import numpy as np
import pytest

from pragdml import (
    AdditiveFeatures,
    AverageDerivative,
    LinearOutcomeLearner,
    LinearRieszLearner,
    PragmaticDML,
    VaryingCoefficientFeatures,
    cross_fit,
    make_folds,
)
from pragdml.functionals import ATE
from pragdml.riesz import PlugInRieszLearner


def _dgp(n=4000, seed=0, tau=2.0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4))
    e = 1.0 / (1.0 + np.exp(-(0.8 * X[:, 0] - 0.5 * X[:, 1] + 0.3 * X[:, 2])))
    D = (rng.uniform(size=n) < e).astype(float)
    Y = tau * D + 1.4 * X[:, 0] - X[:, 1] + 0.6 * X[:, 3] + rng.normal(size=n)
    return Y, D, X


# --------------------------------------------------------------------------
# statsmodels / OLS
# --------------------------------------------------------------------------

def test_average_derivative_equals_ols_slope_exactly_in_sample():
    """With in-sample nuisances the estimator *is* the OLS slope, to 1e-9.

    Both nuisances are fitted on the whole sample, which removes the ``O(1/n)``
    cross-fitting difference and leaves an exact algebraic identity. This pins the
    score, the Riesz solve and the derivative functional simultaneously.
    """
    statsmodels = pytest.importorskip("statsmodels.api")

    rng = np.random.default_rng(1)
    n = 2000
    X = rng.normal(size=(n, 3))
    D = 0.6 * X[:, 0] + rng.normal(size=n)
    Y = 1.9 * D + X[:, 0] - 0.4 * X[:, 2] + rng.normal(size=n)

    func = AverageDerivative()
    g = LinearOutcomeLearner(AdditiveFeatures()).fit(Y, D, X)
    a = LinearRieszLearner(AdditiveFeatures()).fit(None, D, X, func)
    theta = float(np.mean(func.apply(g, D, X) + a(D, X) * (Y - g(D, X))))

    design = statsmodels.add_constant(np.column_stack([D, X]))
    ols = statsmodels.OLS(Y, design).fit()
    assert abs(theta - ols.params[1]) < 1e-9


def test_ate_equals_saturated_ols_contrast_in_sample():
    """A saturated linear outcome model reproduces the regression-adjustment ATE."""
    rng = np.random.default_rng(2)
    n = 3000
    X = rng.normal(size=(n, 3))
    D = (rng.uniform(size=n) < 0.5).astype(float)
    Y = 1.3 * D + X[:, 0] + 0.5 * D * X[:, 1] + rng.normal(size=n)

    func = ATE()
    fm = VaryingCoefficientFeatures()
    g = LinearOutcomeLearner(fm).fit(Y, D, X)
    a = LinearRieszLearner(fm).fit(None, D, X, func)
    theta = float(np.mean(func.apply(g, D, X) + a(D, X) * (Y - g(D, X))))

    # Regression adjustment: fit each arm and average the contrast.
    def arm(mask):
        Z = np.column_stack([np.ones(mask.sum()), X[mask]])
        coef = np.linalg.lstsq(Z, Y[mask], rcond=None)[0]
        return np.column_stack([np.ones(n), X]) @ coef

    g1 = arm(D == 1)
    g0 = arm(D == 0)
    assert abs(theta - float(np.mean(g1 - g0))) < 1e-7


# --------------------------------------------------------------------------
# DoubleML
# --------------------------------------------------------------------------

def test_matches_doubleml_irm_on_shared_folds():
    """Eq. (8) with an analytic representer is ``DoubleML``'s IRM score.

    The two packages are given the **same fold split**, the same saturated linear
    outcome model and the same logistic propensity, with ``DoubleML``'s IPW
    normalisation switched off so both compute the unnormalised AIPW score. The
    estimates should then agree to well within a hundredth of a standard error.

    Tolerance: ``0.02 * se``. Not exact-to-1e-12, because ``DoubleML`` fits the
    outcome model separately by treatment arm while ``pragdml`` fits one model
    saturated in ``D``; those coincide in population and differ only through
    finite-sample regularisation of a rank-deficient arm.
    """
    dml = pytest.importorskip("doubleml")
    from sklearn.linear_model import LinearRegression, LogisticRegression

    Y, D, X = _dgp(4000, seed=3, tau=2.0)
    n = Y.shape[0]
    n_folds = 5
    folds = make_folds(n, n_folds, rng=np.random.default_rng(123))
    all_idx = np.arange(n)
    smpls = [
        (np.setdiff1d(all_idx, test).astype(int), np.asarray(test, dtype=int))
        for test in folds
    ]

    mine = cross_fit(
        Y, D, X, ATE(),
        LinearOutcomeLearner(VaryingCoefficientFeatures()),
        PlugInRieszLearner(LogisticRegression(max_iter=2000), trim=0.0),
        n_folds=n_folds, rng=np.random.default_rng(123),
    )

    import pandas as pd

    frame = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])
    frame["y"] = Y
    frame["d"] = D
    data = dml.DoubleMLData(
        frame, y_col="y", d_cols="d",
        x_cols=[f"x{i}" for i in range(X.shape[1])],
    )
    irm = dml.DoubleMLIRM(
        data,
        ml_g=LinearRegression(),
        ml_m=LogisticRegression(max_iter=2000),
        n_folds=n_folds,
        normalize_ipw=False,
        draw_sample_splitting=False,
    )
    irm.set_sample_splitting([smpls])
    irm.fit()

    theirs = float(np.asarray(irm.coef).reshape(-1)[0])
    assert abs(mine.theta - theirs) < 0.02 * mine.se, (
        f"pragdml {mine.theta:.6f} vs DoubleML {theirs:.6f} "
        f"(se {mine.se:.6f})"
    )


def test_doubleml_and_pragdml_standard_errors_agree():
    """The variance estimators should match as closely as the point estimates."""
    dml = pytest.importorskip("doubleml")
    from sklearn.linear_model import LinearRegression, LogisticRegression
    import pandas as pd

    Y, D, X = _dgp(4000, seed=4, tau=1.5)
    n, n_folds = Y.shape[0], 5
    folds = make_folds(n, n_folds, rng=np.random.default_rng(7))
    all_idx = np.arange(n)
    smpls = [
        (np.setdiff1d(all_idx, t).astype(int), np.asarray(t, dtype=int))
        for t in folds
    ]

    mine = cross_fit(
        Y, D, X, ATE(),
        LinearOutcomeLearner(VaryingCoefficientFeatures()),
        PlugInRieszLearner(LogisticRegression(max_iter=2000), trim=0.0),
        n_folds=n_folds, rng=np.random.default_rng(7),
    )

    frame = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])
    frame["y"], frame["d"] = Y, D
    data = dml.DoubleMLData(
        frame, y_col="y", d_cols="d",
        x_cols=[f"x{i}" for i in range(X.shape[1])],
    )
    irm = dml.DoubleMLIRM(
        data, ml_g=LinearRegression(), ml_m=LogisticRegression(max_iter=2000),
        n_folds=n_folds, normalize_ipw=False, draw_sample_splitting=False,
    )
    irm.set_sample_splitting([smpls])
    irm.fit()

    theirs_se = float(np.asarray(irm.se).reshape(-1)[0])
    assert abs(mine.se - theirs_se) < 0.05 * mine.se


def test_learned_representer_agrees_with_doubleml_within_sampling_error():
    """The paper's *learned* representer should land where IRM does.

    This is the check that matters for the method as such: Riesz regression never
    inverts a propensity score, yet it must target the same parameter. Tolerance
    is three standard errors, because the two use genuinely different first
    stages.
    """
    dml = pytest.importorskip("doubleml")
    from sklearn.linear_model import LinearRegression, LogisticRegression
    import pandas as pd

    Y, D, X = _dgp(6000, seed=5, tau=2.0)
    mine = PragmaticDML(functional="ate", n_folds=5, seed=11).fit(Y, D, X)

    frame = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])
    frame["y"], frame["d"] = Y, D
    data = dml.DoubleMLData(
        frame, y_col="y", d_cols="d",
        x_cols=[f"x{i}" for i in range(X.shape[1])],
    )
    irm = dml.DoubleMLIRM(
        data, ml_g=LinearRegression(), ml_m=LogisticRegression(max_iter=2000),
        n_folds=5,
    )
    irm.fit()
    theirs = float(np.asarray(irm.coef).reshape(-1)[0])
    assert abs(mine.theta - theirs) < 3.0 * mine.se
    assert abs(mine.theta - 2.0) < 4.0 * mine.se
