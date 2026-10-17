"""Exact algebra: the paper's identities must hold to machine precision.

These are the tests that matter most, because the identities are not
approximations -- if one of them is off by more than floating-point noise, the
implementation is wrong, full stop.

Author
------
Dr Merwan Roudane <merwanroudane920@gmail.com>
"""

from __future__ import annotations

import numpy as np
import pytest

from pragdml import (
    ATE,
    AverageDerivative,
    LinearOutcomeLearner,
    LinearRieszLearner,
    VaryingCoefficientFeatures,
    cross_fit,
    excess_risk_identity,
    omitted_information_gap,
    riesz_loss,
    representer_second_moment,
    square_risk,
)
from pragdml.functionals import ATT, AveragePotentialOutcome, PolicyEffect, WeightedATE
from pragdml.riesz import FeatureEvaluable


def _data(n=600, seed=0, binary=True):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    if binary:
        e = 1.0 / (1.0 + np.exp(-0.8 * X[:, 0]))
        D = (rng.uniform(size=n) < e).astype(float)
    else:
        D = 0.5 * X[:, 0] + rng.normal(size=n)
    Y = 1.5 * D + X[:, 0] - 0.5 * X[:, 1] + rng.normal(size=n)
    return Y, D, X


# --------------------------------------------------------------------------
# Linearity of m in its second argument -- the premise of the whole package
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "functional",
    [
        ATE(),
        ATT(),
        AveragePotentialOutcome(1.0),
        WeightedATE(lambda X: X[:, 0] ** 2),
        AverageDerivative(),
        PolicyEffect(lambda D, X: D + 0.5),
    ],
)
def test_m_is_linear_in_h(functional):
    """``m(O, a h1 + b h2) == a m(O, h1) + b m(O, h2)`` exactly.

    Every downstream result -- the Riesz normal equations, the closed-form
    segment weights of Eqs. (36)-(37), the aggregation of ``m`` values by the same
    weights as the predictions -- relies on this and nothing else.
    """
    Y, D, X = _data(200, seed=1)
    a, b = 2.5, -1.25

    class H:
        def __init__(self, c0, c1):
            self.c0, self.c1 = c0, c1

        def __call__(self, d, X):
            return self.c0 * d + self.c1 * X[:, 0] + self.c0 * self.c1

        def deriv_d(self, d, X):
            return np.full(d.shape[0], self.c0)

    h1, h2 = H(1.0, 2.0), H(-0.5, 3.0)

    class Combo:
        def __call__(self, d, X):
            return a * h1(d, X) + b * h2(d, X)

        def deriv_d(self, d, X):
            return a * h1.deriv_d(d, X) + b * h2.deriv_d(d, X)

    lhs = functional.apply(Combo(), D, X)
    rhs = a * functional.apply(h1, D, X) + b * functional.apply(h2, D, X)
    np.testing.assert_allclose(lhs, rhs, rtol=1e-10, atol=1e-10)


def test_average_derivative_exact_vs_numerical():
    """The exact ``deriv_d`` path and the central difference must agree.

    They are different code paths -- one uses the feature map's analytic
    derivative, the other differences the learner -- so agreement is a real check.
    """
    Y, D, X = _data(400, seed=2, binary=False)
    learner = LinearOutcomeLearner().fit(Y, D, X)
    exact = AverageDerivative().apply(learner, D, X)

    class NoDeriv:
        def __call__(self, d, XX):
            return learner(d, XX)

    numeric = AverageDerivative().apply(NoDeriv(), D, X)
    np.testing.assert_allclose(exact, numeric, rtol=1e-5, atol=1e-6)


# --------------------------------------------------------------------------
# Riesz normal equations and the second-moment identity
# --------------------------------------------------------------------------

def test_riesz_normal_equations_hold():
    """The fitted representer satisfies ``P_n[phi alpha] = P_n[m(O, phi)]``.

    This is the empirical version of the Riesz representation of Eqs. (4)-(5) and
    is the defining property of the estimator, so it should hold to solver
    precision.
    """
    Y, D, X = _data(500, seed=3)
    fm = VaryingCoefficientFeatures()
    learner = LinearRieszLearner(fm).fit(None, D, X, ATE())
    phi = learner.features_.transform(D, X)
    alpha = learner(D, X)
    lhs = phi.T @ alpha / phi.shape[0]
    rhs = np.asarray(
        ATE().apply(FeatureEvaluable(learner.features_), D, X)
    ).mean(axis=0)
    np.testing.assert_allclose(lhs, rhs, rtol=1e-8, atol=1e-8)


def test_second_moment_identity_at_the_minimiser():
    """``inf L = -E[alpha^2]``, i.e. ``P_n[2m(a)-a^2] == P_n[a^2]`` at the optimum.

    This identity (Eq. 26) is why Eq. (56) can estimate ``nu_0^2`` by the
    orthogonal form, and why :func:`representer_second_moment` is preferred over
    the naive ``P_n[alpha^2]``.
    """
    Y, D, X = _data(800, seed=4)
    learner = LinearRieszLearner().fit(None, D, X, ATE())
    naive = float(np.mean(learner(D, X) ** 2))
    orthogonal = representer_second_moment(learner, ATE(), D, X)
    assert abs(naive - orthogonal) < 1e-8 * max(1.0, abs(naive))
    assert abs(riesz_loss(learner, ATE(), D, X) + naive) < 1e-8 * max(1.0, naive)


# --------------------------------------------------------------------------
# Proposition 1 / Eqs. (6)-(7)
# --------------------------------------------------------------------------

def test_cauchy_schwarz_bound_dominates_the_gap():
    """Eq. (7): the bound is never smaller than the exact gap of Eq. (6)."""
    rng = np.random.default_rng(5)
    for _ in range(50):
        n = 200
        res = omitted_information_gap(
            rng.normal(size=n), rng.normal(size=n),
            rng.normal(size=n), rng.normal(size=n),
        )
        assert res.bound + 1e-12 >= abs(res.gap)
        assert -1.0 - 1e-12 <= res.rho <= 1.0 + 1e-12


def test_gap_is_zero_when_either_error_vanishes():
    """Proposition 1's headline: preserving *either* object preserves the target."""
    rng = np.random.default_rng(6)
    n = 300
    g = rng.normal(size=n)
    a = rng.normal(size=n)
    only_g = omitted_information_gap(g, np.zeros(n), a, a)
    only_a = omitted_information_gap(g, g, a, np.zeros(n))
    assert abs(only_g.gap) < 1e-14
    assert abs(only_a.gap) < 1e-14


def test_bound_is_tight_when_errors_are_proportional():
    """Cauchy-Schwarz binds exactly when the two error vectors are collinear."""
    rng = np.random.default_rng(7)
    g_err = rng.normal(size=400)
    res = omitted_information_gap(
        g_err, np.zeros(400), 2.5 * g_err, np.zeros(400)
    )
    assert res.is_tight
    assert abs(res.bound - abs(res.gap)) < 1e-10


# --------------------------------------------------------------------------
# Lemma 2 / Eqs. (28)-(29): excess risk == squared L2 error, on H
# --------------------------------------------------------------------------

def test_excess_square_risk_equals_squared_l2_error():
    """Eq. (28), verified against the exact projection in a finite-dim ``H``."""
    Y, D, X = _data(1000, seed=8, binary=False)
    fm = VaryingCoefficientFeatures()
    g0 = LinearOutcomeLearner(fm).fit(Y, D, X)

    rng = np.random.default_rng(9)
    perturbed = LinearOutcomeLearner(fm).fit(Y, D, X)
    perturbed.coef_ = g0.coef_ + 0.1 * rng.normal(size=g0.coef_.shape)

    risk = square_risk(perturbed, Y, D, X)
    risk_min = square_risk(g0, Y, D, X)
    err = float(np.sqrt(np.mean((perturbed(D, X) - g0(D, X)) ** 2)))
    out = excess_risk_identity(risk, risk_min, err, tol=1e-8)
    assert out["holds"], out


def test_excess_riesz_loss_equals_squared_l2_error():
    """Eq. (29), the same statement for the Riesz loss."""
    Y, D, X = _data(1000, seed=10)
    fm = VaryingCoefficientFeatures()
    a0 = LinearRieszLearner(fm).fit(None, D, X, ATE())

    rng = np.random.default_rng(11)
    perturbed = LinearRieszLearner(fm).fit(None, D, X, ATE())
    perturbed.coef_ = a0.coef_ + 0.1 * rng.normal(size=a0.coef_.shape)

    loss = riesz_loss(perturbed, ATE(), D, X)
    loss_min = riesz_loss(a0, ATE(), D, X)
    err = float(np.sqrt(np.mean((perturbed(D, X) - a0(D, X)) ** 2)))
    out = excess_risk_identity(loss, loss_min, err, tol=1e-8)
    assert out["holds"], out


# --------------------------------------------------------------------------
# Eq. (9): the orthogonal score's drift is exactly a mixed-bias term
# --------------------------------------------------------------------------

def test_score_drift_equals_minus_mixed_bias():
    """Eq. (9): ``P_n psi(g, a) - theta_0 == -P_n[(g-g_0)(a-a_0)]``.

    Verified in-sample against the exact projections, which is the only setting
    where the identity is exact rather than asymptotic.
    """
    Y, D, X = _data(1200, seed=12)
    func = ATE()
    fm = VaryingCoefficientFeatures()
    g0 = LinearOutcomeLearner(fm).fit(Y, D, X)
    a0 = LinearRieszLearner(fm).fit(None, D, X, func)
    theta0 = float(np.mean(func.apply(g0, D, X)))

    rng = np.random.default_rng(13)
    g = LinearOutcomeLearner(fm).fit(Y, D, X)
    g.coef_ = g0.coef_ + 0.3 * rng.normal(size=g0.coef_.shape)
    a = LinearRieszLearner(fm).fit(None, D, X, func)
    a.coef_ = a0.coef_ + 0.3 * rng.normal(size=a0.coef_.shape)

    psi = func.apply(g, D, X) + a(D, X) * (Y - g(D, X))
    drift = float(np.mean(psi)) - theta0
    mixed = -float(np.mean((g(D, X) - g0(D, X)) * (a(D, X) - a0(D, X))))
    assert abs(drift - mixed) < 1e-8 * max(1.0, abs(mixed))


# --------------------------------------------------------------------------
# Eqs. (55)-(57) consistency with the stored arrays
# --------------------------------------------------------------------------

def test_sensitivity_components_match_their_definitions():
    """``S_hat_0 == (sigma_hat_0^2 nu_hat_0^2)^{1/2}`` and both match Eqs. (55)-(56)."""
    Y, D, X = _data(600, seed=14)
    out = cross_fit(
        Y, D, X, ATE(), LinearOutcomeLearner(), LinearRieszLearner(),
        n_folds=5, rng=np.random.default_rng(15),
    )
    assert abs(out.sigma2_outcome - np.mean((out.y - out.g_hat) ** 2)) < 1e-14
    assert abs(
        out.nu2_representer - np.mean(2 * out.m_alpha - out.alpha_hat**2)
    ) < 1e-14
    assert abs(out.S - np.sqrt(out.sigma2_outcome * out.nu2_representer)) < 1e-12
    assert abs(out.riesz_loss + out.nu2_representer) < 1e-14
