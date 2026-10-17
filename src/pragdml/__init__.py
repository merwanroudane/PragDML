"""pragdml -- pragmatic DML with AI-learned representations.

A Python implementation of

    A. Aradillas Fernandez, V. Chernozhukov, C. Cinelli, S. Klaassen, W. Newey,
    M. Spindler, J. Teichert-Kluge and S. Vijaykumar,
    *Pragmatic DML with AI-Learned Representations*, arXiv:2610.01935 (2026),

together with the omitted-variable-bias theory it builds on,

    V. Chernozhukov, C. Cinelli, W. K. Newey, A. Sharma and V. Syrgkanis,
    *Long Story Short: Omitted Variable Bias in Causal Machine Learning*,
    Review of Economics and Statistics (2026), doi:10.1162/REST.a.1705.

What the package is for
-----------------------
When rich covariates (text, images, panels) are compressed into a learned
representation and then used as controls, the causal target moves. The paper
shows that it moves by exactly one quantity -- the product of the representation
error in the outcome regression and the representation error in the balancing
weight -- and that three things follow. ``pragdml`` implements all three:

1. **Valid Wald inference** for the representation-specific target, by cross-fitted
   DML with a *learned* Riesz representer (:class:`PragmaticDML`). Under target
   adaptivity the same interval covers the full-information target, and under
   adaptive efficiency it attains the efficiency bound.
2. **Fold-wise representation learning**, compatible with that inference because
   the conditions sit on the fitted functions, not on the tuning algorithm --
   including convex and star aggregation over a menu of pipelines
   (:mod:`pragdml.aggregation`).
3. **Sensitivity regions** with root-n inference for their endpoints when the
   representation error is not negligible (:mod:`pragdml.sensitivity`), plus
   robustness values, contour plots and covariate benchmarking.

Quick start
-----------
>>> import numpy as np
>>> from pragdml import PragmaticDML
>>> rng = np.random.default_rng(0)
>>> X = rng.normal(size=(500, 3))
>>> D = (rng.uniform(size=500) < 0.5).astype(float)
>>> Y = 2.0 * D + X[:, 0] + rng.normal(size=500)
>>> res = PragmaticDML(functional="ate", seed=0).fit(Y, D, X)
>>> bool(res.conf_int()[0] < 2.0 < res.conf_int()[1])
True

Author
------
Dr Merwan Roudane <merwanroudane920@gmail.com>
https://github.com/merwanroudane
"""

from __future__ import annotations

__version__ = "0.1.1"
__author__ = "Dr Merwan Roudane"
__email__ = "merwanroudane920@gmail.com"

from ._colors import (
    PARULA_STOPS,
    bluered_colors,
    list_colorscales,
    matlab_jet_colors,
    parula_cmap,
    parula_colors,
    resolve_colorscale,
    sinha_colors,
    turbo_colors,
)
from ._core import (
    CrossFitOutput,
    combine_repetitions,
    cross_fit,
    debiased_score,
    make_folds,
)
from .diagnostics import (
    OmittedInformation,
    adaptive_efficiency,
    critical_radius_l1_hull,
    excess_risk_identity,
    mixed_bias_drift,
    omitted_information_gap,
    pipeline_target_gap,
    pooling_conditions,
    product_rate,
    target_adaptivity,
)
from .features import (
    AdditiveFeatures,
    FeatureMap,
    PolynomialTreatmentFeatures,
    VaryingCoefficientFeatures,
)
from .functionals import (
    ATE,
    ATT,
    AverageDerivative,
    AveragePotentialOutcome,
    Functional,
    PolicyEffect,
    WeightedATE,
    get_functional,
)
from .learners import (
    LinearOutcomeLearner,
    OutcomeLearner,
    SklearnOutcomeLearner,
    square_risk,
)
from .riesz import (
    LinearRieszLearner,
    PartitionRieszLearner,
    PlugInRieszLearner,
    RieszLearner,
    representer_second_moment,
    riesz_loss,
)

from .aggregation import (  # noqa: E402  (after the primitives it builds on)
    AggregationResult,
    convex_aggregate,
    pooled_convex_aggregate,
    pooled_star_aggregate,
    star_aggregate,
    star_oracle_bound,
)
from .dml import (  # noqa: E402
    InformationSetComparison,
    PipelineDML,
    PragmaticDML,
    compare_information_sets,
    pragmatic_dml,
)
from .sensitivity import (  # noqa: E402
    BenchmarkResult,
    SensitivityAnalysis,
    SensitivityResult,
    benchmark_covariate,
    robustness_value,
)

__all__ = [
    "__version__",
    "__author__",
    "__email__",
    # estimators
    "PragmaticDML",
    "PipelineDML",
    "pragmatic_dml",
    "compare_information_sets",
    "InformationSetComparison",
    # functionals
    "Functional",
    "ATE",
    "ATT",
    "AveragePotentialOutcome",
    "WeightedATE",
    "AverageDerivative",
    "PolicyEffect",
    "get_functional",
    # feature maps
    "FeatureMap",
    "VaryingCoefficientFeatures",
    "PolynomialTreatmentFeatures",
    "AdditiveFeatures",
    # learners
    "OutcomeLearner",
    "LinearOutcomeLearner",
    "SklearnOutcomeLearner",
    "square_risk",
    "RieszLearner",
    "LinearRieszLearner",
    "PlugInRieszLearner",
    "PartitionRieszLearner",
    "riesz_loss",
    "representer_second_moment",
    # core
    "cross_fit",
    "CrossFitOutput",
    "debiased_score",
    "make_folds",
    "combine_repetitions",
    # aggregation
    "star_aggregate",
    "convex_aggregate",
    "pooled_star_aggregate",
    "pooled_convex_aggregate",
    "star_oracle_bound",
    "AggregationResult",
    # sensitivity
    "SensitivityAnalysis",
    "SensitivityResult",
    "robustness_value",
    "benchmark_covariate",
    "BenchmarkResult",
    # diagnostics
    "omitted_information_gap",
    "OmittedInformation",
    "mixed_bias_drift",
    "pipeline_target_gap",
    "excess_risk_identity",
    "target_adaptivity",
    "adaptive_efficiency",
    "product_rate",
    "pooling_conditions",
    "critical_radius_l1_hull",
    # colours
    "parula_colors",
    "parula_cmap",
    "matlab_jet_colors",
    "turbo_colors",
    "bluered_colors",
    "sinha_colors",
    "resolve_colorscale",
    "list_colorscales",
    "PARULA_STOPS",
]
