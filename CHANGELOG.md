# Changelog

All notable changes to `pragdml` are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[semantic versioning](https://semver.org/). The `0.x` series is deliberate: the
paper this implements is a 2026 preprint and no reference implementation exists
to pin behaviour against, so the API may still move.

## [0.1.1] - 2026-10-17

### Fixed

- **README links now render on PyPI.** The two figures and four document links
  used repository-relative paths, which have no meaning on the PyPI project page:
  PyPI dropped both images entirely and the `LICENSE`, `CHANGELOG.md` and
  `docs/methods.md` links pointed nowhere. All six are now absolute
  (`raw.githubusercontent.com` for images, `github.com/.../blob/main` for
  documents), so the project page renders the same content as the repository.

No functional changes; the library code is identical to 0.1.0.

## [0.1.0] - 2026-10-17

First release.

### Added

**Targets and estimators**

- Riesz-linear functionals: `ATE`, `ATT` (a ratio target carrying a delta-method
  correction), `AveragePotentialOutcome`, `WeightedATE`, `AverageDerivative`
  (Eq. 67) and `PolicyEffect`.
- `PragmaticDML` — cross-fitted DML for one information set (Eqs. 8–12,
  Theorem 1), with repeated cross-fitting and median aggregation.
- `PipelineDML` — a menu of pipelines aggregated fold-wise on pooled
  out-of-fold predictions (Definitions 5–6, Theorems 4–5).
- `compare_information_sets` — the paper's Table 1 layout.

**Nuisance learning**

- Riesz regression in closed form over a feature map (`LinearRieszLearner`), the
  analytic inverse-propensity representer (`PlugInRieszLearner`), and a
  tree-partition representer (`PartitionRieszLearner`).
- Feature maps spanning the information space `H`: `VaryingCoefficientFeatures`
  (the paper's Eqs. 68–69), `PolynomialTreatmentFeatures`, `AdditiveFeatures`.

**Aggregation (Section 3)**

- `star_aggregate` / `pooled_star_aggregate` (Definitions 4–5) with the
  closed-form segment weights of Eqs. (36)–(37).
- `convex_aggregate` / `pooled_convex_aggregate` (Definition 6).
- Oracle remainders for both schemes (Propositions 5–8).

**Sensitivity (Section 4 and [9])**

- `SensitivityAnalysis` — the Eq. (50) factorization, the Eq. (51) region, the
  Eqs. (52)–(54) auxiliary scores and the Eq. (64) endpoint influence function.
- Three envelope rules: Bonferroni (Eq. 65), one-sided ([9] Theorem 4) and a
  shorter joint rule computed from the estimated bivariate law.
- `robustness_value` ([9] Definition 2), `benchmark_covariate` ([9] Appendix E)
  and `calibrate_deletion` (the paper's Section 5.4).

**Diagnostics**

- The Proposition 1 and 2 identities, the Lemma 2 excess-risk identities, target
  adaptivity and adaptive efficiency (Definitions 1–2), the product-rate
  condition, the pooling conditions of Eqs. (45)/(49) and the Eq. (35) critical
  radius.

**Output**

- Aligned `summary()` tables with significance stars and rule lines, plus LaTeX
  (booktabs) and Markdown exports rendered from one underlying frame.
- Figures: a forest plot by information set, the [9] Section 5.2.3 sensitivity
  contour with the robustness value marked, the Table 3 scenario grid, an
  influence-value histogram and the Proposition 1 information-loss rectangle.
- MATLAB Parula (from its 64 RGB stops) as the default colour scale, alongside
  Jet, Turbo, BlueRed and Sinha.

**Data and examples**

- `datasets.load_pension` — the 1991 SIPP 401(k) extract, 9,915 households.
- `examples/pension_401k.py` — the full worked analysis.
- `examples/multimodal_template.py` — the paper's specification applied to a
  user-supplied multimodal panel.

### Extensions beyond the papers

- Cluster-robust variance and cluster-respecting folds, which Section 5.1
  requires but does not supply.
- Gaussian-multiplier bootstrap intervals for the estimate and both endpoints.
- Both scenario parameterizations kept distinct (`"cy_cd"` and `"rv"`).
- Numerically evaluated rate and pooling diagnostics.

### Known limitations

- The target paper's multimodal application is **not reproducible**: its Keepa
  data is not public. The package reproduces that paper's Table 3 endpoints from
  the published centre and scale, and its Table 2 arithmetic, but not its data.
- No reference implementation of the target paper exists to cross-check against.
  The ATE special case is checked against `DoubleML`.
- `RieszBoost` (gradient-boosted Riesz regression) is not implemented;
  `PartitionRieszLearner` covers the nonlinear case.

### Notes on the sources

Two internal inconsistencies in arXiv:2610.01935v2 surfaced while building, and
are pinned by tests in `tests/test_sensitivity.py`:

- Table 3's note states `C_Y² = C_D² = r²`, but its printed endpoints are only
  reproducible under `C_D² = r²/(1−r²)` — the [9] Definition 2 convention. All
  eighteen endpoints match to the last printed digit under it.
- Table 2's printed `C_Y² = 0.0017` is the raw numerator `R²_full − R²_text`, not
  the ratio `(R²_full − R²_text)/(1 − R²_text) = 0.001752` that Section 5.4
  specifies.

Propositions 3 and 4 of that paper also carry an unresolved `[?]` citation for
Riesz-regression rates; it affects documentation only, not the implementation.
