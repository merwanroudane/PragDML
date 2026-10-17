# Methods: the complete step → equation map

Every numbered equation, definition, proposition and theorem of

> A. Aradillas Fernández, V. Chernozhukov, C. Cinelli, S. Klaassen, W. Newey,
> M. Spindler, J. Teichert-Kluge & S. Vijaykumar, *Pragmatic DML with AI-Learned
> Representations*, arXiv:2610.01935v2 (2026)

against the code that implements it, plus the results of the parent paper

> V. Chernozhukov, C. Cinelli, W. K. Newey, A. Sharma & V. Syrgkanis, *Long Story
> Short: Omitted Variable Bias in Causal Machine Learning*, Review of Economics
> and Statistics (2026), doi:10.1162/REST.a.1705 — cited below as **[9]**

that `pragdml` also implements.

Where the two papers disagree, or where a paper's own text and numbers disagree,
that is recorded here rather than silently resolved.

---

## 1. Setup and targets (Section 2.1.1)

Observe `O = (Y, D, Z)` with `Z` a rich covariate. A learned representation
`X = x(Z, λ)` gives two information sets:

- full: `W* = (D, Z)`, regression `g*(w*) = E[Y | W* = w*]`
- short: `W_λ = (D, X)`, regression `g_λ = E[Y | W_λ]`, the `L²(P)` projection of
  `g*` onto `σ(W_λ)`

| Paper | Code |
|---|---|
| Eq. (2) `θ* = E[m(O, g*)]` | `functionals.Functional`, evaluated on full-information nuisances |
| Eq. (3) `θ_λ = E[m(O, g_λ)]` | the same, on representation-based nuisances |
| Footnote 1, p. 6: ATE, `m = h(1,z) − h(0,z)` | `functionals.ATE` |
| Sec. 2.1.1, average potential outcome | `functionals.AveragePotentialOutcome` |
| Sec. 2.1.1, weighted ATE | `functionals.WeightedATE` |
| Sec. 2.1.1, policy effects | `functionals.PolicyEffect` |
| Eq. (67), average partial derivative | `functionals.AverageDerivative` |
| ratio targets (beyond the linear class) | `functionals.ATT` |

**The compatibility condition** of Section 2.1.1 — that `m(O, h)` is
`σ(W_λ)`-measurable when `h` is a function of `W_λ` — holds for every shipped
functional because each depends on `W` only through the values of `h` at
observable arguments.

**Linearity of `m` in `h` is the load-bearing assumption of the whole
implementation.** It is what lets `Functional.apply` be called once on a whole
feature map to produce the matrix `M[i,j] = m(O_i, φ_j)`, which is the right-hand
side of the Riesz normal equations; it is also what makes the aggregation of `m`
values by the same weights as the predictions legitimate. Asserted for all six
functionals in `tests/test_identities.py::test_m_is_linear_in_h`.

## 2. Riesz representers (Section 2.1.2)

| Paper | Code |
|---|---|
| Eqs. (4)–(5), the representers `α*`, `α_λ` | the contract every `RieszLearner` satisfies |
| Sec. 2.1.2, the IPW form of `α` for the ATE | `ATE.analytic_representer`, used by `PlugInRieszLearner` |
| Sec. 3.1, the Riesz loss `L(α) = E[α² − 2m(O,α)]` | `riesz.riesz_loss` |
| Eq. (26), `inf L = −E[α²]` | `riesz.representer_second_moment` |

`LinearRieszLearner` solves the **Riesz normal equations** in closed form:

```
Q c = r,    Q = P_n[φ φ'],    r_j = P_n[m(O, φ_j)]
```

This is the exact minimiser of the quadratic Riesz loss over `H = span(φ)`. No
propensity score is inverted, so no near-zero denominator can blow up; the price
is that `α₀` is only the projection of `α*` onto `H`, and the gap is precisely the
`α* − α₀` term of the mixed-bias identity.

The solve rescales columns by their RMS before solving — an exact
reparameterisation (`Q̃ = SQS`, `r̃ = Sr`, `c = Sc̃`), done only for conditioning,
which matters when features mix dollars of income with binary indicators.

## 3. The omitted-information identity (Proposition 1)

**Eqs. (6)–(7):**

```
θ* − θ_λ = E[(g* − g_λ)(α* − α_λ)]
|θ* − θ_λ| ≤ ‖g* − g_λ‖_{P,2} · ‖α* − α_λ‖_{P,2}
```

→ `diagnostics.omitted_information_gap`, returning the exact gap, the
Cauchy–Schwarz bound, both error norms and their normalised alignment `ρ`.

This is the learned-representation specialisation of **[9] Theorem 2**. The
substantive content: a representation preserves the target if it preserves
*either* the regression *or* the balancing weight. Only the product matters, so a
rich object may be compressed enormously without moving the estimand.

Asserted in `tests/test_identities.py`: the bound dominates the gap always; the
gap vanishes when either error vanishes; the bound binds exactly when the two
error vectors are collinear.

## 4. The score and cross-fitted DML (Section 2.1.3)

| Paper | Code |
|---|---|
| Eq. (8), short score `ψ_λ = m(O,g) + α(W){Y − g(W)}` | `_core.debiased_score` |
| Eq. (9), the mixed-bias drift | `diagnostics.mixed_bias_drift` |
| Eq. (10), the long score | the same function, long nuisances |
| Eq. (11), `θ̂` | `CrossFitOutput.theta` |
| Eq. (12), `σ̂²` | `CrossFitOutput.sigma2` |
| Eq. (13), `L²`-boundedness of `m` | assumption; checkable via held-out loss |
| Eq. (14), the product-rate condition | `diagnostics.product_rate` |
| Eq. (16), the Wald interval | `CrossFitOutput.conf_int` |
| Sec. 2.1.3, fold construction | `_core.make_folds` |
| Assumption 1 | documented preconditions on the learners |
| **Theorem 1** | `PragmaticDML.fit` → `DMLResult` |

**Eq. (9) is the reason orthogonality matters:** the population drift of the score
away from its target is *exactly* a mixed-bias term, `−E[(g−g₀)(α−α₀)]`, so
first-stage error enters as a product and not linearly. Asserted exactly (not
asymptotically) in-sample in `test_score_drift_equals_minus_mixed_bias`.

**Cross-fitting discipline.** For each fold `k`, both learners are cloned and
fitted on `I_kᶜ` only, then the score is evaluated on `I_k`. Conditional on the
training fold the fitted functions are fixed — the independence structure the
proofs use. `make_folds` keeps whole clusters inside a fold when `cluster=` is
given, which Section 5.1 requires.

## 5. Target adaptivity and adaptive efficiency (Section 2.2)

| Paper | Code |
|---|---|
| **Definition 1**, Eq. (17): `√n(θ* − θ_λ) = o_P(1)` | `diagnostics.target_adaptivity` |
| Eq. (18), the sufficient product form | the same, `scaled_product` |
| **Definition 2**, Eq. (19): adaptive efficiency | `diagnostics.adaptive_efficiency` |
| Lemma 1, Eqs. (20)–(21): short-score convergence | theory only |
| **Theorem 2**, Eq. (22): efficiency under losslessness | reported via `target_label` |

The hierarchy the package reports honestly on every result:

1. **Theorem 1 alone** — the interval covers `θ_λ`, the representation-specific
   target. Always true under Assumption 1.
2. **+ target adaptivity** — the same interval covers `θ*`.
3. **+ adaptive efficiency** — it does so at the semiparametric efficiency bound.

`target_adaptivity` reports both the exact `√n|gap|` and the conservative product
form, because the two can disagree: errors may be large yet nearly orthogonal, in
which case the target barely moves. The `threshold` is a **reporting convention**,
not a test; no asymptotic statement follows from crossing it.

## 6. Fine-tuned pipelines (Sections 2.3–2.5)

**Definition 3** defines the pipeline targets as projections onto the common
closed linear space `H = span(G ∪ A)`:

```
g₀ = argmin_{g∈H} E(Y − g)²,   α₀ = argmin_{α∈H} E[α² − 2m(O,α)],   θ₀ = E[m(O,g₀)]
```

| Paper | Code |
|---|---|
| Definition 3 | `features.FeatureMap` span; `LinearOutcomeLearner`/`LinearRieszLearner` compute both projections **exactly** |
| **Proposition 2(i)** restricted Riesz representation | asserted in `test_riesz_normal_equations_hold` |
| Proposition 2(ii), Eq. (23) mixed bias | `diagnostics.mixed_bias_drift` |
| Proposition 2(iii), Eq. (24) target gap | `diagnostics.pipeline_target_gap` |
| Sec. 2.4, Eqs. (25)–(26) subordinated tuning | `square_risk`, `riesz_loss` |
| Assumption 2 | preconditions |
| **Theorem 3**, Eq. (27) | `PipelineDML.fit` |

**Why an explicit feature map matters here.** When `H` is the span of a feature
map, both projections have closed forms, so `g₀` and `α₀` are *computed* rather
than approximated — and Lemma 2's identities then hold to machine precision and
can be asserted in tests. With a black-box learner the theory is unchanged but the
space is implicit, and only the product-rate condition remains checkable.

**Section 2.4's geometry.** Profiling out the nuisances leaves exactly the two
information-loss quantities of Proposition 1: minimising population prediction
risk over `λ` is equivalent to minimising `‖g* − g_λ‖²` (Eq. 25, Pythagoras), and
minimising the profiled Riesz loss is equivalent to minimising `‖α* − α_λ‖²`
(Eq. 26). The natural tuning objectives target precisely the right objects.

## 7. Learning over `H` (Section 3)

| Paper | Code |
|---|---|
| **Lemma 2**, Eqs. (28)–(29): excess risk = squared `L²` error | `diagnostics.excess_risk_identity` |
| Eqs. (30)–(31), foldwise oracle inequalities | template |
| Eq. (32), the design constraint `√n (r_g + a_g)(r_α + a_α) → 0` | `diagnostics.product_rate` |
| Proposition 3, Eq. (33): square-loss ERM on a star-shaped class | theory |
| Proposition 4, Eq. (34): Riesz regression rates | theory — **see the citation note below** |
| Eq. (35), the critical radius of a signed `ℓ¹` hull | `diagnostics.critical_radius_l1_hull` |

Lemma 2 is the bridge from learning theory to Assumption 2: on `H` there is no gap
between controlling risk and controlling prediction error — they are the same
statement. Asserted for both heads.

Eq. (35)'s two regimes decide a real design question: the first behaves like
`√(M/n)`, so proper convex aggregation to an `n^{−1/4}` rate typically needs
`M = o(√n)`; the second grows only logarithmically in `M` but pays the envelope
constant. `critical_radius_l1_hull` reports which regime binds.

> **Unresolved citation in the source.** Propositions 3 and 4 of the target paper
> both cite "Theorem 3.2 of [?]" for high-probability Riesz-regression rates under
> bounded-class assumptions. The reference is missing from the v2 bibliography.
> This affects documentation only: Proposition 4 states a *rate*, not an
> algorithm, and `pragdml` implements the Riesz *objective*, which Sections 3.1
> and 5.2 write out in full.

## 8. Aggregation (Sections 3.3–3.4)

| Paper | Code |
|---|---|
| **Definition 4**, the star estimator | `aggregation.star_aggregate` |
| Eq. (36), `β` for the square loss | `aggregation.segment_weight_square` |
| Eq. (37), `β` for the Riesz loss | `aggregation.segment_weight_riesz` |
| Eqs. (38)–(39), Propositions 5–6 | `aggregation.star_oracle_bound` |
| Eqs. (40)–(41), the pooled objectives | `aggregation.pooled_star_aggregate` |
| **Definition 5**, the pooled star estimator | the same |
| Eq. (42), the best-single benchmark | `AggregationResult.best_single` |
| Eqs. (43)–(44), Proposition 7 | `star_oracle_bound(pooled=True)` |
| Assumption 3, Eq. (45) | `diagnostics.pooling_conditions("star")` |
| **Theorem 4** | `PipelineDML(aggregation="star")` |
| **Definition 6**, the pooled convex estimator | `aggregation.pooled_convex_aggregate` |
| Eq. (46), the convex benchmark | returned |
| Eqs. (47)–(48), Proposition 8 | `aggregation.convex_oracle_bound` |
| Eq. (49), **Theorem 5** | `diagnostics.pooling_conditions("convex")` |

Both segment steps are one-dimensional quadratic minimisations with closed-form
solutions, so a finite menu is aggregated by a single scan over `j = 1 … M` with
no iterative optimiser. Both are verified against a brute-force grid search over
`β ∈ [0,1]`.

**Definition 5's own observation** — that the pooled star estimator *is* the
Definition 4 estimator computed from augmented observations — is why
`pooled_star_aggregate` delegates to `star_aggregate` rather than being a second
algorithm. Asserted: the two produce bit-identical weights.

**Remark 1 (what must stay inside the folds).** Only the aggregation step crosses
folds. Each candidate must still be trained on `I_kᶜ` alone; otherwise the
candidates themselves — not merely a low-dimensional choice among them — depend on
the evaluation fold, and nothing here applies. `PipelineDML` enforces this.

**The design trade-off, quantified.** Star weights cost `√(log M)` in Eq. (45),
reducing to `log M = o(√n)`; convex weights cost `√(M log n)` in Eq. (49),
reducing to `M log n = o(√n)`. So a star menu may grow faster than any power of
`n` while a convex menu must stay small — in exchange for a weakly better
approximation benchmark (`ē ≤ e`).

**Oracle remainders are rate shapes, not guarantees.** The universal constants in
Propositions 5–8 are unspecified in the paper. `pragdml` reports the remainders
with `constant=1.0` and says so in every returned dict: the *ratio* of two such
numbers is meaningful for comparing designs; the level is arbitrary.

## 9. Sensitivity (Section 4 and [9])

**Eq. (50)**, following [9] Theorem 2:

```
θ* − θ₀ = ρ · C_Y · C_D · S
```

with all moments **uncentred**:

- `ρ = E[(g*−g₀)(α*−α₀)] / (‖g*−g₀‖ ‖α*−α₀‖) ∈ [−1,1]` — alignment
- `C_Y² = E[(g*−g₀)²] / E[(Y−g₀)²]` — outcome-side strength
- `C_D² = E[(α*−α₀)²] / E[α₀²]` — representer-side strength
- `S² = E[(Y−g₀)²] · E[α₀²]` — **identified**, estimated by DML

| Paper | Code |
|---|---|
| Eq. (50) | `sensitivity.bias_factor` |
| Eq. (51), the identified region | `SensitivityAnalysis.interval` |
| Eqs. (52)–(54), the auxiliary orthogonal scores | `SensitivityAnalysis.scores` |
| Eqs. (55)–(57), `σ̂₀²`, `ν̂₀²`, `Ŝ₀` | `CrossFitOutput.sigma2_outcome`, `.nu2_representer`, `.S` |
| Eq. (58), the endpoint estimates | `SensitivityAnalysis.evaluate` |
| Assumption 4, Eq. (59): `o_P(n^{−1/4})` per nuisance | precondition |
| Lemma 3, Eqs. (60)–(62) | `SensitivityAnalysis.scores` |
| **Theorem 6**, Eqs. (63)–(64) | `SensitivityAnalysis.evaluate` |
| Eq. (65), the Bonferroni envelope | `envelope(method="bonferroni")` |
| [9] Theorem 4, one-sided bounds | `envelope(method="one_sided")` |
| after Eq. (65), a shorter simultaneous rule | `envelope(method="joint")` |
| **[9] Definition 2**, robustness values | `sensitivity.robustness_value` |
| **[9] Section 5.2.3**, contour plots | `contour_data` + `_plots.plot_sensitivity_contour` |
| **[9] Appendix E**, benchmarking | `sensitivity.benchmark_covariate` |
| Section 5.4, deletion calibration | `sensitivity.calibrate_deletion` |

### Why `ν̂₀²` uses the orthogonal form

Eq. (56) estimates `E[α₀²]` by `P_n[2m(O,α) − α²]`, not by `P_n[α²]`. The two
agree at the truth (that is Eq. 26) but differ at a fitted `α`, and only the
former is Neyman-orthogonal with respect to perturbations of `α` — a consequence
of Proposition 2(i) and linearity of `m`. Only it supports root-`n` inference on
the bounds. The identity is asserted to `1e-8` at the minimiser.

### The endpoint influence function

Eq. (64) follows from the delta method applied to `(u,v) ↦ (uv)^{1/2}`, whose
partials are `∂S/∂σ₀² = ν₀²/(2S)` and `∂S/∂ν₀² = σ₀²/(2S)`:

```
ψ_{0,±} = ψ_{θ₀} ± (c / 2S₀)(ν₀² ψ_{σ₀²} + σ₀² ψ_{ν₀²})
```

This was re-derived independently here before [9] was available and matches [9]
Theorem 4 exactly. The partials are checked numerically in
`test_delta_method_factors_match_numerical_derivatives`.

### Three envelopes, three different guarantees

| Method | Critical value | Guarantee |
|---|---|---|
| `"one_sided"` ([9] Thm 4) | `z_{1−a}` | per endpoint, one-sided |
| `"bonferroni"` (Eq. 65, default) | `z_{1−a/2}` | simultaneous for the whole region |
| `"joint"` | from the estimated bivariate law | simultaneous, strictly shorter than Bonferroni when the endpoints are correlated |

They nest: one-sided ⊆ joint ⊆ Bonferroni. Asserted.

### Two parameterizations, deliberately not conflated

The two source papers grid the same region differently:

- **`"cy_cd"`** — the target paper's Table 3: set `C_Y²` and `C_D²` directly.
- **`"rv"`** — [9] Definition 2: one bound `RV` constrains `C_Y² ≤ RV` **and**
  `1 − R²_{α_s} ≤ RV`, which implies `C_D² = RV/(1−RV)`, *not* `C_D² = RV`.

Reporting one under the other's label would misstate every robustness value, so
`pragdml` keeps them separate and labels every result.

### `ρ` is alignment, not strength

[9] Appendix E.4's example: `D = U²`, `Y = D + U` with `U ~ N(0,1)`. Here `U`
explains 100% of the residual variation in both equations, yet `U²` is
uncorrelated with `U`, so the bias is exactly zero. For omitted information to
bias an estimate it must not only be strong in both equations — the *functional
form* of its influence must be similar in both. That is what `ρ` measures, and why
it multiplies the strengths rather than being implied by them.

### Benchmarking vs deletion calibration

Two routines, two different questions, with arithmetic that relates them exactly:

- `calibrate_deletion` — the target paper's Section 5.4 **deletion exercise**: the
  deleted block *is* the omitted information, so `C_Y²` is the partial `R²`
  itself.
- `benchmark_covariate` — [9] Appendix E **relative benchmarking**: the omitted
  information is unobserved and `X_j` only sets the scale, so `G_Y` is the odds
  form `partial/(1 − partial)` of Eq. (26), to be multiplied by `k_Y`.

Hence `G_Y = C_Y²_deletion / (1 − C_Y²_deletion)` exactly. Pinned by test.

> **A `ρ` convention mismatch between the sources.** The sensitivity parameter `ρ`
> of Eq. (50) and [9] Theorem 2 is *uncentred*; the benchmark `ρ_j` of [9]
> Appendix E.4 is the *Pearson* (centred) correlation. These are different
> objects. `benchmark_covariate` returns both (`rho_pearson`, `rho_uncentred`) and
> defaults to Pearson per E.4.

## 10. The application (Section 5)

| Paper | Code |
|---|---|
| Eq. (66), the first-difference specification | `examples/multimodal_template.py` |
| Eq. (67), the average partial derivative | `functionals.AverageDerivative` |
| Eqs. (68)–(69), the varying-coefficient architecture | `features.VaryingCoefficientFeatures` |
| Eq. (70), the Riesz objective for that architecture | `LinearRieszLearner` |
| Eq. (71), the debiased score | `_core.debiased_score` |
| Table 1 layout | `compare_information_sets` |
| Section 5.4 calibration | `sensitivity.calibrate_deletion` |
| Table 3 layout | `SensitivityAnalysis.grid` |

**The paper's application is not reproducible.** Its product-level data is scraped
from Keepa.com and is neither public nor redistributable; v2 announces no
replication archive. `pragdml` reproduces the *arithmetic* of Tables 2 and 3 from
the published values, and ships the parent paper's public 401(k) application
instead.

### Two inconsistencies found in the target paper

Both pinned by tests in `tests/test_sensitivity.py`.

**1. Table 3's stated parameterization does not generate its own numbers.** The
note says `C_Y² = C_D² = r²`. Fitting a single scale `S` by least squares to the
eighteen printed endpoints:

| Convention | implied `S` | max residual |
|---|---|---|
| `c = ρr²` (the note) | 6.8987 | 0.00147 |
| `c = ρr²/√(1−r²)` ([9] Def. 2) | **6.7452** | **0.00044** |

Under the second, all eighteen endpoints reproduce **to the last printed digit**.
Under the note's, several miss the third decimal. The table was computed with the
parent paper's convention.

**2. Table 2's `C_Y²` is the raw numerator, not the stated ratio.** Section 5.4
specifies `C_Y² = (R²_full − R²_text)/(1 − R²_text)`, which gives `0.001752`. The
table prints `0.0017`, which is `R²_full − R²_text` exactly. `C_D²` *does* follow
its stated formula (`2.22/252.02 = 0.008809` against the printed `0.0088`).
`pragdml` implements the stated ratio, since that is what Eq. (50) requires and
what [9] Eq. (24) derives.

---

## Faithfulness traps checked on this build

| Trap | How it was handled |
|---|---|
| **Transformation order** | `m` is applied exactly where the paper writes it, to the feature map directly, so linearity is used rather than assumed. |
| **Conditioning set** | `include_intercept=True` by default on every feature map; a missing constant silently biases everything. |
| **Divisor conventions** | `1/n` throughout, matching the paper's `P_n`. The cluster-robust variance uses `Σ_g (Σ_{i∈g} ψ̃_i)² / n²`. |
| **Indexing** | 0-based arrays against the paper's 1-based notation; folds are index arrays, asserted disjoint and exhaustive. |
| **RNG discipline** | One `np.random.default_rng(seed)`, spawned via `SeedSequence` for repetitions. Never the legacy global API. |
| **Dtype and conditioning** | `float64` everywhere; `solve` with a `pinv` fallback, after exact column rescaling. |
| **Resampling replicates the whole statistic** | The multiplier bootstrap resamples the cross-fitted influence values, and draws one multiplier per cluster when clustering. |
| **Derivative of a step function** | `AverageDerivative` warns when the learner is piecewise constant, where the numerical derivative is identically zero. |
| **Degenerate representer** | `cross_fit` warns when `ν̂² ≤ 0`; `SensitivityAnalysis` refuses to construct. |

## References

1. Aradillas Fernández, Chernozhukov, Cinelli, Klaassen, Newey, Spindler,
   Teichert-Kluge & Vijaykumar (2026). *Pragmatic DML with AI-Learned
   Representations.* arXiv:2610.01935
2. Chernozhukov, Cinelli, Newey, Sharma & Syrgkanis (2026). *Long Story Short.*
   REStat. doi:10.1162/REST.a.1705
3. Chernozhukov, Newey & Singh (2022). *Automatic Debiased Machine Learning of
   Causal and Structural Effects.* Econometrica 90(3):967–1027.
   doi:10.3982/ECTA18515
4. Chernozhukov, Chetverikov, Demirer, Duflo, Hansen, Newey & Robins (2018).
   *Double/Debiased Machine Learning.* Econometrics Journal 21(1):C1–C68.
   doi:10.1111/ectj.12097
5. Vijaykumar (2021). *Localization, Convexity, and Star Aggregation.* NeurIPS 34.
6. Liang, Rakhlin & Sridharan (2015). *Learning with Square Loss.* COLT.
7. Audibert (2007). *Progressive Mixture Rules are Deviation Suboptimal.* NeurIPS 20.
8. Cinelli & Hazlett (2020). *Making Sense of Sensitivity.* JRSS-B 82(1):39–67.
   doi:10.1111/rssb.12348

*All DOIs verified 17 October 2026.*
