# `_SOURCES_NEEDED.md` — pre-build source audit for `pragdml`

Target paper: **Pragmatic DML with AI-Learned Representations**, Aradillas Fernández,
Chernozhukov, Cinelli, Klaassen, Newey, Spindler, Teichert-Kluge, Vijaykumar,
arXiv:2610.01935v2 [econ.EM], 2 October 2026.

Audit run: 17 October 2026. Build folder: `PragDML/`.

---

## Lead: what changes the build

**The one load-bearing gap is now closed. One citation in the paper is still broken.**

1. ~~Reference [9] missing~~ — **RESOLVED, 17 Oct 2026.** `9.pdf` in the folder is
   Chernozhukov, Cinelli, Newey, Sharma & Syrgkanis, **"Long Story Short: Omitted Variable
   Bias in Causal Machine Learning"**, arXiv:2112.13398v6 (21 Sep 2026), forthcoming
   *Review of Economics and Statistics*, `doi:10.1162/REST.a.1705`, 79 pp. This is the
   parent paper. Reading it settled every open question and **changed the build in four
   ways**:

   * **Centring convention — confirmed uncentred.** [9] Theorem 2 defines
     `ρ = E[(g-g_s)(α-α_s)] / sqrt(E(g-g_s)² E(α-α_s)²)`, with `C_Y² = R²_{Y-g_s ~ g-g_s}`
     and `C_D² = (1-R²_{α_s})/R²_{α_s}`, all as **uncentred** second-moment ratios. This
     matches the target paper's Eq. (50) and its remark that "centering is not required".
     `sensitivity.py` uses uncentred moments throughout. *One documented exception:* [9]
     Appendix E.4 benchmarks `ρ` by the **Pearson** (centred) correlation of the two
     induced errors. The two conventions are not the same object; `benchmark_covariate()`
     therefore reports both and defaults to Pearson per E.4, with the discrepancy flagged
     in its docstring.
   * **Eq. (64) — independently re-derived, then confirmed.** The delta-method influence
     function for `S0` was derived here before [9] arrived and matches [9] Theorem 4
     exactly: `ψ_± = ψ_θ ± (|ρ|C_Y C_D / 2S)(ν_s² ψ_{σ²} + σ_s² ψ_{ν²})`. Asserted
     numerically in `tests/test_sensitivity.py`.
   * **Confidence bounds — the two papers state *different* guarantees, and both ship.**
     [9] Theorem 4 uses `z_{1-a}` per endpoint and claims **one-sided** coverage of each
     endpoint. The target paper's Eq. (65) uses `z_{1-α/2}` and claims, by Bonferroni,
     **simultaneous** coverage of the whole region. `envelope(method=...)` implements
     `"bonferroni"` (default, target paper Eq. 65), `"one_sided"` ([9] Thm 4) and
     `"joint"` (the shorter simultaneous critical value the target paper suggests one
     sentence after Eq. 65).
   * **Two things I had listed as my own extensions are actually [9]'s results, and are
     now implemented as such** — see the corrected ranking in §3 below: the robustness
     value is [9] **Definition 2** (not "extension E2"), and the contour plot plus
     leave-one-covariate-out benchmarking are [9] §5.2.3 and **Appendix E** (Eqs. 21-30),
     not inventions. Attribution corrected in code and docs.

   Remaining difference worth knowing: the target paper's Table 3 parameterises scenarios
   as `C_Y² = C_D² = r²`, whereas [9] Definition 2 bounds `C_Y² ≤ RV` **and**
   `1 - R²_{α_s} ≤ RV`, which implies `C_D² = RV/(1-RV)`, not `C_D² = RV`. These are
   genuinely different grids. Both are implemented and the distinction is documented
   (`parameterization="cy_cd"` vs `"rv"`), because silently conflating them would
   misreport every robustness value.

2. **Broken citation inside the paper itself.** Propositions 3 and 4 (§3.2) both cite
   "Theorem 3.2 of `[?]`" for high-probability Riesz-regression rates under bounded-class
   (rather than sub-Gaussian) assumptions. The reference is missing from the v2
   bibliography. Candidates present in the folder: `RieszBoost` (Lee & Schuler,
   arXiv:2501.04871v2) and `Generalized Riesz Regression` (Kato, arXiv:2601.07752v4);
   neither carries a Theorem 3.2 matching that statement, so the intended source is
   probably a fourth Riesz-regression paper.
   **Impact on the build: documentation only.** Proposition 4 states a *rate*, not an
   algorithm — `pragdml` implements the Riesz *objective* (which the paper writes in full
   in §3.1 and §5.2, Eq. 70) and reports the critical-radius bound of Eq. (35) as a
   diagnostic. `docs/methods.md` records the citation as unresolved rather than guessing.

Everything else needed to build the library faithfully is present.

---

## Tier 1 — present and used (all 14 PDFs in the folder)

| File | Source | Role in the build |
|---|---|---|
| `pragmatic dml.pdf` | Aradillas Fernández et al., arXiv:2610.01935v2 (2026) | **The paper.** Every equation implemented. |
| `9.pdf` | Chernozhukov, Cinelli, Newey, Sharma & Syrgkanis, arXiv:2112.13398v6, *REStat* forthcoming, doi:10.1162/REST.a.1705 — ref **[9]** | **The parent paper.** Theorem 2 (the OVB identity that Proposition 1 specialises), the `ρ·C_Y·C_D·S` factorization, the three auxiliary orthogonal scores, Theorem 4 (endpoint inference), Definition 2 (robustness values), §5.2.3 (contour plots) and Appendix E (benchmarking). Drives all of `sensitivity.py`. Its §5 application also supplies the shipped dataset and its published acceptance targets. |
| `1608.00060v7.pdf` | Chernozhukov, Chetverikov, Demirer, Duflo, Hansen, Newey & Robins, *Econometrics J.* 21(1):C1–C68 (2018) — ref [7] | Cross-fitting + Neyman-orthogonality engine; Assumption 1 regularity; the `K`-fold convention in `_core.py`. |
| `1809.05224v5.pdf` | Chernozhukov, Newey & Singh, *Econometrica* 90(3):967–1027 (2022) — ref [8] | **Riesz representer calculus.** The Riesz loss `L(α)=E[α²−2m(O,α)]`, the automatic-debiasing score, and the class of Riesz-linear functionals. Drives `riesz.py` and `functionals.py`. |
| `1802.08667v6.pdf` | Chernozhukov, Newey & Singh, *Econometrics J.* (2022), regularized Riesz representers | ℓ1-regularized Riesz regression; the signed ℓ1-hull `F_B` of §3.2 and the `l1` penalty option in `LinearRieszLearner`. |
| `2105.08866v3.pdf` | Vijaykumar, *NeurIPS* 34:4570–4581 (2021) — ref [20] | **Star aggregation.** Definition 4, the star margin (Lemma 4), and the `log M / n` oracle inequalities of Props. 5–8. Drives `aggregation.py`. |
| `Liang15.pdf` | Liang, Rakhlin & Sridharan, *COLT* (2015) — ref [11] | Offset Rademacher complexity / localization; Proposition 3 and the critical radius in `aggregation.critical_radius_l1_hull`. |
| `NIPS-2007-...-Paper.pdf` | Audibert, *NeurIPS* 20 (2007) — ref [2] | Origin of the star hull; the deviation-suboptimality of progressive mixtures that motivates star over exponential weights. Cited in docs. |
| `21-0862.pdf` | Bach, Chernozhukov, Kurz & Spindler, *JMLR* 23:1–6 (2022) | `DoubleML` Python package. **API idiom to mimic** (two of its authors are on the target paper) and the ecosystem dependency decision in §2.3 below. |
| `2110.03031v3.pdf` | Chernozhukov, Newey, Quintas-Martínez & Syrgkanis (RieszNet / ForestRiesz), ICML (2022) | Non-linear Riesz learners; the architecture behind `riesz.ForestRieszLearner` and the optional neural head. |
| `2501.04871v2.pdf` | Lee & Schuler, *RieszBoost* (2025) | Gradient-boosting Riesz regression; `riesz.BoostedRieszLearner` (extension E8). |
| `2601.07752v4.pdf` | Kato, *Generalized Riesz Regression* (2026) | Bregman-divergence view of Riesz fitting; documents why the squared choice is the paper's default and what the KL variant would change. Reference only. |
| `1903.10075v1.pdf` | Athey & Imbens, *Annual Rev. Econ.* 11:685–725 (2019) — ref [1] | Framing of ML representations in economics; README/theory prose only. |
| `D18-1488.pdf` | Wood-Doughty, Shpitser & Dredze, *EMNLP* (2018) — ref [22] | Pitfalls of text classifiers as causal controls; motivates the `representations.py` warnings and the README's "interpreting the output" cautions. |
| `sciadv.ade6585.pdf` | Sridhar & Blei, *Science Advances* 8(42):eade6585 (2022) — ref [15] | Causal inference from text, commentary; docs only. |

## Tier 2 — missing, affects documentation only (not requested)

Standard background the target paper cites but the build does not need: Bickel, Klaassen,
Ritov & Wellner (1993) [4]; Newey, *Econometrica* 62(6) (1994) [13]; van der Vaart (1998)
[17]; van der Vaart & Wellner (1996) [18]; van der Vaart, Dudoit & van der Laan (2006)
[19]; Breiman (1996) [5]; Wolpert (1992) [21]; van der Laan, Polley & Hubbard (2007) [16];
Belloni, Chernozhukov & Hansen (2014) [3]; and the embedding papers BERT [10], GloVe [14],
word2vec [12], GPT-3 [6]. All are cited in `docs/methods.md` from their published
metadata; none changes a line of code.

## Tier 3 — data

**The target paper's own data is unavailable; the parent paper's is, and it comes with
published numbers.**

*Target paper (not shippable).* The application uses **product-level Amazon data scraped
from Keepa.com** (clothing and shoes; price histories + text + images + tabular
attributes). Not public, not redistributable, no replication archive announced in v2.
**Consequence:** the empirical numbers of Tables 1–3 (full-multimodal −1.069, se 0.015;
star aggregate −1.053; calibration `C_Y²=0.0017`, `C_D²=0.0088`, `ρ̂=0.8654`) **cannot be
reproduced here.** They are recorded in `tests/fixtures/paper_targets.json` as
documentation, and the suite asserts only the *algebra* that produced them — that Eq. (50)
and the Eq. (51) endpoints hold exactly for the reported triple — never the numbers.
**If the authors release the Keepa extract, send it and I will add a replication script
and a numerical cross-check test.**

*Parent paper [9] (shipped).* [9] §5 analyses the effect of **401(k) eligibility on net
financial assets** using the 1991 Survey of Income and Program Participation (SIPP) —
the canonical DML dataset (Poterba, Venti & Wise 1994, 1995; Chernozhukov et al. 2018).
Obtained here via `doubleml.datasets.fetch_401K` (MIT), origin US Census Bureau SIPP
(US Government work, public domain), `n = 9,915`, 14 columns. Shipped as
`src/pragdml/datasets/pension.csv.gz` with full provenance in
`src/pragdml/datasets/SOURCES.md`.

**Verified acceptance target, already checked:** the naive treated-minus-control
difference in `net_tfa` reproduces **$19,559.3**, against the "$19,559" reported in [9]
§5.1 — same data, same sample. Further published targets from [9] §5, asserted loosely in
`tests/test_acceptance_401k.py`: ATE ≈ $9,000 (partially linear) and ≈ $8,000
(nonparametric), and `RV_{θ=0, a=0.05}` = 5.4% and 4.5% respectively. The covariate set
of [9] §5.1 (age, income, family size, education, marital status, two-earner status, IRA
participation, home ownership) is exactly reproducible from the shipped columns.

---

## §2.2 Reconcile with code

No reference implementation of *this* paper exists yet (searched: no code link in v2, no
author GitHub for `pragmatic-dml`, nothing on PyPI — `pragdml` and `pragmatic-dml` both
return 404 on PyPI, confirmed 17 Oct 2026). So there is **no cross-implementation
fixture** for the paper's own estimators; that is stated plainly in the README limitations
and in §6 of the test plan.

What *can* be cross-checked, and is:

- **`DoubleML` (installed, v0.11.4)** — for the binary-treatment ATE with the *analytic*
  (plug-in IPW) representer, `pragdml` must agree with `DoubleML.DoubleMLIRM` on a shared
  dataset and a shared fold split, because the two scores coincide there. This is the
  strongest available compatibility proof and is written as a test.
- **`statsmodels` OLS** — the average-derivative functional with a linear outcome learner
  and no representation compression collapses to the OLS slope; asserted to `1e-10`.
- **Priority when sources disagree:** the target paper's equations first, then
  `DoubleML`/[8], then convenience. Stated in the docstrings where it bites.

Licences recorded: `DoubleML` MIT, `statsmodels` BSD-3, `scikit-learn` BSD-3,
`Liang15`/`Audibert`/`Vijaykumar` are papers (read-for-reference only; no code copied).

---

## §2.3 Existing Python ecosystem

| Library | What it already does | Decision |
|---|---|---|
| `DoubleML` 0.11.4 | Cross-fitted DML for a fixed menu of causal models (PLR, IRM, IIVM, DID, …), with analytic scores | **Mimic its API idiom, do not depend on it.** It has no Riesz-representer *learning*, no representation-error identity, no aggregation, no sensitivity region. Used as a *test oracle* for the ATE special case only (test dependency). |
| `numpy` 2.2.6 / `scipy` 1.13.1 | Linear algebra (`lstsq`, `pinv`), SLSQP/QP, distributions | **Hard dependency.** The Riesz normal equations and the simplex QPs go through these. |
| `pandas` 3.0.5 | Frames in/out, named variables carried into tables | **Hard dependency.** |
| `scikit-learn` 1.8.0 | `KFold`/`GroupKFold`, any outcome learner, pipelines, `PCA`/encoders for representations | **Hard dependency.** Cross-fitting splitters and every pluggable learner come from here. |
| `matplotlib` 3.10.0 | Static journal figures | **Hard dependency.** |
| `plotly` 6.5.2 | Interactive 3-D sensitivity surfaces / contours | **Optional extra** `[interactive]`. |
| `statsmodels` 0.14.6 | OLS/quantile baselines | **Test-only dependency** (OLS collapse check). |
| `xgboost` 3.2.0 / `lightgbm` 4.6.0 | Boosted trees | **Optional extra** `[boost]`, for `BoostedRieszLearner` (RieszBoost, 2501.04871). |
| `torch` 2.10.0 / `sentence-transformers` 6.0.1 | Text/image encoders, RieszNet-style heads | **Optional extra** `[text]`. `representations.py` works on *precomputed* embedding matrices by default so the core never needs torch. |
| `joblib` 1.5.3 | Parallel folds / bootstrap with spawned child seeds | **Hard dependency** (small, and needed for honest RNG under parallelism). |
| `numba` 0.62.1 | JIT | **Not used.** The hot loops are BLAS-bound (`φφ'` and segment scans); no bespoke inner loop needs it. |
| `linearmodels` 5.4 | Panel estimators | **Not used — and currently broken in this environment** (compiled against NumPy 1.x, fails to import under NumPy 2.2.6). Avoided deliberately; the first-difference panel step of §5.1 is three lines of `pandas`. |

Hard dependency set kept to: `numpy`, `scipy`, `pandas`, `scikit-learn`, `matplotlib`,
`joblib`.

---

## §3 Proposed additions beyond the paper (ranked)

Reported before coding; all are implemented except where noted.

**Corrected after reading [9]:** two items I had drafted as my own additions are results of
the parent paper. They are implemented faithfully and attributed to [9], not claimed here.

| # | Feature | Status | Attribution |
|---|---|---|---|
| **F1** | **Robustness value** `RV_{θ,a}` — the minimum bound on `C_Y² ≤ RV`, `1-R²_{α_s} ≤ RV`, `\|ρ\|≤1` at which the confidence bound includes `θ` | **Not an extension — [9] Definition 2.** Implemented exactly, including that it binds `1-R²_{α_s}` (so `C_D² = RV/(1-RV)`) and that it inverts the *confidence* bound, not the point bound. | [9] Def. 2; Cinelli & Hazlett (2020, 2025) |
| **F2** | **Sensitivity contour plot** over `(1-R²_{α_s}, C_Y²)` with the zero contour, the 45° line through `RV`, and benchmark reference points | **Not an extension — [9] §5.2.3.** The target paper has only the Table-3 grid; the contour is the parent paper's recommended display. | [9] §5.2.3; Imbens (2003); Cinelli & Hazlett (2020) |
| **F3** | **Leave-one-covariate-out benchmarking** — `k_{Y,j}`, `k_{D,j}`, `G_{Y,j}`, `G_{D,j}`, `ρ_j`, `Δθ_{s,j}` for "U as strong as `X_j`", or `1/4 ×` it | **Not an extension — [9] Appendix E**, Eqs. (21)-(30). The target paper's §5.4 full-versus-text calibration is the special case `k = 1` with a modality *block* as `X_j`; verified algebraically identical. | [9] App. E; Altonji et al. (2005); Oster (2019) |

Genuine additions, ranked:

| # | Addition | Why defensible | Cost | Attribution |
|---|---|---|---|---|
| **E1** | **Cluster-robust variance + clustered fold assignment** (`cluster=` on every estimator) | The paper *asks for this and does not supply it*: §5.1 says all adjacent changes of a product "should remain in the same training or validation fold, and the variance calculation should allow for within-product dependence". Without it the shipped panel example is wrong. | low | target paper §5.1; [7] |
| **E2** | **Both scenario grids, never conflated** — `parameterization="cy_cd"` (target paper Table 3: `C_Y²=C_D²=r²`) and `"rv"` ([9] Def. 2: `C_D²=RV/(1-RV)`) | The two source papers parameterise the same region differently. Reporting one grid under the other's label would misstate every robustness number. | low | target paper Table 3; [9] Def. 2 |
| **E3** | **Multiplier-bootstrap CIs** for `θ̂` and for both sensitivity endpoints | Both papers give only normal/one-sided Wald intervals. The cross-fitted influence values are already in hand, so a Gaussian-multiplier bootstrap is nearly free and behaves better in small samples. | low | [7] §3; standard |
| **E4** | **Repeated cross-fitting** (`n_rep`, median-aggregated estimate, variance inflated across splits) | Not my invention either: [9] footnote 11 does exactly this (5 folds, repeated 5×, median, variance per Chernozhukov et al. 2018) and the target paper's §5.5 flags it as the natural extension of its single split. Needed to hit [9]'s published 401(k) numbers. | low | [9] fn. 11; [7]; `DoubleML` `n_rep` |
| **E5** | **Numerically evaluated oracle/rate diagnostics** — report the Eq. (32), (45) and (49) left-hand sides, and the Eq. (35) critical radius, from the fitted objects | The paper's conditions are assumptions; a user cannot currently tell whether their `M`, `n` and fold design satisfy `√n r_g r_α → 0` or `log M = o(√n)`. This makes them printable. | medium | Eqs. (32), (35), (45), (49) |
| **E6** | **Target-adaptivity / information-loss diagnostic plot** — the two factors of Eq. (18) and their product against `√n`, for a long-vs-short pair | Makes Definitions 1–2 visual and is exactly the §5.4 deletion exercise, generalised. | low | Eq. (17)–(19); §5.4 |
| **E7** | **More functionals than the paper tabulates** — ATT, weighted ATE, average potential outcome, policy effect, alongside ATE and average derivative | §2.1.1 says the class "includes the ATE and many other causal and policy parameters" but only two are instantiated. Each is a two-line `Functional` subclass. | low | [8] Table 1; target paper §2.1.1 |
| **E8** | **`BoostedRieszLearner` and `ForestRieszLearner`** as optional non-linear representer learners | The paper's §5.2 uses one varying-coefficient architecture; a non-convex menu is precisely what star aggregation is designed for, so the library should be able to *produce* one. | medium | RieszBoost (2501.04871); RieszNet/ForestRiesz (2110.03031) |
| **E9** | **Joint-Gaussian simultaneous envelope** as an alternative to Bonferroni in Eq. (65) | The paper itself suggests it one sentence after Eq. (65) but does not implement it; gives a strictly shorter envelope. | low | target paper, after Eq. (65) |
| **NOT DONE** | Reproduction of the Keepa application | Data not public (Tier 3). Shipped as a template script that runs on any user-supplied panel with the paper's exact specification. | — | — |

## What would make the build stronger — please send if you have it

1. ~~`Long story short` PDF~~ — **received 17 Oct 2026 as `9.pdf`; nothing further needed.**
2. **The source of "Theorem 3.2"** cited as `[?]` in Propositions 3–4 of the target paper.
   *Unlocks:* the Riesz-rate statement on the theory page. Build impact: none (the
   proposition states a rate, not an algorithm).
3. **The Keepa extract, or any multimodal panel** (price, outcome, text/image embeddings).
   *Unlocks:* a replication script for Tables 1–3 and a numerical cross-check test.
   Until then `examples/multimodal_template.py` runs the paper's exact specification on
   any user-supplied panel.

Nothing on this list blocks the build.
