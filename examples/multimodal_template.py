"""The paper's multimodal specification, for your own panel.

The application of arXiv:2610.01935 Section 5 cannot be shipped: its product data
is scraped from Keepa.com and is neither public nor redistributable. This script
is the paper's **exact specification**, ready to run on any panel you supply in
the same shape.

What the paper does, and what this reproduces
---------------------------------------------
1. **First differences** (Eq. 66). With ``R_it`` the sales rank, orient the
   outcome so larger means stronger demand, ``Y_it = -log R_it``, let ``D_it`` be
   log price, and take adjacent changes. Differencing removes time-invariant
   product quality.
2. **Seven information sets** -- text, image and tabular representations and
   their four combinations.
3. **The average partial derivative** (Eq. 67) as the target: with both variables
   in logs it is an elasticity-like response.
4. **The varying-coefficient architecture** (Eqs. 68-69) for both heads, fitted
   by their own losses.
5. **Star aggregation** over the seven candidates (Section 3.4).
6. **Calibration by deletion** (Section 5.4) and a **sensitivity grid**
   (Table 3).

Two things this script enforces that the paper's own write-up only asks for
-----------------------------------------------------------------------------
* **Clustering.** Section 5.1 says every adjacent change of a product must stay
  in one fold and the variance must allow for within-product dependence. Pass
  ``product_id``; the folds and the variance both respect it.
* **The Table 3 parameterization.** The paper's Table 3 note says
  ``C_Y^2 = C_D^2 = r^2``, but its printed endpoints are only reproducible under
  ``C_D^2 = r^2/(1-r^2)``. This script uses ``parameterization="rv"``, which is
  the one that generates the published numbers.

Your data
---------
A long panel with, per product-period: a product id, a period, a numerical sales
rank (or any demand proxy), a price, and -- per modality -- an embedding matrix
aligned row-for-row. Edit :func:`load_your_panel` and run.

    python examples/multimodal_template.py

Author
------
Dr Merwan Roudane <merwanroudane920@gmail.com>
https://github.com/merwanroudane
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pragdml import (
    AverageDerivative,
    LinearOutcomeLearner,
    LinearRieszLearner,
    VaryingCoefficientFeatures,
    compare_information_sets,
    omitted_information_gap,
    target_adaptivity,
)
from pragdml.sensitivity import calibrate_deletion

SEED = 20261017


# ---------------------------------------------------------------------------
# 1. Your data goes here
# ---------------------------------------------------------------------------

def load_your_panel():
    """Return ``(panel, embeddings)``.

    ``panel`` : DataFrame with columns
        ``product`` (id), ``period`` (sortable), ``rank`` (>0), ``price`` (>0).
    ``embeddings`` : dict of ``{"text": A, "image": B, "tabular": C}``
        each an ``(n_rows, k_j)`` array aligned row-for-row with ``panel``.

    The synthetic stand-in below has a **known** true elasticity of ``-1.0`` and
    is built so that the design actually identifies it: run the script unmodified
    and the whole workflow should return about ``-1.0``, which is the point of
    shipping a template you can sanity-check before trusting it on real data.

    How the stand-in is built, and why it has to be built this way
    -------------------------------------------------------------
    The paper's controls are **product-level** representations: text and images
    describe the product, not the week. First differencing already removes any
    time-invariant additive effect of those attributes, so the representations do
    not earn their keep by absorbing a level -- they earn it by explaining
    *heterogeneous demand evolution*, which is exactly what the
    varying-coefficient architecture of Eqs. (68)-(69) encodes:

        dY = dD * gamma(X) + delta(X) + noise.

    The confounding therefore has to live in how *changes* co-move by product
    type: products of a given type both raise prices and enjoy demand growth. That
    is what ``trend`` does below. Conditioning on ``X`` (which carries the product
    type) removes it; ignoring ``X`` does not, and the script prints both so the
    bias is visible.
    """
    rng = np.random.default_rng(SEED)
    n_products, n_periods = 600, 6
    n = n_products * n_periods

    product = np.repeat(np.arange(n_products), n_periods)
    period = np.tile(np.arange(n_periods), n_products)

    # Time-invariant product type, visible through every modality.
    quality_p = rng.normal(size=n_products)
    quality = quality_p[product]

    text = quality[:, None] * rng.normal(size=(1, 8)) + 0.6 * rng.normal(size=(n, 8))
    image = quality[:, None] * rng.normal(size=(1, 6)) + 0.6 * rng.normal(size=(n, 6))
    tabular = np.column_stack([
        quality + rng.normal(size=n) * 0.3,
        np.repeat(rng.normal(size=n_products), n_periods),
        np.repeat((rng.uniform(size=n_products) < 0.3).astype(float), n_periods),
    ])

    # Product-type-specific drift in BOTH price and demand: the confounder that
    # survives differencing, and the one the representations can see.
    trend = 0.25 * quality_p[product]

    # Price path: a product-type drift plus exogenous pricing noise.
    price_noise = rng.normal(size=n) * 0.25
    log_price = 1.5 + trend * period + price_noise

    # Demand path: the true elasticity, the same product-type drift entering
    # demand directly, and idiosyncratic noise independent of the price noise.
    log_demand = (
        -1.0 * log_price
        + 1.1 * quality
        + 0.9 * trend * period
        + rng.normal(size=n) * 0.2
    )
    rank = np.exp(-log_demand)

    panel = pd.DataFrame({
        "product": product, "period": period,
        "rank": rank, "price": np.exp(log_price),
    })
    return panel, {"text": text, "image": image, "tabular": tabular}


# ---------------------------------------------------------------------------
# 2. First differences (Eq. 66)
# ---------------------------------------------------------------------------

def first_difference(panel: pd.DataFrame, embeddings: dict):
    """Build ``(dY, dD, embeddings_at_change, product_id)``.

    ``Y_it = -log R_it`` so that larger means stronger demand (Sec. 5.1), then
    adjacent changes within product. Embeddings are taken at the *later* period of
    each change, since they describe the product at that point.
    """
    panel = panel.sort_values(["product", "period"]).reset_index(drop=True)
    order = panel.index.to_numpy()

    y = -np.log(panel["rank"].to_numpy(dtype=float))
    d = np.log(panel["price"].to_numpy(dtype=float))
    prod = panel["product"].to_numpy()

    # A change is valid when the previous row is the same product.
    same = np.r_[False, prod[1:] == prod[:-1]]
    idx = np.flatnonzero(same)
    prev = idx - 1

    dY = y[idx] - y[prev]
    dD = d[idx] - d[prev]
    emb = {k: v[order][idx] for k, v in embeddings.items()}
    return dY, dD, emb, prod[idx]


def build_information_sets(emb: dict) -> dict:
    """The paper's seven modality combinations."""
    txt, img, tab = emb["text"], emb["image"], emb["tabular"]
    return {
        "text": txt,
        "image": img,
        "tabular": tab,
        "text+image": np.hstack([txt, img]),
        "text+tabular": np.hstack([txt, tab]),
        "image+tabular": np.hstack([img, tab]),
        "full": np.hstack([txt, img, tab]),
    }


# ---------------------------------------------------------------------------
# 3. The analysis
# ---------------------------------------------------------------------------

def main() -> None:
    panel, embeddings = load_your_panel()
    dY, dD, emb, cluster = first_difference(panel, embeddings)
    sets = build_information_sets(emb)

    print("=" * 78)
    print("  Multimodal demand: the specification of arXiv:2610.01935 Section 5")
    print("=" * 78)
    print(f"  Product-period changes   {dY.shape[0]:,}")
    print(f"  Distinct products        {np.unique(cluster).size:,}")
    print(f"  Information sets         {len(sets)}")
    print(
        "\n  Target: average partial derivative (Eq. 67). Both variables are in\n"
        "  logs, so it is an elasticity-like response of the inverse-rank demand\n"
        "  proxy to price.\n"
        "  Clustering on product, as Section 5.1 requires."
    )

    # Is adjustment doing anything? Without it the design is confounded, and a
    # recovery test that passes on unconfounded data proves nothing.
    naive = float(
        np.polyfit(dD, dY, 1)[0]
    )
    print(f"\n  Unadjusted slope of dY on dD   {naive: .4f}")
    print("  (if this already equals the truth, the controls are not being tested)")

    # The paper's own architecture, Eqs. (68)-(69). Both heads share the
    # architecture but not the coefficients: each is fitted by its own loss.
    def head():
        return VaryingCoefficientFeatures()

    print("\n" + "=" * 78)
    print("  Table 1 -- average partial derivatives by information set")
    print("=" * 78)
    comparison = compare_information_sets(
        dY, dD, sets,
        functional=AverageDerivative(),
        outcome_learner=LinearOutcomeLearner(head(), ridge=1e-8),
        riesz_learner=LinearRieszLearner(head(), ridge=1e-8),
        n_folds=5, cluster=cluster, seed=SEED, aggregation="star",
    )
    print(comparison.summary())
    print(
        f"\n  Spread across information sets: {comparison.spread:.4f}\n"
        f"  All intervals on the same side of zero: {comparison.all_same_sign}\n\n"
        "  A narrow spread is itself evidence: the conclusion does not hinge on\n"
        "  which modality carried the product information."
    )

    full = comparison.results["full"]
    text_only = comparison.results["text"]

    # ---- Section 5.4: calibration by deletion -----------------------------
    print("\n" + "=" * 78)
    print("  Table 2 -- full-versus-text calibration (Section 5.4)")
    print("=" * 78)
    cal = calibrate_deletion(full.crossfit_, text_only.crossfit_,
                             name="image + tabular")
    print(f"  Outcome R^2   full {cal['r2_long']: .4f}   text {cal['r2_short']: .4f}")
    print(f"  Riesz loss    full {cal['riesz_loss_long']: .2f}   "
          f"text {cal['riesz_loss_short']: .2f}")
    print(f"\n  C_Y^2 = {cal['cy2']:.4f}    C_D^2 = {cal['cd2']:.4f}    "
          f"rho_hat = {cal['rho']:.4f}")
    print(f"  Equal-strength benchmark r^2 = {cal['r2_equal_strength']:.4f}")
    print(
        "\n  The paper's diagnostic lesson: compare how much the two numbers move.\n"
        "  Information that barely changes the outcome R^2 can still change the\n"
        "  Riesz loss visibly -- so prediction quality alone does not summarise\n"
        "  target preservation."
    )

    # ---- Proposition 1 on the observed deletion ---------------------------
    gap = omitted_information_gap(
        full.crossfit_.g_hat, text_only.crossfit_.g_hat,
        full.crossfit_.alpha_hat, text_only.crossfit_.alpha_hat,
    )
    adapt = target_adaptivity(
        full.crossfit_.n, gap.g_error, gap.alpha_error, gap=gap.gap
    )
    print("\n" + "=" * 78)
    print("  Proposition 1 on the observed deletion")
    print("=" * 78)
    print(f"  ||g_full - g_text||      {gap.g_error:.4f}")
    print(f"  ||a_full - a_text||      {gap.alpha_error:.4f}")
    print(f"  exact gap                {gap.gap:+.6f}")
    print(f"  Cauchy-Schwarz bound     {gap.bound:.6f}")
    print(f"  alignment rho            {gap.rho:+.4f}")
    print(f"  sqrt(n) |gap|            {adapt['scaled_gap']:.4f}")
    print(f"  verdict                  {adapt['verdict']}")
    print(
        "\n  Note how far the exact gap sits below its bound: that distance is\n"
        "  exactly what Proposition 1 is about. Two large errors that are not\n"
        "  aligned move the target very little."
    )

    # ---- Table 3: the sensitivity grid ------------------------------------
    print("\n" + "=" * 78)
    print("  Table 3 -- sensitivity of the full-model estimate")
    print("=" * 78)
    sa = full.sensitivity()
    grid = sa.grid(
        rhos=(1.0, 0.5, 0.2), r2s=(0.010, 0.025, 0.050),
        parameterization="rv",     # the convention that generates the paper's numbers
        tail=0.10,
    )
    show = grid[["rho", "r2", "p10", "lower", "theta", "upper", "p90"]]
    print(show.to_string(index=False, float_format=lambda v: f"{v: .3f}"))
    print(f"\n  Robustness value RV(0, 0.05) = {sa.robustness_value():.4f}")
    print(
        "\n  Read down a block: the sign is typically far more robust than the\n"
        "  magnitude. That asymmetry is the paper's headline empirical claim."
    )

    # ---- Aggregation weights ---------------------------------------------
    if comparison.aggregate is not None:
        print("\n" + "=" * 78)
        print("  Star aggregation weights")
        print("=" * 78)
        extra = comparison.aggregate.extra
        print(f"  Outcome head   {extra.get('Outcome weights', 'n/a')}")
        print(f"  Riesz head     {extra.get('Riesz weights', 'n/a')}")
        print(
            "\n  The two heads may select different candidates. That is expected:\n"
            "  they solve different prediction problems over the same space H."
        )

    print("\n" + "=" * 78)
    print("  Next steps")
    print("=" * 78)
    print(
        "  * Replace load_your_panel() with your data.\n"
        "  * Check pragdml.diagnostics.pooling_conditions() for your n and M.\n"
        "  * Draw the contour: pragdml._plots.plot_sensitivity_contour(sa).\n"
        "  * If you have the paper's Keepa extract, send it to the package author\n"
        "    and a replication script will be added."
    )


if __name__ == "__main__":
    main()
