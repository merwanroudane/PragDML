"""Journal-quality figures.

One shared style: serif-leaning sans family, no top/right spines, light grid,
one- or two-column journal widths, and ``"Parula"`` as the default colour scale
for every contour, heatmap and surface. Every function returns its
``Figure``/``Axes`` so the caller can restyle it.

Author
------
Dr Merwan Roudane <merwanroudane920@gmail.com>
https://github.com/merwanroudane
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from ._colors import parula_cmap, parula_colors

__all__ = [
    "set_style",
    "plot_information_sets",
    "plot_sensitivity_contour",
    "plot_sensitivity_grid",
    "plot_influence",
    "plot_aggregation_weights",
    "plot_information_loss",
    "save_figure",
]

#: One- and two-column journal figure sizes, in inches.
FIGSIZE_ONE_COLUMN = (4.6, 3.4)
FIGSIZE_TWO_COLUMN = (7.2, 4.4)

_ACCENT = "#2166AC"
_ACCENT_2 = "#B2182B"
_GREY = "#4D4D4D"


def set_style(scale: float = 1.0) -> None:
    """Apply the package's figure style to the active matplotlib rcParams.

    Called automatically by every plotting function here, so you only need it
    when drawing your own figures and want them to match.

    Examples
    --------
    >>> from pragdml._plots import set_style
    >>> set_style()
    """
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "font.family": "sans-serif",
            "font.sans-serif": ["Inter", "Segoe UI", "DejaVu Sans", "Arial"],
            "font.size": 10 * scale,
            "axes.titlesize": 11 * scale,
            "axes.labelsize": 10 * scale,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": _GREY,
            "axes.labelcolor": "#1A1A1A",
            "axes.grid": True,
            "grid.color": "#DDDDDD",
            "grid.linewidth": 0.6,
            "grid.alpha": 0.9,
            "xtick.color": _GREY,
            "ytick.color": _GREY,
            "xtick.labelsize": 9 * scale,
            "ytick.labelsize": 9 * scale,
            "legend.frameon": False,
            "legend.fontsize": 9 * scale,
            "lines.linewidth": 1.6,
        }
    )


def save_figure(fig, path: str, formats: Sequence[str] = ("pdf", "png")) -> list[str]:
    """Save a figure as vector **and** raster, the way a journal wants it.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    path : str
        Path without extension.
    formats : sequence of str, default ``("pdf", "png")``

    Returns
    -------
    list of str
        The files written.
    """
    written = []
    for ext in formats:
        name = f"{path}.{ext}"
        fig.savefig(name, format=ext)
        written.append(name)
    return written


def plot_information_sets(
    comparison,
    ax=None,
    level: float = 0.95,
    figsize: tuple = FIGSIZE_ONE_COLUMN,
    title: str = "Estimates by information set",
    xlabel: str = "estimate",
):
    """Forest plot of the paper's Table 1.

    One row per information set with its confidence interval, the aggregate marked
    separately, and a reference line at zero. Because each row is a different
    population target, the figure is a display of *stability across
    representations*, not of agreement about one number.

    Parameters
    ----------
    comparison : InformationSetComparison
    ax : matplotlib.axes.Axes, optional
    level : float, default 0.95
    figsize : tuple
    title, xlabel : str

    Returns
    -------
    matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt

    set_style()
    frame = comparison.to_frame(level)
    if ax is None:
        height = max(figsize[1], 0.42 * len(frame) + 1.2)
        _, ax = plt.subplots(figsize=(figsize[0], height))

    is_agg = frame["information_set"] == comparison.aggregate_label
    positions = np.arange(len(frame))[::-1]

    for pos, (_, row), agg in zip(positions, frame.iterrows(), is_agg):
        colour = _ACCENT_2 if agg else _ACCENT
        ax.plot(
            [row["ci_lower"], row["ci_upper"]], [pos, pos],
            color=colour, linewidth=2.0, solid_capstyle="round", alpha=0.85,
        )
        ax.plot(
            row["estimate"], pos, marker="D" if agg else "o",
            color=colour, markersize=6 if agg else 5,
            markeredgecolor="white", markeredgewidth=0.8, zorder=3,
        )

    ax.axvline(0.0, color=_GREY, linewidth=0.9, linestyle="--", alpha=0.8)
    ax.set_yticks(positions)
    ax.set_yticklabels(frame["information_set"])
    ax.set_xlabel(xlabel)
    ax.set_title(title, loc="left")
    ax.grid(axis="y", visible=False)
    ax.margins(y=0.08)
    return ax


def plot_sensitivity_contour(
    analysis,
    ax=None,
    cy2_max: float = 0.3,
    cd_share_max: float = 0.3,
    n_grid: int = 60,
    rho: float = 1.0,
    level: float = 0.95,
    bound: str = "lower",
    method: str = "one_sided",
    benchmarks: Optional[Sequence] = None,
    figsize: tuple = FIGSIZE_ONE_COLUMN,
    colorscale: str = "Parula",
    show_robustness_value: bool = True,
):
    """The sensitivity contour plot of [9] Sec. 5.2.3.

    The canonical display: the horizontal axis is the share of representer
    variation omitted information could explain (``1 - R^2_{alpha_s}``), the
    vertical axis is ``C_Y^2``, and the contours give a confidence limit for
    ``theta^*``. The **zero contour** is drawn in red: anywhere beyond it, the
    sign of the result is no longer safe. The robustness value is marked where the
    45-degree line crosses that contour, which is exactly [9]'s construction.

    Parameters
    ----------
    analysis : SensitivityAnalysis
    ax : matplotlib.axes.Axes, optional
    cy2_max, cd_share_max : float, default 0.3
    n_grid : int, default 60
    rho : float, default 1.0
        Alignment bound. [9]'s figures show both ``1`` (worst case) and ``0.5``.
    level : float, default 0.95
    bound : {"lower", "upper", "point_lower", "point_upper"}, default ``"lower"``
    method : str, default ``"one_sided"``
    benchmarks : sequence of BenchmarkResult, optional
        Reference points from :func:`~pragdml.sensitivity.benchmark_covariate`,
        annotated on the plot as in [9] Fig. 3 ("1 x Income", "1/4 x Income").
    figsize : tuple
    colorscale : str, default ``"Parula"``
    show_robustness_value : bool, default True

    Returns
    -------
    matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt

    set_style()
    data = analysis.contour_data(
        cy2_max=cy2_max, cd_share_max=cd_share_max, n_grid=n_grid,
        rho=rho, level=level, bound=bound, method=method,
    )
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    xx, yy = np.meshgrid(data["cd_share"], data["cy2"])
    values = data["values"]

    filled = ax.contourf(
        xx, yy, values, levels=18, cmap=parula_cmap(colorscale), alpha=0.92
    )
    lines = ax.contour(
        xx, yy, values, levels=10, colors="white", linewidths=0.6, alpha=0.75
    )
    ax.clabel(lines, inline=True, fontsize=7, fmt="%.2f")

    if float(values.min()) < 0.0 < float(values.max()):
        zero = ax.contour(
            xx, yy, values, levels=[0.0], colors=[_ACCENT_2],
            linewidths=2.0, linestyles="--",
        )
        ax.clabel(zero, inline=True, fontsize=8, fmt="0")

    if show_robustness_value:
        rv = analysis.robustness_value(level=level, rho=rho)
        if np.isfinite(rv) and 0.0 < rv <= max(cy2_max, cd_share_max):
            top = min(cy2_max, cd_share_max)
            ax.plot([0, top], [0, top], color="white", linewidth=0.9, alpha=0.7)
            ax.plot(
                rv, rv, marker="*", color=_ACCENT_2, markersize=13,
                markeredgecolor="white", markeredgewidth=0.8, zorder=5,
            )
            ax.annotate(
                f"RV = {rv:.3f}", xy=(rv, rv), xytext=(6, 6),
                textcoords="offset points", fontsize=8, color="white",
                fontweight="bold",
            )

    for bm in benchmarks or []:
        x, y = float(bm.cd_share), float(bm.cy2)
        if x <= cd_share_max and y <= cy2_max:
            ax.plot(
                x, y, marker="^", color="white", markersize=7,
                markeredgecolor=_GREY, markeredgewidth=0.9, zorder=5,
            )
            label = bm.name if bm.k_y == 1.0 else f"{bm.k_y:g}x {bm.name}"
            ax.annotate(
                label, xy=(x, y), xytext=(6, -10), textcoords="offset points",
                fontsize=8, color="white", fontweight="bold",
            )

    cbar = ax.figure.colorbar(filled, ax=ax, pad=0.02)
    kind = "point bound" if bound.startswith("point") else f"{level:.0%} bound"
    cbar.set_label(f"{bound.replace('point_', '')} {kind} on theta*")

    ax.set_xlabel(r"omitted share of representer variation  $1-R^2_{\alpha_s}$")
    ax.set_ylabel(r"omitted share of outcome variation  $C_Y^2$")
    ax.set_title(
        rf"Sensitivity of $\theta^*$   ($\rho$ = {rho:g})", loc="left"
    )
    ax.grid(visible=False)
    return ax


def plot_sensitivity_grid(
    analysis,
    rhos: Sequence[float] = (1.0, 0.5, 0.2),
    r2s: Sequence[float] = (0.010, 0.025, 0.050),
    ax=None,
    level: float = 0.95,
    figsize: tuple = FIGSIZE_TWO_COLUMN,
    parameterization: str = "cy_cd",
):
    """The paper's Table 3 as a figure: nested intervals by scenario.

    Each row is one ``(rho, r^2)`` scenario; the thick band is the identified
    region of Eq. (51) and the thin whiskers add sampling uncertainty. Reading
    down a block shows what the paper emphasises -- the sign is usually far more
    robust than the magnitude.

    Returns
    -------
    matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt

    set_style()
    frame = analysis.grid(
        rhos=rhos, r2s=r2s, level=level, parameterization=parameterization
    )
    if ax is None:
        height = max(figsize[1], 0.34 * len(frame) + 1.4)
        _, ax = plt.subplots(figsize=(figsize[0], height))

    palette = parula_colors(max(len(rhos), 2))
    colour_of = {float(r): palette[i] for i, r in enumerate(rhos)}
    positions = np.arange(len(frame))[::-1]
    labels = []

    for pos, (_, row) in zip(positions, frame.iterrows()):
        colour = colour_of.get(float(row["rho"]), _ACCENT)
        ax.plot(
            [row["ci_lower"], row["ci_upper"]], [pos, pos],
            color=colour, linewidth=1.0, alpha=0.55,
        )
        ax.plot(
            [row["lower"], row["upper"]], [pos, pos],
            color=colour, linewidth=3.4, solid_capstyle="butt", alpha=0.95,
        )
        labels.append(rf"$\rho$={row['rho']:g}, $r^2$={row['r2']:.3f}")

    ax.plot(
        [frame["theta"].iloc[0]] * len(frame), positions,
        marker="|", linestyle="none", color="black", markersize=9, zorder=4,
    )
    ax.axvline(0.0, color=_ACCENT_2, linewidth=1.0, linestyle="--", alpha=0.9)
    ax.set_yticks(positions)
    ax.set_yticklabels(labels)
    ax.set_xlabel(r"$\theta^*$")
    ax.set_title("Sensitivity region by scenario", loc="left")
    ax.grid(axis="y", visible=False)
    ax.margins(y=0.04)
    return ax


def plot_influence(result, ax=None, bins: int = 40, figsize: tuple = FIGSIZE_ONE_COLUMN):
    """Histogram of the cross-fitted influence values, with a normal overlay.

    The Wald interval of Eq. (16) rests on the CLT for these values, so heavy
    tails or visible skew are a warning that the asymptotics are not yet reliable
    -- most often caused by a representer with extreme values.

    Returns
    -------
    matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt
    from scipy import stats

    set_style()
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    infl = np.asarray(result.crossfit_.influence, dtype=np.float64)
    ax.hist(
        infl, bins=bins, density=True, color=_ACCENT, alpha=0.75,
        edgecolor="white", linewidth=0.5,
    )
    grid = np.linspace(infl.min(), infl.max(), 300)
    ax.plot(
        grid, stats.norm.pdf(grid, 0.0, infl.std()),
        color=_ACCENT_2, linewidth=1.6, label="normal reference",
    )
    ax.set_xlabel("cross-fitted influence value")
    ax.set_ylabel("density")
    ax.set_title("Score distribution (Eq. 8)", loc="left")
    ax.legend()
    return ax


def plot_aggregation_weights(
    outcome_aggregation,
    riesz_aggregation=None,
    names: Optional[Sequence[str]] = None,
    ax=None,
    figsize: tuple = FIGSIZE_ONE_COLUMN,
):
    """Side-by-side aggregation weights for the two heads.

    The figure that makes the paper's Sec. 5.3 lesson visible: the outcome
    regression and the Riesz representer solve different prediction problems and
    may put their weight on different candidates. Seeing the two sets of bars
    disagree is the expected behaviour, not a bug.

    Returns
    -------
    matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt

    set_style()
    w_g = np.asarray(outcome_aggregation.weights, dtype=np.float64)
    M = w_g.shape[0]
    labels = list(names) if names is not None else [f"cand {i}" for i in range(M)]
    if ax is None:
        _, ax = plt.subplots(figsize=(figsize[0], max(figsize[1], 0.34 * M + 1.3)))

    pos = np.arange(M)[::-1]
    if riesz_aggregation is None:
        ax.barh(pos, w_g, color=_ACCENT, alpha=0.9, height=0.6)
    else:
        w_a = np.asarray(riesz_aggregation.weights, dtype=np.float64)
        ax.barh(pos + 0.19, w_g, height=0.36, color=_ACCENT,
                alpha=0.9, label="outcome  $g$")
        ax.barh(pos - 0.19, w_a, height=0.36, color=_ACCENT_2,
                alpha=0.9, label=r"representer  $\alpha$")
        ax.legend(loc="lower right")
    ax.set_yticks(pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel("aggregation weight")
    ax.set_xlim(0, 1.02)
    ax.set_title("Aggregation weights by head", loc="left")
    ax.grid(axis="y", visible=False)
    return ax


def plot_information_loss(
    omitted,
    n: int,
    ax=None,
    figsize: tuple = FIGSIZE_ONE_COLUMN,
    n_grid: Sequence[int] = (),
):
    """Visualise Proposition 1 and Definition 1 (extension E6).

    Shows the two representation errors as the sides of a rectangle whose area is
    the Cauchy-Schwarz bound of Eq. (7), with the **actual** gap marked inside it.
    The distance between the two is the practical content of Proposition 1: errors
    can be large and still move the target barely, if they are not aligned.

    Parameters
    ----------
    omitted : OmittedInformation
        From :func:`~pragdml.diagnostics.omitted_information_gap`.
    n : int
        Sample size, used to draw the ``sqrt(n)``-scaled comparison.
    ax : matplotlib.axes.Axes, optional
    figsize : tuple
    n_grid : sequence of int, optional
        Extra sample sizes at which to mark the scaled gap, to show how the
        Definition 1 requirement tightens with ``n``.

    Returns
    -------
    matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    set_style()
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    gx, ax_err = float(omitted.g_error), float(omitted.alpha_error)
    ax.add_patch(
        Rectangle(
            (0, 0), gx, ax_err, facecolor=_ACCENT, alpha=0.18,
            edgecolor=_ACCENT, linewidth=1.4,
        )
    )
    frac = abs(omitted.rho)
    ax.add_patch(
        Rectangle(
            (0, 0), gx, ax_err * frac, facecolor=_ACCENT_2, alpha=0.35,
            edgecolor=_ACCENT_2, linewidth=1.4,
        )
    )
    ax.annotate(
        f"Cauchy-Schwarz bound\n{omitted.bound:.4g}",
        xy=(gx * 0.52, ax_err * 0.82), fontsize=8, ha="center", color=_ACCENT,
    )
    ax.annotate(
        f"actual |gap|\n{abs(omitted.gap):.4g}   ($\\rho$={omitted.rho:.3f})",
        xy=(gx * 0.52, ax_err * frac * 0.45), fontsize=8, ha="center",
        color=_ACCENT_2,
    )
    scaled = omitted.scaled_gap(n)
    extra = "".join(
        f"\n  n={m}:  {omitted.scaled_gap(m):.3f}" for m in n_grid
    )
    ax.set_title(
        f"Omitted information (Prop. 1)\n"
        rf"$\sqrt{{n}}\,|$gap$|$ = {scaled:.3f} at n={n}{extra}",
        loc="left", fontsize=9,
    )
    ax.set_xlabel(r"$\|g^*-g_\lambda\|_{P,2}$")
    ax.set_ylabel(r"$\|\alpha^*-\alpha_\lambda\|_{P,2}$")
    ax.set_xlim(0, gx * 1.12 if gx > 0 else 1)
    ax.set_ylim(0, ax_err * 1.12 if ax_err > 0 else 1)
    return ax
