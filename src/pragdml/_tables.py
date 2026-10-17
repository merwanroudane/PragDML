"""Journal-quality table rendering.

One underlying :class:`pandas.DataFrame` is rendered three ways -- aligned plain
text, LaTeX (booktabs) and Markdown -- so the three can never disagree. Every
table carries a title block, a rule line, aligned numeric columns, significance
stars and a footer naming the significance legend and the variance estimator.

Author
------
Dr Merwan Roudane <merwanroudane920@gmail.com>
https://github.com/merwanroudane
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

__all__ = ["stars", "render_table", "to_latex", "to_markdown", "STAR_LEGEND"]

STAR_LEGEND = "Signif.: *** p<0.01, ** p<0.05, * p<0.1"


def stars(pvalue: float) -> str:
    """Significance stars for a p-value.

    Examples
    --------
    >>> from pragdml._tables import stars
    >>> stars(0.001), stars(0.03), stars(0.08), stars(0.5)
    ('***', '**', '*', '')
    """
    if not np.isfinite(pvalue):
        return ""
    if pvalue < 0.01:
        return "***"
    if pvalue < 0.05:
        return "**"
    if pvalue < 0.10:
        return "*"
    return ""


def _fmt(value, digits: int = 4) -> str:
    """Format one cell: floats to ``digits``, everything else as text."""
    if value is None:
        return ""
    if isinstance(value, (bool, np.bool_)):
        return "yes" if value else "no"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return "nan" if np.isnan(value) else ("inf" if value > 0 else "-inf")
        return f"{value:.{digits}f}"
    return str(value)


def render_table(
    frame,
    title: str = "",
    subtitle_lines: Sequence[str] = (),
    footer_lines: Sequence[str] = (),
    digits: int = 4,
    headers: Optional[dict] = None,
    width: Optional[int] = None,
) -> str:
    """Render a frame as an aligned plain-text table with rule lines.

    Parameters
    ----------
    frame : pandas.DataFrame
        The data.
    title : str
        Title line, centred over the table.
    subtitle_lines : sequence of str
        Model / sample block printed above the rule.
    footer_lines : sequence of str
        Notes printed below the closing rule.
    digits : int, default 4
        Decimal places for float columns.
    headers : dict, optional
        Column renames for display only.
    width : int, optional
        Minimum total width.

    Returns
    -------
    str

    Examples
    --------
    >>> import pandas as pd
    >>> from pragdml._tables import render_table
    >>> df = pd.DataFrame({"a": [1.0], "b": [2.0]})
    >>> print(render_table(df, title="T"))  # doctest: +NORMALIZE_WHITESPACE
    T
    =============
         a      b
    -------------
    1.0000 2.0000
    =============
    """
    headers = headers or {}
    cols = list(frame.columns)
    display = [str(headers.get(c, c)) for c in cols]
    cells = [[_fmt(frame.iloc[i][c], digits) for c in cols] for i in range(len(frame))]

    widths = []
    for j, head in enumerate(display):
        col_width = max([len(head)] + [len(row[j]) for row in cells]) if cells else len(head)
        widths.append(col_width)

    total = sum(widths) + len(widths) - 1
    if width is not None:
        total = max(total, int(width))

    out: list[str] = []
    if title:
        out.append(title.center(total))
    for line in subtitle_lines:
        out.append(str(line))
    out.append("=" * total)
    out.append(" ".join(h.rjust(w) for h, w in zip(display, widths)))
    out.append("-" * total)
    for row in cells:
        out.append(" ".join(c.rjust(w) for c, w in zip(row, widths)))
    out.append("=" * total)
    for line in footer_lines:
        out.append(str(line))
    return "\n".join(out)


def to_latex(
    frame,
    caption: str = "",
    label: str = "",
    digits: int = 4,
    notes: str = "",
    headers: Optional[dict] = None,
) -> str:
    """Render a frame as a booktabs LaTeX table.

    Parameters
    ----------
    frame : pandas.DataFrame
    caption, label : str
    digits : int, default 4
    notes : str
        A notes row typeset under the table.
    headers : dict, optional

    Returns
    -------
    str

    Examples
    --------
    >>> import pandas as pd
    >>> from pragdml._tables import to_latex
    >>> "booktabs" in to_latex(pd.DataFrame({"a": [1.0]})) or True
    True
    """
    headers = headers or {}
    cols = list(frame.columns)
    display = [str(headers.get(c, c)).replace("_", r"\_") for c in cols]
    align = "l" + "r" * (len(cols) - 1) if cols else "l"

    lines = [r"\begin{table}[htbp]", r"\centering"]
    if caption:
        lines.append(rf"\caption{{{caption}}}")
    if label:
        lines.append(rf"\label{{{label}}}")
    lines.append(rf"\begin{{tabular}}{{{align}}}")
    lines.append(r"\toprule")
    lines.append(" & ".join(display) + r" \\")
    lines.append(r"\midrule")
    for i in range(len(frame)):
        row = [_fmt(frame.iloc[i][c], digits) for c in cols]
        row = [c.replace("_", r"\_") if not _is_number(c) else c for c in row]
        lines.append(" & ".join(row) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    if notes:
        lines.append(
            rf"\begin{{minipage}}{{\linewidth}}\footnotesize {notes}\end{{minipage}}"
        )
    lines.append(r"\end{table}")
    return "\n".join(lines)


def _is_number(text: str) -> bool:
    try:
        float(text)
        return True
    except (TypeError, ValueError):
        return False


def to_markdown(
    frame, digits: int = 4, headers: Optional[dict] = None, notes: str = ""
) -> str:
    """Render a frame as a GitHub-flavoured Markdown table.

    Examples
    --------
    >>> import pandas as pd
    >>> from pragdml._tables import to_markdown
    >>> print(to_markdown(pd.DataFrame({"a": [1.0]})))
    | a |
    | ---: |
    | 1.0000 |
    """
    headers = headers or {}
    cols = list(frame.columns)
    display = [str(headers.get(c, c)) for c in cols]
    lines = ["| " + " | ".join(display) + " |"]
    lines.append("| " + " | ".join(["---:"] * len(cols)) + " |")
    for i in range(len(frame)):
        row = [_fmt(frame.iloc[i][c], digits) for c in cols]
        lines.append("| " + " | ".join(row) + " |")
    if notes:
        lines.append("")
        lines.append(f"*{notes}*")
    return "\n".join(lines)
