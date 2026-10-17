"""Shipped real datasets.

Author
------
Dr Merwan Roudane <merwanroudane920@gmail.com>
https://github.com/merwanroudane
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

__all__ = ["load_pension", "PENSION_DESCR", "PENSION_COVARIATES", "data_path"]

_HERE = Path(__file__).parent

#: The covariate set of [9] Sec. 5.1, in the order listed there.
PENSION_COVARIATES: list[str] = [
    "age", "inc", "fsize", "educ", "marr", "twoearn", "pira", "hown",
]

PENSION_DESCR = """401(k) eligibility and net financial assets (SIPP 1991)
========================================================

The canonical dataset of the double machine learning literature, and the
empirical application of

    V. Chernozhukov, C. Cinelli, W. K. Newey, A. Sharma and V. Syrgkanis,
    "Long Story Short: Omitted Variable Bias in Causal Machine Learning",
    Review of Economics and Statistics (2026), doi:10.1162/REST.a.1705,

Section 5, whose published numbers this package uses as acceptance targets.

Observations
    9,915 households.

Source
    1991 Survey of Income and Program Participation (SIPP), US Census Bureau.
    Used by Poterba, Venti & Wise (1994, 1995) and by Chernozhukov, Chetverikov,
    Demirer, Duflo, Hansen, Newey & Robins (2018). Retrieved through
    ``doubleml.datasets.fetch_401K`` (MIT licensed). US Government work; see
    ``SOURCES.md`` for the full provenance and licence note.

The causal question
    401(k) plans are employer-sponsored tax-deferred savings vehicles. Eligibility
    is not randomised -- it depends on whom you work for -- so a naive comparison
    is confounded. Poterba et al. argued that, at the time these plans appeared,
    job choice was driven by salary rather than by 401(k) availability, so
    eligibility may be treated as ignorable conditional on income and related
    covariates. [9] asks how strong unobserved *firm* characteristics would have
    to be to overturn that conclusion.

Columns
    net_tfa   net total financial assets, in dollars (the outcome Y). Defined as
              IRA + 401(k) balances + chequing + savings bonds + other
              interest-earning accounts and assets + stocks + mutual funds, less
              non-mortgage debt.
    e401      1 if eligible for a 401(k) plan (the treatment D).
    p401      1 if participating in a 401(k) plan. Note: participation, not
              eligibility. It is a *downstream* choice, so using it as the
              treatment changes the estimand and reintroduces selection.
    age       age in years (25-64).
    inc       income in dollars.
    fsize     family size.
    educ      years of education.
    marr      1 if married.
    twoearn   1 if two-earner household.
    pira      1 if participating in an IRA.
    hown      1 if home owner.
    db        1 if covered by a defined-benefit pension.
    nifa      non-401(k) financial assets, in dollars.
    tw        total wealth, in dollars.

Published reference values
    Naive treated-minus-control difference in net_tfa: $19,559
    ([9] Sec. 5.1; reproduced exactly by this copy of the data).
    ATE under conditional ignorability: about $9,000 in a partially linear
    specification and about $8,000 in a nonparametric one, with random forests,
    5-fold cross-fitting repeated 5 times and median aggregation ([9] Table 1 and
    footnote 11).
    Robustness values RV(theta=0, a=0.05): 5.4% and 4.5% respectively.

Suggested use
    Y = net_tfa, D = e401, X = PENSION_COVARIATES.
"""


def data_path(name: str) -> Path:
    """Absolute path of a shipped data file."""
    path = _HERE / name
    if not path.exists():
        raise FileNotFoundError(
            f"{name} is not shipped with pragdml; available: "
            f"{sorted(p.name for p in _HERE.glob('*.csv.gz'))}"
        )
    return path


def load_pension(as_frame: bool = True, return_X_y: bool = False):
    """Load the 401(k) / SIPP 1991 dataset.

    Parameters
    ----------
    as_frame : bool, default True
        Return a :class:`pandas.DataFrame`. ``False`` returns a dict of arrays.
    return_X_y : bool, default False
        Return ``(Y, D, X)`` instead, with ``X`` restricted to
        :data:`PENSION_COVARIATES` -- the covariate set of [9] Sec. 5.1 -- so the
        shipped example and the published numbers line up.

    Returns
    -------
    pandas.DataFrame or dict or tuple
        See the parameters. The description is in :data:`PENSION_DESCR` and on the
        returned frame's ``attrs["DESCR"]``.

    Examples
    --------
    >>> from pragdml.datasets import load_pension
    >>> df = load_pension()
    >>> df.shape
    (9915, 14)
    >>> Y, D, X = load_pension(return_X_y=True)
    >>> X.shape
    (9915, 8)
    >>> round(float(Y[D == 1].mean() - Y[D == 0].mean()), 1)   # [9] reports 19559
    19559.3
    """
    import pandas as pd

    frame = pd.read_csv(data_path("pension.csv.gz"))
    if return_X_y:
        y = frame["net_tfa"].to_numpy(dtype=np.float64)
        d = frame["e401"].to_numpy(dtype=np.float64)
        x = frame[PENSION_COVARIATES].to_numpy(dtype=np.float64)
        return y, d, x
    if as_frame:
        frame.attrs["DESCR"] = PENSION_DESCR
        return frame
    return {
        "data": frame.to_numpy(dtype=np.float64),
        "columns": list(frame.columns),
        "DESCR": PENSION_DESCR,
    }
