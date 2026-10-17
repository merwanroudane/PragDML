# Data provenance

## `pension.csv.gz` -- 401(k) eligibility and net financial assets

| | |
|---|---|
| **Rows / columns** | 9,915 x 14 |
| **Original source** | 1991 Survey of Income and Program Participation (SIPP), US Census Bureau |
| **Retrieved via** | `doubleml.datasets.fetch_401K()` (DoubleML 0.11.4, MIT licence) |
| **Retrieved on** | 17 October 2026 |
| **File size** | ~143 KB gzipped |
| **Transformations applied** | None to the values. Columns reordered so the outcome and treatment come first; no rows dropped, no imputation, no rescaling. |

### Licence and redistribution

The underlying data are a **work of the US federal government** (US Census Bureau,
SIPP) and are therefore not subject to domestic copyright. The extract has been
redistributed for years as part of open-source packages -- `DoubleML` for Python
(MIT) and `hdm` for R -- and accompanies the replication material of
Chernozhukov et al. (2018). Redistributing it inside `pragdml` under MIT is
consistent with that practice.

### Literature using exactly this extract

- J. Poterba, S. Venti and D. Wise (1994, 1995) -- the original analyses.
- V. Chernozhukov, D. Chetverikov, M. Demirer, E. Duflo, C. Hansen, W. Newey and
  J. Robins (2018), *The Econometrics Journal* 21(1):C1-C68 -- the DML benchmark
  application.
- V. Chernozhukov, C. Cinelli, W. K. Newey, A. Sharma and V. Syrgkanis (2026),
  *Review of Economics and Statistics*, doi:10.1162/REST.a.1705, Section 5 -- the
  sensitivity analysis this package reproduces.

### Verified reference value

The naive treated-minus-control difference in `net_tfa` is **$19,559.3**, against
the "$19,559" reported in Chernozhukov et al. (2026) Section 5.1. This confirms the
shipped copy is the same extract those papers use, and it is asserted in
`tests/test_acceptance_401k.py`.

---

## Not shipped: the paper's own multimodal data

The target paper (arXiv:2610.01935) uses product-level Amazon data scraped from
**Keepa.com** -- price histories plus product text, images and tabular attributes
for clothing and shoes. That data is neither public nor redistributable, and
version 2 of the paper announces no replication archive.

Consequently **Tables 1-3 of that paper cannot be reproduced here.** Their reported
values are recorded in `tests/fixtures/paper_targets.json` purely as documentation,
and the test suite checks only the algebra that produced them -- that the Eq. (50)
factorization and the Eq. (51) endpoints are internally consistent with the
reported triple -- never the numbers themselves.

`examples/multimodal_template.py` runs the paper's exact specification (first
differences, seven modality combinations, average partial derivative, star
aggregation, sensitivity grid) on any panel the user supplies in that shape.
