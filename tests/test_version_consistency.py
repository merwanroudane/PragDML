"""The version number must agree everywhere it is written down.

`pragdml` states its version in six places: ``pyproject.toml``,
``__init__.py``, ``CITATION.cff``, ``CHANGELOG.md``, and the BibTeX block in
both ``README.md`` and ``docs/index.html``. A release bump that misses one of
them ships a package telling users to cite a version that is not the one they
installed -- which is exactly what happened between 0.1.0 and 0.1.1, where the
two BibTeX blocks were left behind and the stale citation reached the published
PyPI page.

These tests are cheap and they make that class of mistake impossible to repeat.

Author
------
Dr Merwan Roudane <merwanroudane920@gmail.com>
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

import pragdml

ROOT = Path(__file__).resolve().parents[1]


def _read(relative: str) -> str:
    path = ROOT / relative
    if not path.exists():
        pytest.skip(f"{relative} is not present (installed package, not a checkout)")
    return path.read_text(encoding="utf-8")


def test_pyproject_matches_dunder_version():
    """``pyproject.toml`` is what the wheel is built from."""
    text = _read("pyproject.toml")
    match = re.search(r'^version\s*=\s*"([^"]+)"', text, re.M)
    assert match, "no version field found in pyproject.toml"
    assert match.group(1) == pragdml.__version__


def test_citation_cff_matches_dunder_version():
    """``CITATION.cff`` is what GitHub's 'Cite this repository' button reads."""
    text = _read("CITATION.cff")
    match = re.search(r"^version:\s*(\S+)", text, re.M)
    assert match, "no version field found in CITATION.cff"
    assert match.group(1).strip("\"'") == pragdml.__version__


@pytest.mark.parametrize("relative", ["README.md", "docs/index.html"])
def test_bibtex_block_matches_dunder_version(relative):
    """The BibTeX a user copies must name the version they installed.

    The README's block is embedded in the published PyPI long description, so a
    stale value there is visible on the project page and cannot be corrected
    without a new release.
    """
    text = _read(relative)
    versions = re.findall(r"version\s*=\s*\{([^}]+)\}", text)
    assert versions, f"no BibTeX version field found in {relative}"
    for found in versions:
        assert found.strip() == pragdml.__version__, (
            f"{relative} cites {found.strip()!r} but the package is "
            f"{pragdml.__version__!r}"
        )


def test_changelog_documents_the_current_version():
    """The top release heading in the changelog is the version being shipped."""
    text = _read("CHANGELOG.md")
    headings = re.findall(r"^##\s*\[([^\]]+)\]", text, re.M)
    assert headings, "no release headings found in CHANGELOG.md"
    assert headings[0] == pragdml.__version__, (
        f"newest changelog entry is {headings[0]!r} but the package is "
        f"{pragdml.__version__!r}"
    )


def test_version_is_a_sane_release_number():
    """Guard against a placeholder or a development suffix reaching a release."""
    assert re.fullmatch(r"\d+\.\d+\.\d+", pragdml.__version__), (
        f"{pragdml.__version__!r} is not a plain X.Y.Z release version"
    )
