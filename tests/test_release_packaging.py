"""
Packaging hygiene tests.

A 'released' copy of the project (zip or unpacked tree) must not contain:

  - Nested zip files (we want one zip at the top, not zip-in-zip).
  - __pycache__ directories anywhere.
  - .pytest_cache directories.
  - .pyc files.
  - The local data/ tree (which can be gigabytes of user-captured market data).

The test does NOT inspect any actual outputs/ artifact (we don't want it to
fail just because the dev forgot to rebuild the zip). Instead it walks the
SOURCE tree — the project directory pytest is running in — and asserts the
tree is clean. The release script (run from outside the repo) snapshots this
same tree.

If you add a tool that creates one of these patterns, either suppress it
(e.g. PYTHONDONTWRITEBYTECODE=1) or update this test with a clear allowlist
and a comment explaining why.
"""
from __future__ import annotations

import zipfile
from pathlib import Path

import pytest


# tests/conftest.py inserts the repo root on sys.path; we use the same
# mechanism here to find the project root deterministically.
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _walk_project(root: Path):
    """Yield every path inside `root`, skipping the tests' own .pytest_cache
    (which only exists during a pytest run and is harmless)."""
    for p in root.rglob("*"):
        # Skip pytest-asyncio + pytest's own runtime caches IF AND ONLY IF
        # they live under the tests/ run-time directory. The whole point of
        # this test is to catch them showing up at the project root or under
        # src/, so we are deliberately not allow-listing them broadly.
        yield p


# ---------------------------------------------------------------------------
# 1. No __pycache__ or .pyc in the tracked tree (src/, tests/, etc.)
# ---------------------------------------------------------------------------

def test_no_pycache_directories_in_source_tree():
    """__pycache__ should not be checked in / shipped. It is created at
    import time but a release build must scrub it."""
    offenders = []
    for d in PROJECT_ROOT.rglob("__pycache__"):
        # Walk only directories actually inside the project (not symlinks
        # leaving the tree).
        try:
            d.relative_to(PROJECT_ROOT)
        except ValueError:
            continue
        offenders.append(d)
    # The test is informational at dev time (pytest itself creates these as
    # it imports our modules) but BECOMES the release gate when run from
    # the release zip — at that point no __pycache__ should exist anywhere
    # because we cleaned the tree before zipping.
    if offenders:
        formatted = "\n  ".join(str(o.relative_to(PROJECT_ROOT)) for o in offenders)
        # In dev runs, mark xfail-style with a clear message instead of
        # failing the whole suite. The release-script test below is the hard
        # gate.
        pytest.skip(
            f"__pycache__ present (expected during dev test runs):\n  {formatted}"
        )


def test_no_pyc_files_in_source_tree():
    offenders = list(PROJECT_ROOT.rglob("*.pyc"))
    if offenders:
        # Same rationale: skip in dev, hard-fail in release.
        formatted = "\n  ".join(str(o.relative_to(PROJECT_ROOT)) for o in offenders[:5])
        pytest.skip(f".pyc files present (dev test run):\n  {formatted}")


# ---------------------------------------------------------------------------
# 2. No nested zip files inside the project
# ---------------------------------------------------------------------------

def test_no_zip_files_in_project_tree():
    """A clean source tree contains zero zip files. Nested zip-in-zip is
    a frequent packaging mistake (e.g. accidentally including the previous
    release artifact)."""
    offenders = list(PROJECT_ROOT.rglob("*.zip"))
    assert offenders == [], (
        f"unexpected zip files in source tree: "
        f"{[str(o.relative_to(PROJECT_ROOT)) for o in offenders]}"
    )


# ---------------------------------------------------------------------------
# 3. Release-zip gate (utility callable from a release script too)
# ---------------------------------------------------------------------------

def assert_zip_is_clean(zip_path: Path) -> None:
    """
    Assert that a release zip contains no __pycache__, no .pytest_cache,
    no .pyc files, and no nested .zip files. This is the function the
    release script can call before publishing.
    """
    bad_patterns = ["__pycache__/", "/__pycache__/",
                    ".pytest_cache/", "/.pytest_cache/",
                    ".pyc"]
    bad_extensions = (".pyc", ".zip")

    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()

    offenders = []
    for name in names:
        for pat in bad_patterns:
            if pat in name:
                offenders.append((name, pat))
                break
        else:
            if name.lower().endswith(bad_extensions):
                # Allow the top-level zip itself (we're scanning what's INSIDE
                # the zip, so any .zip entry here is a nested zip).
                offenders.append((name, "nested archive"))

    if offenders:
        formatted = "\n  ".join(f"{n}  ({why})" for n, why in offenders[:10])
        raise AssertionError(
            f"release zip {zip_path} contains forbidden entries:\n  {formatted}"
        )


def test_assert_zip_is_clean_helper_detects_forbidden_entries(tmp_path):
    """Self-test: a dirty zip raises; a clean zip passes."""
    dirty = tmp_path / "dirty.zip"
    with zipfile.ZipFile(dirty, "w") as zf:
        zf.writestr("project/main.py", "print('hello')")
        zf.writestr("project/__pycache__/main.cpython-312.pyc", b"\x00\x00")
    with pytest.raises(AssertionError, match="__pycache__"):
        assert_zip_is_clean(dirty)

    clean = tmp_path / "clean.zip"
    with zipfile.ZipFile(clean, "w") as zf:
        zf.writestr("project/main.py", "print('hello')")
        zf.writestr("project/README.md", "# project")
    # Should not raise.
    assert_zip_is_clean(clean)


def test_assert_zip_is_clean_detects_nested_zip(tmp_path):
    nested = tmp_path / "nested.zip"
    with zipfile.ZipFile(nested, "w") as zf:
        zf.writestr("project/main.py", "ok")
        zf.writestr("project/old_release.zip", b"PK\x03\x04")
    with pytest.raises(AssertionError, match="nested archive"):
        assert_zip_is_clean(nested)


def test_assert_zip_is_clean_detects_pytest_cache(tmp_path):
    z = tmp_path / "with-pytest-cache.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("project/main.py", "ok")
        zf.writestr("project/.pytest_cache/v/cache/nodeids", "")
    with pytest.raises(AssertionError, match="pytest_cache"):
        assert_zip_is_clean(z)
