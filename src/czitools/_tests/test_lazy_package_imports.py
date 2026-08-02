"""Regression tests for the lazy top-level package API."""

import subprocess
import sys


def test_top_level_package_does_not_eagerly_import_subpackages() -> None:
    code = """
import sys
import czitools

subpackages = {"metadata_tools", "read_tools", "utils", "visu_tools"}
assert subpackages <= set(dir(czitools))
assert all(f"czitools.{name}" not in sys.modules for name in subpackages)
assert "matplotlib" not in sys.modules
assert "dask" not in sys.modules

from czitools import metadata_tools

assert metadata_tools is czitools.metadata_tools
assert "czitools.read_tools" not in sys.modules
assert "czitools.visu_tools" not in sys.modules
assert "matplotlib" not in sys.modules
"""

    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
