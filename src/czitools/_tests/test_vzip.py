"""GUI-independent CZI reference archive export tests."""

import builtins
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from czitools.export_tools import create_czi_vzip


@pytest.fixture
def source(tmp_path):
    """Provide a local source for validation and mocked refkv calls."""
    path = tmp_path / "sample.czi"
    path.write_bytes(b"ZISRAWFILE\x00\x00\x00\x00\x00\x00mock CZI")
    return path


@pytest.fixture
def fake_refkv(monkeypatch):
    """Avoid requiring optional dependencies for boundary tests."""
    calls = []

    def save(path):
        Path(path).write_bytes(b"archive")

    def zarrify(path):
        calls.append(path)
        return SimpleNamespace(save=save)

    module = SimpleNamespace(zarrify=zarrify)
    monkeypatch.setitem(sys.modules, "refkv.zarrify", module)
    return calls, module


@pytest.mark.parametrize("custom", [False, True])
def test_create_destination(source, fake_refkv, custom):
    """Default and explicit outputs return absolute paths and leave no temp files."""
    output = source.parent / "custom.vzip" if custom else source.with_suffix(".vzip")
    result = create_czi_vzip(str(source), output if custom else None)
    assert result == output
    assert result.read_bytes() == b"archive"
    assert fake_refkv[0] == [str(source.resolve())]
    assert source.read_bytes() == b"ZISRAWFILE\x00\x00\x00\x00\x00\x00mock CZI"
    assert sorted(path.name for path in source.parent.iterdir()) == sorted([source.name, output.name])


def test_existing_output_is_protected(source, fake_refkv):
    """Refuse overwrite before invoking refkv."""
    output = source.with_suffix(".vzip")
    output.write_bytes(b"existing")
    with pytest.raises(FileExistsError):
        create_czi_vzip(source)
    assert output.read_bytes() == b"existing"
    assert not fake_refkv[0]
    assert create_czi_vzip(source, overwrite=True).read_bytes() == b"archive"


@pytest.mark.parametrize("overwrite", [False, True])
def test_failure_cleans_temporary_output(source, fake_refkv, overwrite):
    """A save failure leaves neither partial archives nor damaged old output."""
    output = source.with_suffix(".vzip")
    if overwrite:
        output.write_bytes(b"existing")

    def fail(path):
        Path(path).write_bytes(b"partial")
        raise RuntimeError("unsupported layout")

    fake_refkv[1].zarrify = lambda path: SimpleNamespace(save=fail)
    with pytest.raises(RuntimeError, match="unsupported layout"):
        create_czi_vzip(source, overwrite=overwrite)
    assert list(source.parent.glob(".*.vzip")) == []
    if overwrite:
        assert output.read_bytes() == b"existing"
    else:
        assert not output.exists()


def test_input_validation(source):
    """Reject missing sources, directories, invalid suffixes and missing parents."""
    with pytest.raises(FileNotFoundError):
        create_czi_vzip(source.parent / "missing.czi")
    with pytest.raises(ValueError, match=".czi"):
        create_czi_vzip(source.with_suffix(".tif"))
    directory = source.parent / "directory.czi"
    directory.mkdir()
    with pytest.raises(FileNotFoundError):
        create_czi_vzip(directory)
    with pytest.raises(ValueError, match=".vzip"):
        create_czi_vzip(source, source)
    with pytest.raises(FileNotFoundError, match="Destination directory"):
        create_czi_vzip(source, source.parent / "missing" / "output.vzip")


def test_wrong_file_content_is_rejected(source, fake_refkv):
    """A different format renamed to .czi must not be silently exported."""
    source.write_bytes(b"not a CZI")
    with pytest.raises(ValueError, match="CZI file header"):
        create_czi_vzip(source)
    assert not fake_refkv[0]


def test_concurrent_output_is_not_overwritten(source, fake_refkv):
    """Recheck non-overwrite protection when publishing, not only before export."""
    output = source.with_suffix(".vzip")

    def save(path):
        Path(path).write_bytes(b"archive")
        output.write_bytes(b"concurrent archive")

    fake_refkv[1].zarrify = lambda path: SimpleNamespace(save=save)
    with pytest.raises(FileExistsError):
        create_czi_vzip(source)
    assert output.read_bytes() == b"concurrent archive"
    assert list(source.parent.glob(".*.vzip")) == []


def test_public_import_needs_no_optional_packages():
    """Importing the public function must not load refkv, Qt or other exporters."""
    code = """
import sys
from czitools.export_tools import create_czi_vzip
assert callable(create_czi_vzip)
for name in ('refkv', 'napari', 'qtpy', 'PySide6', 'ngff_zarr', 'ome_zarr'):
    assert name not in sys.modules, name
"""
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)


def test_missing_dependency_has_install_hint(source, monkeypatch):
    """A missing refkv installation produces actionable optional-extra guidance."""
    original_import = builtins.__import__

    def import_without_refkv(name, *args, **kwargs):
        if name == "refkv.zarrify":
            raise ModuleNotFoundError("No module named refkv", name="refkv")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_refkv)
    with pytest.raises(ImportError, match=r"czitools\[vzip\]"):
        create_czi_vzip(source)
    assert not source.with_suffix(".vzip").exists()


def test_real_czi_roundtrip(tmp_path):
    """Reopen an archive independently and compare source-backed pixel reads."""
    pytest.importorskip("refkv.zarr")
    import numpy as np
    import refkv.zarr
    from refkv.zarrify import zarrify
    import zarr

    source = Path(__file__).resolve().parents[3] / "data" / "CellDivision_T10_Z15_CH2_DCV_small.czi"
    if not source.is_file():
        pytest.skip("Local CZI test data is unavailable")
    output = create_czi_vzip(source, tmp_path / "image.vzip")
    reopened = zarr.open_group(refkv.zarr.open(str(output)), mode="r")
    expected = zarr.open_group(refkv.zarr.store(zarrify(str(source))), mode="r")
    actual_image = reopened.get_array("0/0")
    expected_image = expected.get_array("0/0")
    assert actual_image.shape == expected_image.shape
    assert actual_image.metadata.dimension_names == expected_image.metadata.dimension_names
    selection = (0,) * (actual_image.ndim - 2) + (slice(0, 16), slice(0, 16))
    np.testing.assert_array_equal(actual_image[selection], expected_image[selection])
