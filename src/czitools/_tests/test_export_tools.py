"""Tests for the OME-Zarr export tools (Stage 5).

These tests are skipped automatically when the optional export dependencies
(``ngff-zarr``, ``ome-zarr``, ``ome-zarr-models``, ``zarr``) are not installed.
"""

import ast
import zipfile
from inspect import signature
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from dask.array import Array as DaskArray
from numcodecs import Blosc as NumcodecsBlosc

pytest.importorskip("ngff_zarr")
pytest.importorskip("ome_zarr")
pytest.importorskip("ome_zarr_models")

from czitools.export_tools import (
    convert_czi2hcs_ngff,
    resolve_hcs_layout,
    validate_ome_zarr,
    write_omezarr_ngff,
)
from czitools.export_tools import conversion
from czitools.metadata_tools.czi_metadata import CziMetadata

BASEDIR = Path(__file__).resolve().parents[3]
WELLPLATE = BASEDIR / "data" / "WP96_4Pos_B4-10_DAPI.czi"


def test_legacy_export_options_are_not_public() -> None:
    assert "zarr_format" not in signature(conversion.write_omezarr).parameters
    assert "zarr_format" not in signature(conversion.convert_czi2hcs_omezarr).parameters
    assert "normalize_level_paths" not in signature(conversion.convert_czi2hcs_omezarr).parameters
    assert "normalize_level_paths" not in signature(conversion.convert_czi2hcs_ngff).parameters


def test_gui_single_image_writers_share_conversion_log() -> None:
    gui_path = BASEDIR / "src" / "czitools" / "export_tools" / "gui.py"
    tree = ast.parse(gui_path.read_text(encoding="utf-8"))
    writer_names = {"write_omezarr", "write_omezarr_ngff"}
    writer_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in writer_names
    ]

    assert {call.func.id for call in writer_calls} == writer_names
    for call in writer_calls:
        log_keywords = [keyword for keyword in call.keywords if keyword.arg == "log_file_path"]
        assert len(log_keywords) == 1
        assert isinstance(log_keywords[0].value, ast.Call)
        assert isinstance(log_keywords[0].value.func, ast.Name)
        assert log_keywords[0].value.func.id == "str"


def test_resolve_layout_prefers_stage1_model() -> None:
    mdata = CziMetadata(str(WELLPLATE))
    layout = resolve_hcs_layout(mdata, pad_columns=True)

    assert layout.source == "hcs"
    assert layout.row_names == ["B"]
    assert layout.col_names == ["04", "05", "06", "07", "08", "09", "10"]
    assert len(layout.wells) == 7
    assert layout.field_count == 4
    # Every well has 4 fields, each mapping to a distinct scene index.
    for well in layout.wells:
        assert len(well.fields) == 4
    all_scenes = sorted(scene for well in layout.wells for _, scene in well.fields)
    assert all_scenes == list(range(28))


def test_resolve_layout_without_pad_columns() -> None:
    mdata = CziMetadata(str(WELLPLATE))
    layout = resolve_hcs_layout(mdata, pad_columns=False)
    assert layout.col_names == ["4", "5", "6", "7", "8", "9", "10"]
    assert layout.wells[0].path == "B/4"


def test_convert_czi2hcs_ngff_and_validate(tmp_path: Path) -> None:
    output_dir = tmp_path / "exports"
    output = convert_czi2hcs_ngff(
        WELLPLATE,
        overwrite=True,
        output_dir=output_dir,
        pad_columns=True,
    )
    assert output.exists()
    assert output.name == "WP96_4Pos_B4-10_DAPI_ngff_plate_zarr3.ome.zarr"
    assert validate_ome_zarr(output) is True


def test_convert_czi2hcs_ngff_direct_ozx(tmp_path: Path) -> None:
    output = convert_czi2hcs_ngff(
        WELLPLATE,
        overwrite=True,
        output_dir=tmp_path,
        write_ozx_directly=True,
    )

    assert output.is_file()
    assert output.name == "WP96_4Pos_B4-10_DAPI_ngff_plate.ozx"
    assert zipfile.is_zipfile(output)
    with zipfile.ZipFile(output) as archive:
        assert archive.testzip() is None
        assert "zarr.json" in archive.namelist()


def test_write_omezarr_ngff_to_local_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Compression options must not be forwarded as FSSpec storage options."""
    metadata = SimpleNamespace(
        filename="test.czi",
        scale=SimpleNamespace(X=1.0, Y=1.0, Z=1.0),
        image=None,
        channelinfo=None,
    )
    output = tmp_path / "image.ome.zarr"
    multiscales = SimpleNamespace(metadata=SimpleNamespace(omero=None))
    captured: dict = {}

    def capture_image(data, *args, **kwargs) -> str:
        captured["image_data"] = data
        return "image"

    monkeypatch.setattr(conversion.nz, "to_ngff_image", capture_image)
    monkeypatch.setattr(conversion.nz, "to_multiscales", lambda *args, **kwargs: multiscales)

    def capture_write(store, **kwargs) -> None:
        captured["store"] = store
        captured.update(kwargs)

    monkeypatch.setattr(conversion.nz, "to_ngff_zarr", capture_write)

    image = write_omezarr_ngff(
        np.zeros((1, 3, 1, 8, 8), dtype=np.uint16),
        output,
        metadata,
        scale_factors=[2],
        chunks=(1, 1, 1, 4, 4),
        chunks_per_shard=None,
        overwrite=True,
        use_tensorstore=False,
    )

    assert image == "image"
    assert isinstance(captured["image_data"], DaskArray)
    assert captured["image_data"].chunks[1] == (1, 1, 1)
    assert captured["store"] == output
    assert isinstance(captured["compressor"], NumcodecsBlosc)
    assert "storage_options" not in captured
