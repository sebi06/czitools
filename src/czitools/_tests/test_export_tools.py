"""Tests for the OME-Zarr export tools (Stage 5).

These tests are skipped automatically when the optional export dependencies
(``ngff-zarr``, ``ome-zarr``, ``ome-zarr-models``, ``zarr``) are not installed.
"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

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

    monkeypatch.setattr(conversion.nz, "to_ngff_image", lambda *args, **kwargs: "image")
    monkeypatch.setattr(conversion.nz, "to_multiscales", lambda *args, **kwargs: multiscales)

    def capture_write(store, **kwargs) -> None:
        captured["store"] = store
        captured.update(kwargs)

    monkeypatch.setattr(conversion.nz, "to_ngff_zarr", capture_write)

    image = write_omezarr_ngff(
        np.zeros((1, 1, 1, 8, 8), dtype=np.uint16),
        output,
        metadata,
        scale_factors=[2],
        chunks=(1, 1, 1, 4, 4),
        chunks_per_shard=None,
        overwrite=True,
        use_tensorstore=False,
    )

    assert image == "image"
    assert captured["store"] == output
    assert captured["compressor"] is not None
    assert "storage_options" not in captured
