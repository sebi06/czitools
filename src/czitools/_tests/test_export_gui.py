"""Tests for the optional CZI to OME-Zarr converter GUI."""

import os
from inspect import signature
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

gui = pytest.importorskip("czitools.export_tools.gui")


def test_ngff_zarr_is_default_backend() -> None:
    assert gui.czi_to_omezarr_converter.package_choice.value == gui.omezarr_package.NGFF_ZARR
    assert gui.czi_to_omezarr_converter.conversion_preset.value is gui.NgffConversionPreset.FAST_BALANCED
    assert "use_tensorstore" not in signature(gui.perform_conversion).parameters


def test_controls_require_selected_czi() -> None:
    gui.on_file_changed(Path())

    assert all(not control.enabled for control in gui._conversion_controls())
    assert gui.czi_to_omezarr_converter.czi_file.enabled


def test_file_selection_loads_metadata_and_hcs_details(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    filepath = tmp_path / "plate.czi"
    filepath.touch()
    fake_metadata = SimpleNamespace(
        image=SimpleNamespace(
            SizeX=100,
            SizeY=80,
            SizeC=2,
            SizeZ=3,
            SizeT=1,
        ),
        pyczi_dims="STCZYX",
        hcs=object(),
    )
    monkeypatch.setattr(
        gui,
        "read_czi_metadata",
        lambda _: (fake_metadata, 4),
    )
    monkeypatch.setattr(gui, "_format_hcs_details", lambda _: "HCS DETAILS")

    gui.czi_to_omezarr_converter.czi_file.value = filepath

    assert gui.metadata is fake_metadata
    assert gui.selected_file == filepath
    assert gui.convert_button.enabled
    assert gui.czi_to_omezarr_converter.write_hcs.enabled
    assert "HCS layout detected" in gui.info_display.value
    assert "HCS DETAILS" in gui.info_display.value


def test_non_hcs_file_disables_hcs_option(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    filepath = tmp_path / "image.czi"
    filepath.touch()
    fake_metadata = SimpleNamespace(
        image=SimpleNamespace(
            SizeX=100,
            SizeY=80,
            SizeC=1,
            SizeZ=1,
            SizeT=1,
        ),
        pyczi_dims="TCZYX",
        hcs=None,
    )
    monkeypatch.setattr(gui, "read_czi_metadata", lambda _: (fake_metadata, 1))
    gui.czi_to_omezarr_converter.write_hcs.value = True

    gui.czi_to_omezarr_converter.czi_file.value = filepath

    assert not gui.czi_to_omezarr_converter.write_hcs.enabled
    assert not gui.czi_to_omezarr_converter.write_hcs.value
    assert "No HCS plate layout detected" in gui.info_display.value


def test_hcs_mode_allows_single_ozx_option(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(gui, "metadata", object())
    monkeypatch.setattr(gui, "selected_file", tmp_path / "plate.czi")
    gui.czi_to_omezarr_converter.package_choice.value = gui.omezarr_package.NGFF_ZARR
    gui.czi_to_omezarr_converter.write_hcs.value = True

    gui.update_use_ozx_format_enabled_state()

    assert gui.czi_to_omezarr_converter.use_ozx_format.enabled
    assert not hasattr(gui.czi_to_omezarr_converter, "use_ozx_write_directly")
    assert not hasattr(gui.czi_to_omezarr_converter, "use_ozx_after_writing")

    gui.czi_to_omezarr_converter.use_ozx_format.value = True

    assert not gui.czi_to_omezarr_converter.show_napari.enabled


def test_fast_balanced_preset_allows_ozx_and_enforces_blosc(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(gui, "metadata", object())
    monkeypatch.setattr(gui, "selected_file", tmp_path / "image.czi")
    gui.czi_to_omezarr_converter.package_choice.value = gui.omezarr_package.NGFF_ZARR
    gui.czi_to_omezarr_converter.write_hcs.value = False
    gui.czi_to_omezarr_converter.use_ozx_format.value = True
    gui.czi_to_omezarr_converter.compression_choice.value = gui.compression_type.NONE

    gui.czi_to_omezarr_converter.conversion_preset.value = gui.NgffConversionPreset.FAST_BALANCED
    gui.update_use_ozx_format_enabled_state()

    assert gui.czi_to_omezarr_converter.use_ozx_format.value
    assert gui.czi_to_omezarr_converter.use_ozx_format.enabled
    assert gui.czi_to_omezarr_converter.compression_choice.value is gui.compression_type.BLOSC
    assert not gui.czi_to_omezarr_converter.compression_choice.enabled


def test_fast_balanced_preset_forwards_writer_settings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    filepath = tmp_path / "image.czi"
    filepath.touch()
    array = xr.DataArray(
        np.zeros((1, 1, 1, 2, 8, 8), dtype=np.uint16),
        dims=("S", "T", "C", "Z", "Y", "X"),
    )
    captured: dict = {}
    monkeypatch.setattr(gui.read_tools, "read_6darray", lambda *args, **kwargs: (array, object()))

    def capture_writer(*args, **kwargs) -> str:
        captured["path"] = args[1]
        captured.update(kwargs)
        return "image"

    monkeypatch.setattr(gui, "write_omezarr_ngff", capture_writer)
    monkeypatch.setattr(gui, "validate_ome_zarr", lambda _: True)

    output = gui.perform_conversion(
        filepath=filepath,
        use_ozx_format=True,
        write_hcs=False,
        package_choice=gui.omezarr_package.NGFF_ZARR,
        scene_id=0,
        compression_choice=gui.compression_type.NONE,
        conversion_preset=gui.NgffConversionPreset.FAST_BALANCED,
    )

    assert output is not None and output.endswith("_ngff.ozx")
    assert captured["path"] == Path(output)
    assert captured["compression"] is gui.compression_type.BLOSC
    assert captured["chunks"] == (1, 1, 4, 1024, 1024)
    assert captured["chunks_per_shard"] == {"y": 2, "x": 2}
    assert captured["downsampling_method"] is gui.nz.Methods.DASK_BIN_SHRINK


def test_hcs_details_are_plain_text() -> None:
    field = SimpleNamespace(
        field_index=0,
        scene_index=4,
        scene_center_x=38687.5,
        scene_center_y=15687.5,
        position_unit="micrometer",
    )
    well = SimpleNamespace(canonical_name="B5", fields=[field])
    plate = SimpleNamespace(
        id="plate:test.czi",
        name="Multichamber 96",
        schema_version="1.0",
        declared_rows=8,
        declared_columns=12,
        observed_row_indices=(1,),
        observed_column_indices=(3, 4),
        wells=[well],
    )
    metadata = SimpleNamespace(
        hcs=plate,
        hcs_status=SimpleNamespace(detected=True, reason="HCS metadata found."),
        sample=SimpleNamespace(
            scene_count=4,
            well_unique_number=1,
            multipos_per_well=True,
        ),
    )

    output = gui._format_hcs_details(metadata)

    assert "HCS PLATE INFORMATION" in output
    assert "Total wells: 1" in output
    assert "Field 0: scene 4" in output
    assert "\x1b" not in output
    assert "╭" not in output


def test_metadata_display_precedes_conversion_options() -> None:
    container = gui.create_gui()

    assert gui.info_display.native.lineWrapMode() == gui.QTextEdit.LineWrapMode.NoWrap
    assert container[1].labels
    assert container[2] is gui.info_display
    conversion_options = container[3]
    assert conversion_options.labels
    assert conversion_options[0] is gui.czi_to_omezarr_converter.write_hcs
    assert conversion_options[1] is gui.czi_to_omezarr_converter.scene_id
    assert conversion_options[2] is gui.czi_to_omezarr_converter.conversion_preset
    assert conversion_options[3] is gui.czi_to_omezarr_converter.use_ozx_format
    assert conversion_options[4] is gui.czi_to_omezarr_converter.compression_choice
    assert conversion_options[5] is gui.czi_to_omezarr_converter.show_napari
    assert not hasattr(gui.czi_to_omezarr_converter, "use_tensorstore")
