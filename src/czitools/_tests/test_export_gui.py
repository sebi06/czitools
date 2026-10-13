"""Tests for the optional CZI to OME-Zarr converter GUI."""

import os
from inspect import signature
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import dask.array as da
import pytest
import xarray as xr

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

gui = pytest.importorskip("czitools.export_tools.gui")


def test_ngff_zarr_is_default_backend() -> None:
    assert gui.czi_to_omezarr_converter.package_choice.value == gui.omezarr_package.NGFF_ZARR
    assert gui.czi_to_omezarr_converter.conversion_preset.value is gui.NgffConversionPreset.QUALITY
    assert gui.czi_to_omezarr_converter.pyramid_levels.value == 3
    assert gui.czi_to_omezarr_converter.spatial_chunk_size.value == 512
    assert gui.czi_to_omezarr_converter.spatial_chunk_size.choices == (256, 512, 1024)
    assert gui.czi_to_omezarr_converter.reuse_czi_pyramid.value is True
    assert gui.czi_to_omezarr_converter.worker_count.value == 0
    assert gui.czi_to_omezarr_converter.worker_count.native.specialValueText() == "Auto"
    assert gui.czi_to_omezarr_converter.worker_count.label == "Dask workers (0 = Auto)"
    assert "0 = Auto" in gui.czi_to_omezarr_converter.worker_count.tooltip
    assert "1-32" in gui.czi_to_omezarr_converter.worker_count.tooltip
    assert "up to 8 workers" in gui.czi_to_omezarr_converter.worker_count.tooltip
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
    assert gui.czi_to_omezarr_converter.conversion_preset.native.isHidden()
    assert gui.czi_to_omezarr_converter.pyramid_levels.native.isHidden()
    assert gui.hcs_pyramid_policy.visible
    assert "CZI-stored levels" in gui.hcs_pyramid_policy.value
    assert "512 px" in gui.hcs_pyramid_policy.value
    assert not hasattr(gui.czi_to_omezarr_converter, "use_ozx_write_directly")
    assert not hasattr(gui.czi_to_omezarr_converter, "use_ozx_after_writing")

    gui.czi_to_omezarr_converter.use_ozx_format.value = True

    assert not gui.czi_to_omezarr_converter.show_napari.enabled

    gui.czi_to_omezarr_converter.write_hcs.value = False
    gui.update_use_ozx_format_enabled_state()

    assert not gui.czi_to_omezarr_converter.conversion_preset.native.isHidden()
    assert not gui.czi_to_omezarr_converter.pyramid_levels.native.isHidden()
    assert not gui.hcs_pyramid_policy.visible


def test_omezarr_hcs_mode_shows_fixed_pyramid_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(gui, "metadata", object())
    monkeypatch.setattr(gui, "selected_file", tmp_path / "plate.czi")
    gui.czi_to_omezarr_converter.package_choice.value = gui.omezarr_package.OME_ZARR
    gui.czi_to_omezarr_converter.write_hcs.value = True

    gui.update_use_ozx_format_enabled_state()

    assert gui.hcs_pyramid_policy.visible
    assert "CZI-stored levels" in gui.hcs_pyramid_policy.value
    assert "512 px" in gui.hcs_pyramid_policy.value


def test_fast_balanced_preset_allows_ozx_and_enforces_blosc(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(gui, "metadata", object())
    monkeypatch.setattr(gui, "selected_file", tmp_path / "image.czi")
    gui.czi_to_omezarr_converter.package_choice.value = gui.omezarr_package.NGFF_ZARR
    gui.czi_to_omezarr_converter.write_hcs.value = False
    gui.czi_to_omezarr_converter.reuse_czi_pyramid.value = False
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
    monkeypatch.setattr(gui.read_tools, "read_stacks", lambda *args, **kwargs: ([array.squeeze("S")], {}, 1, object()))

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
        pyramid_levels=2,
        reuse_czi_pyramid=False,
    )

    assert output is not None and output.endswith("_ngff.ozx")
    assert captured["path"] == Path(output)
    assert captured["compression"] is gui.compression_type.BLOSC
    assert captured["chunks"] == (1, 1, 4, 512, 512)
    assert captured["chunks_per_shard"] == {"z": 2, "y": 2, "x": 2}
    assert captured["pyramid_levels"] == 2
    assert captured["downsampling_method"] is gui.nz.Methods.DASK_BIN_SHRINK


def test_quality_preset_forwards_3d_layout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    filepath = tmp_path / "volume.czi"
    filepath.touch()
    array = xr.DataArray(
        np.zeros((1, 1, 2, 16, 32, 32), dtype=np.uint16),
        dims=("S", "T", "C", "Z", "Y", "X"),
    )
    captured: dict = {}
    monkeypatch.setattr(gui.read_tools, "read_stacks", lambda *args, **kwargs: ([array.squeeze("S")], {}, 1, object()))

    def capture_writer(*args, **kwargs) -> str:
        captured.update(kwargs)
        return "image"

    monkeypatch.setattr(gui, "write_omezarr_ngff", capture_writer)
    monkeypatch.setattr(gui, "validate_ome_zarr", lambda _: True)

    gui.perform_conversion(
        filepath=filepath,
        use_ozx_format=False,
        write_hcs=False,
        package_choice=gui.omezarr_package.NGFF_ZARR,
        scene_id=0,
        conversion_preset=gui.NgffConversionPreset.QUALITY,
        pyramid_levels=3,
        reuse_czi_pyramid=False,
    )

    assert captured["chunks"] == (1, 1, 8, 512, 512)
    assert captured["chunks_per_shard"] == {"z": 4, "y": 4, "x": 4}
    assert captured["pyramid_levels"] == 3


@pytest.mark.parametrize("chunk_size", [256, 512, 1024])
@pytest.mark.parametrize("backend", [gui.omezarr_package.NGFF_ZARR, gui.omezarr_package.OME_ZARR])
def test_conversion_forwards_lazy_roi_and_chunk_settings(tmp_path, monkeypatch, chunk_size, backend) -> None:
    """Both standard backends receive lazy tiled pixels and selected chunks."""
    array = xr.DataArray(da.zeros((1, 2, 1, 16, 17), chunks=(1, 1, 1, 8, 8)), dims=("T", "C", "Z", "Y", "X"))
    captured = {}

    def read_scene(*args, **kwargs):
        captured["reader"] = kwargs
        return [array], {}, 1, object()

    def write_scene(data, *args, **kwargs):
        assert isinstance(data.data, da.Array)
        captured["writer"] = kwargs
        return tmp_path / "result.ome.zarr"

    monkeypatch.setattr(gui.read_tools, "read_stacks", read_scene)
    monkeypatch.setattr(gui.read_tools, "read_6darray", lambda *args, **kwargs: pytest.fail("Eager reader used"))
    monkeypatch.setattr(gui, "write_omezarr_ngff", write_scene)
    monkeypatch.setattr(gui, "write_omezarr", write_scene)
    monkeypatch.setattr(gui, "validate_ome_zarr", lambda _: True)
    result = gui.perform_conversion(
        tmp_path / "large.czi",
        False,
        False,
        backend,
        2,
        spatial_chunk_size=chunk_size,
        reuse_czi_pyramid=False,
    )
    assert result is not None
    assert captured["reader"]["planes"] == {"S": (2, 2)}
    assert captured["reader"]["use_dask"] is True
    assert captured["reader"]["tile_size"] == 4096
    assert captured["reader"]["chunk_memory_limit"] == 32 * 1024 * 1024
    assert captured["reader"]["planes_per_chunk"] == 1
    if backend is gui.omezarr_package.NGFF_ZARR:
        assert captured["writer"]["chunks"][-2:] == (chunk_size, chunk_size)
    else:
        assert captured["writer"]["spatial_chunk_size"] == chunk_size


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
        image=SimpleNamespace(
            SizeX_scene=2752,
            SizeY_scene=2208,
            SizeC=3,
            SizeZ=1,
            SizeT=1,
        ),
        sample=SimpleNamespace(
            scene_count=4,
            well_unique_number=1,
            multipos_per_well=True,
        ),
    )

    output = gui._format_hcs_details(metadata)

    assert "HCS PLATE INFORMATION" in output
    assert "Total wells: 1" in output
    assert "Image size: 2752 × 2208 pixels (X × Y)" in output
    assert "Channels: 3" in output
    assert "Field 0: scene 4" in output
    assert "\x1b" not in output
    assert "╭" not in output


@pytest.mark.parametrize("backend", [gui.omezarr_package.NGFF_ZARR, gui.omezarr_package.OME_ZARR])
@pytest.mark.parametrize("worker_count", [0, 4, 8])
def test_conversion_reuses_czi_pyramid_by_default(tmp_path, monkeypatch, backend, worker_count) -> None:
    """Both backends receive lazy CZI levels and bounded read settings."""
    image = gui.nz.to_ngff_image(da.zeros((1, 2, 1, 27, 31), chunks=8), dims=["t", "c", "z", "y", "x"])
    source = SimpleNamespace(images=[image])
    captured = {}
    selected_workers = []
    metadata = object()
    monkeypatch.setattr(gui, "_resolve_worker_count", lambda value: value or 6)
    monkeypatch.setattr(gui, "CziMetadata", lambda _: metadata)
    monkeypatch.setattr(gui.read_tools, "get_pyramid_zooms", lambda _: [1.0, 0.333336])

    def read_levels(path, scene, mdata, zooms, chunks, planes, **kwargs):
        assert scene == 2 and mdata is metadata
        assert zooms == [1.0, 0.333336]
        assert chunks == 1024 and planes == 1
        assert kwargs["tile_size"] == 4096
        assert kwargs["chunk_memory_limit"] == 32 * 1024 * 1024
        return source

    def write_levels(array, *args, **kwargs):
        assert isinstance(array, xr.DataArray)
        assert isinstance(array.data, da.Array)
        captured.update(kwargs)
        return tmp_path / "image.ome.zarr"

    monkeypatch.setattr(gui, "_read_single_scene_multiscales", read_levels)
    monkeypatch.setattr(gui.read_tools, "read_stacks", lambda *args, **kwargs: pytest.fail("Base-only reader used"))
    monkeypatch.setattr(gui, "write_omezarr_ngff", write_levels)
    monkeypatch.setattr(gui, "write_omezarr", write_levels)
    monkeypatch.setattr(gui, "validate_ome_zarr", lambda _: True)
    assert gui.perform_conversion(
        tmp_path / "image.czi",
        False,
        False,
        backend,
        2,
        spatial_chunk_size=1024,
        worker_count=worker_count,
        on_workers_selected=selected_workers.append,
    )
    assert captured["source_multiscales"] is source
    assert captured["worker_count"] == (worker_count or 6)
    assert selected_workers == [worker_count or 6]
    log_text = (tmp_path / "image_conversion.log").read_text(encoding="utf-8")
    mode = "Auto" if worker_count == 0 else "manual"
    assert f"Using {worker_count or 6} Dask workers ({mode})." in log_text


@pytest.mark.parametrize("backend", [gui.omezarr_package.NGFF_ZARR, gui.omezarr_package.OME_ZARR])
def test_reuse_checkbox_controls_regeneration(tmp_path, monkeypatch, backend) -> None:
    """Reuse disables regeneration settings, but leaves compression selectable."""
    monkeypatch.setattr(gui, "metadata", object())
    monkeypatch.setattr(gui, "selected_file", tmp_path / "image.czi")
    gui.czi_to_omezarr_converter.package_choice.value = backend
    gui.czi_to_omezarr_converter.write_hcs.value = False
    gui.czi_to_omezarr_converter.reuse_czi_pyramid.value = True
    gui.update_use_ozx_format_enabled_state()
    assert gui.czi_to_omezarr_converter.reuse_czi_pyramid.enabled
    assert not gui.czi_to_omezarr_converter.pyramid_levels.enabled
    assert not gui.czi_to_omezarr_converter.conversion_preset.enabled
    assert gui.czi_to_omezarr_converter.compression_choice.enabled
    gui.czi_to_omezarr_converter.reuse_czi_pyramid.value = False
    assert gui.czi_to_omezarr_converter.pyramid_levels.enabled
    gui.czi_to_omezarr_converter.reuse_czi_pyramid.value = True


@pytest.mark.parametrize("requested,resolved,expected", [(0, 6, "6 (Auto)"), (4, 4, "4 (manual)")])
def test_conversion_displays_resolved_workers(tmp_path, monkeypatch, requested, resolved, expected) -> None:
    """The main-thread timer displays the exact worker count reported by conversion."""
    filepath = tmp_path / "image.czi"
    filepath.touch()
    metadata = SimpleNamespace(image=None, pyczi_dims="TCZYX", hcs=None)
    monkeypatch.setattr(gui, "read_czi_metadata", lambda _: (metadata, 1))
    gui.czi_to_omezarr_converter.czi_file.value = filepath
    gui.czi_to_omezarr_converter.write_hcs.value = False
    gui.czi_to_omezarr_converter.worker_count.value = requested
    callbacks = []
    monkeypatch.setattr(
        gui,
        "QTimer",
        lambda: SimpleNamespace(
            timeout=SimpleNamespace(connect=callbacks.append),
            start=lambda _: None,
            stop=lambda: None,
        ),
    )
    monkeypatch.setattr(gui.threading, "Thread", lambda target, daemon: SimpleNamespace(start=target))
    monkeypatch.setattr(gui, "finish_conversion", lambda *args: None)

    def convert(**kwargs):
        kwargs["on_workers_selected"](resolved)
        return str(tmp_path / "image.ome.zarr")

    monkeypatch.setattr(gui, "perform_conversion", convert)
    gui.on_convert_clicked()
    assert gui.worker_status.visible
    callbacks[0]()
    assert gui.worker_status.value == expected
    assert gui.czi_to_omezarr_converter.worker_count.value == requested
    gui.on_file_changed(Path())
    assert gui.worker_status.value == ""
    assert not gui.worker_status.visible


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
    assert conversion_options[3] is gui.czi_to_omezarr_converter.pyramid_levels
    assert conversion_options[4] is gui.czi_to_omezarr_converter.spatial_chunk_size
    assert conversion_options[5] is gui.czi_to_omezarr_converter.reuse_czi_pyramid
    assert conversion_options[6] is gui.czi_to_omezarr_converter.worker_count
    assert conversion_options[7] is gui.worker_status
    assert conversion_options[8] is gui.hcs_pyramid_policy
    assert conversion_options[9] is gui.czi_to_omezarr_converter.use_ozx_format
    assert conversion_options[10] is gui.czi_to_omezarr_converter.compression_choice
    assert conversion_options[11] is gui.czi_to_omezarr_converter.show_napari
    assert not hasattr(gui.czi_to_omezarr_converter, "use_tensorstore")
