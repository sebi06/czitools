"""Tests for the OME-Zarr export tools (Stage 5).

These tests are skipped automatically when the optional export dependencies
(``ngff-zarr``, ``ome-zarr``, ``ome-zarr-models``, ``zarr``) are not installed.
"""

import ast
import json
import logging
import zipfile
from inspect import signature
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr
import dask.array as da
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
from czitools.export_tools import _logging as export_logging
from czitools.metadata_tools.czi_metadata import CziMetadata

BASEDIR = Path(__file__).resolve().parents[3]
WELLPLATE = BASEDIR / "data" / "WP96_4Pos_B4-10_DAPI.czi"


def test_legacy_export_options_are_not_public() -> None:
    assert "zarr_format" not in signature(conversion.write_omezarr).parameters
    assert "zarr_format" not in signature(conversion.convert_czi2hcs_omezarr).parameters
    assert "normalize_level_paths" not in signature(conversion.convert_czi2hcs_omezarr).parameters
    assert "normalize_level_paths" not in signature(conversion.convert_czi2hcs_ngff).parameters
    assert "use_tensorstore" not in signature(conversion.write_omezarr_ngff).parameters
    assert signature(conversion.convert_czi2hcs_ngff).parameters["chunks_per_shard"].default == {"y": 4, "x": 4}
    assert signature(conversion.convert_czi2hcs_ngff).parameters["max_workers"].default == 4
    assert signature(conversion.convert_czi2hcs_omezarr).parameters["logging_detail"].default == "basic"
    assert signature(conversion.convert_czi2hcs_ngff).parameters["logging_detail"].default == "basic"


def test_hcs_basic_logging_bounds_progress_messages(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger=conversion.__name__):
        for completed in range(101):
            conversion._log_hcs_progress(completed, 100, 0.0)

    progress_records = [record for record in caplog.records if record.message.startswith("HCS conversion progress:")]
    assert len(progress_records) == 11
    assert "(0/100 fields" in progress_records[0].message
    assert "(100/100 fields" in progress_records[-1].message


def test_hcs_full_logging_reports_every_field(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger=conversion.__name__):
        for completed in range(6):
            conversion._log_hcs_progress(completed, 5, 0.0, full_logging=True)

    progress_records = [record for record in caplog.records if record.message.startswith("HCS conversion progress:")]
    assert len(progress_records) == 6


@pytest.mark.parametrize(
    ("include_internal_info", "logger_name", "level", "expected"),
    [
        (False, "czitools.export_tools.conversion", logging.INFO, True),
        (False, "czitools", logging.INFO, False),
        (False, "root", logging.INFO, False),
        (False, "czitools", logging.WARNING, True),
        (True, "czitools", logging.INFO, True),
        (True, "root", logging.INFO, True),
    ],
)
def test_export_log_filter(
    include_internal_info: bool,
    logger_name: str,
    level: int,
    expected: bool,
) -> None:
    record = logging.LogRecord(logger_name, level, __file__, 1, "message", (), None)
    log_filter = export_logging._ExportLogFilter(include_internal_info)

    assert bool(log_filter.filter(record)) is expected


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


def test_gui_hcs_converters_use_basic_logging() -> None:
    gui_path = BASEDIR / "src" / "czitools" / "export_tools" / "gui.py"
    tree = ast.parse(gui_path.read_text(encoding="utf-8"))
    converter_names = {"convert_czi2hcs_omezarr", "convert_czi2hcs_ngff"}
    converter_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in converter_names
    ]

    assert {call.func.id for call in converter_calls} == converter_names
    for call in converter_calls:
        detail_keywords = [keyword for keyword in call.keywords if keyword.arg == "logging_detail"]
        assert len(detail_keywords) == 1
        assert isinstance(detail_keywords[0].value, ast.Constant)
        assert detail_keywords[0].value.value == "basic"


def test_gui_starts_each_conversion_with_fresh_log() -> None:
    gui_path = BASEDIR / "src" / "czitools" / "export_tools" / "gui.py"
    tree = ast.parse(gui_path.read_text(encoding="utf-8"))
    setup_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "setup_logging"
    ]
    conversion_setup = next(call for call in setup_calls if call.args)
    truncate_keywords = [keyword for keyword in conversion_setup.keywords if keyword.arg == "truncate_log_file"]

    assert len(truncate_keywords) == 1
    assert isinstance(truncate_keywords[0].value, ast.Constant)
    assert truncate_keywords[0].value.value is True


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

    array_metadata_path = output / "B" / "04" / "0" / "scale0" / WELLPLATE.name / "zarr.json"
    array_metadata = json.loads(array_metadata_path.read_text(encoding="utf-8"))
    assert array_metadata["chunk_grid"]["configuration"]["chunk_shape"][-2:] == [640, 640]
    sharding_codec = array_metadata["codecs"][0]
    assert sharding_codec["name"] == "sharding_indexed"
    assert sharding_codec["configuration"]["chunk_shape"][-2:] == [320, 320]
    assert sharding_codec["configuration"]["codecs"][1]["name"] == "blosc"


def test_convert_czi2hcs_ngff_direct_ozx(tmp_path: Path) -> None:
    log_path = tmp_path / "conversion.log"
    output = convert_czi2hcs_ngff(
        WELLPLATE,
        overwrite=True,
        output_dir=tmp_path,
        write_ozx_directly=True,
        log_file_path=log_path,
    )

    assert output.is_file()
    assert output.name == "WP96_4Pos_B4-10_DAPI_ngff_plate.ozx"
    assert zipfile.is_zipfile(output)
    with zipfile.ZipFile(output) as archive:
        assert archive.testzip() is None
        assert "zarr.json" in archive.namelist()

    log_text = log_path.read_text(encoding="utf-8")
    assert "HCS conversion progress:" in log_text
    assert "100% (28/28 fields" in log_text
    assert "OZX archive finalization progress:" in log_text
    assert "Total conversion time:" in log_text


def test_hcs_multiscales_reuse_stored_czi_levels(monkeypatch: pytest.MonkeyPatch) -> None:
    dims = ("S", "T", "C", "Z", "Y", "X")
    level0 = xr.DataArray(da.zeros((1, 1, 2, 1, 256, 384), chunks=(1, 1, 1, 1, 256, 384)), dims=dims)
    level1 = xr.DataArray(da.zeros((1, 1, 2, 1, 128, 192), chunks=(1, 1, 1, 1, 128, 192)), dims=dims)
    captured: dict = {}

    def fake_read_stacks_multiscale(**kwargs):
        captured.update(kwargs)
        infos = [
            SimpleNamespace(zoom=1.0, stored=True),
            SimpleNamespace(zoom=0.5, stored=True),
        ]
        return [level0, level1], infos, list(dims), 1, None

    monkeypatch.setattr(conversion.read_tools, "read_stacks_multiscale", fake_read_stacks_multiscale)
    monkeypatch.setattr(
        conversion.nz,
        "to_multiscales",
        lambda *args, **kwargs: pytest.fail("stored levels must not be resampled"),
    )
    metadata = SimpleNamespace(
        filename="plate.czi",
        scale=SimpleNamespace(X=0.5, Y=0.5, Z=2.0),
    )

    multiscales = conversion._read_single_scene_multiscales(
        "plate.czi",
        scene_index=3,
        metadata=metadata,
        pyramid_zooms=[1.0, 0.5],
        spatial_chunk_size=512,
        planes_per_chunk=64,
    )

    assert captured["planes"] == {"S": (3, 3)}
    assert captured["zooms"] == [1.0, 0.5]
    assert captured["max_coarse_edge"] == 512
    assert captured["planes_per_chunk"] == 64
    assert captured["metadata"] is metadata
    assert [image.data.shape for image in multiscales.images] == [
        (1, 2, 1, 256, 384),
        (1, 2, 1, 128, 192),
    ]
    assert all(isinstance(image.data, DaskArray) for image in multiscales.images)
    assert multiscales.images[0].data.chunksize == (1, 1, 1, 256, 384)
    assert multiscales.images[1].scale == {"t": 1.0, "c": 1.0, "z": 2.0, "y": 1.0, "x": 1.0}
    assert multiscales.images[1].translation == {"t": 0.0, "c": 0.0, "z": 0.0, "y": 0.25, "x": 0.25}
    assert multiscales.generated_data_keys is None
    assert multiscales.method is None


def test_hcs_multiscales_drop_singleton_h_dimension(monkeypatch: pytest.MonkeyPatch) -> None:
    dims = ("S", "H", "T", "C", "Z", "Y", "X")
    data = da.arange(3, chunks=1).reshape((1, 1, 1, 3, 1, 1, 1))
    level = xr.DataArray(data, dims=dims)

    monkeypatch.setattr(
        conversion.read_tools,
        "read_stacks_multiscale",
        lambda **kwargs: ([level], [SimpleNamespace(zoom=1.0, stored=True)], list(dims), 1, None),
    )
    metadata = SimpleNamespace(
        filename="airy.czi",
        scale=SimpleNamespace(X=1.0, Y=1.0, Z=1.0),
    )

    multiscales = conversion._read_single_scene_multiscales(
        "airy.czi",
        scene_index=0,
        metadata=metadata,
        pyramid_zooms=[1.0],
        spatial_chunk_size=512,
        planes_per_chunk=64,
    )

    assert multiscales.images[0].dims == ["t", "c", "z", "y", "x"]
    assert multiscales.images[0].data.shape == (1, 3, 1, 1, 1)
    np.testing.assert_array_equal(multiscales.images[0].data.compute().ravel(), np.arange(3))


def test_hcs_multiscales_reject_multiple_h_phases(monkeypatch: pytest.MonkeyPatch) -> None:
    dims = ("S", "H", "T", "C", "Z", "Y", "X")
    level = xr.DataArray(da.zeros((1, 2, 1, 1, 1, 1, 1), chunks=1), dims=dims)
    monkeypatch.setattr(
        conversion.read_tools,
        "read_stacks_multiscale",
        lambda **kwargs: ([level], [SimpleNamespace(zoom=1.0, stored=True)], list(dims), 1, None),
    )
    metadata = SimpleNamespace(
        filename="airy.czi",
        scale=SimpleNamespace(X=1.0, Y=1.0, Z=1.0),
    )

    with pytest.raises(ValueError, match=r"scene 4 has SizeH=2.*CZI-specific phase axis.*Only SizeH=1"):
        conversion._read_single_scene_multiscales(
            "airy.czi",
            scene_index=4,
            metadata=metadata,
            pyramid_zooms=[1.0],
            spatial_chunk_size=512,
            planes_per_chunk=64,
        )


def test_hcs_rejects_multiple_h_phases_from_metadata() -> None:
    metadata = SimpleNamespace(image=SimpleNamespace(SizeH=3))

    with pytest.raises(ValueError, match=r"metadata has SizeH=3.*common OME-Zarr viewers do not support it"):
        conversion._validate_metadata_h_dimension(metadata)


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
    )

    assert image == "image"
    assert isinstance(captured["image_data"], DaskArray)
    assert captured["image_data"].chunks[1] == (1, 1, 1)
    assert captured["store"] == output
    assert isinstance(captured["compressor"], NumcodecsBlosc)
    assert "storage_options" not in captured
    assert "use_tensorstore" not in captured


def test_write_omezarr_ngff_directory_uses_bounded_default_chunks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata = SimpleNamespace(
        filename="large.czi",
        scale=SimpleNamespace(X=1.0, Y=1.0, Z=1.0),
        image=None,
        channelinfo=None,
    )
    captured: dict = {}
    multiscales = SimpleNamespace(metadata=SimpleNamespace(omero=None))

    def capture_image(data, *args, **kwargs) -> str:
        captured["image_data"] = data
        return "image"

    def capture_multiscales(*args, **kwargs):
        captured["chunks"] = kwargs["chunks"]
        captured["method"] = kwargs["method"]
        return multiscales

    def capture_write(store, **kwargs) -> None:
        captured["store"] = store

    monkeypatch.setattr(conversion.nz, "to_ngff_image", capture_image)
    monkeypatch.setattr(conversion.nz, "to_multiscales", capture_multiscales)
    monkeypatch.setattr(conversion.nz, "to_ngff_zarr", capture_write)

    output = tmp_path / "large.ome.zarr"
    write_omezarr_ngff(
        np.zeros((1, 4, 2, 600, 700), dtype=np.uint16),
        output,
        metadata,
        scale_factors=[2],
        downsampling_method=conversion.nz.Methods.DASK_BIN_SHRINK,
        overwrite=True,
    )

    assert captured["chunks"] == (1, 1, 1, 512, 512)
    assert captured["method"] is conversion.nz.Methods.DASK_BIN_SHRINK
    assert captured["image_data"].chunksize == (1, 1, 1, 512, 512)
    assert captured["store"] == output


def test_write_omezarr_ngff_overwrites_ozx_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "image.ozx"
    output.write_bytes(b"incomplete archive")
    metadata = SimpleNamespace(
        filename="test.czi",
        scale=SimpleNamespace(X=1.0, Y=1.0, Z=1.0),
        image=None,
        channelinfo=None,
    )
    multiscales = SimpleNamespace(metadata=SimpleNamespace(omero=None))
    monkeypatch.setattr(conversion.nz, "to_ngff_image", lambda *args, **kwargs: "image")
    monkeypatch.setattr(conversion.nz, "to_multiscales", lambda *args, **kwargs: multiscales)
    monkeypatch.setattr(conversion.nz, "to_ngff_zarr", lambda *args, **kwargs: None)

    write_omezarr_ngff(
        np.zeros((1, 1, 1, 8, 8), dtype=np.uint16),
        output,
        metadata,
        scale_factors=[2],
        overwrite=True,
    )

    assert not output.exists()


def test_write_omezarr_ngff_logs_progress_and_total_time(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata = SimpleNamespace(
        filename="test.czi",
        scale=SimpleNamespace(X=1.0, Y=1.0, Z=1.0),
        image=None,
        channelinfo=None,
    )
    multiscales = SimpleNamespace(metadata=SimpleNamespace(omero=None))
    monkeypatch.setattr(conversion.nz, "to_ngff_image", lambda *args, **kwargs: "image")
    monkeypatch.setattr(conversion.nz, "to_multiscales", lambda *args, **kwargs: multiscales)

    def write_with_progress(*args, **kwargs) -> None:
        progress = kwargs["progress"]
        progress.add_multiscales_task("Writing scales", 2)
        progress.update_multiscales_task_completed(1)
        progress.update_multiscales_task_completed(2)

    monkeypatch.setattr(conversion.nz, "to_ngff_zarr", write_with_progress)

    log_path = tmp_path / "conversion.log"
    write_omezarr_ngff(
        np.zeros((1, 1, 1, 8, 8), dtype=np.uint16),
        tmp_path / "image.ome.zarr",
        metadata,
        scale_factors=[2],
        overwrite=True,
        log_file_path=log_path,
    )

    log_text = log_path.read_text(encoding="utf-8")
    assert "NGFF write progress: [>" in log_text
    assert "] 0% (0/2 scales" in log_text
    assert "] 50% (1/2 scales" in log_text
    assert "] 100% (elapsed " in log_text
    assert "Total conversion time:" in log_text


def test_logged_progress_bar_logs_transitions_only(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO):
        with conversion._logged_progress_bar("Test write") as progress:
            progress.add_multiscales_task("Writing scales", 2)
            progress.update_multiscales_task_completed(1)

    progress_records = [record for record in caplog.records if "Test write progress" in record.message]
    assert len(progress_records) == 4
    assert "100%" in progress_records[-1].message


def test_logged_progress_bar_reports_within_scale_tasks(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO):
        with conversion._logged_progress_bar("Test write") as progress:
            progress.add_multiscales_task("Writing scales", 1)
            progress.update_multiscales_task_completed(1)
            description = "[green]Writing scale 1 of 1"
            progress.add_callback_task(description)
            task_id = progress.tasks[description]
            assert task_id is not None
            progress.update(task_id, total=100, completed=40)
            progress._log_progress()

    progress_records = [record for record in caplog.records if "Test write progress" in record.message]
    assert any("40% (40/100 tasks, Writing scale 1 of 1" in record.message for record in progress_records)


def test_logged_progress_does_not_round_up_or_regress(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO):
        with conversion._logged_progress_bar("Test write") as progress:
            progress.add_multiscales_task("Writing scales", 1)
            progress.update_multiscales_task_completed(1)
            description = "[green]Writing scale 1 of 1"
            progress.add_callback_task(description)
            task_id = progress.tasks[description]
            assert task_id is not None
            progress.update(task_id, total=10873, completed=10872)
            progress.update(task_id, total=10873, completed=10873)
            progress.update(task_id, visible=False)

    progress_records = [record.message for record in caplog.records if "Test write progress" in record.message]
    assert any("99% (10872/10873 tasks" in message for message in progress_records)
    finalizing_records = [
        message for message in progress_records if "finalizing storage (Dask tasks complete" in message
    ]
    assert len(finalizing_records) == 1
    assert "100%" in progress_records[-1]


def test_logged_progress_reports_ten_percent_increments(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO):
        with conversion._logged_progress_bar("Test write") as progress:
            description = "[green]Writing scale 1 of 1"
            progress.add_callback_task(description)
            task_id = progress.tasks[description]
            assert task_id is not None
            for completed in range(101):
                progress.update(task_id, total=100, completed=completed)

    task_records = [record.message for record in caplog.records if "tasks, Writing scale 1 of 1" in record.message]
    assert [message.split("] ", 1)[1].split("%", 1)[0] for message in task_records] == [
        "0",
        "10",
        "20",
        "30",
        "40",
        "50",
        "60",
        "70",
        "80",
        "90",
        "99",
    ]
