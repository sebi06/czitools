"""Tests for the OME-Zarr export tools (Stage 5).

These tests are skipped automatically when the optional export dependencies
(``ngff-zarr``, ``ome-zarr``, ``ome-zarr-models``, ``zarr``) are not installed.
"""

import ast
import dask
import json
import logging
import threading
import zipfile
from inspect import signature
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr
import dask.array as da
import zarr
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
import czitools.export_tools as export_tools
from czitools.export_tools import conversion
from czitools.export_tools import _logging as export_logging
from czitools.export_tools import validation
from czitools.metadata_tools.czi_metadata import CziMetadata

BASEDIR = Path(__file__).resolve().parents[3]
WELLPLATE = BASEDIR / "data" / "WP96_4Pos_B4-10_DAPI.czi"


def test_gui_import_without_qt_binding_has_install_hint(monkeypatch) -> None:
    class QtBindingsNotFoundError(RuntimeError):
        pass

    def raise_missing_binding(*_args, **_kwargs):
        raise QtBindingsNotFoundError("No Qt bindings could be found")

    monkeypatch.setattr(export_tools.importlib, "import_module", raise_missing_binding)

    with pytest.raises(ImportError, match=r'pip install "czitools\[omezarr-gui\]"'):
        export_tools.__getattr__("run_gui")


def test_legacy_export_options_are_not_public() -> None:
    assert "zarr_format" not in signature(conversion.write_omezarr).parameters
    assert "zarr_format" not in signature(conversion.convert_czi2hcs_omezarr).parameters
    assert "normalize_level_paths" not in signature(conversion.convert_czi2hcs_omezarr).parameters
    assert "normalize_level_paths" not in signature(conversion.convert_czi2hcs_ngff).parameters
    assert "use_tensorstore" not in signature(conversion.write_omezarr_ngff).parameters
    assert signature(conversion.write_omezarr).parameters["version"].default == "0.6"
    assert signature(conversion.write_omezarr_ngff).parameters["version"].default == "0.6"
    assert signature(conversion.convert_czi2hcs_ngff).parameters["version"].default == "0.5"
    assert signature(conversion.convert_czi2hcs_ngff).parameters["chunks_per_shard"].default == {"y": 4, "x": 4}
    assert signature(conversion.convert_czi2hcs_ngff).parameters["max_workers"].default == 4
    assert signature(conversion.convert_czi2hcs_omezarr).parameters["logging_detail"].default == "basic"
    assert signature(conversion.convert_czi2hcs_ngff).parameters["logging_detail"].default == "basic"


def test_ngff_hcs_rejects_unsupported_version() -> None:
    with pytest.raises(ValueError, match="HCS writing currently supports"):
        convert_czi2hcs_ngff(WELLPLATE, version="0.7")


def test_validation_dispatches_ome_ngff_06_to_ngff_zarr(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "image.ome.zarr"
    group = zarr.open_group(output, mode="w")
    group.attrs["ome"] = {"version": "0.6"}
    parsed = SimpleNamespace(metadata=SimpleNamespace(version="0.6"))
    monkeypatch.setattr(validation.nz, "from_ngff_zarr", lambda path: parsed)

    assert validate_ome_zarr(output) is True


def test_validation_dispatches_ome_ngff_06_hcs_to_hcs_reader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "plate.ome.zarr"
    group = zarr.open_group(output, mode="w")
    group.attrs["ome"] = {"version": "0.6", "plate": {}}
    calls: list[tuple[str, bool]] = []

    def fake_from_hcs_zarr(path: str, validate: bool = False) -> object:
        calls.append((path, validate))
        return object()

    monkeypatch.setattr(validation.nz, "from_hcs_zarr", fake_from_hcs_zarr)
    monkeypatch.setattr(
        validation.nz,
        "from_ngff_zarr",
        lambda _path: pytest.fail("HCS validation used the image reader"),
    )

    assert validate_ome_zarr(output) is True
    assert calls == [(str(output), True)]


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


def test_hcs_scene_dimensions_are_logged(caplog: pytest.LogCaptureFixture) -> None:
    metadata = SimpleNamespace(
        image=SimpleNamespace(
            SizeX_scene=2752,
            SizeY_scene=2208,
            SizeC=3,
            SizeZ=1,
            SizeT=1,
        )
    )

    with caplog.at_level(logging.INFO, logger=conversion.__name__):
        conversion._log_hcs_scene_dimensions(metadata)  # type: ignore[arg-type]

    assert "Individual scene dimensions: X=2752, Y=2208, C=3, Z=1, T=1" in caplog.text


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
    assert array_metadata["shape"][-2:] == [640, 640]
    assert array_metadata["chunk_grid"]["configuration"]["chunk_shape"][-2:] == [1024, 1024]
    sharding_codec = array_metadata["codecs"][0]
    assert sharding_codec["name"] == "sharding_indexed"
    assert sharding_codec["configuration"]["chunk_shape"][-2:] == [512, 512]
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
    monkeypatch.setattr(conversion, "_bounded_ngff_multiscales", lambda *args, **kwargs: multiscales)

    def capture_write(store, **kwargs) -> None:
        captured["store"] = store
        captured.update(kwargs)

    monkeypatch.setattr(conversion.nz, "to_ngff_zarr", capture_write)
    monkeypatch.setattr(
        conversion,
        "_write_ngff_directory",
        lambda store, image, factors, chunks, method, version, shards, compressor, channels: capture_write(
            store,
            version=version,
            chunks_per_shard=shards,
            compressor=compressor,
        ),
    )

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
    assert captured["version"] == "0.6"
    assert "storage_options" not in captured
    assert "use_tensorstore" not in captured


def test_write_omezarr_defaults_to_real_ome_ngff_06_store(tmp_path: Path) -> None:
    metadata = SimpleNamespace(
        filename="test.czi",
        scale=SimpleNamespace(X=1.0, Y=1.0, Z=1.0),
        image=None,
        channelinfo=None,
    )
    output = tmp_path / "image.ome.zarr"
    array = xr.DataArray(
        np.zeros((1, 1, 1, 32, 32), dtype=np.uint16),
        dims=("T", "C", "Z", "Y", "X"),
    )
    log_path = tmp_path / "conversion.log"

    conversion.write_omezarr(
        array,
        output,
        metadata,
        compression=None,
        log_file_path=log_path,
    )

    root_metadata = json.loads((output / "zarr.json").read_text(encoding="utf-8"))
    assert root_metadata["attributes"]["ome"]["version"] == "0.6"
    datasets = root_metadata["attributes"]["ome"]["multiscales"][0]["datasets"]
    assert len(datasets) == 4
    for dataset in datasets:
        array_metadata = json.loads((output / dataset["path"] / "zarr.json").read_text(encoding="utf-8"))
        assert array_metadata["codecs"][0]["name"] == "sharding_indexed"
    verified = [
        line for line in log_path.read_text(encoding="utf-8").splitlines() if "OME-Zarr layout verified" in line
    ]
    assert len(verified) == 4
    assert all("inner_chunk=" in line and "shard=" in line for line in verified)


def test_write_omezarr_ngff_defaults_to_real_ome_ngff_06_store(tmp_path: Path) -> None:
    metadata = SimpleNamespace(
        filename="test.czi",
        scale=SimpleNamespace(X=1.0, Y=1.0, Z=1.0),
        image=None,
        channelinfo=None,
    )
    output = tmp_path / "image.ome.zarr"

    write_omezarr_ngff(
        np.zeros((1, 1, 1, 32, 32), dtype=np.uint16),
        output,
        metadata,
        scale_factors=[],
        chunks=(1, 1, 1, 16, 16),
        chunks_per_shard=None,
        compression=None,
    )

    root_metadata = json.loads((output / "zarr.json").read_text(encoding="utf-8"))
    assert root_metadata["attributes"]["ome"]["version"] == "0.6"
    assert validate_ome_zarr(output) is True


@pytest.mark.parametrize("method", [conversion.nz.Methods.DASK_IMAGE_GAUSSIAN, conversion.nz.Methods.DASK_BIN_SHRINK])
@pytest.mark.parametrize("suffix", [".ome.zarr", ".ozx"])
def test_ngff_pyramid_reads_source_once_and_cleans_cache(tmp_path, monkeypatch, method, suffix) -> None:
    """Generated levels use disk-backed predecessors, including OZX output."""
    reads = []
    staged_reads = []
    cache_image = conversion._cache_ngff_image

    def capture_cache(*args, **kwargs):
        cached = cache_image(*args, **kwargs)
        if not staged_reads:
            staged_reads.append(len(reads))
        return cached

    monkeypatch.setattr(conversion, "_cache_ngff_image", capture_cache)
    write_store = conversion.nz.to_ngff_zarr

    def capture_write(*args, **kwargs):
        if suffix == ".ome.zarr":
            assert not list(tmp_path.glob(".czitools-cache-*"))
        write_store(*args, **kwargs)
        if suffix == ".ome.zarr" and kwargs.get("start_level") == 0:
            staged_reads.append(len(reads))

    monkeypatch.setattr(conversion.nz, "to_ngff_zarr", capture_write)
    pixels = np.arange(2 * 65 * 79, dtype=np.uint16).reshape(1, 2, 1, 65, 79)

    @dask.delayed
    def source(channel):
        reads.append(channel)
        return pixels[:, channel : channel + 1]

    data = da.concatenate(
        [da.from_delayed(source(channel), shape=(1, 1, 1, 65, 79), dtype=pixels.dtype) for channel in range(2)],
        axis=1,
    ).rechunk((1, 1, 1, 16, 16))
    metadata = SimpleNamespace(
        filename="large.czi", scale=SimpleNamespace(X=0.5, Y=0.5, Z=2.0), image=None, channelinfo=None
    )
    output = tmp_path / f"image{suffix}"
    previous_store = conversion.ngff_config.cache_store
    previous_target = conversion.ngff_config.memory_target
    write_omezarr_ngff(data, output, metadata, chunks=(1, 1, 1, 16, 16), pyramid_levels=2, downsampling_method=method)
    assert set(reads) == {0, 1}
    assert len(reads) == staged_reads[0]
    assert not list(tmp_path.glob(".czitools-cache-*"))
    assert conversion.ngff_config.cache_store == previous_store
    assert conversion.ngff_config.memory_target == previous_target
    multiscales = conversion.nz.from_ngff_zarr(output)
    assert [image.data.shape for image in multiscales.images] == [
        (1, 2, 1, 65, 79),
        (1, 2, 1, 32, 39),
        (1, 2, 1, 16, 19),
    ]
    np.testing.assert_array_equal(multiscales.images[0].data.compute(), pixels)
    reference = conversion.nz.to_multiscales(
        conversion.nz.to_ngff_image(
            da.from_array(pixels, chunks=(1, 1, 1, 16, 16)),
            dims=["t", "c", "z", "y", "x"],
        ),
        scale_factors=[{"z": 1, "y": 2, "x": 2}, {"z": 1, "y": 4, "x": 4}],
        chunks=(1, 1, 1, 16, 16),
        method=method,
        cache=False,
    )
    for written, expected in zip(multiscales.images[1:], reference.images[1:], strict=True):
        np.testing.assert_array_equal(written.data.compute(), expected.data.compute())
    assert multiscales.images[-1].scale["y"] == 2.0
    assert multiscales.images[-1].scale["z"] == 2.0
    log_text = (tmp_path / f"{output.stem}_ngff.log").read_text(encoding="utf-8")
    if suffix == ".ozx":
        assert "Temporary NGFF cache:" in log_text
        assert "NGFF cache base image progress: [>" in log_text
        assert "Caching x-strips" in log_text
        assert "NGFF cache level 1/2 progress:" in log_text
        assert "NGFF cache level 2/2 progress:" in log_text
    else:
        assert "Temporary NGFF cache:" not in log_text
        assert "Writing level 0/2 directly to final store" in log_text
        assert "Building level 1 from the preceding written level" in log_text
        assert "NGFF write level 2/2 progress:" in log_text
    assert "[blue]" not in log_text


@pytest.mark.parametrize(
    "available_gib,total_gib,cpus,expected",
    [(56, 96, 24, 8), (6, 16, 24, 2), (1, 16, 24, 1), (64, 96, 4, 2), (64, 96, 1, 1)],
)
def test_auto_workers_balance_memory_and_cpus(monkeypatch, available_gib, total_gib, cpus, expected) -> None:
    """Auto reserves RAM and respects CPU and concurrency limits."""
    monkeypatch.setattr(conversion.os, "cpu_count", lambda: cpus)
    monkeypatch.setattr(
        conversion.psutil, "virtual_memory",
        lambda: SimpleNamespace(available=available_gib * 2**30, total=total_gib * 2**30),
    )
    assert conversion._resolve_worker_count() == expected
    assert conversion._resolve_worker_count(12) == 12


def test_auto_workers_falls_back_when_memory_is_unavailable(monkeypatch) -> None:
    """Unavailable memory metrics select a small CPU-bounded fallback."""
    monkeypatch.setattr(conversion.os, "cpu_count", lambda: 8)

    def unavailable_memory():
        raise OSError("Memory metrics unavailable")

    monkeypatch.setattr(conversion.psutil, "virtual_memory", unavailable_memory)
    assert conversion._resolve_worker_count() == 2
    monkeypatch.setattr(conversion.os, "cpu_count", lambda: 1)
    assert conversion._resolve_worker_count() == 1


@pytest.mark.parametrize("backend", ["ngff", "ome"])
def test_writers_run_selected_workers_concurrently(tmp_path, backend) -> None:
    """Four independent source tasks must execute together, not with a fixed two-worker limit."""
    barrier = threading.Barrier(4, timeout=10)
    thread_ids = set()
    lock = threading.Lock()

    @dask.delayed
    def source(channel):
        with lock:
            thread_ids.add(threading.get_ident())
        barrier.wait()
        return np.full((1, 1, 1, 8, 8), channel, dtype=np.uint16)

    data = da.concatenate([
        da.from_delayed(source(channel), shape=(1, 1, 1, 8, 8), dtype=np.uint16)
        for channel in range(4)
    ], axis=1)
    array = xr.DataArray(data, dims=("T", "C", "Z", "Y", "X"))
    metadata = SimpleNamespace(filename="parallel.czi", scale=None, image=None, channelinfo=None)
    writer = conversion.write_omezarr_ngff if backend == "ngff" else conversion.write_omezarr
    writer(array, tmp_path / "parallel.ome.zarr", metadata, pyramid_levels=0, worker_count=4)
    assert len(thread_ids) == 4


@pytest.mark.parametrize("writer", [conversion.write_omezarr_ngff, conversion.write_omezarr])
def test_writers_reject_negative_worker_count(tmp_path, writer) -> None:
    """Invalid overrides fail before creating or overwriting output."""
    output = tmp_path / "invalid.ome.zarr"
    with pytest.raises(ValueError, match="worker_count must be non-negative"):
        writer(np.zeros((1, 1, 1, 8, 8)), output, object(), worker_count=-1)
    assert not output.exists()


@pytest.mark.parametrize("worker_count", [1, 2, 4, 8])
def test_ngff_failure_restores_resources_and_removes_cache(tmp_path, monkeypatch, worker_count) -> None:
    """A failed disk write must not leave caches or change global settings."""
    previous_store = conversion.ngff_config.cache_store
    previous_target = conversion.ngff_config.memory_target
    previous_workers = dask.config.get("num_workers", default=None)

    def fail(*args, **kwargs):
        assert dask.config.get("num_workers") == worker_count
        assert conversion.ngff_config.memory_target == 256 * 1024 * 1024
        raise OSError("Disk full")

    monkeypatch.setattr(conversion.nz, "to_ngff_zarr", fail)
    metadata = SimpleNamespace(
        filename="test.czi", scale=SimpleNamespace(X=1.0, Y=1.0, Z=1.0), image=None, channelinfo=None
    )
    with pytest.raises(OSError, match="Disk full"):
        write_omezarr_ngff(
            da.zeros((1, 1, 1, 16, 16), chunks=8), tmp_path / "image.ome.zarr", metadata,
            pyramid_levels=1, worker_count=worker_count,
        )
    assert conversion.ngff_config.cache_store == previous_store
    assert conversion.ngff_config.memory_target == previous_target
    assert dask.config.get("num_workers", default=None) == previous_workers
    assert not list(tmp_path.glob(".czitools-cache-*"))


@pytest.fixture
def native_czi_pyramid():
    """Provide distinct native 3x pixels not reproducible from the base."""
    images = [
        conversion.nz.to_ngff_image(
            da.full((1, 2, 1, height, width), index + 11, dtype=np.uint16, chunks=(1, 1, 1, 16, 16)),
            dims=["t", "c", "z", "y", "x"],
            scale={"t": 1.0, "c": 1.0, "z": 2.0, "y": 0.5 * factor, "x": 0.25 * factor},
            translation={"t": 0.0, "c": 0.0, "z": 0.0, "y": 0.25 * (factor - 1), "x": 0.125 * (factor - 1)},
            axes_units={"t": "second", "z": "micrometer", "y": "micrometer", "x": "micrometer"},
            name="native.czi",
        )
        for index, (height, width, factor) in enumerate([(82, 94, 1), (27, 31, 3), (9, 10, 9)])
    ]
    return conversion._ngff_multiscales_metadata(images, [], (1, 1, 1, 16, 16), None)


@pytest.mark.parametrize("backend,suffix", [("ngff", ".ome.zarr"), ("ngff", ".ozx"), ("ome", ".ome.zarr")])
def test_writers_preserve_native_czi_pyramid(tmp_path, monkeypatch, native_czi_pyramid, backend, suffix) -> None:
    """Both libraries preserve stored pixels and transforms without resampling."""
    source = native_czi_pyramid
    expected = [(level.data.compute(), dict(level.scale), dict(level.translation)) for level in source.images]
    monkeypatch.setattr(conversion.nz, "to_multiscales", lambda *args, **kwargs: pytest.fail("NGFF resampling used"))
    monkeypatch.setattr(
        conversion, "OMEZarrMultiscale", lambda *args, **kwargs: pytest.fail("ome-zarr resampling used")
    )
    channels = [
        {"label": label, "color": color, "window": {"min": 0, "max": 65535, "start": 0, "end": 65535}}
        for label, color in [("DAPI", "0000FF"), ("EGFP", "00FF00")]
    ]
    monkeypatch.setattr(conversion, "create_channel_list", lambda _: channels)
    metadata = SimpleNamespace(
        filename="native.czi", scale=SimpleNamespace(X=0.25, Y=0.5, Z=2.0), image=None, channelinfo=None
    )
    output = tmp_path / f"image{suffix}"
    writer = conversion.write_omezarr_ngff if backend == "ngff" else conversion.write_omezarr
    writer(
        source.images[0].data,
        output,
        metadata,
        source_multiscales=source,
        compression=None,
        pyramid_levels=8,
    )
    written = conversion.nz.from_ngff_zarr(output)
    assert len(written.images) == 3
    for level, (pixels, scale, translation) in zip(written.images, expected, strict=True):
        np.testing.assert_array_equal(level.data.compute(), pixels)
        assert level.scale == scale
        assert level.translation == translation
        assert all(level.axes_units.get(dim) == source.images[0].axes_units.get(dim) for dim in level.dims)
    if suffix == ".ozx":
        with zipfile.ZipFile(output) as archive:
            root_metadata = json.loads(archive.read("zarr.json"))
    else:
        root_metadata = json.loads((output / "zarr.json").read_text(encoding="utf-8"))
    written_channels = root_metadata["attributes"]["ome"]["omero"]["channels"]
    assert [(channel["label"], channel["color"]) for channel in written_channels] == [
        ("DAPI", "0000FF"),
        ("EGFP", "00FF00"),
    ]
    assert not list(tmp_path.glob(".czitools-cache-*"))


@pytest.mark.parametrize("backend", ["ngff", "ome"])
def test_hcs_writers_preserve_native_czi_pyramid(tmp_path, monkeypatch, native_czi_pyramid, backend) -> None:
    """Both HCS entry points use the supplied native field levels."""
    metadata = SimpleNamespace(
        filename="plate.czi", scale=SimpleNamespace(X=0.25, Y=0.5, Z=2.0), image=None, channelinfo=None
    )
    layout = SimpleNamespace(
        source="hcs",
        row_names=["B"],
        col_names=["04"],
        field_count=1,
        wells=[SimpleNamespace(row="B", column="04", path="B/04", fields=[(0, 3)])],
    )
    calls = []

    def read_levels(path, scene, *args, **kwargs):
        calls.append(scene)
        return native_czi_pyramid

    monkeypatch.setattr(conversion, "CziMetadata", lambda _: metadata)
    monkeypatch.setattr(conversion, "resolve_hcs_layout", lambda *args, **kwargs: layout)
    monkeypatch.setattr(conversion.read_tools, "get_pyramid_zooms", lambda _: [1.0, 1 / 3, 1 / 9])
    monkeypatch.setattr(conversion, "_read_single_scene_multiscales", read_levels)
    monkeypatch.setattr(
        conversion, "_read_single_scene", lambda *args, **kwargs: pytest.fail("Base-only field reader used")
    )
    monkeypatch.setattr(conversion.nz, "to_multiscales", lambda *args, **kwargs: pytest.fail("Stored fields resampled"))
    if backend == "ngff":
        output = conversion.convert_czi2hcs_ngff(tmp_path / "plate.czi", output_dir=tmp_path, compression=None)
    else:
        output = conversion.convert_czi2hcs_omezarr(tmp_path / "plate.czi", compression=None)
    assert calls == [3]
    root = zarr.open_group(output, mode="r")
    field = root["B/04/0"]
    datasets = field.attrs["ome"]["multiscales"][0]["datasets"]
    assert len(datasets) == 3
    for dataset, level in zip(datasets, native_czi_pyramid.images, strict=True):
        np.testing.assert_array_equal(field[dataset["path"]][:], level.data.compute())
        assert dataset["coordinateTransformations"][0]["scale"] == [level.scale[dim] for dim in level.dims]


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

    def capture_multiscales(image, scale_factors, chunks, method):
        captured["chunks"] = chunks
        captured["method"] = method
        return multiscales

    def capture_write(store, **kwargs) -> None:
        captured["store"] = store

    monkeypatch.setattr(conversion.nz, "to_ngff_image", capture_image)
    monkeypatch.setattr(conversion, "_bounded_ngff_multiscales", capture_multiscales)
    monkeypatch.setattr(conversion.nz, "to_ngff_zarr", capture_write)

    def capture_directory(store, image, factors, chunks, method, version, shards, compressor, channels):
        capture_multiscales(image, factors, chunks, method)
        capture_write(store)

    monkeypatch.setattr(conversion, "_write_ngff_directory", capture_directory)

    output = tmp_path / "large.ome.zarr"
    write_omezarr_ngff(
        np.zeros((1, 4, 2, 600, 700), dtype=np.uint16),
        output,
        metadata,
        scale_factors=[2],
        downsampling_method=conversion.nz.Methods.DASK_BIN_SHRINK,
        overwrite=True,
    )

    assert captured["chunks"] == (1, 1, 8, 512, 512)
    assert captured["method"] is conversion.nz.Methods.DASK_BIN_SHRINK
    assert captured["image_data"].chunksize == (1, 1, 2, 512, 512)
    assert captured["store"] == output


def test_write_omezarr_ngff_shards_every_pyramid_level(tmp_path: Path) -> None:
    metadata = SimpleNamespace(
        filename="volume.czi",
        scale=SimpleNamespace(X=1.0, Y=1.0, Z=1.0),
        image=None,
        channelinfo=None,
    )
    output = tmp_path / "volume.ome.zarr"
    log_path = tmp_path / "conversion.log"

    write_omezarr_ngff(
        np.zeros((1, 1, 9, 64, 64), dtype=np.uint16),
        output,
        metadata,
        pyramid_levels=2,
        compression=None,
        overwrite=True,
        log_file_path=log_path,
    )

    root_metadata = json.loads((output / "zarr.json").read_text(encoding="utf-8"))
    arrays = [
        metadata
        for metadata in root_metadata["consolidated_metadata"]["metadata"].values()
        if metadata.get("node_type") == "array"
    ]
    assert len(arrays) == 3
    for array_metadata in arrays:
        sharding_codec = array_metadata["codecs"][0]
        assert sharding_codec["name"] == "sharding_indexed"
        assert sharding_codec["configuration"]["chunk_shape"][:3] == [1, 1, 8]
    verified = [
        line for line in log_path.read_text(encoding="utf-8").splitlines() if "OME-Zarr layout verified" in line
    ]
    assert len(verified) == 3
    assert all("codec=sharding_indexed" in line for line in verified)


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
    monkeypatch.setattr(conversion, "_bounded_ngff_multiscales", lambda *args, **kwargs: multiscales)
    captured: dict = {}

    def capture_write(*args, **kwargs) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(conversion.nz, "to_ngff_zarr", capture_write)

    write_omezarr_ngff(
        np.zeros((1, 1, 1, 8, 8), dtype=np.uint16),
        output,
        metadata,
        scale_factors=[2],
        overwrite=True,
    )

    assert not output.exists()
    assert captured["version"] == "0.5"


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
    monkeypatch.setattr(conversion, "_bounded_ngff_multiscales", lambda *args, **kwargs: multiscales)

    def write_with_progress(*args, **kwargs) -> None:
        progress = kwargs["progress"]
        progress.add_multiscales_task("Writing scales", 2)
        progress.update_multiscales_task_completed(1)
        progress.update_multiscales_task_completed(2)

    monkeypatch.setattr(conversion.nz, "to_ngff_zarr", write_with_progress)

    def capture_directory(*args):
        with conversion._logged_progress_bar("NGFF write") as progress:
            write_with_progress(progress=progress)

    monkeypatch.setattr(conversion, "_write_ngff_directory", capture_directory)

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
