"""Regression tests for the modular reader API and lazy 6D reads."""

from contextlib import contextmanager
from importlib import import_module
from pathlib import Path
from typing import TypeAlias

import dask.array as da
import numpy as np

import czitools.read_tools as public_api
from czitools.read_tools import _helpers, array6d, attachments, field_well, stacks, tiles

BASEDIR = Path(__file__).resolve().parents[3]


def test_split_modules_preserve_public_and_legacy_imports() -> None:
    """Every public reader resolves to its focused implementation module."""
    legacy_api = import_module("czitools.read_tools.read_tools")
    implementations = {
        "read_6darray": array6d,
        "read_attachments": attachments,
        "read_field": field_well,
        "read_well": field_well,
        "read_tiles": tiles,
        "read_stacks": stacks,
        "read_stacks_list": stacks,
        "read_stacks_stacked": stacks,
    }

    for name, module in implementations.items():
        implementation = getattr(module, name)
        assert getattr(public_api, name) is implementation
        assert getattr(legacy_api, name) is implementation


def test_reader_aliases_are_declared_as_type_aliases() -> None:
    """Shared reader aliases are explicit rather than untyped assignments."""
    aliases = {
        "CziPath",
        "Array6D",
        "StackArray",
        "StackList",
        "ReadStacksReturn",
        "ReadStacksWithMetaReturn",
        "LazyReadStrategy",
    }

    assert aliases <= _helpers.__annotations__.keys()
    assert all(_helpers.__annotations__[name] is TypeAlias for name in aliases)


def test_read_6darray_defers_plane_reads_until_compute(monkeypatch) -> None:
    """Building a Dask-backed 6D array must not perform pixel I/O."""
    filepath = BASEDIR / "data" / "CellDivision_T3_Z5_CH2_X240_Y170.czi"
    original_read_plane = array6d._read_plane
    calls = []

    def tracked_read_plane(*args, **kwargs):
        calls.append((args, kwargs))
        return original_read_plane(*args, **kwargs)

    monkeypatch.setattr(array6d, "_read_plane", tracked_read_plane)

    result, _ = array6d.read_6darray(filepath, use_dask=True, use_xarray=False)

    assert isinstance(result, da.Array)
    assert calls == []

    pixel = result[0, 0, 0, 0, 0, 0].compute(scheduler="synchronous")

    assert pixel.shape == ()
    assert len(calls) == 1
    assert calls[0][0][3] == (0, 0, 240, 170)


def test_read_6darray_constrains_eager_reads_to_nonpyramid_roi(monkeypatch) -> None:
    """Pyramid overhangs must not enlarge regular full-resolution arrays."""
    filepath = BASEDIR / "data" / "CellDivision_T3_Z5_CH2_X240_Y170.czi"
    original_open_czi = array6d.pyczi.open_czi
    observed_rois = []

    class ReaderProxy:
        def __init__(self, reader) -> None:
            self._reader = reader

        def __getattr__(self, name):
            return getattr(self._reader, name)

        def read(self, *args, **kwargs):
            roi = kwargs.get("roi")
            observed_rois.append(roi)
            image = self._reader.read(*args, **kwargs)
            if roi is None:
                return np.pad(image, ((0, 1), (0, 1), (0, 0)))
            return image

    @contextmanager
    def tracked_open_czi(*args, **kwargs):
        with original_open_czi(*args, **kwargs) as reader:
            yield ReaderProxy(reader)

    monkeypatch.setattr(array6d.pyczi, "open_czi", tracked_open_czi)

    result, _ = array6d.read_6darray(
        filepath,
        planes={"S": (0, 0), "T": (0, 0), "C": (0, 0), "Z": (0, 0)},
        use_xarray=False,
    )

    assert result is not None
    assert result.shape == (1, 1, 1, 1, 170, 240)
    assert observed_rois == [(0, 0, 240, 170)]


def test_spatial_regions_preserve_per_scene_origins() -> None:
    """Equal-sized scenes must retain their distinct absolute ROI origins."""
    filepath = BASEDIR / "data" / "S2_3x3_CH2.czi"
    metadata = array6d.czimd.CziMetadata(filepath)

    first_roi, first_shape = array6d._spatial_region(metadata, scene=0, zoom=1.0)
    second_roi, second_shape = array6d._spatial_region(metadata, scene=1, zoom=1.0)

    assert first_roi == (494104, 354104, 1792, 1792)
    assert second_roi == (584104, 354104, 1792, 1792)
    assert first_shape == second_shape == (1792, 1792)
