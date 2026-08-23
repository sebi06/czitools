"""Regression tests for the modular reader API and lazy 6D reads."""

from importlib import import_module
from pathlib import Path
from typing import TypeAlias

import dask.array as da

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
