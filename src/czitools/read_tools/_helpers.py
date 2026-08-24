"""Shared types, constants, and coordinate helpers for CZI readers."""

import os
from typing import Any, Literal, TypeAlias

import dask.array as da
import numpy as np
import xarray as xr

from czitools.metadata_tools import czi_metadata as czimd
from czitools.utils import logging_tools

logger = logging_tools.set_logging()

CziPath: TypeAlias = str | os.PathLike[str]
Array6D: TypeAlias = np.ndarray | da.Array | xr.DataArray
StackArray: TypeAlias = xr.DataArray | np.ndarray | da.Array
StackList: TypeAlias = list[StackArray]
ReadStacksReturn: TypeAlias = StackList | StackArray
ReadStacksWithMetaReturn: TypeAlias = tuple[ReadStacksReturn, list[str], int, czimd.CziMetadata]
LazyReadStrategy: TypeAlias = Literal["chunk", "plane"]


def _as_int(value: Any, default: int = 1) -> int:
    """Convert optional numeric values to int with a safe fallback."""
    if value is None:
        return default
    return int(value)


def _as_float(value: Any, default: float = 1.0) -> float:
    """Convert optional numeric values to float with a safe fallback."""
    if value is None:
        return default
    return float(value)


def _channel_names_or_default(mdata: czimd.CziMetadata, size_c: int) -> list[str]:
    """Return channel names with deterministic fallback labels."""
    names: list[str] = []
    channelinfo = mdata.channelinfo
    if channelinfo is not None and channelinfo.names is not None:
        names = [str(name) for name in channelinfo.names[:size_c]]

    if len(names) < size_c:
        names.extend(f"ch{i}" for i in range(len(names), size_c))

    return names


def _get_axis_coord_step(scale: Any, axis: str, zoom: float = 1.0) -> float:
    """Return physical coordinate spacing for a given axis."""
    if axis == "Z":
        return _as_float(getattr(scale, "Z", None), 1.0)
    if axis == "Y":
        return _as_float(
            getattr(scale, "Y_sf", None),
            _as_float(getattr(scale, "Y", None), 1.0) * (1.0 / zoom),
        )
    if axis == "X":
        return _as_float(
            getattr(scale, "X_sf", None),
            _as_float(getattr(scale, "X", None), 1.0) * (1.0 / zoom),
        )
    raise ValueError(f"Unsupported axis for coordinate step: {axis}")


_EXTRA_DIMS = ["V", "R", "I", "H", "M"]
_CORE_DIMS = ["T", "C", "Z"]
_PLANE_DIMS_READ = ["T", "Z", "C", "V", "R", "I", "H", "M"]
