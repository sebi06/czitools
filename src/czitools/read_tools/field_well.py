"""HCS well and field convenience readers."""

from typing import Any, cast

import dask.array as da
import numpy as np
import xarray as xr

from czitools.metadata_tools import czi_metadata as czimd
from czitools.metadata_tools.hcs import CziPlate, resolve_field, resolve_well

from ._helpers import Array6D, CziPath, logger
from .array6d import read_6darray


def _require_hcs_plate(mdata: czimd.CziMetadata) -> CziPlate:
    """Return the detected HCS plate or raise a clear error explaining its absence."""

    if mdata.hcs is None:
        reason = mdata.hcs_status.reason if mdata.hcs_status is not None else "No HCS plate detected."
        raise ValueError(f"CZI has no usable HCS plate metadata: {reason}")
    return mdata.hcs


def read_field(
    filepath: CziPath,
    well: str,
    field: int | str = 0,
    *,
    use_dask: bool = False,
    chunk_zyx: bool = False,
    planes: dict[str, tuple[int, int]] | None = None,
    zoom: float = 1.0,
    use_xarray: bool = True,
    adapt_metadata: bool = False,
) -> tuple[Array6D | None, czimd.CziMetadata]:
    """Read a single well/field of an HCS plate as a 6D array (STCZYX(A)).

    The field is resolved through the canonical HCS model (``CziMetadata.hcs``)
    and read via :func:`read_6darray` using ``planes={"S": (scene, scene)}``, so
    it reuses the existing single-scene code path.

    Args:
        filepath (CziPath): Path to the CZI image file.
        well (str): Well name such as ``"B4"``, ``"b04"`` or ``"B/4"``.
        field (Union[int, str]): Well-local 0-based field index (``int``) or a
            source-scoped ``RegionId`` (``str``). Defaults to 0.
        use_dask (bool): See :func:`read_6darray`. Defaults to False.
        chunk_zyx (bool): See :func:`read_6darray`. Defaults to False.
        planes (Optional[Dict[str, Tuple[int, int]]]): Additional T/C/Z substack
            ranges. Any ``"S"`` entry is overridden by the resolved field.
        zoom (float): See :func:`read_6darray`. Defaults to 1.0.
        use_xarray (bool): See :func:`read_6darray`. Defaults to True.
        adapt_metadata (bool): See :func:`read_6darray`. Defaults to False.

    Returns:
        Tuple[Optional[Array6D], CziMetadata]: Same shape as
            :func:`read_6darray`; the array holds a single scene (``S == 1``).

    Raises:
        ValueError: If the CZI has no usable HCS plate metadata.
        KeyError, IndexError, TypeError: If the well/field cannot be resolved.
    """

    mdata = czimd.CziMetadata(str(filepath))
    plate = _require_hcs_plate(mdata)
    resolved = resolve_field(plate, well, field)

    field_planes: dict[str, tuple[int, int]] = dict(planes) if planes else {}
    field_planes["S"] = (resolved.scene_index, resolved.scene_index)

    return read_6darray(
        filepath,
        use_dask=use_dask,
        chunk_zyx=chunk_zyx,
        planes=field_planes,
        zoom=zoom,
        use_xarray=use_xarray,
        adapt_metadata=adapt_metadata,
    )


def read_well(
    filepath: CziPath,
    well: str,
    *,
    fields: list[int | str] | None = None,
    stack: bool = False,
    use_dask: bool = False,
    chunk_zyx: bool = False,
    planes: dict[str, tuple[int, int]] | None = None,
    zoom: float = 1.0,
    use_xarray: bool = True,
    adapt_metadata: bool = False,
) -> tuple[list[Array6D] | Array6D | None, czimd.CziMetadata]:
    """Read all (or selected) fields of one well of an HCS plate.

    By default this returns a list of per-field 6D arrays (in well-local field
    order) because fields may legitimately have different shapes. Set
    ``stack=True`` to concatenate the fields along the ``S`` axis, which is only
    allowed when every selected field has an identical shape.

    Args:
        filepath (CziPath): Path to the CZI image file.
        well (str): Well name such as ``"B4"``, ``"b04"`` or ``"B/4"``.
        fields (Optional[List[Union[int, str]]]): Specific fields to read, given
            as local indices and/or ``RegionId`` strings. Defaults to all fields.
        stack (bool): Concatenate fields along ``S`` when True. Requires equal
            field shapes. Defaults to False.
        use_dask (bool): See :func:`read_6darray`. Defaults to False.
        chunk_zyx (bool): See :func:`read_6darray`. Defaults to False.
        planes (Optional[Dict[str, Tuple[int, int]]]): Additional T/C/Z substack
            ranges applied to every field. Any ``"S"`` entry is overridden.
        zoom (float): See :func:`read_6darray`. Defaults to 1.0.
        use_xarray (bool): See :func:`read_6darray`. Defaults to True.
        adapt_metadata (bool): See :func:`read_6darray`. Defaults to False.

    Returns:
        Tuple[Union[List[Array6D], Array6D, None], CziMetadata]: A list of
            per-field arrays (``stack=False``), a single stacked array
            (``stack=True``), or ``None`` if no field could be read. The returned
            metadata is the plate-level metadata for ``filepath``.

    Raises:
        ValueError: If the CZI has no usable HCS plate metadata, or if
            ``stack=True`` is requested with differing field shapes.
    """

    mdata = czimd.CziMetadata(str(filepath))
    plate = _require_hcs_plate(mdata)
    target_well = resolve_well(plate, well)

    if fields is None:
        selectors: list[int | str] = [item.field_index for item in target_well.fields]
    else:
        selectors = list(fields)

    arrays: list[Array6D] = []
    for selector in selectors:
        resolved = resolve_field(plate, well, selector)
        field_planes: dict[str, tuple[int, int]] = dict(planes) if planes else {}
        field_planes["S"] = (resolved.scene_index, resolved.scene_index)
        array, _ = read_6darray(
            filepath,
            use_dask=use_dask,
            chunk_zyx=chunk_zyx,
            planes=field_planes,
            zoom=zoom,
            use_xarray=use_xarray,
            adapt_metadata=adapt_metadata,
        )
        if array is None:
            logger.warning(f"Field {selector!r} of well {well!r} could not be read; skipping.")
            continue
        arrays.append(array)

    if not arrays:
        return None, mdata

    if not stack:
        return arrays, mdata

    first_shape = arrays[0].shape
    if any(item.shape != first_shape for item in arrays):
        raise ValueError(
            "Cannot stack fields with differing shapes; call read_well(..., stack=False) "
            "to obtain a list of per-field arrays."
        )

    if use_xarray:
        stacked: Array6D = xr.concat(cast(Any, arrays), dim="S")
    elif use_dask:
        stacked = da.concatenate(cast(Any, arrays), axis=0)
    else:
        stacked = np.concatenate(cast(Any, arrays), axis=0)

    return stacked, mdata
