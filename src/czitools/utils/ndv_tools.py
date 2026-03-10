from __future__ import annotations

from typing import Protocol, Sequence, TypeAlias

from cmap import Colormap


NDVLutEntry: TypeAlias = dict[str, Colormap]
NDVLutMapping: TypeAlias = dict[int, NDVLutEntry]


class _ImageLike(Protocol):
    """Minimal image metadata contract used by NDV helper functions.

    The real metadata object in ``czitools`` contains more fields, but NDV LUT
    and scaling creation only depends on ``SizeC``.
    """

    @property
    def SizeC(self) -> int | None: ...


class _ChannelInfoLike(Protocol):
    """Minimal channel info contract used by NDV helper functions."""

    @property
    def names(self) -> Sequence[str] | None: ...

    @property
    def colors(self) -> Sequence[str] | None: ...


class _ScaleLike(Protocol):
    """Minimal spatial scaling contract for Z/Y/X dimensions."""

    @property
    def Z(self) -> float | int | None: ...

    @property
    def Y(self) -> float | int | None: ...

    @property
    def X(self) -> float | int | None: ...


class NDVMetadataLike(Protocol):
    """Structural typing interface for metadata consumed by this module.

    Any object that exposes the required ``image``, ``channelinfo``, and
    ``scale`` attributes is accepted, independent of its concrete class.
    """

    @property
    def image(self) -> _ImageLike: ...

    @property
    def channelinfo(self) -> _ChannelInfoLike: ...

    @property
    def scale(self) -> _ScaleLike: ...


def normalize_luts(luts_like: NDVLutMapping | Sequence[Colormap]) -> NDVLutMapping:
    """Normalize LUT definitions to NDV's channel-indexed mapping format.

    NDV expects LUTs in the shape ``{channel_index: {"cmap": Colormap}}``.
    This helper accepts either an already normalized mapping or a sequence of
    ``Colormap`` objects and converts it to the expected mapping.

    Args:
        luts_like: Either a normalized mapping or a sequence of ``Colormap``
            instances ordered by channel index.

    Returns:
        A normalized channel-indexed LUT mapping.

    Raises:
        TypeError: If ``luts_like`` is neither a mapping nor a sequence.
    """
    if isinstance(luts_like, dict):
        return luts_like

    if isinstance(luts_like, (list, tuple)):
        return {i: {"cmap": cmap} for i, cmap in enumerate(luts_like)}

    raise TypeError("luts must be a dict or list")


def _to_rgb_hex_from_argb(color_argb: str) -> str:
    """Convert ARGB-like channel metadata to ``#RRGGBB``.

    CZI channel colors are commonly provided as 8-character ARGB strings
    (``AARRGGBB``). NDV expects HTML-like RGB hex values (``#RRGGBB``), so we
    keep only the trailing RGB part. If metadata is missing or malformed,
    return a deterministic fallback color.
    """
    color_value = str(color_argb)
    if len(color_value) >= 8:
        return f"#{color_value[-6:]}"

    # Fallback to green when metadata is empty or malformed.
    return "#00FF00"


def create_luts_ndv(mdata: NDVMetadataLike) -> NDVLutMapping:
    """Create per-channel NDV LUT definitions from CZI metadata.

    For each channel, this function builds a two-stop colormap from black to
    the channel's display color provided in metadata.

    Args:
        mdata: Metadata object that satisfies the ``NDVMetadataLike`` protocol.

    Returns:
        NDV-compatible LUT mapping ``{channel_index: {"cmap": Colormap}}``.
    """
    luts: NDVLutMapping = {}

    size_c = int(getattr(mdata.image, "SizeC", 0) or 0)
    names = list(getattr(mdata.channelinfo, "names", None) or [])
    colors = list(getattr(mdata.channelinfo, "colors", None) or [])

    for ch_index in range(size_c):
        # Be defensive: metadata arrays can be shorter than ``SizeC``.
        chname = names[ch_index] if ch_index < len(names) else f"ch{ch_index}"
        color_argb = colors[ch_index] if ch_index < len(colors) else "FF00FF00"
        rgb = _to_rgb_hex_from_argb(color_argb)
        luts[ch_index] = {"cmap": Colormap(["#000000", rgb], name="cm_" + chname)}

    return normalize_luts(luts)


def create_scales_ndv(mdata: NDVMetadataLike) -> dict[str, float]:
    """Create NDV scale mapping for the spatial dimensions (Z, Y, X).

    NDV expects plain floats for dimension scaling. Missing or ``None`` values
    are replaced with ``1.0`` to keep visualization functional.

    Args:
        mdata: Metadata object that satisfies the ``NDVMetadataLike`` protocol.

    Returns:
        Dictionary with ``Z``, ``Y``, and ``X`` scale values.
    """
    return {
        "Z": float(getattr(mdata.scale, "Z", 1.0) or 1.0),
        "Y": float(getattr(mdata.scale, "Y", 1.0) or 1.0),
        "X": float(getattr(mdata.scale, "X", 1.0) or 1.0),
    }
