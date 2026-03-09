from __future__ import annotations

from typing import Protocol, Sequence, TypeAlias

from cmap import Colormap


NDVLutEntry: TypeAlias = dict[str, Colormap]
NDVLutMapping: TypeAlias = dict[int, NDVLutEntry]


class _ImageLike(Protocol):
    @property
    def SizeC(self) -> int | None: ...


class _ChannelInfoLike(Protocol):
    @property
    def names(self) -> Sequence[str] | None: ...

    @property
    def colors(self) -> Sequence[str] | None: ...


class _ScaleLike(Protocol):
    @property
    def Z(self) -> float | int | None: ...

    @property
    def Y(self) -> float | int | None: ...

    @property
    def X(self) -> float | int | None: ...


class NDVMetadataLike(Protocol):
    @property
    def image(self) -> _ImageLike: ...

    @property
    def channelinfo(self) -> _ChannelInfoLike: ...

    @property
    def scale(self) -> _ScaleLike: ...


def normalize_luts(luts_like: NDVLutMapping | Sequence[Colormap]) -> NDVLutMapping:
    """Normalize LUT definitions to NDV's channel-indexed mapping format."""
    if isinstance(luts_like, dict):
        return luts_like

    if isinstance(luts_like, (list, tuple)):
        return {i: {"cmap": cmap} for i, cmap in enumerate(luts_like)}

    raise TypeError("luts must be a dict or list")


def _to_rgb_hex_from_argb(color_argb: str) -> str:
    """Convert ARGB-like channel metadata to ``#RRGGBB`` with a safe fallback."""
    color_value = str(color_argb)
    if len(color_value) >= 8:
        return f"#{color_value[-6:]}"

    # Fallback to green when metadata is empty or malformed.
    return "#00FF00"


def create_luts_ndv(mdata: NDVMetadataLike) -> NDVLutMapping:
    """Create per-channel NDV LUT definitions from CZI metadata."""
    luts: NDVLutMapping = {}

    size_c = int(getattr(mdata.image, "SizeC", 0) or 0)
    names = list(getattr(mdata.channelinfo, "names", None) or [])
    colors = list(getattr(mdata.channelinfo, "colors", None) or [])

    for ch_index in range(size_c):
        chname = names[ch_index] if ch_index < len(names) else f"ch{ch_index}"
        color_argb = colors[ch_index] if ch_index < len(colors) else "FF00FF00"
        rgb = _to_rgb_hex_from_argb(color_argb)
        luts[ch_index] = {"cmap": Colormap(["#000000", rgb], name="cm_" + chname)}

    return normalize_luts(luts)


def create_scales_ndv(mdata: NDVMetadataLike) -> dict[str, float]:
    """Create NDV scale mapping for the spatial dimensions (Z, Y, X)."""
    return {
        "Z": float(getattr(mdata.scale, "Z", 1.0) or 1.0),
        "Y": float(getattr(mdata.scale, "Y", 1.0) or 1.0),
        "X": float(getattr(mdata.scale, "X", 1.0) or 1.0),
    }
