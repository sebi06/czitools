from dataclasses import dataclass

import pytest
from cmap import Colormap

from czitools.utils.ndv_tools import create_luts_ndv, create_scales_ndv, normalize_luts


@dataclass
class _Image:
    SizeC: int | None


@dataclass
class _ChannelInfo:
    names: list[str] | None
    colors: list[str] | None


@dataclass
class _Scale:
    Z: float | int | None
    Y: float | int | None
    X: float | int | None


@dataclass
class _Metadata:
    image: _Image
    channelinfo: _ChannelInfo
    scale: _Scale


def test_normalize_luts_from_dict() -> None:
    luts = {0: {"cmap": Colormap(["#000000", "#FF0000"], name="cm_0")}}
    result = normalize_luts(luts)
    assert result == luts


def test_normalize_luts_from_list() -> None:
    cmaps = [
        Colormap(["#000000", "#FF0000"], name="cm_0"),
        Colormap(["#000000", "#00FF00"], name="cm_1"),
    ]

    result = normalize_luts(cmaps)

    assert list(result.keys()) == [0, 1]
    assert result[0]["cmap"] is cmaps[0]
    assert result[1]["cmap"] is cmaps[1]


def test_normalize_luts_invalid_type() -> None:
    with pytest.raises(TypeError, match="luts must be a dict or list"):
        normalize_luts(123)  # type: ignore[arg-type]


def test_normalize_luts_rejects_string_input() -> None:
    with pytest.raises(TypeError, match="luts must be a dict or list"):
        normalize_luts("invalid")  # type: ignore[arg-type]


def test_create_luts_ndv_uses_channel_names_and_colors() -> None:
    mdata = _Metadata(
        image=_Image(SizeC=2),
        channelinfo=_ChannelInfo(names=["DAPI", "FITC"], colors=["FFFF0000", "FF00FF00"]),
        scale=_Scale(Z=1.0, Y=1.0, X=1.0),
    )

    luts = create_luts_ndv(mdata)

    assert list(luts.keys()) == [0, 1]
    assert isinstance(luts[0]["cmap"], Colormap)
    assert isinstance(luts[1]["cmap"], Colormap)


def test_create_luts_ndv_fallbacks_for_missing_channel_info() -> None:
    mdata = _Metadata(
        image=_Image(SizeC=2),
        channelinfo=_ChannelInfo(names=["OnlyOne"], colors=["BAD"]),
        scale=_Scale(Z=1.0, Y=1.0, X=1.0),
    )

    luts = create_luts_ndv(mdata)

    assert list(luts.keys()) == [0, 1]
    assert isinstance(luts[0]["cmap"], Colormap)
    assert isinstance(luts[1]["cmap"], Colormap)


def test_create_scales_ndv_values_and_defaults() -> None:
    mdata = _Metadata(
        image=_Image(SizeC=1),
        channelinfo=_ChannelInfo(names=None, colors=None),
        scale=_Scale(Z=2, Y=None, X=0.5),
    )

    scales = create_scales_ndv(mdata)

    assert scales == {"Z": 2.0, "Y": 1.0, "X": 0.5}
