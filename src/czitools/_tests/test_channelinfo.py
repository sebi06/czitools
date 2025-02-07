from pathlib import Path
from czitools.metadata_tools import czi_metadata as czimd
from czitools.metadata_tools.channel import hex_to_rgb
import pytest
from typing import List, Dict
from pylibCZIrw import czi as pyczi


basedir = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize(
    "czifile, clims, colors, gamma, names, dyes, ch_disp, is_rgb, is_consistent, pxtypes",
    [
        (
            "Tumor_HE_Orig_small.czi",
            [[0.0, 0.5]],
            ["#80808000"],
            [0.85],
            ["CH1"],
            ["Brigh"],
            {
                0: pyczi.ChannelDisplaySettingsDataClass(
                    True,
                    pyczi.TintingMode.none,
                    pyczi.Rgb8Color(r=128, g=128, b=0),
                    0.0,
                    1.0,
                )
            },
            {0: True},
            True,
            {0: "Bgr24"},
        ),
        (
            "CellDivision_T3_Z5_CH2_X240_Y170.czi",
            [[0.0, 0.05983062485694667], [0.0, 0.24975967040512703]],
            ["#FFFF7E00", "#FF00FF33"],
            [0.7999999999999998, 0.7999999999999998],
            ["LED555", "LED470"],
            ["AF555", "AF488"],
            {
                0: pyczi.ChannelDisplaySettingsDataClass(
                    True,
                    pyczi.TintingMode.Color,
                    pyczi.Rgb8Color(r=255, g=126, b=0),
                    0.0,
                    0.05983062485694667,
                ),
                1: pyczi.ChannelDisplaySettingsDataClass(
                    True,
                    pyczi.TintingMode.Color,
                    pyczi.Rgb8Color(r=0, g=255, b=51),
                    0.0,
                    0.24975967040512703,
                ),
            },
            {0: False, 1: False},
            True,
            {0: "Gray16", 1: "Gray16"},
        ),
        (
            "Al2O3_SE_020_sp.czi",
            [[0.0, 0.5]],
            ["#80808000"],
            [0.85],
            ["CH1"],
            ["Dye-CH1"],
            {
                0: pyczi.ChannelDisplaySettingsDataClass(
                    True,
                    pyczi.TintingMode.Color,
                    pyczi.Rgb8Color(r=128, g=128, b=0),
                    0.0,
                    0.5,
                )
            },
            {0: False},
            True,
            {0: "Gray8"},
        ),
        (
            "w96_A1+A2.czi",
            [
                [0.000871455799315693, 0.044245974575704575],
                [0.000881881329185286, 0.05011349562051524],
            ],
            ["#FFFF1800", "#FF00FF33"],
            [0.7999999999999998, 0.7999999999999998],
            ["AF568", "AF488"],
            ["AF568", "AF488"],
            {
                0: pyczi.ChannelDisplaySettingsDataClass(
                    True,
                    pyczi.TintingMode.Color,
                    pyczi.Rgb8Color(r=255, g=24, b=0),
                    0.000871455799315693,
                    0.044245974575704575,
                ),
                1: pyczi.ChannelDisplaySettingsDataClass(
                    True,
                    pyczi.TintingMode.Color,
                    pyczi.Rgb8Color(r=0, g=255, b=51),
                    0.000881881329185286,
                    0.05011349562051524,
                ),
            },
            {0: False, 1: False},
            True,
            {0: "Gray16", 1: "Gray16"},
        ),
    ],
)
def test_channelinfo(
    czifile: str,
    clims: List[List[float]],
    colors: List[str],
    gamma: List[float],
    names: List[str],
    dyes: List[str],
    ch_disp: Dict[int, pyczi.ChannelDisplaySettingsDataClass],
    is_rgb: Dict[int, bool],
    is_consistent: bool,
    pxtypes: Dict[int, str],
) -> None:
    # get the CZI filepath
    filepath = basedir / "data" / czifile
    czi_channels = czimd.CziChannelInfo(filepath, verbose=False)

    assert czi_channels.clims == clims
    assert czi_channels.colors == colors
    assert czi_channels.gamma == gamma
    assert czi_channels.names == names
    assert czi_channels.dyes == dyes
    assert czi_channels.czi_disp_settings == ch_disp
    assert czi_channels.isRGB == is_rgb
    assert czi_channels.consistent_pixeltypes == is_consistent
    assert czi_channels.pixeltypes == pxtypes


@pytest.mark.parametrize(
    "hex_string, results",
    [
        ("#80808000", (128, 128, 0)),
        ("#FFFF7E000", (255, 126, 0)),
        ("00FF3", (128, 128, 128)),
        ("FF00FF", (255, 0, 255)),
        ("##FF00FF", (255, 0, 255)),
        ("#FF00FF33", (0, 255, 51)),
        ("#", (128, 128, 128)),
        ("", (128, 128, 128)),
    ],
)
def test_channelinfo_hexstring(hex_string: str, results: tuple[int, int, int]) -> None:

    assert hex_to_rgb(hex_string) == results
