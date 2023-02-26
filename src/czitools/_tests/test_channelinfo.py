from pathlib import Path
from czitools import pylibczirw_metadata as czimd
import pytest
from typing import List, Dict, Tuple, Optional, Type, Any, Union, Mapping

basedir = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize(
    "czifile, clims, colors, gamma, names, dyes",
    [
        ("CellDivision_T=3_Z=5_CH=2_X=240_Y=170.czi",
         [[0.0, 0.05983062485694667], [0.0, 0.24975967040512703]],
         ["#FFFF7E00", "#FF00FF33"],
         [0.7999999999999998, 0.7999999999999998],
         ["LED555", "LED470"],
         ["AF555", "AF488"]),

        ("Al2O3_SE_020_sp.czi",
         [[0.0, 0.5]],
         ["#80808000"],
         [0.85],
         ["CH1"],
         ["Dye-CH1"]),

        ("w96_A1+A2.czi",
         [[0.000871455799315693, 0.044245974575704575], [0.000881881329185286, 0.05011349562051524]],
         ['#FFFF1800', '#FF00FF33'],
         [0.7999999999999998, 0.7999999999999998],
         ['AF568', 'AF488'],
         ['AF568', 'AF488'])
    ]
)
def test_channelinfo(czifile: str, clims: List, colors: List, gamma: List, names: List, dyes: List) -> None:
    # get the CZI filepath
    filepath = basedir / "data" / czifile
    czi_channels = czimd.CziChannelInfo(filepath)

    assert (czi_channels.clims == clims)
    assert (czi_channels.colors == colors)
    assert (czi_channels.gamma == gamma)
    assert (czi_channels.names == names)
    assert (czi_channels.dyes == dyes)
