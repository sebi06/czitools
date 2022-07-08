from czitools import pylibczirw_metadata as czimd
import os
from pathlib import Path

basedir = Path(__file__).resolve().parents[1]


def test_channelinfo():

    # get the CZI filepath
    filepath = os.path.join(basedir, r"_data/CellDivision_T=3_Z=5_CH=2_X=240_Y=170.czi")
    czi_channels = czimd.CziChannelInfo(filepath)

    assert (czi_channels.clims == [[0.0, 0.05983062485694667], [0.0, 0.24975967040512703]])
    assert (czi_channels.colors == ["#FFFF7E00", "#FF00FF33"])
    assert (czi_channels.gamma == [0.7999999999999998, 0.7999999999999998])
    assert (czi_channels.names == ["LED555", "LED470"])
    assert (czi_channels.dyes == ["AF555", "AF488"])

    # get the CZI filepath
    filepath = os.path.join(basedir, r"_data/Al2O3_SE_020_sp.czi")
    czi_channels = czimd.CziChannelInfo(filepath)

    assert (czi_channels.clims == [[0.1, 0.5]])
    assert (czi_channels.colors == ["#80808000"])
    assert (czi_channels.gamma == [0.85])
    assert (czi_channels.names == ["C1"])
    assert (czi_channels.dyes == ["Dye1"])

    # get the CZI filepath
    filepath = os.path.join(basedir, r"_data/w96_A1+A2.czi")
    czi_channels = czimd.CziChannelInfo(filepath)

    assert (czi_channels.clims == [[0.000871455799315693, 0.044245974575704575], [
            0.000881881329185286, 0.05011349562051524]])
    assert (czi_channels.colors == ['#FFFF1800', '#FF00FF33'])
    assert (czi_channels.gamma == [0.7999999999999998, 0.7999999999999998])
    assert (czi_channels.names == ['AF568', 'AF488'])
    assert (czi_channels.dyes == ['AF568', 'AF488'])


test_channelinfo()
