from czitools.metadata import pylibczirw_metadata as czimd
import os
from pathlib import Path

basedir = Path(__file__).resolve().parents[1]

# get the CZI filepath
filepath = os.path.join(
    basedir, r"data/CellDivision_T=3_Z=5_CH=2_X=240_Y=170.czi")


def test_channelinfo():

    czi_channels = czimd.CziChannelInfo(filepath)

    print("Channel - clims: ", czi_channels.clims)
    print("Channel - colors: ", czi_channels.gamma)
    print("Channel - gamma: ", czi_channels.colors)
    print("Channel - names: ", czi_channels.names)
    print("Channel - shortnames: ", czi_channels.shortnames)

    assert (czi_channels.clims == [[0.0, 0.05983062485694667], [0.0, 0.24975967040512703]])
    assert (czi_channels.colors == ['#FFFF7E00', '#FF00FF33'])
    assert (czi_channels.gamma == [0.7999999999999998, 0.7999999999999998])
    assert (czi_channels.names == ['LED555', 'LED470'])
    assert (czi_channels.shortnames == ['AF555', 'AF488'])
