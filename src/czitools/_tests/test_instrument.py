from czitools import pylibczirw_metadata as czimd
from pylibCZIrw import czi as pyczi
from pathlib import Path
import numpy as np
from box import Box
from box import BoxList

basedir = Path(__file__).resolve().parents[3]

def test_microscope():

    # get the CZI filepath
    filepath = basedir / r"data/CellDivision_T=3_Z=5_CH=2_X=240_Y=170.czi"
    mic = czimd.CziMicroscope(filepath)

    assert(mic.ID == 'Microscope:1')
    assert(mic.Name == 'Castor.Stand')

    # get the CZI filepath
    filepath = basedir / r"data/Airyscan.czi"
    mic = czimd.CziMicroscope(filepath)

    assert(mic.ID == 'Microscope:0')
    assert(mic.Name is None)

        # get the CZI filepath
    filepath = basedir / r"data/newCZI_zloc.czi"
    mic = czimd.CziMicroscope(filepath)

    assert(mic.ID is None)
    assert(mic.Name is None)


def test_detectors():

    # get the CZI filepath
    filepath = basedir / r"data/CellDivision_T=3_Z=5_CH=2_X=240_Y=170.czi"
    det = czimd.CziDetector(filepath)

    assert(det.ID == ['Detector:Axiocam 506'])
    assert(det.model == [None])
    assert(det.modeltype == [None])
    assert(det.name == ['Axiocam 506'])

    # get the CZI filepath
    filepath = basedir / r"data/Airyscan.czi"
    det = czimd.CziDetector(filepath)

    assert(det.ID == ['Detector:0:0', 'Detector:1:0'])
    assert(det.model == [None, None])
    assert(det.modeltype == ['Airyscan', 'Airyscan'])
    assert(det.name == [None, None])

        # get the CZI filepath
    filepath = basedir / r"data/newCZI_zloc.czi"
    det = czimd.CziDetector(filepath)

    assert(det.ID is None)
    assert(det.model is None)
    assert(det.modeltype is None)
    assert(det.name is None)

test_microscope()
test_detectors()