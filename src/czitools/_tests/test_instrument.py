from czitools import pylibczirw_metadata as czimd
from pathlib import Path

basedir = Path(__file__).resolve().parents[3]


def test_instrument():

    to_test = {0: r"data/CellDivision_T=3_Z=5_CH=2_X=240_Y=170.czi",
               1: r"data/Airyscan.czi",
               2: r"data/newCZI_zloc.czi"}

    results_mic = {0: ['Microscope:1', 'Castor.Stand'],
                   1: ['Microscope:0', None],
                   2: [None, None]
                   }

    results_det = {0: [['Detector:Axiocam 506'], [None], [None], ['Axiocam 506']],
                   1: [['Detector:0:0', 'Detector:1:0'], [None, None], ['Airyscan', 'Airyscan'], [None, None]],
                   2: [[None], [None], [None], [None]]
                   }

    for t in range(len(to_test)):

        # get the filepath and the metadata
        filepath = basedir / to_test[t]
        mic = czimd.CziMicroscope(filepath)
        det = czimd.CziDetector(filepath)

        # check the microscope data
        assert (mic.ID == results_mic[t][0])
        assert (mic.Name == results_mic[t][1])

        # check the detector(s) data
        assert (det.ID == results_det[t][0])
        assert (det.model == results_det[t][1])
        assert (det.modeltype == results_det[t][2])
        assert (det.name == results_det[t][3])
