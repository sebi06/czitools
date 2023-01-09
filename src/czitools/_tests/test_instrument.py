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


def test_objectives():

    to_test = {0: r"data/CellDivision_T=3_Z=5_CH=2_X=240_Y=170.czi",
               1: r"data/Al2O3_SE_020_sp.czi",
               2: r"data/w96_A1+A2.czi",
               3: r"data/Airyscan.czi",
               4: r"data/newCZI_zloc.czi",
               5: r"data/FOV7_HV110_P0500510000.czi",
               6: r"data/Tumor_HE_RGB.czi"
               }

    results = {0: {'name': 'Plan-Apochromat 50x/1.2', 'immersion': 'Water', 'NA': 1.2, 'ID': 'Objective:1', 'objmag': 50.0, 'tubelensmag': 1.0, 'totalmag': 50.0},
               1: {},
               2: {'name': 'Plan-Apochromat 20x/0.95', 'immersion': 'Air', 'NA': 0.95, 'ID': 'Objective:1', 'objmag': 20.0, 'tubelensmag': 0.5, 'totalmag': 10.0},
               3: {'name': 'Plan-Apochromat 63x/1.4 Oil DIC M27', 'immersion': 'Oil', 'NA': 1.4000000000000001, 'ID': 'Objective:0', 'objmag': 63.0, 'totalmag': 63.0},
               4: {},
               5: {},
               6: {},
               }

    for t in range(len(to_test)):

        # get the filepath and the metadata
        filepath = basedir / to_test[t]
        obj = czimd.CziObjectives(filepath)

        out = obj.__dict__
        del out['filepath']

        assert(out == results[t])
