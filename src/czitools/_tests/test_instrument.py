from czitools.metadata_tools import czi_metadata as czimd
from pathlib import Path
import pytest
from typing import Dict

basedir = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize(
    "czifile, result_mic, result_det",
    [
        ("CellDivision_T3_Z5_CH2_X240_Y170.czi",
         ['Microscope:1', 'Castor.Stand'],
         [['Detector:Axiocam 506'], [None], [None], ['Axiocam 506']]),

        ("Airyscan.czi",
         ['Microscope:0', None],
         [['Detector:0:0', 'Detector:1:0'], [None, None], ['Airyscan', 'Airyscan'], [None, None]]),

        ("newCZI_zloc.czi",
         [None, None],
         [[None], [None], [None], [None]]),

    ]
)
def test_instrument(czifile: str, result_mic: Dict, result_det: Dict) -> None:

    # get the filepath and the metadata_tools
    filepath = basedir / "data" / czifile
    mic = czimd.CziMicroscope(filepath)
    det = czimd.CziDetector(filepath)

    # check the microscope data
    assert (mic.Id == result_mic[0])
    assert (mic.Name == result_mic[1])

    # check the detector(s) data
    assert (det.Id == result_det[0])
    assert (det.model == result_det[1])
    assert (det.modeltype == result_det[2])
    assert (det.name == result_det[3])


@pytest.mark.parametrize(
    "czifile, result",
    [
        ("LLS7_small.czi", {'name': ['Objective Reflected Light', 'Objective Observation'],
                            'model': [],
                            'immersion': ['Water', 'Water', None],
                            'NA': [0.44, 1.0],
                            'Id': ['Objective:1', 'Objective:0', 'Objective:2'],
                            'objmag': [13.3, 44.83],
                            'tubelensmag': [],
                            'totalmag': [13.3, 44.83]}),

        ("CellDivision_T3_Z5_CH2_X240_Y170.czi", {'name': ['Plan-Apochromat 50x/1.2'],
                                                       'model': [],
                                                       'immersion': ['Water'],
                                                       'NA': [1.2],
                                                       'Id': ['Objective:1'],
                                                       'objmag': [50.0],
                                                       'tubelensmag': [1.0],
                                                       'totalmag': [50.0]}),

        ("Al2O3_SE_020_sp.czi", {'name': [],
                                 'model': [],
                                 'immersion': [],
                                 'NA': [],
                                 'Id': [],
                                 'objmag': [],
                                 'tubelensmag': [],
                                 'totalmag': []}),

        ("w96_A1+A2.czi", {'name': ['Plan-Apochromat 20x/0.95'],
                           'model': [],
                           'immersion': ['Air'],
                           'NA': [0.95],
                           'Id': ['Objective:1'],
                           'objmag': [20.0],
                           'tubelensmag': [0.5],
                           'totalmag': [10.0]}),

        ("Airyscan.czi",  {'name': ['Plan-Apochromat 63x/1.4 Oil DIC M27'],
                           'model': [],
                           'immersion': ['Oil'],
                           'NA': [1.4000000000000001],
                           'Id': ['Objective:0'],
                           'objmag': [63.0],
                           'tubelensmag': [],
                           'totalmag': [63.0]}),

        ("newCZI_zloc.czi", {'name': [],
                             'model': [],
                             'immersion': [],
                             'NA': [],
                             'Id': [],
                             'objmag': [],
                             'tubelensmag': [],
                             'totalmag': []}),

        ("FOV7_HV110_P0500510000.czi", {'name': [],
                                        'model': [],
                                        'immersion': [],
                                        'NA': [],
                                        'Id': [],
                                        'objmag': [],
                                        'tubelensmag': [],
                                        'totalmag': []}),

        ("Tumor_HE_RGB.czi", {'name': [],
                              'model': [],
                              'immersion': [],
                              'NA': [],
                              'Id': [],
                              'objmag': [],
                              'tubelensmag': [],
                              'totalmag': []})

    ]
)
def test_objectives(czifile: str, result: Dict) -> None:

    # get the filepath and the metadata_tools
    filepath = basedir / "data" / czifile
    obj = czimd.CziObjectives(filepath)

    out = obj.__dict__
    del out['czisource']
    del out["verbose"]

    assert (out == result)
