from czitools import pylibczirw_metadata as czimd
from pathlib import Path

basedir = Path(__file__).resolve().parents[3]


def test_information():

    to_test = {0: r"data/CellDivision_T=3_Z=5_CH=2_X=240_Y=170.czi",
               1: r"data/Airyscan.czi",
               2: r"data/newCZI_zloc.czi"}

    results = {0: ['ZEN 3.2 (blue edition)',
                   '3.2.0.00001',
                   '2016-02-12T09:41:02.4915604Z',
                   True,
                   'CellDivision_T=3_Z=5_CH=2_X=240_Y=170.czi'],
               1: ['ZEN (blue edition)',
                   '3.5.093.00000',
                   None,
                   True,
                   'Airyscan.czi'],
               2: ['pylibCZIrw',
                   '3.2.1',
                   None,
                   True,
                   'newCZI_zloc.czi']
               }

    for t in range(len(to_test)):

        # get the filepath
        filepath = basedir / to_test[t]
        md = czimd.CziMetadata(filepath)

        assert (md.software_name == results[t][0])
        assert (md.software_version == results[t][1])
        assert (md.acquisition_date == results[t][2])
        assert (Path.exists(Path(md.dirname)) == results[t][3])
        assert (md.filename == results[t][4])


test_information()
