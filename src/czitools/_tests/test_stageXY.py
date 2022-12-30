from czitools import pylibczirw_metadata as czimd
from pathlib import Path

basedir = Path(__file__).resolve().parents[3]

files = [r"data/1_tileregion.czi",
         r"data/2_tileregions.czi",
         r"data/CellDivision_T=3_Z=5_CH=2_X=240_Y=170.czi"
         ]

results = [{"X": [47729.178], "Y": [35660.098]},
           {"X": [48757.762, 47729.178], "Y": [35127.654, 35660.098]},
           {"X": 16977.153, "Y": 18621.489}
           ]


def test_stage_xy():

    for file, result in zip(files, results):

        # get the CZI filepath
        filepath = basedir / file

        # read the metadata
        md = czimd.CziMetadata(filepath.as_posix())

        if md.image.SizeS is not None:
            assert (md.sample.scene_stageX == result["X"])
            assert (md.sample.scene_stageY == result["Y"])
        if md.image.SizeS is None:
            assert (md.sample.image_stageX == result["X"])
            assert (md.sample.image_stageY == result["Y"])

