
from czitools import pylibczirw_metadata as czimd
import os
from pathlib import Path

basedir = Path(__file__).resolve().parents[3]

files = {0: r"data/1_tileregion.czi",
         1: r"data/2_tileregions.czi",
         2: r"data/CellDivision_T=3_Z=5_CH=2_X=240_Y=170.czi"
         }

results = {0: {"X": [47729.178], "Y": [35660.098]},
           1: {"X": [48757.762, 47729.178], "Y": [35127.654, 35660.098]},
           2: {"X": 16977.153, "Y": 18621.489}
           }


def test_stage_xy():

    for key, value in files.items():

        # get the CZI filepath
        filepath = os.path.join(basedir, value)

        # read the metadata
        md = czimd.CziMetadata(filepath)

        if md.image.SizeS is not None:
            assert (md.sample.scene_stageX == results[key]["X"])
            assert (md.sample.scene_stageY == results[key]["Y"])
        if md.image.SizeS is None:
            assert (md.sample.image_stageX == results[key]["X"])
            assert (md.sample.image_stageY == results[key]["Y"])


test_stage_xy()