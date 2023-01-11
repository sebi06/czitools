from czitools import pylibczirw_metadata as czimd
from pathlib import Path

basedir = Path(__file__).resolve().parents[3]


files = ["WellD6_S=1.czi",
         "WellD6-7_S=2.czi",
         "CellDivision_T=3_Z=5_CH=2_X=240_Y=170.czi",
         "Al2O3_SE_020_sp.czi",
         "1_tileregion.czi",
         "2_tileregions.czi",
         "w96_A1+A2.czi",
         "Airyscan.czi",
         "newCZI_zloc.czi",
         "FOV7_HV110_P0500510000.czi",
         "Tumor_HE_RGB.czi",
         "S=3_1Pos_2Mosaic_T=2=Z=3_CH=2_sm.czi",
         "DAPI_GFP.czi"
         ]

results = [{"X": [49500.0], "Y": [35500.0]},
           {"X": [49500.0, 58500.0], "Y": [35500.0, 35500.0]},
           {"X": 16977.153, "Y": 18621.489},
           {"X": 0.0, "Y": 0.0},
           {"X": [47729.178], "Y": [35660.098]},
           {"X": [48757.762, 47729.178], "Y": [35127.654, 35660.098]},
           {"X": [4261.294, 13293.666], "Y": [8361.406, 8372.441]},
           {"X": 0.0, "Y": 0.0},
           {"X": [], "Y": []},
           {"X": 0.0, "Y": 0.0},
           {"X": 0.0, "Y": 0.0},
           {"X": [38397.296, 37980.0, 38679.412], "Y": [12724.718, 13020.0, 13260.298]},
           {"X": 0.0, "Y": 0.0},
           ]


def test_stage_xy():

    for file, result in zip(files, results):

        # get the CZI filepath
        filepath = basedir / "data" / file

        # read the metadata
        md = czimd.CziMetadata(filepath)

        if md.image.SizeS is not None:
            assert (md.sample.scene_stageX == result["X"])
            assert (md.sample.scene_stageY == result["Y"])
        if md.image.SizeS is None:
            assert (md.sample.image_stageX == result["X"])
            assert (md.sample.image_stageY == result["Y"])
