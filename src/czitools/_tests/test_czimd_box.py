from czitools import pylibczirw_metadata as czimd
from pathlib import Path, PurePath


# adapt to your needs
defaultdir = Path(__file__).resolve().parents[3] / "data"

def test_general_attr():

    filepath = defaultdir / "CellDivision_T=3_Z=5_CH=2_X=240_Y=170.czi"

    czibox = czimd.get_czimd_box(filepath)

    assert(hasattr(czibox, "has_customattr"))
    assert(hasattr(czibox, "has_exp = False"))
    assert(hasattr(czibox, "has_disp"))
    assert(hasattr(czibox, "has_hardware"))
    assert(hasattr(czibox, "has_scale"))
    assert(hasattr(czibox, "has_instrument"))
    assert(hasattr(czibox, "has_microscopes"))
    assert(hasattr(czibox, "has_detectors"))
    assert(hasattr(czibox, "has_objectives"))
    assert(hasattr(czibox, "has_tubelenses"))
    assert(hasattr(czibox, "has_disp"))
    assert(hasattr(czibox, "has_channels"))
    assert(hasattr(czibox, "has_info"))
    assert(hasattr(czibox, "has_app"))
    assert(hasattr(czibox, "has_doc"))
    assert(hasattr(czibox, "has_image"))
    assert(hasattr(czibox, "has_scenes"))
    assert(hasattr(czibox, "has_dims"))


def test_box():

    czifiles = ["CellDivision_T=3_Z=5_CH=2_X=240_Y=170.czi",
                "Al2O3_SE_020_sp.czi",
                "w96_A1+A2.czi",
                "Airyscan.czi",
                "newCZI_zloc.czi",
                "does_not_exist.czi",
                "FOV7_HV110_P0500510000.czi",
                "Tumor_HE_RGB.czi",
                "WP96_4Pos_B4-10_DAPI.czi",
                "S=3_1Pos_2Mosaic_T=2=Z=3_CH=2_sm.czi"
                ]

    results = [True, True, True, True, True, False, True, True, True, True]

    for czifile, result in zip(czifiles, results):

        filepath = defaultdir / czifile

        try:
            czibox = czimd.get_czimd_box(filepath)
            ok = True
        except Exception:
            ok = False

        assert(ok == result)

test_box()