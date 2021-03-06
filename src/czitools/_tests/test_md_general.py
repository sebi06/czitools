
from czitools import pylibczirw_metadata as czimd
from pylibCZIrw import czi as pyczi
import os
from pathlib import Path
import numpy as np

basedir = Path(__file__).resolve().parents[3]


def test_pixeltypes():

    # get the CZI filepath
    filepath = os.path.join(basedir, r"data/Tumor_HE_RGB.czi")
    md = czimd.CziMetadata(filepath)

    print("PixelTypes: ", md.pixeltypes)
    print("NumPy dtype: ", md.npdtype)
    print("Max. Pixel Value: ", md.maxvalue)

    assert (md.pixeltypes == {0: 'Bgr24'})
    assert (md.isRGB == True)

    # check the function to get npdytes and maxvalues

    pts = ["gray16",
           "Gray16",
           "gray8",
           "Gray8",
           "bgr48",
           "Bgr48",
           "bgr24",
           "Bgr24",
           "bgr96float",
           "Bgr96Float",
           "abc",
           None]

    dts = [np.dtype(np.uint16),
           np.dtype(np.uint16),
           np.dtype(np.uint8),
           np.dtype(np.uint8),
           np.dtype(np.uint16),
           np.dtype(np.uint16),
           np.dtype(np.uint8),
           np.dtype(np.uint8),
           np.dtype(np.uint16),
           np.dtype(np.uint16),
           None,
           None]

    mvs = [65535, 65535, 255, 255, 65535, 65535, 255, 255, 65535, 65535, None, None]

    for pt, dt, mv in zip(pts, dts, mvs):

        out = czimd.CziMetadata.get_dtype_fromstring(pt)
        assert (out[0] == dt)
        assert (out[1] == mv)


def test_dimorder():

    # get the CZI filepath
    filepath = os.path.join(basedir, r"data/S=2_3x3_CH=2.czi")
    md = czimd.CziMetadata(filepath)

    assert(md.aics_dim_order == {'R': -1, 'I': -1, 'M': 5, 'H': 0, 'V': -1,
           'B': -1, 'S': 1, 'T': 2, 'C': 3, 'Z': 4, 'Y': 6, 'X': 7, 'A': -1})
    assert(md.aics_dim_index == [-1, -1, 5, 0, -1, -1, 1, 2, 3, 4, 6, 7, -1])
    assert(md.aics_dim_valid == 8)


def test_scene_shape():

    files = [r"data/S=3_1Pos_2Mosaic_T=2=Z=3_CH=2_sm.czi",
             r"data/CellDivision_T=3_Z=5_CH=2_X=240_Y=170.czi",
             r"data/WP96_4Pos_B4-10_DAPI.czi"]

    shapes = [False, True, True]

    for file, sc in zip(files, shapes):

        # get the CZI filepath
        filepath = os.path.join(basedir, file)

        # get the complete metadata at once as one big class
        md = czimd.CziMetadata(filepath)

        assert(md.scene_shape_is_consistent == sc)
