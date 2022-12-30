
from czitools import pylibczirw_metadata as czimd
from pylibCZIrw import czi as pyczi
from pathlib import Path
import numpy as np

basedir = Path(__file__).resolve().parents[3]


def test_pixeltypes():

    # get the CZI filepath
    filepath = basedir / r"data/Tumor_HE_RGB.czi"
    md = czimd.CziMetadata(filepath)

    print("PixelTypes: ", md.pixeltypes)
    print("NumPy dtype: ", md.npdtype)
    print("Max. Pixel Value: ", md.maxvalue)

    assert (md.pixeltypes == {0: 'Bgr24'})
    assert (md.isRGB is True)

    # check the function to get npdtypes and maxvalues

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
    filepath = basedir / r"data/S=2_3x3_CH=2.czi"
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
        filepath = basedir / file

        # get the complete metadata at once as one big class
        md = czimd.CziMetadata(filepath)

        assert(md.scene_shape_is_consistent == sc)


def test_reading_czi_fresh():

    filepath = basedir / r"data/A01_segSD.czi"

    # get the complete metadata at once as one big class
    mdata = czimd.CziMetadata(filepath)

    assert (mdata.sample.well_array_names == [])
    assert (mdata.sample.well_indices == [])
    assert (mdata.sample.well_position_names == [])
    assert (mdata.sample.well_colID == [])
    assert (mdata.sample.well_rowID == [])
    assert (mdata.sample.well_counter == [])
    assert (mdata.sample.scene_stageX == [])
    assert (mdata.sample.scene_stageY == [])
    assert (mdata.sample.image_stageX is None)
    assert (mdata.sample.image_stageY is None)


def test_get_image_dimensions():

    test_dict = {"ImageDocument":
                 {"Metadata":
                  {"Information":
                   {"Image":
                    {"SizeX": 0,
                     "SizeY": 300,
                     "SizeS": 500,
                     "SizeT": 3,
                     "SizeZ": 20,
                     "SizeC": 2,
                     "SizeM": None,
                     "SizeR": 0,
                     "SizeH": -1,
                     "SizeI": None,
                     "SizeV": None,
                     "SizeB": 1}}}}}

    dimensions = ["SizeX",
                  "SizeY",
                  "SizeS",
                  "SizeT",
                  "SizeZ",
                  "SizeC",
                  "SizeM",
                  "SizeR",
                  "SizeH",
                  "SizeI",
                  "SizeV",
                  "SizeB"]

    results = [None, 300, 500, 3, 20, 2, None, None, None, None, None, 1]

    dim_dict = czimd.CziDimensions.get_image_dimensions(test_dict)

    for d, v in zip(dimensions, results):
        print((d, v))
        assert (dim_dict[d] == v)


def test_scaling():

    # get the CZI filepath
    filepath = basedir / r"data/DAPI_GFP.czi"
    md = czimd.CziMetadata(filepath)

    assert(md.scale.X == 1.0)
    assert(md.scale.Y == 1.0)
    assert(md.scale.Z == 1.0)
    assert(md.scale.ratio == {'xy': 1.0, 'zx': 1.0})

    scaling = czimd.CziScaling(filepath, dim2none=True)
    assert(scaling.X is None)
    assert(scaling.Y is None)
    assert(scaling.Z is None)
    assert(scaling.ratio == {'xy': None, 'zx': None})

    scaling = czimd.CziScaling(filepath, dim2none=False)
    assert(scaling.X == 1.0)
    assert(scaling.Y == 1.0)
    assert(scaling.Z == 1.0)
    assert(scaling.ratio == {'xy': 1.0, 'zx': 1.0})
