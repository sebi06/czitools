from czitools import pylibczirw_tools, misc
from czitools import pylibczirw_metadata as czimd
from pylibCZIrw import czi as pyczi
import os
from pathlib import Path
from tifffile import imread, TiffFile
import numpy as np
from tqdm.contrib import itertools as it
from tqdm import tqdm

basedir = Path(__file__).resolve().parents[3]

# get some data to write
filepath = os.path.join(basedir, r"data/z=16_ch=3.czi")
mdarray, mdata, dimstring = pylibczirw_tools.read_6darray(filepath,
                                                          output_dask=False,
                                                          chunks_auto=False,
                                                          output_order="STZCYX",
                                                          remove_adim=True)
numCH = mdata.image.SizeC
numZ = mdata.image.SizeZ


def test_write_1():

    # get the CZI filepath
    filepath1 = os.path.join(basedir, r"data/CH=1_16bit.tif")
    filepath2 = os.path.join(basedir, r"data/Fluorescence_RGB.tif")

    files = [filepath1, filepath2]
    sps = [1, 3]

    for file, sp in zip(files, sps):

        czi_path = os.path.join(os.path.dirname(file), (misc.get_fname_woext(file) + ".czi"))

        # remove the CZI if it already exits
        if os.path.exists(czi_path):
            os.remove(czi_path)

        # read the TIF image
        tif_image = imread(file)

        tiff_tags = {}

        with TiffFile(file) as tif:
            for page in tif.pages:
                for tag in page.tags:
                    if not isinstance(tag.value, (bytes, bytearray)):
                        tiff_tags[tag.name] = tag.value

        # reverse RGB since ZEN is using BGR
        if tiff_tags["SamplesPerPixel"] == 3:
            print("Reversing RGB to BGR")
            tif_image = tif_image[..., ::-1]

        if tiff_tags["SamplesPerPixel"] == 1:
            print("Adding new axis")
            tif_image = tif_image[..., np.newaxis]

        assert (tiff_tags["SamplesPerPixel"] == sp)

        # open a new CZI and allow overwrite (!!!) to play around ...
        with pyczi.create_czi(czi_path, exist_ok=True) as czidoc_w:

            # write the plane
            czidoc_w.write(data=tif_image)

        # check if CZI was written
        assert (os.path.exists(czi_path) is True)

        # remove the files
        os.remove(czi_path)


def test_write_2():

    # create the filename for the new CZI image file
    newczi_4dstack = os.path.join(os.getcwd(), "z=16_ch=3.czi")

    # open a new CZI and allow overwrite (!!!) to play around ...
    with pyczi.create_czi(newczi_4dstack, exist_ok=True) as czidoc_w:

        # loop over all z-planes and channels
        for z, ch in it.product(range(numZ), range(numCH)):
            # get the 2d array for the current plane and add axis to get (Y, X, 1) as shape
            # write the plane with shape (Y, X, 1) to the new CZI file
            czidoc_w.write(
                data=mdarray[0, 0, z, ch, ...],
                plane={"Z": z, "C": ch}
            )

    # get the complete metadata at once as one big class
    mdata = czimd.CziMetadata(newczi_4dstack)

    assert (mdata.pyczi_dims == {'T': (0, 1), 'Z': (0, 16),
            'C': (0, 3), 'X': (0, 512), 'Y': (0, 512)})
    assert (mdata.pixeltypes == {0: 'Gray16', 1: 'Gray16', 2: 'Gray16'})
    assert (mdata.isRGB is False)
    assert (mdata.ismosaic is False)

    os.remove(newczi_4dstack)


def test_write_3():

    # create the filename for the new CZI image file
    newczi_zloc = os.path.join(os.getcwd(), "newCZI_zloc.czi")
    xstart = 0
    ch = 0

    with pyczi.create_czi(newczi_zloc, exist_ok=True) as czidoc_w:
        # loop over all z-planes
        for z in tqdm(range(numZ)):

            # for fun - write the z-planes to different locations
            czidoc_w.write(
                data=mdarray[0, 0, z, ch, ...],
                plane={"C": ch},
                location=(xstart, 0)
            )

            # change the x-position for the next round to write "side-by-side"
            xstart = xstart + 512

    # get the complete metadata at once as one big class
    mdata = czimd.CziMetadata(newczi_zloc)

    assert (mdata.pyczi_dims == {'T': (0, 1), 'Z': (0, 1),
            'C': (0, 1), 'X': (0, 8192), 'Y': (0, 512)})
    assert (mdata.pixeltypes == {0: 'Gray16'})
    assert (mdata.isRGB is False)
    assert (mdata.ismosaic is True)
    assert (mdata.bbox.total_bounding_box == {'T': (0, 1), 'Z': (
        0, 1), 'C': (0, 1), 'X': (0, 8192), 'Y': (0, 512)})

    os.remove(newczi_zloc)


def test_write_4():

    # first step is to create some kind of grid of locations
    locx = []
    locy = []
    xystart = 0
    offset = 700

    # create the list for the 4x4 grid locations for the 16 planes
    for x, y in it.product(range(4), range(4)):
        locx.append(xystart + offset * x)
        locy.append(xystart + offset * y)

    # create the filename for the new CZI image file
    newczi_zscenes = os.path.join(os.getcwd(), "newCZI_scenes.czi")
    ch = 0

    with pyczi.create_czi(newczi_zscenes, exist_ok=True) as czidoc_w:
        # loop over all z-planes
        for z in tqdm(range(numZ)):

            # for "fun" - write the z-planes to different locations using the locations we just created
            czidoc_w.write(
                data=mdarray[0, 0, z, ch, ...],
                plane={"C": ch},
                scene=z,
                location=(locx[z], locy[z])
            )

        # get the complete metadata at once as one big class
    mdata = czimd.CziMetadata(newczi_zscenes)

    assert (mdata.pyczi_dims == {'T': (0, 1), 'Z': (0, 1),
            'C': (0, 1), 'X': (0, 2612), 'Y': (0, 2612)})
    assert (mdata.pixeltypes == {0: 'Gray16'})
    assert (mdata.isRGB is False)
    assert (mdata.ismosaic is False)
    assert (mdata.bbox.total_bounding_box == {'T': (0, 1), 'Z': (
        0, 1), 'C': (0, 1), 'X': (0, 2612), 'Y': (0, 2612)})

    os.remove(newczi_zscenes)

test_write_1()
