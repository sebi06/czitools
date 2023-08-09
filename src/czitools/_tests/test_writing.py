from czitools import read_tools, misc_tools
from czitools import metadata_tools as czimd
from pylibCZIrw import czi as pyczi
# import os
from pathlib import Path
from tifffile import imread, TiffFile
import numpy as np
from tqdm.contrib import itertools as it
from tqdm import tqdm
import pytest
from typing import List, Dict, Tuple, Optional, Type, Any, Union, Mapping

basedir = Path(__file__).resolve().parents[3] / "data"


@pytest.mark.parametrize(
    "tiff_file, sp",
    [
        ("CH=1_16bit.tif", 1),
        ("Fluorescence_RGB.tif", 3)
    ]
)
def test_write_1(tiff_file: str, sp: int) -> None:

    # get the TIFF filepath
    filepath = basedir / tiff_file

    czi_path = filepath.parent / Path(misc_tools.get_fname_woext(str(filepath)) + ".czi")

    # remove the CZI if it already exits
    if czi_path.exists():
        Path.unlink(czi_path)

    # read the TIF image and tags
    tif_image = imread(filepath)
    tiff_tags = {}

    with TiffFile(filepath) as tif:
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
    with pyczi.create_czi(str(czi_path), exist_ok=True) as czidoc_w:

        # write the plane
        czidoc_w.write(data=tif_image)

    # check if CZI was written
    assert (czi_path.exists() is True)

    # remove the files
    Path.unlink(Path(czi_path))


@pytest.mark.parametrize(
    "czifile, pyczi_dims, pix_types, is_rgb, is_mosaic",
    [
        ("z=16_ch=3.czi",
         {'T': (0, 1), 'Z': (0, 16), 'C': (0, 3), 'X': (0, 512), 'Y': (0, 512)},
         {0: 'Gray16', 1: 'Gray16', 2: 'Gray16'},
         False,
         False
         )
    ]
)
def test_write_2(czifile: str,
                 pyczi_dims: Dict[str, Tuple[int, int]],
                 pix_types: Dict[int, str],
                 is_rgb: bool,
                 is_mosaic: bool) -> None:

    # get the czifile path
    filepath = basedir / czifile

    mdarray, mdata, dimstring = read_tools.read_6darray(filepath,
                                                        output_dask=False,
                                                        chunks_auto=False,
                                                        output_order="STZCYX",
                                                        remove_adim=True)

    # create the filename for the new CZI image file
    newczi = str(Path.cwd() / czifile)

    # open a new CZI and allow overwrite (!!!) to play around ...
    with pyczi.create_czi(newczi, exist_ok=True) as czidoc_w:

        # loop over all z-planes and channels
        for z, ch in it.product(range(mdata.image.SizeZ), range(mdata.image.SizeC)):

            # get the 2d array for the current plane and add axis to get (Y, X, 1) as shape
            # write the plane with shape (Y, X, 1) to the new CZI file
            czidoc_w.write(data=mdarray[0, 0, z, ch, ...], plane={"Z": z, "C": ch})

    # get the complete metadata at once as one big class
    mdata = czimd.CziMetadata(newczi)

    assert (mdata.pyczi_dims == pyczi_dims)
    assert (mdata.pixeltypes == pix_types)
    assert (mdata.isRGB is is_rgb)
    assert (mdata.ismosaic is is_mosaic)

    # remove file
    Path.unlink(Path(newczi))


@pytest.mark.parametrize(
    "czifile, xstart, ch, pyczi_dims, pix_types, is_rgb, is_mosaic, tbox",
    [
        ("z=16_ch=3.czi",
         0,
         0,
         {'T': (0, 1), 'Z': (0, 1), 'C': (0, 1), 'X': (0, 8192), 'Y': (0, 512)},
         {0: 'Gray16'},
         False,
         True,
         {'T': (0, 1), 'Z': (0, 1), 'C': (0, 1), 'X': (0, 8192), 'Y': (0, 512)}
         )
    ]
)
def test_write_3(czifile: str,
                 xstart: int,
                 ch: int,
                 pyczi_dims: Dict[str, Tuple[int, int]],
                 pix_types: Dict[int, str],
                 is_rgb: bool,
                 is_mosaic: bool,
                 tbox: Dict[str, Tuple[int, int]]) -> None:

    # get the czifile path
    filepath = basedir / czifile

    mdarray, mdata, dimstring = read_tools.read_6darray(filepath,
                                                        output_dask=False,
                                                        chunks_auto=False,
                                                        output_order="STZCYX",
                                                        remove_adim=True)

    # create the filename for the new CZI image file
    newczi_zloc = str(Path.cwd() / "newCZI_zloc.czi")

    with pyczi.create_czi(newczi_zloc, exist_ok=True) as czidoc_w:

        # loop over all z-planes
        for z in tqdm(range(mdata.image.SizeZ)):

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

    assert (mdata.pyczi_dims == pyczi_dims)
    assert (mdata.pixeltypes == pix_types)
    assert (mdata.isRGB is is_rgb)
    assert (mdata.ismosaic is is_mosaic)
    assert (mdata.bbox.total_bounding_box == tbox)

    # remove file
    Path.unlink(Path(newczi_zloc))


@pytest.mark.parametrize(
    "czifile, ch, gx, gy, xystart, offset, pyczi_dims, pix_types, is_rgb, is_mosaic, tbox",
    [
        ("z=16_ch=3.czi",
         0,
         4,
         4,
         0,
         700,
         {'T': (0, 1), 'Z': (0, 1), 'C': (0, 1), 'X': (0, 2612), 'Y': (0, 2612)},
         {0: 'Gray16'},
         False,
         False,
         {'T': (0, 1), 'Z': (0, 1), 'C': (0, 1), 'X': (0, 2612), 'Y': (0, 2612)}
         )
    ]
)
def test_write_4(czifile: str,
                 ch: int,
                 gx: int,
                 gy: int,
                 xystart: int,
                 offset: int,
                 pyczi_dims: Dict[str, Tuple[int, int]],
                 pix_types: Dict[int, str],
                 is_rgb: bool,
                 is_mosaic: bool,
                 tbox: Dict[str, Tuple[int, int]]) -> None:

    # get the czifile path
    filepath = basedir / czifile

    mdarray, mdata, dimstring = read_tools.read_6darray(filepath,
                                                        output_dask=False,
                                                        chunks_auto=False,
                                                        output_order="STZCYX",
                                                        remove_adim=True)

    # first step is to create some kind of grid of locations
    locx = []
    locy = []

    # create the list for the 4x4 grid locations for the 16 planes
    for x, y in it.product(range(gx), range(gy)):
        locx.append(xystart + offset * x)
        locy.append(xystart + offset * y)

    # create the filename for the new CZI image file
    newczi_zscenes = str(Path.cwd() / "newCZI_scenes.czi")

    with pyczi.create_czi(newczi_zscenes, exist_ok=True) as czidoc_w:
        # loop over all z-planes
        for z in tqdm(range(mdata.image.SizeZ)):

            # for "fun" - write the z-planes to different locations using the locations we just created
            czidoc_w.write(data=mdarray[0, 0, z, ch, ...],
                           plane={"C": ch},
                           scene=z,
                           location=(locx[z], locy[z])
                           )

        # get the complete metadata at once as one big class
    mdata = czimd.CziMetadata(newczi_zscenes)

    assert (mdata.pyczi_dims == pyczi_dims)
    assert (mdata.pixeltypes == pix_types)
    assert (mdata.isRGB is is_rgb)
    assert (mdata.ismosaic is is_mosaic)
    assert (mdata.bbox.total_bounding_box == tbox)

    # remove files
    Path.unlink(Path(newczi_zscenes))
