from czitools import pylibczirw_tools, misc
from pylibCZIrw import czi as pyczi
import os
from pathlib import Path
from tifffile import imread, TiffFile
import numpy as np

basedir = Path(__file__).resolve().parents[3]


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
                    #print(tag.name, tag.value)
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
        assert (os.path.exists(czi_path) == True)


test_write_1()
