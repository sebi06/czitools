# -*- coding: utf-8 -*-

#################################################################
# File        : read_czi_simple.py
# Author      : sebi06
#
# Disclaimer: This code is purely experimental. Feel free to
# use it at your own risk.
#
#################################################################

from pylibCZIrw import czi as pyczi
from czitools.metadata_tools.czi_metadata import CziMetadata, create_md_dict_red
import matplotlib as mpl

mpl.use("Qt5Agg")
from matplotlib import pyplot as plt
import matplotlib.cm as cm

# filepath = r"F:\Testdata_Zeiss\Mindpeak\28448e7c-3dbb-486e-ab57-c4fb3ade3f1d.czi"
# filepath = r"F:\Testdata_Zeiss\Mindpeak\sample.czi"
# filepath = r"F:\Testdata_Zeiss\CZI_Testfiles\W96_A1+A2_S=2_4x2_Z=5_CH=2.czi"
# filepath = r"F:\Testdata_Zeiss\CZI_Testfiles\W96_A1+A2_S=2_Pos=8_Z=5_CH=2.czi"
filepath = r"F:\Github\czitools\data\WellD6_S1.czi"


# determine shape of combines stack
mdata = CziMetadata(filepath)

md_red = create_md_dict_red(mdata)

# zoom_level = [1.0, 0.5, 0.25, 0.125, 0.1]
zoom_level = [1.0]
read_czi = False
show_czi = False
# open the CZI document to read the
with pyczi.open_czi(filepath) as czidoc:

    # # get the image dimensions as a dictionary, where the key identifies the dimension
    total_bounding_box = czidoc.total_bounding_box
    print(total_bounding_box)

    # # get the total bounding box for all scenes
    total_bounding_rectangle = czidoc.total_bounding_rectangle
    print(total_bounding_rectangle)

    # get the bounding boxes for each individual scene
    scenes_bounding_rectangle = czidoc.scenes_bounding_rectangle
    print(scenes_bounding_rectangle)

    print(czidoc.pixel_types)

    if read_czi:
        for zl in zoom_level:
            print("Trying to read with ZoomLevel:", zl)
            image2d = czidoc.read(plane={"T": 0, "Z": 0, "C": 0}, zoom=zl)
            # image2d = czidoc.read(plane={"Z": 0, "C": 0, "V": 1}, zoom=zl)
            print(f"Shape image2d: {image2d.shape} ZoomLevel: {zl}")

    if show_czi:

        # show the 2D image plane
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.imshow(
            image2d[..., 0], cmap=cm.inferno, vmin=image2d.min(), vmax=image2d.max()
        )
        plt.show()
