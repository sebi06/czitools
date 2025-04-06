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
from czitools.metadata_tools.czi_metadata import CziMetadata
from czitools.metadata_tools.sample import get_scenes_for_well
import matplotlib as mpl

mpl.use("Qt5Agg")
from matplotlib import pyplot as plt
import matplotlib.cm as cm


# Define a simple Rectangle class (if not already provided)
class SceneRectangle:
    def __init__(self, x, y, w, h):
        self.x = x  # Top-left x
        self.y = y  # Top-left y
        self.w = w  # Width
        self.h = h  # Height

    def __repr__(self):
        return f"Rectangle(x={self.x}, y={self.y}, w={self.w}, h={self.h})"

    @property
    def x2(self):
        # Rightmost x coordinate
        return self.x + self.w

    @property
    def y2(self):
        # Bottom y coordinate
        return self.y + self.h


def get_overall_rectangle(rect_dict, start_key, end_key):
    """
    Computes the overall rectangle that tightly encloses all individual rectangles
    in the given key range [start_key, end_key] (inclusive).
    """
    # Filter the rectangles using the specified key range
    selected_rects = [
        rect_dict[key] for key in range(start_key, end_key + 1) if key in rect_dict
    ]

    if not selected_rects:
        return None  # Or raise an exception if no keys are found

    # Calculate min and max coordinates
    min_x = min(rect.x for rect in selected_rects)
    min_y = min(rect.y for rect in selected_rects)
    max_x = max(rect.x + rect.w for rect in selected_rects)  # Calculate x2 as (x + w)
    max_y = max(rect.y + rect.h for rect in selected_rects)  # Calculate y2 as (y + h)

    # Construct the overall rectangle
    return SceneRectangle(min_x, min_y, max_x - min_x, max_y - min_y)


filepath = r"F:\Testdata_Zeiss\CZI_Testfiles\W96_A1+A2_S=2_4x2_Z=5_CH=2.czi"
# filepath = r"F:\Testdata_Zeiss\CZI_Testfiles\W96_A1+A2_S=2_Pos=8_Z=5_CH=2.czi"

# determine shape of combines stack
mdata = CziMetadata(filepath)

read_czi = True


# het all scene indicies for that specific well
scenes_for_well = get_scenes_for_well(mdata.sample, "A1")

print(f"Scenes Index for well A1: {scenes_for_well}")

# open the CZI document to read the
with pyczi.open_czi(filepath) as czidoc:

    # get the bounding boxes for each individual scene
    scenes_bounding_rectangles = czidoc.scenes_bounding_rectangle
    overall_rect = get_overall_rectangle(
        scenes_bounding_rectangles, scenes_for_well[0], scenes_for_well[-1]
    )
    print(overall_rect)

    if read_czi:
        image2d = czidoc.read(
            plane={"T": 0, "Z": 0, "C": 0},
            roi=(overall_rect.x, overall_rect.y, overall_rect.w, overall_rect.h),
        )
        print(f"Shape of image2d: {image2d.shape}")

        # show the 2D image plane
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.imshow(
            image2d[..., 0], cmap=cm.inferno, vmin=image2d.min(), vmax=image2d.max()
        )
        plt.show()
