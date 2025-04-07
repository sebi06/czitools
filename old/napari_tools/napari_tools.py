# -*- coding: utf-8 -*-

#################################################################
# File        : napari_tools.py
# Author      : sebi06
#
# Disclaimer: This code is purely experimental. Feel free to
# use it at your own risk.
#
#################################################################

import sys
from czitools.utils import logging_tools
from czitools.metadata_tools.helper import ValueRange

logger = logging_tools.set_logging()

# check if Napari is actually installed
try:
    import napari
except (ImportError, ModuleNotFoundError) as error:
    # Output expected ImportErrors.
    logger.error(error.__class__.__name__ + ": " + error.msg)
    sys.exit(1)
else:

    from PyQt5.QtWidgets import (
        QVBoxLayout,
        QWidget,
        QTableWidget,
        QTableWidgetItem,
    )

    from PyQt5.QtCore import Qt
    from PyQt5 import QtWidgets
    from PyQt5.QtGui import QFont
    from czitools.metadata_tools import czi_metadata as czimd
    from czitools.utils import misc
    from czitools.utils.datatreewiget import DataTreeWidget
    import numpy as np
    from typing import (
        List,
        Dict,
        Tuple,
        # Optional,
        # Type,
        # Any,
        Union,
        Literal,
        # Mapping,
        Annotated,
    )
    from napari.utils.colormaps import Colormap
    from napari.utils import resize_dask_cache
    import dask.array as da
    import warnings

    warnings.filterwarnings("ignore", category=DeprecationWarning)


class MdTableWidget(QWidget):
    """
    MdTableWidget is a custom QWidget that displays metadata in a table format using QTableWidget.
    Methods:
        __init__() -> None:
            Initializes the MdTableWidget with a vertical layout and a QTableWidget.
        update_metadata(md_dict: Dict) -> None:
            Updates the table with metadata from the provided dictionary.
                md_dict (Dict): Metadata dictionary where keys are parameters and values are their corresponding values.
        update_style() -> None:
            Updates the style of the table, including font size, type, and header items.
    """

    def __init__(self) -> None:
        super(QWidget, self).__init__()

        self.layout = QVBoxLayout(self)
        self.mdtable = QTableWidget()
        self.layout.addWidget(self.mdtable)
        self.mdtable.setShowGrid(True)
        self.mdtable.setHorizontalHeaderLabels(["Parameter", "Value"])
        header = self.mdtable.horizontalHeader()
        header.setDefaultAlignment(Qt.AlignLeft)

    def update_metadata(self, md_dict: Dict) -> None:
        """
        Update the table with the metadata from the dictionary.
        This method updates the table widget (`self.mdtable`) with the metadata
        provided in the `md_dict` dictionary. Each key-value pair in the dictionary
        is added as a row in the table, with the key in the first column and the
        value in the second column. The table is resized to fit the content after
        updating.
            md_dict (Dict): Metadata dictionary where keys are the metadata fields
                            and values are the corresponding metadata values.
        """

        # number of rows is set to number of metadata_tools entries
        row_count = len(md_dict)
        col_count = 2
        self.mdtable.setColumnCount(col_count)
        self.mdtable.setRowCount(row_count)

        row = 0

        # update the table with the entries from metadata_tools dictionary
        for key, value in md_dict.items():
            newkey = QTableWidgetItem(key)
            self.mdtable.setItem(row, 0, newkey)
            newvalue = QTableWidgetItem(str(value))
            self.mdtable.setItem(row, 1, newvalue)
            row += 1

        # fit columns to content
        self.mdtable.resizeColumnsToContents()

    def update_style(self) -> None:
        """
        Updates the style of the table headers in the `mdtable` widget.
        This method sets the font size, type, and boldness for the table headers.
        It also sets the text for the headers to "Parameter" and "Value".
        Parameters:
        None
        Returns:
        None
        """

        # define font size and type
        fnt = QFont()
        fnt.setPointSize(11)
        fnt.setBold(True)
        fnt.setFamily("Arial")

        # update both header items
        # fc = (25, 25, 25)
        item1 = QtWidgets.QTableWidgetItem("Parameter")
        # item1.setForeground(QtGui.QColor(25, 25, 25))
        item1.setFont(fnt)
        self.mdtable.setHorizontalHeaderItem(0, item1)

        item2 = QtWidgets.QTableWidgetItem("Value")
        # item2.setForeground(QtGui.QColor(25, 25, 25))
        item2.setFont(fnt)
        self.mdtable.setHorizontalHeaderItem(1, item2)


class MdTreeWidget(QWidget):
    """
    A custom QWidget that contains a DataTreeWidget for displaying hierarchical data.
    Attributes:
        layout (QVBoxLayout): The main layout of the widget.
        mdtree (DataTreeWidget): The tree widget used to display the data.
    Args:
        data (optional): The data to be displayed in the tree widget. Defaults to None.
        expandlevel (int, optional): The level to which the tree should be expanded initially. Defaults to 0.
    """

    def __init__(self, data=None, expandlevel=0) -> None:
        super(QWidget, self).__init__()

        self.layout = QVBoxLayout(self)
        self.mdtree = DataTreeWidget(data=data)
        self.mdtree.setData(data, expandlevel=expandlevel, hideRoot=True)
        self.mdtree.setAlternatingRowColors(False)
        # self.mdtree.expandToDepth(expandlevel)
        self.layout.addWidget(self.mdtree)


def show(
    viewer: napari.Viewer,
    array: Union[np.ndarray, List[da.Array], da.Array],
    metadata: czimd.CziMetadata,
    dim_string: str = "STCZYX",
    blending: str = "additive",
    contrast: Literal["calc", "napari_auto", "from_czi"] = "calc",
    gamma: float = 0.85,
    show_metadata: Literal["none", "tree", "table"] = "tree",
    name_sliders: bool = False,
    dask_cache_size: Annotated[float, ValueRange(0.5, 0.9)] = 0.5,
    verbose: bool = True,
) -> List:
    """Display the multidimensional array inside the Napari viewer.
    Optionally the CziMetadata class will be used show a table with the metadata_tools.
    Every channel will be added as a new layer to the viewer.

    Args:
        viewer (Any): Napari viewer object
        array (Union[np.ndarray, List[da.Array], da.Array]): multi-dimensional array containing the pixel data (Numpy, List of Dask Array or Dask array.
        metadata (czimd.CziMetadata): CziMetadata class
        dim_string (str): dimension string for the array to be shown
        blending (str, optional): blending mode for viewer. Defaults to "additive".
        contrast (Literal[str], optional): method to be used to calculate an appropriate display scaling.
            - "calc" : real min & max calculation (might be slow) be calculated (slow)
            - "napari_auto" : Let Napari figure out a display scaling. Will look in the center of an image!
            - "from_czi" : use the display scaling from ZEN stored inside the CZI metadata_tools. Defaults to "calc".
        gamma (float, optional): gamma value for the Viewer for all layers Defaults to 0.85.
        show_metadata (mdviewoption, optional): Option to show metadata_tools as tree or table. Defaults to "tree".
        name_sliders (bool, optional): option to use the dimension letters as slider labels for the viewer. Defaults to False.
        dask_cache_size(float, optional): option to resize the dask cache used for opportunistic caching. Range [0 - 1]
        verbose (bool): Flag to enable verbose logging. Initialized to True.


    Returns:
        List: List of napari layers
    """

    # set napari dask cache size
    cache = resize_dask_cache(dask_cache_size)

    dim_order, dim_index, dim_valid = czimd.pixels.get_dimorder(dim_string)

    # check if contrast mode
    if contrast not in ["calc", "napari_auto", "from_czi"]:
        if verbose:
            logger.info(
                contrast, "is not valid contrast method. Use napari_auto instead."
            )
            contrast = "napari_auto"

    # create empty list for the napari layers
    napari_layers = []

    # create scale factor with all ones
    scalefactors = [1.0] * len(array.shape)

    # the "strange factor" is added due to an open (rounding) bug on the Napari side:
    # https://github.com/napari/napari/issues/4861
    # https://forum.image.sc/t/image-layer-in-napari-showing-the-wrong-dimension-size-one-plane-is-missing/69939/12

    scalefactors[dim_order["Z"]] = metadata.scale.ratio["zx_sf"] * 1.001

    if show_metadata.lower != "none":
        # add PyQTGraph DataTreeWidget to Napari viewer to show the metadata_tools
        if show_metadata == "tree":
            md_dict = czimd.create_md_dict_nested(metadata, sort=True, remove_none=True)
            mdtree = MdTreeWidget(data=md_dict, expandlevel=1)
            viewer.window.add_dock_widget(mdtree, name="MetadataTree", area="right")

        # add QTableWidget DataTreeWidget to Napari viewer to show the metadata_tools
        if show_metadata == "table":
            md_dict = czimd.create_md_dict_red(metadata, sort=True, remove_none=True)
            mdtable = MdTableWidget()
            mdtable.update_metadata(md_dict)
            mdtable.update_style()
            viewer.window.add_dock_widget(mdtable, name="MetadataTable", area="right")

    # add all channels as individual layers
    if metadata.image.SizeC is None:
        size_c = 1
    else:
        size_c = metadata.image.SizeC

    # loop over all channels and add them as layers
    for ch in range(size_c):
        try:
            # get the channel name
            chname = metadata.channelinfo.names[ch]
            # inside the CZI metadata_tools colors are defined as ARGB hexstring
            rgb = "#" + metadata.channelinfo.colors[ch][3:]
            ncmap = Colormap(["#000000", rgb], name="cm_" + chname)
        except (KeyError, IndexError) as e:
            logger.warning(e)
            # or use CH1 etc. as string for the name
            chname = "CH" + str(ch + 1)
            ncmap = Colormap(["#000000", "#ffffff"], name="cm_" + chname)

        # cut out channel
        if metadata.image.SizeC is not None:
            channel = misc.slicedim(array, ch, dim_order["C"])
        if metadata.image.SizeC is None:
            channel = array

        # actually show the image array
        if verbose:
            logger.info(f"Adding Channel: {chname}")
            logger.info(f"Shape Channel: {ch} , {channel.shape}")
            logger.info(f"Scaling Factors: {scalefactors}")

        if contrast == "calc":
            # really calculate the min and max values - might be slow
            sc = misc.calc_scaling(channel, corr_min=1.1, corr_max=0.9)
            if verbose:
                logger.info(f"Calculated Display Scaling (min & max): {sc}")

            # add channel to napari viewer
            new_layer = viewer.add_image(
                channel,
                name=chname,
                scale=scalefactors,
                contrast_limits=sc,
                blending=blending,
                gamma=gamma,
                colormap=ncmap,
            )

        if contrast == "napari_auto":
            # let Napari figure out what the best display scaling is
            # Attention: It will measure in the center of the image !!!

            # add channel to napari viewer
            new_layer = viewer.add_image(
                channel,
                name=chname,
                scale=scalefactors,
                blending=blending,
                gamma=gamma,
                colormap=ncmap,
            )
        if contrast == "from_czi":
            # guess an appropriate scaling from the display setting embedded in the CZI
            try:
                lower = np.round(
                    metadata.channelinfo.clims[ch][0] * metadata.maxvalue_list[ch], 0
                )
                higher = np.round(
                    metadata.channelinfo.clims[ch][1] * metadata.maxvalue_list[ch], 0
                )
            except IndexError as e:
                logger.warning(
                    "Calculation from display setting from CZI failed. Use 0-Max instead."
                )
                lower = 0
                higher = metadata.maxvalue[ch]

            # simple validity check
            if lower >= higher:
                logger.info("Fancy Display Scaling detected. Use Defaults")
                lower = 0
                higher = np.round(metadata.maxvalue[ch] * 0.25, 0)

            if verbose:
                logger.info(
                    f"Display Scaling from CZI for CH: {ch} Min-Max: {lower}-{higher}"
                )

            # add channel to Napari viewer
            new_layer = viewer.add_image(
                channel,
                name=chname,
                scale=scalefactors,
                contrast_limits=[lower, higher],
                blending=blending,
                gamma=gamma,
                colormap=ncmap,
            )

        # append the current layer
        napari_layers.append(new_layer)

    if name_sliders:
        # get the label of the sliders (as a tuple) ad rename it
        sliderlabels = rename_sliders(viewer.dims.axis_labels, dim_order)
        viewer.dims.axis_labels = sliderlabels

    return napari_layers


def rename_sliders(sliders: Tuple, dim_order: Dict) -> Tuple:
    """Rename the sliders inside the Napari viewer based on the metadata_tools


    Args:
        sliders (Tuple): labels of sliders from viewer
        dim_order (Dict): dictionary indication the dimension string and its position inside the array

    Returns:
        Tuple: tuple with renamed sliders
    """

    # update the labels with the correct dimension strings
    slidernames = ["S", "T", "Z"]

    # convert to list()
    tmp_sliders = list(sliders)

    for s in slidernames:
        try:
            if dim_order[s] >= 0:
                # assign the dimension labels
                tmp_sliders[dim_order[s]] = s

        except KeyError as e:
            logger.info(f"{e}: No {s} Dimension found")

    # convert back to tuple
    sliders = tuple(tmp_sliders)

    return sliders
