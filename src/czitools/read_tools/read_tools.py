# -*- coding: utf-8 -*-

#################################################################
# File        : read_tools.py
# Author      : sebi06
#
# Disclaimer: This code is purely experimental. Feel free to
# use it at your own risk.
#
#################################################################

from typing import (
    Dict,
    Tuple,
    Optional,
    Union,
    Annotated,
)
from pylibCZIrw import czi as pyczi
from czitools.metadata_tools import czi_metadata as czimd
from czitools.utils import misc
import numpy as np
from pathlib import Path
import dask.array as da
import dask.delayed
import os
from tqdm import tqdm
from tqdm.contrib.itertools import product
import tempfile
import shutil
from czitools.utils import logging_tools
from czitools.metadata_tools.helper import ValueRange
from czitools.metadata_tools.helper import AttachmentType

# from memory_profiler import profile

logger = logging_tools.set_logging()

# code for which memory has to be monitored
# instantiating the decorator
# @profile
def read_6darray(
    filepath: Union[str, os.PathLike[str]],
    use_dask: bool = False,
    chunk_zyx: bool = False,
    planes: Dict[str, tuple[int, int]] = {},
    zoom: Annotated[float, ValueRange(0.01, 1.0)] = 1.0,
) -> Tuple[Optional[Union[np.ndarray, da.Array]], czimd.CziMetadata]:
    """Read a CZI image file as 6D dask array.
    Important: Currently supported are only scenes with equal size and CZIs with consistent pixel types.
    The output array has always the dimension order: STCZYX(A)

    Args:
        filepath (str | Path): filepath for the CZI image
        use_dask (bool, optional): Option to store image data as dask array with delayed reading
        chunk_zyx (bool, optional): Option to chunk dask array along z-stacks. Defaults to False.
        zoom (float, optional): Downscale images using a factor [0.01 - 1.0]. Defaults to 1.0
        planes (dict, optional): Allowed keys S, T, Z, C and their start and end values must be >=0 (zero-based).
                                 planes = {"Z":(0, 2)} will return 3 z-plane with indices (0, 1, 2).
                                 Respectively {"Z":(5, 5)} will return a single z-plane with index 5.
        zoom (float, options): Downsample CZI image by a factor [0.01 - 1.0]. Defaults to 1.0.

    Returns:
        Tuple[array6d, mdata]: output as 6D dask array and metadata_tools
    """

    if isinstance(filepath, Path):
        # convert to string
        filepath = str(filepath)

    # check zoom factor
    if zoom > 1.0:
        logger.warning(
            f"Zoom factor f{zoom} is not in valid range [0.01 - 1.0]. Using 1.0 instead."
        )
        zoom = 1.0
    if zoom < 0.01:
        logger.warning(
            f"Zoom factor f{zoom} is not in valid range [0.01 - 1.0]. Using 0.01 instead."
        )
        zoom = 0.01

    # get the complete metadata_tools at once as one big class
    mdata = czimd.CziMetadata(filepath)

    if not mdata.consistent_pixeltypes:
        logger.info("Detected PixelTypes ar not consistent. Cannot create array6d")
        return None, mdata

    # check zoomlevels
    if zoom < 0.01:
        logger.warning(
            f"Zoomlevel: {zoom} is outside allowed range [0.05 - 1.0]. Using 0.5 instead"
        )
    elif zoom > 1.0:
        logger.warning(
            f"Zoomlevel: {zoom} is outside allowed range [0.05 - 1.0]. Using 1.0 instead"
        )

    # update scaling
    mdata.scale.X_sf = np.round(mdata.scale.X * (1 / zoom), 3)
    mdata.scale.Y_sf = np.round(mdata.scale.Y * (1 / zoom), 3)
    mdata.scale.ratio["zx_sf"] = np.round(mdata.scale.Z / mdata.scale.X_sf, 3)

    # check planes
    if not planes is False:
        for k in ["S", "T", "C", "Z"]:
            if k in planes.keys() and k in mdata.bbox.total_bounding_box.keys():
                if mdata.bbox.total_bounding_box[k][1] - 1 < planes[k][1]:
                    logger.info(
                        f"Planes indices (zero-based) for {planes[k]} are invalid. BBox for {[k]}: {mdata.bbox.total_bounding_box[k]}"
                    )
                    return None, mdata

    if mdata.consistent_pixeltypes:
        # use pixel type from first channel
        use_pixeltype = mdata.npdtype[0]

    if not mdata.scene_shape_is_consistent and not "S" in planes.keys():
        logger.info("Scenes have inconsistent shape. Cannot read 6D array")
        return None, mdata

    # open the CZI document to read the
    with pyczi.open_czi(filepath, mdata.pyczi_readertype) as czidoc:
        # if mdata.image.SizeS is not None:
        #     # get size for a single scene using the 1st
        #     # works only if scene shape is consistent

        #     # use the size of the 1st scenes_bounding_rectangle
        #     size_x = czidoc.scenes_bounding_rectangle[0].w
        #     size_y = czidoc.scenes_bounding_rectangle[0].h

        # if mdata.image.SizeS is None:
        #     # use the size of the total_bounding_rectangle
        #     size_x = czidoc.total_bounding_rectangle.w
        #     size_y = czidoc.total_bounding_rectangle.h

        # check if dimensions are None (because they do not exist for that image)
        size_c = misc.check_dimsize(mdata.image.SizeC, set2value=1)
        size_z = misc.check_dimsize(mdata.image.SizeZ, set2value=1)
        size_t = misc.check_dimsize(mdata.image.SizeT, set2value=1)
        size_s = misc.check_dimsize(mdata.image.SizeS, set2value=1)

        s_start = 0
        s_end = size_s
        t_start = 0
        t_end = size_t
        c_start = 0
        c_end = size_c
        z_start = 0
        z_end = size_z

        # check for additional arguments to create substacks
        if (
            not planes is False
            and mdata.image.SizeS is not None
            and "S" in planes.keys()
        ):
            size_s = planes["S"][1] - planes["S"][0] + 1
            mdata.image.SizeS = size_s
            s_start = planes["S"][0]
            s_end = planes["S"][1] + 1

        if not planes is False and "T" in planes.keys():
            size_t = planes["T"][1] - planes["T"][0] + 1
            mdata.image.SizeT = size_t
            t_start = planes["T"][0]
            t_end = planes["T"][1] + 1

        if not planes is False and "Z" in planes.keys():
            size_z = planes["Z"][1] - planes["Z"][0] + 1
            mdata.image.SizeZ = size_z
            z_start = planes["Z"][0]
            z_end = planes["Z"][1] + 1

        if not planes is False and "C" in planes.keys():
            size_c = planes["C"][1] - planes["C"][0] + 1
            mdata.image.SizeC = size_c
            c_start = planes["C"][0]
            c_end = planes["C"][1] + 1

        # check if ADim can be removed because image is grayscale
        remove_adim = False if mdata.isRGB else True

        if mdata.is_url:
            logger.info("Reading pixel data via network from link location.")

        planecount = 0
        shape = ()

        # read array for the scene 2Dplane-by-2Dplane
        for s, t, c, z in product(
            enumerate(range(s_start, s_end)),
            enumerate(range(t_start, t_end)),
            enumerate(range(c_start, c_end)),
            enumerate(range(z_start, z_end)),
            desc="Reading 2D planes",
            unit=" 2Dplanes",
        ):
            planecount += 1

            # read a 2D image plane from the CZI
            if mdata.image.SizeS is None:
                image2d = czidoc.read(
                    plane={"T": t[1], "Z": z[1], "C": c[1]}, zoom=zoom
                )
            else:
                image2d = czidoc.read(
                    plane={"T": t[1], "Z": z[1], "C": c[1]}, scene=s[1], zoom=zoom
                )

            if planecount == 1:
                # allocate array based on the expected size incl. down scaling
                shape = (
                    size_s,
                    size_t,
                    size_c,
                    size_z,
                    image2d.shape[0],
                    image2d.shape[1],
                    3 if mdata.isRGB else 1,
                )

                # in case of numpy array
                if not use_dask:
                    array6d = np.empty(shape, dtype=use_pixeltype)

                # in case of normal dask array
                elif use_dask:
                    array6d = da.empty(shape, dtype=use_pixeltype, chunks=shape)

            # insert 2D image plane into the array6d
            array6d[s[0], t[0], c[0], z[0], ...] = image2d

        # remove the A dimension
        if remove_adim:
            array6d = np.squeeze(array6d, axis=-1)
            # dim_string = dim_string.replace("A", "")

            if use_dask and chunk_zyx:
                # for testing
                array6d = array6d.rechunk(chunks=(1, 1, 1, size_z, image2d.shape[0], image2d.shape[1]))

        if not remove_adim:
            if use_dask and chunk_zyx:
                # for testing
                array6d = array6d.rechunk(chunks=(1, 1, 1, size_z, image2d.shape[0], image2d.shape[1], 3))

    # update metadata_tools
    mdata.array6d_size = array6d.shape

    return array6d, mdata


# code for which memory has to be monitored
# instantiating the decorator
# @profile
def read_6darray_lazy(
    filepath: Union[str, os.PathLike[str]],
    chunk_zyx=False,
    planes: Dict[str, tuple[int, int]] = {},
) -> Tuple[Optional[Union[np.ndarray, da.Array]], czimd.CziMetadata]:
    """Read a CZI image file as 6D dask array with delayed plane reading.
    Important: Currently supported are only scenes with equal size and CZIs with consistent pixel types.
    The output array has always the dimension order: STCZYX(A)

    Args:
        filepath (str | Path): filepath for the CZI image
        chunk_zyx (bool, optional): Option to chunk dask array along z-stacks. Defaults to False.
        planes (dict, optional): Allowed keys S, T, Z, C and their start and end values must be >=0 (zero-based).
                                 planes = {"Z":(0, 2)} will return 3 z-plane with indices (0, 1, 2).
                                 Respectively {"Z":(5, 5)} will return a single z-plane with index 5.

    Returns:
        Tuple[array6d, mdata]: output as 6D dask array and metadata_tools
    """

    if isinstance(filepath, Path):
        # convert to string
        filepath = str(filepath)

    # get the complete metadata_tools at once as one big class
    mdata = czimd.CziMetadata(filepath)

    if not mdata.consistent_pixeltypes:
        logger.info("Detected PixelTypes ar not consistent. Cannot create array6d")
        return None, mdata

    # check planes
    if not planes is False:
        for k in ["S", "T", "C", "Z"]:
            if k in planes.keys() and k in mdata.bbox.total_bounding_box.keys():
                if mdata.bbox.total_bounding_box[k][1] - 1 < planes[k][1]:
                    logger.info(
                        f"Planes indices (zero-based) for {planes[k]} are invalid. BBox for {[k]}: {mdata.bbox.total_bounding_box[k]}"
                    )
                    return None, mdata

    if not mdata.scene_shape_is_consistent and not "S" in planes.keys():
        logger.info("Scenes have inconsistent shape. Cannot read 6D array")
        return None, mdata

    # open the CZI document to read the
    with pyczi.open_czi(filepath, mdata.pyczi_readertype) as czidoc:
        if mdata.image.SizeS is not None:
            # get size for a single scene using the 1st
            # works only if scene shape is consistent

            # use the size of the 1st scenes_bounding_rectangle
            size_x = czidoc.scenes_bounding_rectangle[0].w
            size_y = czidoc.scenes_bounding_rectangle[0].h

        if mdata.image.SizeS is None:
            # use the size of the total_bounding_rectangle
            size_x = czidoc.total_bounding_rectangle.w
            size_y = czidoc.total_bounding_rectangle.h

        # check if dimensions are None (because they do not exist for that image)
        size_c = misc.check_dimsize(mdata.image.SizeC, set2value=1)
        size_z = misc.check_dimsize(mdata.image.SizeZ, set2value=1)
        size_t = misc.check_dimsize(mdata.image.SizeT, set2value=1)
        size_s = misc.check_dimsize(mdata.image.SizeS, set2value=1)

        s_start = 0
        s_end = size_s
        t_start = 0
        t_end = size_t
        c_start = 0
        c_end = size_c
        z_start = 0
        z_end = size_z

        # check for additional arguments to create substacks
        if (
            not planes is False
            and mdata.image.SizeS is not None
            and "S" in planes.keys()
        ):
            size_s = planes["S"][1] - planes["S"][0] + 1
            mdata.image.SizeS = size_s
            s_start = planes["S"][0]
            s_end = planes["S"][1] + 1

        if not planes is False and "T" in planes.keys():
            size_t = planes["T"][1] - planes["T"][0] + 1
            mdata.image.SizeT = size_t
            t_start = planes["T"][0]
            t_end = planes["T"][1] + 1

        if not planes is False and "Z" in planes.keys():
            size_z = planes["Z"][1] - planes["Z"][0] + 1
            mdata.image.SizeZ = size_z
            z_start = planes["Z"][0]
            z_end = planes["Z"][1] + 1

        if not planes is False and "C" in planes.keys():
            size_c = planes["C"][1] - planes["C"][0] + 1
            mdata.image.SizeC = size_c
            c_start = planes["C"][0]
            c_end = planes["C"][1] + 1

        if mdata.isRGB:
            shape2d = (size_y, size_x, 3)
        elif not mdata.isRGB:
            shape2d = (size_y, size_x)

        # check if ADim can be removed because image is grayscale
        remove_adim = False if mdata.isRGB else True

        # initialise empty list to hold the dask arrays
        img = []

        with tqdm(
            total=size_s * size_t * size_c * size_z,
            desc="Reading 2D planes",
            unit=" 2dplanes",
        ) as pbar:
            for s in enumerate(range(s_start, s_end)):
                # for s in range(size_s):
                time_stack = []

                for time in enumerate(range(t_start, t_end)):
                    # for time in range(size_t):
                    ch_stack = []

                    for ch in enumerate(range(c_start, c_end)):
                        # for ch in range(size_c):
                        z_stack = []

                        # for z in range(size_z):
                        for z in enumerate(range(z_start, z_end)):
                            # if mdata.image.SizeS is not None:
                            z_slice = da.from_delayed(
                                read_2dplane(
                                    czidoc,
                                    s=s[1],
                                    t=time[1],
                                    c=ch[1],
                                    z=z[1],
                                    has_scenes=mdata.has_scenes,
                                    remove_adim=remove_adim,
                                ),
                                shape=shape2d,
                                dtype=mdata.npdtype[0],
                            )
                            pbar.update()

                            # if mdata.image.SizeS is None:
                            #     z_slice = da.from_delayed(
                            #         read_2dplane(
                            #             czidoc,
                            #             s=s[1],
                            #             t=time[1],
                            #             c=ch[1],
                            #             z=z[1],
                            #             has_scenes=False,
                            #             remove_adim=remove_adim,
                            #         ),
                            #         shape=shape2d,
                            #         dtype=mdata.npdtype[0],
                            #     )
                            #     pbar.update()

                            # create 2d array
                            z_slice = da.squeeze(z_slice)

                            # append to the z-stack
                            z_stack.append(z_slice)

                        # stack the array and create new dask array
                        z_stack = da.stack(z_stack, axis=0)

                        # create CZYX list of dask array
                        ch_stack.append(z_stack)

                    # create TCZYX list of dask array
                    time_stack.append(ch_stack)

                # create STCZYX list of dask array
                img.append(time_stack)

            # create final STCZYX dask array
            array6d = da.stack(img, axis=0)

        # dim_string = "STCZYXA"

        # remove the A dimension
        if remove_adim:
            # dim_string = dim_string.replace("A", "")
            if chunk_zyx:
                # for testing
                array6d = array6d.rechunk(chunks=(1, 1, 1, size_z, size_y, size_x))

        if not remove_adim:
            if chunk_zyx:
                # for testing
                array6d = array6d.rechunk(chunks=(1, 1, 1, size_z, size_y, size_x, 3))

    return array6d, mdata


@dask.delayed
def read_2dplane(
    czidoc: pyczi.CziReader,
    s: int = 0,
    t: int = 0,
    c: int = 0,
    z: int = 0,
    has_scenes: bool = True,
    remove_adim: bool = True,
):
    """Dask delayed function to read a 2d plane from a CZI image, which has the shape
       (Y, X, 1) or (Y, X, 3).
       If the image is a grayscale image the last array dimension will be removed, when
       the option is ste to True.

    Args:
        czidoc (pyczi.CziReader): CziReader objects
        s (int, optional): Scene index. Defaults to 0.
        t (int, optional): Time index. Defaults to 0.
        c (int, optional): Channel index. Defaults to 0.
        z (int, optional): Z-Plane index. Defaults to 0.
        has_scenes (bool, optional): Defines if the CZI actually contains scenes. Defaults to True.
        remove_adim (bool, optional): Option to remove the last dimension of the 2D array. Defaults to True.

    Returns:
        dask.array: 6d dask.array with delayed reading for individual 2d planes
    """

    # initialize 2d array with some values
    image2d = np.zeros([10, 10], dtype=np.int16)

    if has_scenes:
        # read a 2d plane using the scene index
        image2d = czidoc.read(plane={"T": t, "C": c, "Z": z}, scene=s)
    if not has_scenes:
        # reading a 2d plane in case the CZI has no scenes
        image2d = czidoc.read(plane={"T": t, "C": c, "Z": z})

    # remove a last "A" dimension when desired
    if remove_adim:
        return image2d[..., 0]
    if not remove_adim:
        return image2d


def read_attachments(
    czi_filepath: [str, os.PathLike],
    attachment_type: AttachmentType = AttachmentType.SlidePreview,
    copy: bool = True,
) -> Tuple[np.ndarray, Optional[str]]:
    """Read attachment images from a CZI image as numpy array

    Args:
        czi_filepath (str): FilePath of the CZI image file
        attachment_type (str, optional): Type of the attachment to be read. Defaults to "SlidePreview".
        copy (bool, optional): Option to copy the attachments as CZI image directly. Defaults to True.

    Raises:
        Exception: If specified attachment is not found and exception is raised.

    Returns:
        Tuple[nd,array, [Optional[str]]: Tuple containing the 2d image array and optionally the location of the copied image.
    """

    try:
        import czifile

        if attachment_type not in AttachmentType:
            # if attachment_type not in ["SlidePreview", "Label", "]:
            raise Exception(
                f"{attachment_type} is not supported. Valid types are: SlidePreview, Label or Prescan."
            )

        att = czimd.CziAttachments(czi_filepath)

        if attachment_type == AttachmentType.Label and not att.has_label:
            # if attachment_type == "Label" and not att.has_label:
            return np.array([]), None

        if attachment_type == AttachmentType.SlidePreview and not att.has_preview:
            # if attachment_type == "SlidePreview" and not att.has_preview:
            return np.array([]), None

        if attachment_type == AttachmentType.Prescan and not att.has_prescan:
            # if attachment_type == "SlidePreview" and not att.has_preview:
            return np.array([]), None

        # create CZI-object using czifile library
        with czifile.CziFile(czi_filepath) as cz:
            with tempfile.TemporaryDirectory() as tmpdirname:
                # save attachments to temporary directory
                cz.save_attachments(directory=tmpdirname)

                # iterate over attachments
                for att in cz.attachments():
                    if att.attachment_entry.name == attachment_type.name:
                        # get the full path of the attachment image
                        full_path = Path(tmpdirname) / att.attachment_entry.filename

                        if copy:
                            # create path to store the attachment image
                            att_path = (
                                str(czi_filepath)[:-4]
                                + "_"
                                + att.attachment_entry.name
                                + ".czi"
                            )

                            # copy the file
                            dest = shutil.copyfile(full_path, att_path)

                        # open the CZI document to read array
                        with pyczi.open_czi(str(full_path)) as czidoc:
                            img2d = czidoc.read()

                        if copy:
                            return img2d, dest
                        if not copy:
                            return img2d

    except ImportError as e:
        logger.warning(
            "Package czifile not found. Cannot extract information about attached images."
        )

        return None, None
