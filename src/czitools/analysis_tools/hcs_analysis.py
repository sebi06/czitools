# -*- coding: utf-8 -*-
"""HCS OME-Zarr analysis utilities for czitools analysis tools.

Provides :func:`process_hcs_omezarr` to count objects per well in an HCS
OME-Zarr plate.

Vendored (with light edits) from ``czi_omezarr_utils.processing`` in the
``omezarr_playground`` repository.

These features require optional dependencies. Install them with::

    pip install "czitools[analysis]"
"""

from typing import Dict, Tuple
import numpy as np
import ngff_zarr as nz

from czitools.analysis_tools.processing import ArrayProcessor
from czitools.utils import logging_tools

logger = logging_tools.set_logging()


def process_hcs_omezarr(
    hcs_omezarr_path: str,
    channel2analyze: int = 0,
    measure_properties: Tuple[str, ...] = ("label", "area", "centroid", "bbox"),
) -> Dict[str, int]:
    """Process an HCS OME-ZARR file to count objects per well.

    Iterates over every well and field in the plate, applies Otsu thresholding
    followed by connected-component labelling, and returns the total object count
    per well. Currently only 2D images (squeezed from the channel dimension) are
    supported.

    Args:
        hcs_omezarr_path (str): Path to the HCS OME-ZARR file or directory.
        channel2analyze (int): Index of the channel to analyse (default: 0).
        measure_properties (Tuple[str, ...]): regionprops properties to measure
            (default: label/area/centroid/bbox).

    Returns:
        Dict[str, int]: Dictionary mapping well path strings (e.g. ``"B/4"``) to
            total object counts.

    Raises:
        Exception: Re-raises any validation error from ``nz.from_hcs_zarr``.
    """
    try:
        logger.info("Validating HCS-ZARR file against schema...")
        hcs_plate = nz.from_hcs_zarr(hcs_omezarr_path, validate=True)
        logger.info("Validation successful.")
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        raise

    results_obj: Dict[str, int] = {}
    results_mean: Dict[str, float] = {}

    logger.info(f"Number of wells in metadata: {len(hcs_plate.metadata.wells)}")
    logger.info(f"Wells in metadata: {[w.path for w in hcs_plate.metadata.wells]}")

    for well_meta in hcs_plate.metadata.wells:
        row, col = well_meta.path.split("/")
        logger.info(f"\nProcessing well: {well_meta.path} (Row: {row}, Column: {col})")

        well = hcs_plate.get_well(row, col)

        if not well:
            logger.warning(f"  Well {well_meta.path} not found in plate, skipping")
            continue

        if not well.images or len(well.images) == 0:
            logger.warning(f"  Well {well_meta.path} has no images, skipping")
            continue

        field_intensities: list = []
        field_num_objects: list = []

        logger.info(f"  Found {len(well.images)} field(s) in well {well_meta.path}")

        for field_idx in range(len(well.images)):
            image = well.get_image(field_idx)

            if image:
                data = image.images[0].data.compute()
                logger.info(f"  Processing field {field_idx}: shape={data.shape}, dtype={data.dtype}")

                ap = ArrayProcessor(np.squeeze(data[:, channel2analyze, ...]))
                pro2d = ap.apply_otsu_threshold()
                ap = ArrayProcessor(pro2d)
                pro2d, num_objects, _props = ap.label_objects(
                    min_size=100,
                    label_rgb=False,
                    orig_image=None,
                    bg_label=0,
                    measure_params=True,
                    measure_properties=measure_properties,
                )

                field_num_objects.append(int(num_objects))
                field_intensities.append(float(np.mean(data)))

        results_mean[f"{row}/{col}"] = float(np.mean(field_intensities)) if field_intensities else 0.0
        results_obj[f"{row}/{col}"] = int(np.sum(field_num_objects))

    logger.info(f"Total wells processed: {len(results_mean)}")

    return results_obj
