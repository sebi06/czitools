import logging
import ngff_zarr as nz
import numpy as np
from processing_tools import ArrayProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def process_hcs_omezarr(
    hcs_omezarr_path: str, channel2analyze: int = 0, measure_properties=("label", "area", "centroid", "bbox")
):
    """
    Process an HCS OME-ZARR file to analyze wells and generate heatmaps.

    Parameters:
        hcs_omezarr_path (str): Path to the HCS OME-ZARR file.
        channel2analyze (int): Index of the channel to analyze.
    """

    try:
        logger.info("Validating created HCS-ZARR file against schema...")
        hcs_plate = nz.from_hcs_zarr(hcs_omezarr_path, validate=True)
        logger.info("Validation successful.")
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        raise e

    # run some processing
    results_obj = {}
    results_mean = {}
    measure_properties = ("label", "area", "centroid", "bbox")

    # Debug: Print plate metadata information
    logger.info(f"Number of wells in metadata: {len(hcs_plate.metadata.wells)}")
    logger.info(f"Wells in metadata: {[w.path for w in hcs_plate.metadata.wells]}")

    # Iterate through all wells that actually have data
    # Use the well path directly since it's always correct (e.g., "B/4")
    for well_meta in hcs_plate.metadata.wells:

        # Extract row and column from the path (format: "B/4")
        row, col = well_meta.path.split("/")
        logger.info(f"\nProcessing well: {well_meta.path} (Row: {row}, Column: {col})")

        # Get the well object for the current row/column position
        well = hcs_plate.get_well(row, col)

        # Only process if the well exists and has data
        if not well:
            logger.warning(f"  Warning: Well {well_meta.path} not found in plate, skipping")
            continue

        if not well.images or len(well.images) == 0:
            logger.warning(f"  Warning: Well {well_meta.path} has no images, skipping")
            continue

        # Store intensities for all fields (positions) within the well
        field_intensities = []
        field_num_objects = []

        print(f"  Found {len(well.images)} field(s) in well {well_meta.path}")

        # Process each field (microscope position) in the current well
        for field_idx in range(len(well.images)):

            image = well.get_image(field_idx)

            if image:

                # Load the image data into memory (compute() for dask arrays)
                data = image.images[0].data.compute()
                logger.info(
                    f"Processing well: {well_meta.path} - Field {field_idx} data shape: {data.shape}, dtype: {data.dtype}"
                )

                # count objects
                ap = ArrayProcessor(np.squeeze(data[:, channel2analyze, ...]))  # 2D data as input
                pro2d = ap.apply_otsu_threshold()
                ap = ArrayProcessor(pro2d)
                pro2d, num_objects, props = ap.label_objects(
                    min_size=100,
                    label_rgb=False,
                    orig_image=None,
                    bg_label=0,
                    measure_params=True,
                    measure_properties=measure_properties,
                )

                # store number of objects for this field
                field_num_objects.append(int(num_objects))

                # Calculate mean intensity for this field
                mean_intensity = np.mean(data)
                field_intensities.append(mean_intensity)

        # Store the average intensity across all fields for this well
        results_mean[f"{row}/{col}"] = np.mean(field_intensities)

        # Store the total number of objects across all fields for this well
        results_obj[f"{row}/{col}"] = np.sum(field_num_objects)

    # Report the number of wells processed
    logger.info(f"Total Size of results: {len(results_mean)}")

    return results_obj
