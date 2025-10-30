"""
CZI to OME-ZARR HCS Converter

This script converts CZI (Carl Zeiss Image) files containing High Content Screening (HCS)
plate data into the OME-ZARR format. The output follows the OME-NGFF specification for
HCS data, organizing images in a plate/well/field hierarchy.

Usage:
    python convert2hcs_omezarr.py --czifile path/to/file.czi [OPTIONS]

Example:
    python convert2hcs_omezarr.py --czifile WP96_plate.czi --plate_name "Experiment_001" --overwrite
"""

import argparse

import logging
from pathlib import Path
from ome_zarr_utils import convert_czi_to_hcsplate
import ngff_zarr as nz


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Convert CZI files to OME-ZARR HCS (High Content Screening) format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic conversion with default settings
    python convert2hcs_omezarr.py --czifile WP96_plate.czi
    
    # Specify custom output path and plate name
    python convert2hcs_omezarr.py --czifile WP96_plate.czi --zarr_output /path/to/output.ome.zarr --plate_name "Experiment_001"
    
    # Enable overwrite mode to replace existing files
    python convert2hcs_omezarr.py --czifile WP96_plate.czi --overwrite

Notes:
    - The output format follows the OME-NGFF specification for HCS data
    - Data is organized in a plate/well/field hierarchy
    - All conversion logs are saved to 'czi_hcs_omezarr.log'
        """,
    )

    # Required arguments
    parser.add_argument(
        "--czifile",
        type=str,
        required=True,
        help="Path to the input CZI file to convert (required)",
    )

    # Optional arguments
    parser.add_argument(
        "--zarr_output",
        type=str,
        default=None,
        help="Output path for the OME-ZARR file (default: <czifile>_ngff_plate.ome.zarr)",
    )
    parser.add_argument(
        "--plate_name",
        type=str,
        default="Automated Plate",
        help="Name of the well plate for metadata (default: 'Automated Plate')",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing OME-ZARR files if they exist (default: False)",
    )

    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate the output OME-ZARR files (default: False)",
    )

    args = parser.parse_args()

    # Validate input CZI file exists
    czi_filepath = Path(args.czifile)
    if not czi_filepath.exists():
        print(f"Input CZI file not found: {czi_filepath}")
        raise FileNotFoundError(f"CZI file does not exist: {czi_filepath}")

    # Derive log file path from CZI file location and name
    log_file_path = czi_filepath.parent / f"{czi_filepath.stem}_hcs_omezarr.log"

    # Configure logging with both console and file output
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),  # Output to console
            logging.FileHandler(str(log_file_path)),  # Output to log file
        ],
    )
    logger = logging.getLogger(__name__)

    logger.info("=" * 80)
    logger.info("CZI to OME-ZARR HCS Conversion Started")
    logger.info("=" * 80)

    if not czi_filepath.suffix.lower() == ".czi":
        logger.warning(f"Input file does not have .czi extension: {czi_filepath}")

    logger.info(f"Input CZI file: {czi_filepath.absolute()}")

    # Determine output path
    if args.zarr_output is None:
        # Generate default output path based on input filename
        zarr_output_path = str(czi_filepath.with_suffix("")) + "_ngff_plate.ome.zarr"
        logger.info(f"No output path specified, using default: {zarr_output_path}")
    else:
        zarr_output_path = args.zarr_output
        logger.info(f"Using specified output path: {zarr_output_path}")

    # Log plate name and overwrite settings
    logger.info(f"Plate name: {args.plate_name}")
    logger.info(f"Overwrite mode: {args.overwrite}")

    if args.overwrite:
        logger.warning("Overwrite enabled: Existing OME-ZARR files will be removed!")

    # Perform the conversion
    try:
        logger.info("Starting conversion process...")
        result_path = convert_czi_to_hcsplate(
            czi_filepath=str(czi_filepath), plate_name=args.plate_name, overwrite=args.overwrite
        )

        # Log successful completion
        logger.info("=" * 80)
        logger.info("Conversion completed successfully!")
        logger.info(f"Output OME-ZARR file: {result_path}")
        logger.info("=" * 80)

        # Optional validation step
        if args.validate:
            logger.info("Validating created HCS-ZARR file against schema...")
            hcs_plate = nz.from_hcs_zarr(zarr_output_path, validate=args.validate)
            logger.info("Validation successful.")

    except Exception as e:
        # Log any errors that occur during conversion
        logger.error("=" * 80)
        logger.error(f"Conversion failed with error: {type(e).__name__}")
        logger.error(f"Error message: {str(e)}")
        logger.error("=" * 80)
        raise
