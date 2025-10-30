import zarr
from pathlib import Path
from czitools.read_tools import read_tools
from czitools.metadata_tools.czi_metadata import CziMetadata
from ome_zarr.io import parse_url
from ome_zarr.writer import write_image, write_plate_metadata, write_well_metadata
import ome_zarr.writer
import ome_zarr.format
import shutil
import ngff_zarr as nz
from ngff_zarr.v04.zarr_metadata import Plate, PlateColumn, PlateRow, PlateWell
from ngff_zarr.hcs import HCSPlate, to_hcs_zarr
from dataclasses import dataclass
from typing import Dict
from enum import Enum
import numpy as np
import xarray as xr
from typing import Union, Optional
import dask.array as da


def convert_czi_to_hcs_zarr(czi_filepath: str, overwrite: bool = True) -> str:
    """Convert CZI file to OME-ZARR HCS (High Content Screening) format.

    This function converts a CZI (Carl Zeiss Image) file containing plate data into
    the OME-ZARR HCS format. It handles multi-well plates with multiple fields per well.

    Args:
        czi_filepath: Path to the input CZI file
        overwrite: If True, removes existing zarr files at the output path.
                  If False, skips conversion if output exists.

    Returns:
        str: Path to the output ZARR file (.ngff_plate.zarr)

    Note:
        The output format follows the OME-NGFF specification for HCS data,
        organizing the data in a plate/row/column/field hierarchy.
    """
    # Define output path
    zarr_output_path = Path(czi_filepath[:-4] + "_ngff_plate.ome.zarr")

    # Handle existing files
    if zarr_output_path.exists():
        if overwrite:
            shutil.rmtree(zarr_output_path)
        else:
            print(f"File exists at {zarr_output_path}. Set overwrite=True to remove.")
            return str(zarr_output_path)

    # Read CZI file
    array6d, mdata = read_tools.read_6darray(czi_filepath, use_xarray=True)

    # Extract plate layout
    row_names, col_names, well_paths = extract_well_coordinates(mdata.sample.well_counter)
    field_paths = [str(i) for i in range(mdata.sample.well_counter[mdata.sample.well_array_names[0]])]

    # Initialize zarr storage and write plate metadata with proper row/column objects
    store = parse_url(zarr_output_path, mode="w").store
    root = zarr.group(store=store)

    # Create PlateRow and PlateColumn objects (required for proper metadata)
    # This ensures plate.metadata.rows and plate.metadata.columns are populated
    columns_metadata = [PlateColumn(name=str(col)) for col in sorted(col_names, key=int)]
    rows_metadata = [PlateRow(name=row) for row in sorted(row_names)]

    # Write plate metadata using the standard ome-zarr-py function
    write_plate_metadata(root, row_names, col_names, well_paths)

    # Additionally, store the rows and columns in the metadata for compatibility
    # This is what ngff-zarr expects to find
    plate_attrs = root.attrs.asdict()
    plate_attrs["rows"] = [{"name": r.name} for r in rows_metadata]
    plate_attrs["columns"] = [{"name": c.name} for c in columns_metadata]
    root.attrs.update(plate_attrs)

    # Process wells
    for wp in well_paths:
        row, col = wp.split("/")
        well_group = root.require_group(row).require_group(col)
        write_well_metadata(well_group, field_paths)

        current_well_id = wp.replace("/", "")
        for fi, field in enumerate(field_paths):
            image_group = well_group.require_group(str(field))
            current_scene_index = mdata.sample.well_scene_indices[current_well_id][fi]

            write_image(
                image=array6d[current_scene_index, ...],
                group=image_group,
                axes=array6d.axes[1:].lower(),
                storage_options=dict(chunks=(1, 1, 1, array6d.Y.size, array6d.X.size)),
            )

    return str(zarr_output_path)


def extract_well_coordinates(
    well_counter: dict,
) -> tuple[list[str], list[str], list[str]]:
    """Extract unique row and column names from a well counter dictionary.

    This function parses well positions (e.g., 'B4', 'B5') to extract unique row letters
    and column numbers, and generates corresponding well paths.

    Args:
        well_counter (dict): Dictionary with well positions as keys (e.g., {'B4': 4, 'B5': 4})

    Returns:
        tuple[list[str], list[str], list[str]]: A tuple containing:
            - row_names: Sorted list of unique row letters
            - col_names: Sorted list of unique column numbers
            - well_paths: List of well paths in format "row/column"
    """
    # Initialize empty sets for rows and columns
    rows = set()
    cols = set()

    # Iterate through well names
    for well in well_counter.keys():
        # Extract row (letters) and column (numbers)
        row = "".join(filter(str.isalpha, well))
        col = "".join(filter(str.isdigit, well))

        rows.add(row)
        cols.add(col)

    # Convert to sorted lists
    row_names = sorted(list(rows))
    col_names = sorted(list(cols))

    # Generate well_paths from the extracted coordinates
    well_paths = [f"{row}/{col}" for row in row_names for col in col_names]

    return row_names, col_names, well_paths


@dataclass
class PlateConfiguration:
    """Configuration for standard microplate formats"""

    rows: int
    columns: int
    name: str

    @property
    def total_wells(self) -> int:
        return self.rows * self.columns

    @property
    def row_labels(self) -> list:
        """Generate row labels (A, B, C, ...)"""
        return [chr(ord("A") + i) for i in range(self.rows)]

    @property
    def column_labels(self) -> list:
        """Generate column labels (1, 2, 3, ...)"""
        return [str(i) for i in range(1, self.columns + 1)]


class PlateType(Enum):
    """Standard microplate formats with their configurations"""

    PLATE_6 = PlateConfiguration(2, 3, "6-Well Plate")
    PLATE_24 = PlateConfiguration(4, 6, "24-Well Plate")
    PLATE_48 = PlateConfiguration(6, 8, "48-Well Plate")
    PLATE_96 = PlateConfiguration(8, 12, "96-Well Plate")
    PLATE_384 = PlateConfiguration(16, 24, "384-Well Plate")
    PLATE_1536 = PlateConfiguration(32, 48, "1536-Well Plate")


# Dictionary for easy lookup by well count
PLATE_FORMATS: Dict[int, PlateConfiguration] = {
    6: PlateType.PLATE_6.value,
    24: PlateType.PLATE_24.value,
    48: PlateType.PLATE_48.value,
    96: PlateType.PLATE_96.value,
    384: PlateType.PLATE_384.value,
    1536: PlateType.PLATE_1536.value,
}


def define_plate(plate_type: PlateType, field_count: int = 1) -> Plate:
    """
    Create a plate metadata object for any standard plate format

    Args:
        plate_type: PlateType enum value specifying the plate format
        field_count: Number of fields per well (default: 1)

    Returns:
        Plate metadata object
    """
    config = plate_type.value

    # Create columns and rows based on configuration
    columns = [PlateColumn(name=label) for label in config.column_labels]
    rows = [PlateRow(name=label) for label in config.row_labels]

    # Generate all wells
    wells = [
        PlateWell(path=f"{row.name}/{col.name}", rowIndex=row_idx, columnIndex=col_idx)
        for row_idx, row in enumerate(rows)
        for col_idx, col in enumerate(columns)
    ]

    # Create plate metadata
    plate_metadata = Plate(name=config.name, columns=columns, rows=rows, wells=wells, field_count=field_count)

    return plate_metadata


def define_plate_by_well_count(well_count: int, field_count: int = 1) -> Plate:
    """
    Create a plate by specifying the number of wells

    Args:
        well_count: Number of wells (6, 24, 48, 96, 384, or 1536)
        field_count: Number of fields per well (default: 1)

    Returns:
        Plate metadata object

    Raises:
        ValueError: If well_count is not a standard format
    """
    if well_count not in PLATE_FORMATS:
        available = list(PLATE_FORMATS.keys())
        raise ValueError(f"Unsupported well count: {well_count}. Available formats: {available}")

    config = PLATE_FORMATS[well_count]

    # Create columns and rows based on configuration
    columns = [PlateColumn(name=label) for label in config.column_labels]
    rows = [PlateRow(name=label) for label in config.row_labels]

    # Generate all wells
    wells = [
        PlateWell(path=f"{row.name}/{col.name}", rowIndex=row_idx, columnIndex=col_idx)
        for row_idx, row in enumerate(rows)
        for col_idx, col in enumerate(columns)
    ]

    # Create plate metadata
    plate_metadata = Plate(name=config.name, columns=columns, rows=rows, wells=wells, field_count=field_count)

    return plate_metadata


def convert_czi_to_hcsplate(czi_filepath: str, plate_name: str = "Automated Plate", overwrite: bool = True) -> str:

    # Define output path
    zarr_output_path = Path(czi_filepath[:-4] + "_ngff_plate.ome.zarr")

    # Handle existing files
    if zarr_output_path.exists():
        if overwrite:
            shutil.rmtree(zarr_output_path)
        else:
            print(f"File exists at {zarr_output_path}. Set overwrite=True to remove.")
            return str(zarr_output_path)

    # Read CZI file
    array6d, mdata = read_tools.read_6darray(czi_filepath, use_xarray=True)

    # Extract plate layout
    row_names, col_names, well_paths = extract_well_coordinates(mdata.sample.well_counter)
    field_paths = [str(i) for i in range(mdata.sample.well_counter[mdata.sample.well_array_names[0]])]

    columns = [PlateColumn(name=str(col)) for col in sorted(col_names, key=int)]
    rows = [PlateRow(name=row) for row in sorted(row_names)]

    # Build wells list
    wells = []
    for row in rows:
        # Calculate row index for multi-character rows (A=0, B=1, ..., Z=25, AA=26, AB=27, etc.)
        row_index = 0
        for i, char in enumerate(reversed(row.name.upper())):
            row_index += (ord(char) - ord("A") + 1) * (26**i)
        row_index -= 1  # Convert to 0-based indexing

        for col in columns:
            # Column index: 1->0, 2->1, ... so 4->3, 10->9
            col_index = int(col.name) - 1
            wells.append(
                PlateWell(
                    path=f"{row.name}/{col.name}",
                    rowIndex=row_index,
                    columnIndex=col_index,
                )
            )

    plate = Plate(columns=columns, rows=rows, wells=wells, name=plate_name, field_count=len(field_paths))

    # Create the HCS plate structure
    hcs_plate = HCSPlate(store=zarr_output_path, plate_metadata=plate)
    to_hcs_zarr(hcs_plate, zarr_output_path)

    for well in wells:
        print(f"Creatingw Well: {well.path}")
        row_name, col_name = well.path.split("/")
        current_well_id = well.path.replace("/", "")
        print(f"Current WellID: {current_well_id} Row: {row_name}, Column: {col_name}")
        for fi, field in enumerate(field_paths):
            current_scene_index = mdata.sample.well_scene_indices[current_well_id][fi]

            # create current field image
            current_field_image = nz.NgffImage(
                data=array6d[current_scene_index, ...].data,
                dims=["t", "c", "z", "y", "x"],
                scale={"y": mdata.scale.Y, "x": mdata.scale.X, "z": mdata.scale.Z},
                translation={"t": 0.0, "c": 0.0, "z": 0.0, "y": 0.0, "x": 0.0},
                name=mdata.filename,
            )

            # create multi-scaled, chunked data structure from the image
            multiscales = nz.to_multiscales(
                current_field_image, scale_factors=[2, 2, 2], method=nz.Methods.DASK_IMAGE_GAUSSIAN
            )

            # write to wells
            nz.write_hcs_well_image(
                store=zarr_output_path,
                multiscales=multiscales,
                plate_metadata=plate,
                row_name=row_name,
                column_name=col_name,
                field_index=fi,  # First field of view
                acquisition_id=0,
            )

    return str(zarr_output_path)


def get_display(metadata: CziMetadata, channel_index: int) -> tuple[float, float, float]:
    """
    Extract display range settings for a specific channel from CZI metadata.

    This function retrieves the intensity display window settings (min, max) for a given
    channel from the CZI file metadata. These settings control how the image appears
    when displayed in image viewers.

    Args:
        metadata: CziMetadata object containing channel information and display settings
        channel_index: Zero-based index of the channel to extract display settings for

    Returns:
        tuple[float, float, float]: A tuple containing:
            - lower: Minimum intensity value for display window
            - higher: Maximum intensity value for display window
            - maxvalue: Absolute maximum intensity value for the channel

    Note:
        If display settings cannot be read from the CZI metadata (e.g., missing or
        corrupted data), the function falls back to using 0 as minimum and the
        channel's maximum value as both the display maximum and absolute maximum.
    """

    # Try to read the display settings embedded in the CZI file
    try:
        # Calculate actual intensity values from normalized display limits (0.0-1.0)
        # clims contains normalized values that need to be scaled by the max intensity
        lower = np.round(
            metadata.channelinfo.clims[channel_index][0] * metadata.maxvalue_list[channel_index],
            0,
        )
        higher = np.round(
            metadata.channelinfo.clims[channel_index][1] * metadata.maxvalue_list[channel_index],
            0,
        )

        # Get the absolute maximum intensity value for this channel
        maxvalue = metadata.maxvalue_list[channel_index]

    except IndexError:
        # Fallback when display settings are missing or inaccessible
        print("Calculation from display setting from CZI failed. Use 0-Max instead.")
        lower = 0
        # Use the channel's maximum value from the alternative metadata location
        higher = metadata.maxvalue[channel_index]
        maxvalue = higher

    return lower, higher, maxvalue


def write_omezarr(
    array5d: Union[np.ndarray, xr.DataArray, da.Array],
    zarr_path: Union[str, Path],
    metadata: CziMetadata,
    overwrite: bool = False,
) -> Optional[str]:
    """
    Write a 5D array to OME-ZARR format.

    This function writes a multi-dimensional array (typically from microscopy data)
    to the OME-ZARR format, which is a cloud-optimized format for storing and
    accessing large microscopy datasets.

    Args:
        array5d: Input array with up to 5 dimensions. Can be a numpy array or
                xarray DataArray or dask Array. Expected dimension order is typically TCZYX
                (Time, Channel, Z, Y, X) or similar.
        zarr_path: Path where the OME-ZARR file should be written. Can be a
                  string or Path object.
        metadata: Metadata object containing information about the image.
        overwrite: If True, remove existing file at zarr_path before writing.
                  If False and file exists, return None without writing.
                  Default is False.

    Returns:
        str: Path to the written OME-ZARR file if successful, None if failed.

    Raises:
        None: Function handles errors gracefully and returns None on failure.

    Examples:
        >>> import numpy as np
        >>> data = np.random.rand(10, 2, 5, 512, 512)  # TCZYX
        >>> result = write_omezarr(data, "output.ome.zarr", madata, overwrite=True)
        >>> print(f"Written to: {result}")

    Notes:
        - The function uses chunking strategy (1, 1, 1, Y, X) which keeps
          individual Z-slices as chunks for efficient access.
        - Requires the array to have an 'axes' attribute (typical for xarray)
          or the function will use default axes handling.
        - Uses the current NGFF (Next Generation File Format) specification.
    """

    # Validate input array dimensions - OME-ZARR supports up to 5D
    if len(array5d.shape) > 5:
        print("Input array as more than 5 dimensions.")
        return None

    # Handle existing files based on overwrite parameter
    if Path(zarr_path).exists() and overwrite:
        # Remove existing zarr store completely
        shutil.rmtree(zarr_path, ignore_errors=False, onerror=None)
    elif Path(zarr_path).exists() and not overwrite:
        # Exit early if file exists and overwrite is disabled
        print(f"File already exists at {zarr_path}. Set overwrite=True to remove.")
        return None

    # Display the NGFF specification version being used
    ngff_version = ome_zarr.format.CurrentFormat().version
    print(f"Using ngff format version: {ngff_version}")

    # Initialize zarr store and create root group
    store = parse_url(zarr_path, mode="w").store
    root = zarr.group(store=store, overwrite=overwrite)

    # Write the main image data to zarr
    # Uses chunking strategy that keeps full XY planes together for efficient access
    ome_zarr.writer.write_image(
        image=array5d,
        group=root,
        axes=array5d.axes[1:].lower(),  # Skip first axis (Scene) and convert to lowercase
        storage_options=dict(chunks=(1, 1, 1, array5d.Y.size, array5d.X.size)),
    )

    # Build channel metadata for OMERO visualization
    channels_list = []

    # Process each channel to extract display settings and metadata
    for ch_index in range(metadata.image.SizeC):
        # Extract RGB color from channel metadata (skip first 3 chars, get hex color)
        rgb = metadata.channelinfo.colors[ch_index][3:]
        # Get channel name for display
        chname = metadata.channelinfo.names[ch_index]

        # Calculate display range (min/max intensity values) from CZI metadata
        lower, higher, maxvalue = get_display(metadata, ch_index)

        # Create channel configuration for OMERO viewer
        channels_list.append(
            {
                "color": rgb,  # Hex color code for visualization
                "label": chname,  # Display name for the channel
                "active": True,  # Channel visible by default
                "window": {  # Intensity display range
                    "min": lower,  # Absolute minimum value
                    "start": lower,  # Display window start
                    "end": higher,  # Display window end
                    "max": maxvalue,  # Absolute maximum value
                },
            }
        )

    # Add OMERO metadata for proper visualization in compatible viewers
    ome_zarr.writer.add_metadata(
        root,
        {
            "omero": {
                "name": metadata.filename,  # Original filename for reference
                "channels": channels_list,  # Channel display configurations
            }
        },
    )

    return zarr_path


def write_omezarr_ngff(
    array5d, zarr_output_path: str, metadata: CziMetadata, scale_factors: list = [2, 4, 6], overwrite: bool = False
) -> Optional[nz.NgffImage]:
    """
    Write a 5D array to OME-ZARR NGFF format with multi-scale pyramids.
    This function converts a 5D array (with dimensions t, c, z, y, x) to the OME-ZARR
    Next Generation File Format (NGFF) specification, creating multi-scale representations
    for efficient visualization and analysis.
    Parameters
    ----------
    array5d : array-like
        5D array with dimensions in order [t, c, z, y, x] representing time, channels,
        z-depth, y-coordinate, and x-coordinate respectively.
    zarr_output_path : str
        File path where the OME-ZARR file will be written. Should end with '.ome.zarr'
        extension by convention.
    metadata : CziMetadata
        Metadata object containing scale information (X, Y, Z pixel sizes) and filename
        for the source image.
    scale_factors : list, optional
        List of downsampling factors for creating multi-scale pyramid levels.
        Default is [2, 4, 6].
    overwrite : bool, optional
        If True, existing files at zarr_output_path will be removed before writing.
        If False and file exists, function returns None without writing.
        Default is False.
    Returns
    -------
    image or None
        Returns the NGFF image object if successful, or None if the file already
        exists and overwrite=False.
    Notes
    -----
    - Creates multi-scale representations using Gaussian downsampling via dask-image
    - Automatically sets proper dimension names and scale metadata
    - Uses chunked storage for efficient access patterns
    - Follows OME-ZARR NGFF specification for interoperability
    """

    # Validate input array dimensions - OME-ZARR supports up to 5D
    if len(array5d.shape) > 5:
        print("Input array as more than 5 dimensions.")
        return None

    # check if zarr_path already exits
    if Path(zarr_output_path).exists() and overwrite:
        shutil.rmtree(zarr_output_path, ignore_errors=False, onerror=None)
    elif Path(zarr_output_path).exists() and not overwrite:
        print(f"File already exists at {zarr_output_path}. Set overwrite=True to remove.")
        return None

    # create NGFF image from the array
    image = nz.to_ngff_image(
        array5d.data,
        dims=["t", "c", "z", "y", "x"],
        scale={"y": metadata.scale.Y, "x": metadata.scale.X, "z": metadata.scale.Z},
        name=metadata.filename[:-4] + ".ome.zarr",
    )

    # create multi-scaled, chunked data structure from the image
    multiscales = nz.to_multiscales(image, scale_factors=scale_factors, method=nz.Methods.DASK_IMAGE_GAUSSIAN)

    # write using ngff-zarr
    nz.to_ngff_zarr(zarr_output_path, multiscales)

    return image
