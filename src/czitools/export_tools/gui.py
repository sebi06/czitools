#################################################################
# File        : czi_to_omezarr_gui.py
# Author      : sebi06
# Institution : Carl Zeiss Microscopy GmbH
#
# Copyright(c) 2025 Carl Zeiss AG, Germany. All Rights Reserved.
#
# Permission is granted to use, modify and distribute this code,
# as long as this copyright notice remains part of the code.
#################################################################

"""
MagicGUI Application for CZI to OME-ZARR Conversion

This application provides a graphical user interface for converting Carl Zeiss Image (CZI)
files to OME-ZARR format with support for:
- Single-file OME-ZARR (.ozx) format (NGFF-ZARR only right now)
- HCS (High Content Screening) multi-well plate layouts
- Multiple conversion backends (ome-zarr-py and ngff-zarr)
- Interactive visualization with napari (optional)
"""

from pathlib import Path
from typing import Optional
from magicgui import magicgui, widgets
from czitools.metadata_tools.czi_metadata import CziMetadata

# Internal imports from sibling modules (avoid lazy-loading parent package)
from ._logging import omezarr_package, setup_logging
from .conversion import (
    convert_czi2hcs_omezarr,
    convert_czi2hcs_ngff,
    write_omezarr,
    write_omezarr_ngff,
)
from .plate import convert_hcs_omezarr2ozx
from .validation import validate_ome_zarr

from czitools.read_tools import read_tools
import xarray as xr
import logging
import shutil
import tempfile
import threading
from qtpy.QtCore import QTimer
import ome_zarr.format
import zarr
import ngff_zarr as nz
from importlib.metadata import version

logger = logging.getLogger(__name__)


# ============================================================================
# Module-level Global Variables
# ============================================================================
# These variables maintain application state across callbacks and threads

# Metadata from the currently loaded CZI file
metadata: Optional[CziMetadata] = None

# Maximum number of scenes in the current CZI file
max_scenes: int = 1

# Path to the currently selected CZI file
selected_file: Optional[Path] = None

# Flag indicating whether a conversion is currently in progress
conversion_running: bool = False

# Path to the current conversion log file
log_file_path: Optional[Path] = None

# Current read position in the log file for incremental updates
log_last_position: int = 0

# QTimer instance for periodic log file polling
log_timer: Optional[QTimer] = None

# Path to napari viewer output (unused - kept for compatibility)
napari_viewer_path: Optional[str] = None

# Re-entrancy guard for update_ozx_child_states to prevent signal cascades
_ozx_state_updating: bool = False

# Default parent directory for file browser
try:
    parent_dir: Optional[Path] = Path(__file__).resolve().parents[4] / "data"
    if not parent_dir.exists():
        parent_dir = None
except (ValueError, IndexError):
    parent_dir = None


def update_log_display() -> None:
    """Update log viewer widget with new content from the log file.

    This function performs incremental reading of the log file by seeking to the
    last read position and only reading new content. It's called periodically by
    a QTimer during conversion to provide live log updates.

    The function is thread-safe and designed to be called from the main Qt thread.

    Note:
        Uses global variables log_last_position and log_file_path to track state.
    """
    global log_last_position, log_file_path

    if log_file_path and log_file_path.exists():
        try:
            # Open log file and seek to last read position
            with open(log_file_path, "r", encoding="utf-8") as f:
                f.seek(log_last_position)
                new_content = f.read()

                # Append new content to log viewer if any
                if new_content:
                    log_viewer.value += new_content
                    log_last_position = f.tell()
        except Exception as e:
            logger.warning("Log update error: %s", e)


def read_czi_metadata(filepath: Path) -> tuple[Optional[CziMetadata], int]:
    """Read metadata from a CZI file and determine the number of scenes.

    Args:
        filepath: Path to the CZI file to read

    Returns:
        tuple[Optional[CziMetadata], int]: A tuple containing:
            - CziMetadata object if successful, None if reading fails
            - Maximum number of scenes (defaults to 1 if not specified or on error)

    Note:
        Returns (None, 1) if metadata reading fails. The function prints progress
        messages to console for user feedback.
    """
    try:
        # Read CZI metadata using czitools
        mdata = CziMetadata(filepath)

        # Determine number of scenes
        image = mdata.image
        num_scenes = image.SizeS if (image is not None and hasattr(image, "SizeS")) else None

        # Calculate max_scenes: if None or 0, default to 1
        max_scenes = num_scenes if num_scenes and num_scenes > 0 else 1

        # Build a dimension summary from available CziDimensions attributes
        _dim_keys = ("SizeS", "SizeT", "SizeC", "SizeZ", "SizeY", "SizeX")
        _dims = {k: getattr(image, k, None) for k in _dim_keys if getattr(image, k, None) is not None}
        _dims_str = ", ".join(f"{k}={v}" for k, v in _dims.items())

        logger.info("Metadata loaded successfully")
        logger.info("  - File: %s", filepath.name)
        logger.info("  - Dimensions: %s", _dims_str)
        logger.info("  - Number of scenes: %d", max_scenes)

        return mdata, max_scenes

    except Exception as e:
        logger.error("Error reading metadata: %s", e)
        return None, 1


def perform_conversion(
    filepath: Path,
    use_ozx_format: bool,
    write_hcs: bool,
    package_choice: omezarr_package,
    scene_id: int,
    write_ozx_directly: bool,
    write_ozx_afterwards: bool,
    zarr_format: int = 3,
    use_tensorstore: bool = True,
) -> Optional[str]:
    """
    Perform the CZI to OME-ZARR conversion with specified parameters.

    Args:
        filepath: Path to input CZI file
        use_ozx_format: Enable single-file OME-ZARR format (.ozx)
        write_hcs: Enable HCS (multi-well plate) layout
        write_ozx_directly: Create OZX archive during writing (NGFF-ZARR only)
        write_ozx_afterwards: Convert to OZX after writing (NGFF-ZARR only)
        package_choice: Backend package (OME_ZARR or NGFF_ZARR)
        scene_id: Scene index to convert (for non-HCS mode with multiple scenes)
        zarr_format: Zarr storage format for the ome-zarr-py backend (2 = OME-NGFF
            v0.4 / zarr v2 for legacy viewers, 3 = zarr v3, default). Ignored by the
            ngff-zarr backend, which always writes OME-NGFF v0.5 / zarr v3.
        use_tensorstore: Use the tensorstore backend for parallel chunk I/O in the
            ngff-zarr single-image path. Ignored by the ome-zarr-py backend and by
            the HCS ngff path.

    Returns:
        str: Path to output OME-ZARR file, or None if conversion failed
    """
    try:
        # Setup logging
        log_file_path = filepath.parent / f"{filepath.stem}_conversion.log"
        setup_logging(str(log_file_path), force_reconfigure=True)

        logger.info("=" * 80)
        logger.info("CZI to OME-ZARR Conversion Started")
        logger.info("=" * 80)
        logger.info(f"Input file: {filepath}")
        logger.info(f"Package: {package_choice.name}")
        logger.info(f"HCS mode: {write_hcs}")
        logger.info(f"Single-file (.ozx): {use_ozx_format}")
        logger.info(f"Scene ID: {scene_id}")

        output_path = None

        # ========== HCS Format Conversion ==========
        if write_hcs:
            logger.info("Converting to HCS-ZARR format using %s...", package_choice.name)

            if package_choice == omezarr_package.OME_ZARR:
                output_path = convert_czi2hcs_omezarr(
                    czi_filepath=str(filepath),
                    overwrite=True,
                    log_file_path=str(log_file_path),
                    zarr_format=zarr_format,
                )
            elif package_choice == omezarr_package.NGFF_ZARR:
                if use_ozx_format and write_ozx_afterwards and not write_ozx_directly:
                    # OZX-after mode: write the intermediate .ome.zarr to a temp directory
                    # so it never collides with (or deletes) an existing _ngff_plate.ome.zarr
                    # produced by a previous conversion in the same session.
                    tmp_dir = tempfile.mkdtemp(prefix="omezarr_hcs_tmp_")
                    try:
                        tmp_zarr = convert_czi2hcs_ngff(
                            czi_filepath=str(filepath),
                            overwrite=True,
                            write_ozx_directly=False,
                            log_file_path=str(log_file_path),
                            output_dir=tmp_dir,
                        )
                        # Zip the temp zarr to an ozx also inside tmp_dir
                        tmp_ozx = convert_hcs_omezarr2ozx(tmp_zarr, remove_omezarr=True)
                        if tmp_ozx is not None:
                            # Move the finished ozx to the proper output directory
                            final_ozx = filepath.parent / tmp_ozx.name
                            shutil.move(str(tmp_ozx), str(final_ozx))
                            output_path = str(final_ozx)
                    finally:
                        # Clean up temp directory (should be empty after move)
                        shutil.rmtree(tmp_dir, ignore_errors=True)
                else:
                    output_path = convert_czi2hcs_ngff(
                        czi_filepath=str(filepath),
                        overwrite=True,
                        write_ozx_directly=write_ozx_directly,
                        log_file_path=str(log_file_path),
                    )

            logger.info("HCS-ZARR created: %s", output_path)

        # ========== Standard OME-ZARR Conversion ==========
        else:
            logger.info("Converting scene %d to OME-ZARR format using %s...", scene_id, package_choice.name)

            # Read the CZI file as a 6D array
            array, mdata = read_tools.read_6darray(str(filepath), planes={"S": (scene_id, scene_id)}, use_xarray=True)

            # Extract the specified scene (remove Scene dimension to get 5D array)
            assert isinstance(array, xr.DataArray), "Expected xarray DataArray from read_6darray with use_xarray=True"
            array = array.squeeze("S")
            logger.info("Array shape: %s, dtype: %s", array.shape, array.dtype)

            if package_choice == omezarr_package.OME_ZARR:
                # Generate output path with zarr format suffix for ome-zarr-py backend
                zarr_suffix = "zarr2" if zarr_format == 2 else "zarr3"
                zarr_output_path = Path(str(filepath)[:-4] + f"_{zarr_suffix}.ome.zarr")

                # Write OME-ZARR using ome-zarr-py backend

                output_path = write_omezarr(
                    array,
                    zarr_path=str(zarr_output_path),
                    metadata=mdata,
                    overwrite=True,
                    zarr_format=zarr_format,
                )

                logger.info("OME-ZARR created: %s", output_path)

            elif package_choice == omezarr_package.NGFF_ZARR:

                if write_ozx_directly:
                    # Generate output path with _ngff.ozx extension
                    zarr_output_path: Path = Path(str(filepath)[:-4] + "_ngff.ozx")
                else:
                    # Generate output path with _ngff_zarr3.ome.zarr extension (ngff-zarr always writes v3)
                    zarr_output_path: Path = Path(str(filepath)[:-4] + "_ngff_zarr3.ome.zarr")

                # Write OME-ZARR using ngff-zarr backend.
                # scale_factors=None -> size-aware, Y/X-only pyramid depth derived
                # from the plane size (see compute_pyramid_scale_factors).
                _ = write_omezarr_ngff(
                    array,
                    zarr_output_path,
                    mdata,
                    scale_factors=None,
                    overwrite=True,
                    use_tensorstore=use_tensorstore,
                )

                output_path = str(zarr_output_path)

                logger.info("OME-ZARR created: %s", output_path)

        # Note: napari viewer will be opened on main thread after conversion completes

        # ========== Validate the generated OME-ZARR ==========
        if output_path is not None:
            logger.info("-" * 80)
            if str(output_path).lower().endswith(".ozx"):
                # .ozx is a zipped single-file archive; the OME-NGFF validator opens
                # directory/zip stores via zarr.open_group and does not support the
                # .ozx layout directly, so validation is skipped for these outputs.
                logger.info("Validation skipped: .ozx archives are not validated (%s)", output_path)
            else:
                logger.info("Validating OME-ZARR output against OME-NGFF v0.5...")
                try:
                    is_valid = validate_ome_zarr(output_path)
                    if is_valid:
                        logger.info("Validation result: VALID ✅")
                    else:
                        logger.warning("Validation result: INVALID ❌ (see messages above)")
                except Exception as ve:
                    logger.error("Validation raised an error: %s", ve, exc_info=True)

        logger.info("=" * 80)
        logger.info("Conversion completed successfully!")
        logger.info(f"Output: {output_path}")
        logger.info("=" * 80)

        return str(output_path) if output_path is not None else None

    except Exception as e:
        logger.error("Conversion failed: %s", e, exc_info=True)
        return None


# ============================================================================
# MagicGUI Widget Definition
# ============================================================================


@magicgui(
    call_button=False,
    layout="vertical",
    czi_file={
        "label": "CZI File",
        "mode": "r",
        "filter": "*.czi",
    },
    package_choice={
        "label": "OME-ZARR Package",
        "choices": [("ome-zarr-py", omezarr_package.OME_ZARR), ("ngff-zarr", omezarr_package.NGFF_ZARR)],
        "tooltip": "Choose the backend library for OME-ZARR writing",
    },
    write_hcs={
        "label": "Write HCS Layout",
        "tooltip": "Enable HCS (High Content Screening) multi-well plate format",
    },
    use_ozx_format={
        "label": "Use Single-File OME-ZARR (.ozx)",
        "tooltip": "Enable OZX format for single-file OME-ZARR storage",
    },
    use_ozx_write_directly={
        "label": "Create OZX archive during writing",
        "tooltip": "Write directly into a single-file OZX archive (NGFF-ZARR only, not available in HCS mode)",
    },
    use_ozx_after_writing={
        "label": "Create OZX archive after writing",
        "tooltip": "Enable OZX format for single-file OME-ZARR storage after writing",
    },
    use_zarr_v2={
        "label": "Write zarr v2 (legacy / OME-NGFF v0.4)",
        "tooltip": (
            "Write an OME-NGFF v0.4 / zarr v2 store instead of zarr v3, for legacy "
            "viewers that do not support zarr v3 (ome-zarr-py backend only)."
        ),
    },
    use_tensorstore={
        "label": "Use tensorstore (parallel I/O)",
        "tooltip": (
            "Use the tensorstore backend for async/parallel chunk writes "
            "(ngff-zarr backend, non-HCS only; requires the tensorstore package)."
        ),
    },
    scene_id={
        "label": "Scene ID",
        "min": 0,
        "max": 0,
        "tooltip": "Select scene to convert (only for non-HCS mode with multiple scenes)",
        "visible": False,
    },
    show_napari={
        "label": "Show in napari After Conversion (Experimental !!!)",
        "tooltip": "Automatically open the result in napari viewer",
    },
)
def czi_to_omezarr_converter(
    czi_file: Path = Path(),
    package_choice: omezarr_package = omezarr_package.OME_ZARR,
    write_hcs: bool = False,
    use_ozx_format: bool = False,
    use_ozx_write_directly: bool = False,
    use_ozx_after_writing: bool = False,
    use_zarr_v2: bool = False,
    use_tensorstore: bool = True,
    scene_id: int = 0,
    show_napari: bool = False,
):
    """
    Main widget for CZI to OME-ZARR conversion configuration.

    This widget holds all the conversion parameters.
    The @magicgui decorator creates the actual widget from the parameter definitions above.
    The function parameters must match the decorator configuration keys.
    """
    pass  # This function doesn't need to do anything - it just holds the widgets


# ============================================================================
# Additional Control Widgets
# ============================================================================

# Create "Read Metadata" button
read_metadata_button = widgets.PushButton(
    text="Read Metadata",
    tooltip="Load CZI file metadata and enable conversion options",
)

# Create info display widget
info_display = widgets.TextEdit(
    value="Select a CZI file and click 'Read Metadata' to begin",
    label="Status",
    enabled=False,
)

# Create "Convert to OME-ZARR" button (separate from the main widget)
convert_button = widgets.PushButton(
    text="Convert to OME-ZARR",
    tooltip="Start the conversion process",
    enabled=False,  # Disabled until metadata is read
)

# Create log viewer widget
log_viewer = widgets.TextEdit(
    value="",
    label="Conversion Log",
    enabled=True,  # Enable to allow scrolling
)
log_viewer.min_height = 200  # Set minimum height for the log viewer
log_viewer.read_only = True  # Make it read-only but scrollable

# Create version info widget
try:
    version_info = f"""NGFF Version: {ome_zarr.format.CurrentFormat().version}
ZARR Package: {zarr.__version__}
NGFF-ZARR Package: {nz.__version__}
OME-ZARR Package: {version('ome-zarr')}"""
except Exception:
    version_info = "Version information unavailable"

version_grid = widgets.TextEdit(
    value=version_info,
    label="Package Versions",
    enabled=False,
)
version_grid.min_height = 60
version_grid.max_height = 80


def on_read_metadata_clicked() -> None:
    """Callback function for the 'Read Metadata' button.

    Reads CZI file metadata and updates GUI state:
    - Validates file existence
    - Loads and parses CZI metadata
    - Updates scene selector visibility and range
    - Displays metadata summary in info widget
    - Enables convert button when successful

    Note:
        Updates global variables: metadata, max_scenes, selected_file
    """
    global metadata, max_scenes, selected_file

    # Get current file path from widget
    filepath = czi_to_omezarr_converter.czi_file.value

    # Validate file existence
    if not filepath.exists():
        info_display.value = "❌ Error: File does not exist"
        return

    # Read metadata from CZI file
    info_display.value = "⏳ Reading metadata..."
    metadata, max_scenes = read_czi_metadata(filepath)
    selected_file = filepath

    # Handle metadata reading failure
    if metadata is None:
        info_display.value = "❌ Error: Failed to read metadata"
        return

    # Bind to a local variable so the type checker can narrow to CziMetadata (not Optional)
    mdata = metadata

    # Determine scene selector visibility
    # Show only if: NOT in HCS mode AND file has multiple scenes
    write_hcs = czi_to_omezarr_converter.write_hcs.value
    scene_selector_visible = (not write_hcs) and (max_scenes > 1)

    # Configure scene_id widget properties
    czi_to_omezarr_converter.scene_id.visible = scene_selector_visible
    if max_scenes > 1:
        czi_to_omezarr_converter.scene_id.max = max_scenes - 1
        czi_to_omezarr_converter.scene_id.value = 0

    # Enable the convert button now that metadata is loaded
    convert_button.enabled = True

    # Bind image info to a local variable so the type checker can narrow away Optional
    image = mdata.image
    size_x = image.SizeX if image is not None else "N/A"
    size_y = image.SizeY if image is not None else "N/A"
    size_c = image.SizeC if image is not None else "N/A"
    size_z = image.SizeZ if image is not None else "N/A"
    size_t = image.SizeT if image is not None else "N/A"

    # Build and display metadata summary
    info_text = f"""✅ Metadata loaded successfully!

📁 File: {filepath.name}
📐 Dimensions: {mdata.pyczi_dims}
🔢 Number of scenes: {max_scenes}
📊 Image size: {size_x} × {size_y}
🎨 Channels: {size_c}
📚 Z-slices: {size_z}
⏱️ Time points: {size_t}

Ready to convert
"""
    info_display.value = info_text


def finish_conversion(output_path: Optional[str], should_open_napari: bool = False) -> None:
    """Finalize conversion process and update UI state.

    This function is called from the main Qt thread after the background conversion
    thread completes. It performs cleanup, final log reading, and optionally opens
    the result in napari viewer.

    Args:
        output_path: Path to the generated OME-ZARR file, or None if conversion failed
        should_open_napari: If True, open the output in napari viewer

    Note:
        Must be called from the main Qt thread to safely update UI widgets.
        Uses global variables log_timer and log_file_path.
    """
    global log_timer, log_file_path

    # Stop the log polling timer
    if log_timer:
        log_timer.stop()
        log_timer = None

    # Perform final complete read of log file to capture all content
    if log_file_path and log_file_path.exists():
        try:
            with open(log_file_path, "r", encoding="utf-8") as f:
                log_viewer.value = f.read()
        except Exception as e:
            log_viewer.value += f"\n⚠️ Could not read log file: {e}"

    # Open napari viewer if requested (on main thread)
    if should_open_napari and output_path:
        import napari
        import json
        from napari.utils.colormaps import Colormap

        logger.info("Opening in napari viewer...")
        try:
            viewer = napari.Viewer()
            viewer.open(output_path, plugin="napari-ome-zarr")

            # napari-ome-zarr does not reliably apply OMERO channel colors from NGFF 0.5.
            # Read colors from zarr.json and apply them manually to each layer.
            zarr_json_path = Path(output_path) / "zarr.json"
            if zarr_json_path.exists():
                try:
                    with open(zarr_json_path, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                    channels = meta.get("attributes", {}).get("ome", {}).get("omero", {}).get("channels", [])
                    for layer, ch in zip(viewer.layers, channels):
                        hex_color = ch.get("color", "FFFFFF")
                        r = int(hex_color[0:2], 16) / 255
                        g = int(hex_color[2:4], 16) / 255
                        b = int(hex_color[4:6], 16) / 255
                        layer.colormap = Colormap(
                            colors=[[0, 0, 0, 1], [r, g, b, 1]],
                            name=f"ch_{hex_color}",
                        )
                except Exception as ce:
                    logger.warning("Could not apply channel colors: %s", ce)

            logger.info("Napari viewer opened successfully")
        except Exception as e:
            logger.warning("Failed to open in napari: %s", e)

    # Update UI
    if output_path:
        info_display.value = f"✅ Conversion successful!\n\nOutput: {output_path}"
    else:
        info_display.value = "❌ Conversion failed. Check console for details."

    # Re-enable convert button
    convert_button.enabled = True


def on_convert_clicked() -> None:
    """Callback function for the 'Convert to OME-ZARR' button.

    This function orchestrates the entire conversion process:
    1. Validates that a file is selected and metadata has been read
    2. Clears previous log content and updates UI state
    3. Starts a background thread for conversion
    4. Sets up a QTimer to poll for conversion completion and update logs

    The conversion runs in a separate thread to prevent UI freezing, while
    a QTimer on the main thread handles UI updates (thread-safe approach).

    Note:
        Uses multiple global variables to coordinate between UI thread and
        conversion thread. Disables the convert button during processing.
    """
    global metadata, selected_file, conversion_running, log_file_path, log_last_position, log_timer

    # Get current values from the widget
    czi_file = czi_to_omezarr_converter.czi_file.value
    use_ozx_format = czi_to_omezarr_converter.use_ozx_format.value
    write_hcs = czi_to_omezarr_converter.write_hcs.value
    show_napari = czi_to_omezarr_converter.show_napari.value
    package_choice = czi_to_omezarr_converter.package_choice.value
    scene_id = czi_to_omezarr_converter.scene_id.value
    use_zarr_v2 = czi_to_omezarr_converter.use_zarr_v2.value
    use_tensorstore = czi_to_omezarr_converter.use_tensorstore.value

    # Validate that file exists
    if not czi_file.exists():
        info_display.value = "❌ Error: Selected file does not exist"
        return

    # Validate that metadata has been read
    if metadata is None or selected_file != czi_file:
        info_display.value = "⚠️ Please click 'Read Metadata' first"
        return

    # Validate OZX sub-option selection
    if use_ozx_format:
        _write_directly = czi_to_omezarr_converter.use_ozx_write_directly.value
        _write_afterwards = czi_to_omezarr_converter.use_ozx_after_writing.value
        if not _write_directly and not _write_afterwards:
            info_display.value = (
                "⚠️ 'Use Single-File OME-ZARR (.ozx)' is enabled — " "select at least one OZX write option."
            )
            return

    # Clear log viewer and update status
    log_viewer.value = "Starting conversion...\n"
    info_display.value = "⏳ Converting... Please wait."
    log_last_position = 0

    # Disable convert button during conversion
    convert_button.enabled = False

    # Setup log file path
    log_file_path = czi_file.parent / f"{czi_file.stem}_conversion.log"
    conversion_running = True

    # Store conversion result
    conversion_result = {"output_path": None, "completed": False, "show_napari": show_napari}

    # Start timer to update log display every 500ms
    log_timer = QTimer()

    def check_conversion_status() -> None:
        """Check if conversion is complete and update UI accordingly.

        This function is called periodically by QTimer. It updates the log display
        and checks if the background conversion thread has completed. When complete,
        it triggers UI finalization on the main thread.
        """
        update_log_display()  # Update log with new content

        # Check if conversion is complete
        if conversion_result["completed"]:
            finish_conversion(conversion_result["output_path"], conversion_result["show_napari"])

    log_timer.timeout.connect(check_conversion_status)
    log_timer.start(500)  # Poll every 500ms

    def run_conversion() -> None:
        """Run conversion in background thread.

        This function executes in a separate daemon thread to prevent blocking
        the Qt main thread. It performs the actual conversion and stores the
        result in conversion_result dict for the main thread to process.
        """
        global conversion_running

        # Perform the conversion operation
        output_path = perform_conversion(
            filepath=czi_file,
            use_ozx_format=use_ozx_format,
            write_ozx_afterwards=czi_to_omezarr_converter.use_ozx_after_writing.value,
            write_ozx_directly=czi_to_omezarr_converter.use_ozx_write_directly.value,
            write_hcs=write_hcs,
            package_choice=package_choice,
            scene_id=scene_id,
            zarr_format=(2 if use_zarr_v2 else 3),
            use_tensorstore=use_tensorstore,
        )

        # Store result and mark as complete
        conversion_result["output_path"] = output_path
        conversion_result["completed"] = True
        conversion_running = False

    # Start conversion in a separate thread
    conversion_thread = threading.Thread(target=run_conversion, daemon=True)
    conversion_thread.start()


def update_ozx_child_states() -> None:
    """Synchronize dependent OZX options with the master toggle.

    A re-entrancy guard (_ozx_state_updating) prevents the signal cascade that
    would otherwise occur when the auto-select logic sets a child value and that
    change immediately fires another callback that calls this function again.
    Without the guard, checking 'Create OZX archive during writing' would be
    immediately undone by the cascade: auto-select sets after=True →
    on_use_ozx_after_writing_changed(True) clears directly=False.
    """
    global _ozx_state_updating
    if _ozx_state_updating:
        return
    _ozx_state_updating = True
    try:
        master_active = bool(czi_to_omezarr_converter.use_ozx_format.value)
        hcs_enabled = bool(czi_to_omezarr_converter.write_hcs.value)

        allow_direct = master_active and not hcs_enabled
        allow_after = master_active

        czi_to_omezarr_converter.use_ozx_write_directly.enabled = allow_direct
        czi_to_omezarr_converter.use_ozx_after_writing.enabled = allow_after

        if not allow_direct and czi_to_omezarr_converter.use_ozx_write_directly.value:
            czi_to_omezarr_converter.use_ozx_write_directly.value = False

        if not allow_after and czi_to_omezarr_converter.use_ozx_after_writing.value:
            czi_to_omezarr_converter.use_ozx_after_writing.value = False

        if master_active:
            if (
                czi_to_omezarr_converter.use_ozx_write_directly.value
                and czi_to_omezarr_converter.use_ozx_after_writing.value
            ):
                # Mutual exclusion: prefer 'after' in HCS mode, 'directly' otherwise
                if hcs_enabled:
                    czi_to_omezarr_converter.use_ozx_write_directly.value = False
                else:
                    czi_to_omezarr_converter.use_ozx_after_writing.value = False
            elif (
                not czi_to_omezarr_converter.use_ozx_write_directly.value
                and not czi_to_omezarr_converter.use_ozx_after_writing.value
            ):
                # At least one sub-option must be active — default to 'after writing'
                czi_to_omezarr_converter.use_ozx_after_writing.value = True
        else:
            czi_to_omezarr_converter.use_ozx_write_directly.value = False
            czi_to_omezarr_converter.use_ozx_after_writing.value = False
    finally:
        _ozx_state_updating = False

    update_show_napari_enabled_state()


def update_show_napari_enabled_state() -> None:
    """Enable or disable 'Show in napari' based on whether the output will be an .ozx archive.

    napari (via napari-ome-zarr) can only open directory-based OME-ZARR stores, not
    zip-based .ozx archives. The checkbox is therefore disabled and unchecked whenever
    the conversion is configured to produce an .ozx file.
    """
    will_produce_ozx = czi_to_omezarr_converter.use_ozx_format.value and (
        czi_to_omezarr_converter.use_ozx_write_directly.value or czi_to_omezarr_converter.use_ozx_after_writing.value
    )

    czi_to_omezarr_converter.show_napari.enabled = not will_produce_ozx

    if will_produce_ozx and czi_to_omezarr_converter.show_napari.value:
        czi_to_omezarr_converter.show_napari.value = False


def update_use_ozx_format_enabled_state() -> None:
    """Enable or disable OZX controls based on backend capabilities."""

    package_choice = czi_to_omezarr_converter.package_choice.value

    can_use_ozx = package_choice != omezarr_package.OME_ZARR
    czi_to_omezarr_converter.use_ozx_format.enabled = can_use_ozx

    if not can_use_ozx and czi_to_omezarr_converter.use_ozx_format.value:
        czi_to_omezarr_converter.use_ozx_format.value = False

    update_ozx_child_states()
    update_zarr_v2_enabled_state()


def update_zarr_v2_enabled_state() -> None:
    """Enable the zarr v2 option only for the ome-zarr-py backend.

    The ngff-zarr backend always writes OME-NGFF v0.5 / zarr v3, so the zarr v2
    legacy option is disabled (and unchecked) whenever it is selected.
    """
    is_ome_zarr = czi_to_omezarr_converter.package_choice.value == omezarr_package.OME_ZARR
    czi_to_omezarr_converter.use_zarr_v2.enabled = is_ome_zarr

    if not is_ome_zarr and czi_to_omezarr_converter.use_zarr_v2.value:
        czi_to_omezarr_converter.use_zarr_v2.value = False

    # tensorstore parallel I/O only applies to the ngff-zarr backend.
    is_ngff = czi_to_omezarr_converter.package_choice.value == omezarr_package.NGFF_ZARR
    czi_to_omezarr_converter.use_tensorstore.enabled = is_ngff


def on_use_ozx_format_changed(_: bool) -> None:
    """React to master OZX toggle changes."""

    update_ozx_child_states()
    update_show_napari_enabled_state()


def on_use_ozx_write_directly_changed(value: bool) -> None:
    """Ensure mutually exclusive OZX modes when direct write is toggled."""
    # Only clear the other side if it is currently checked; avoids emitting
    # spurious signals that feed back into update_ozx_child_states.
    if value and czi_to_omezarr_converter.use_ozx_after_writing.value:
        czi_to_omezarr_converter.use_ozx_after_writing.value = False

    update_ozx_child_states()


def on_use_ozx_after_writing_changed(value: bool) -> None:
    """Ensure mutually exclusive OZX modes when post-write archive is toggled."""
    # Only clear the other side if it is currently checked.
    if value and czi_to_omezarr_converter.use_ozx_write_directly.value:
        czi_to_omezarr_converter.use_ozx_write_directly.value = False

    update_ozx_child_states()


def on_write_hcs_changed(value: bool) -> None:
    """Callback for write_hcs checkbox changes.

    Controls UI state based on HCS mode selection:
    - Hides scene selector in HCS mode (HCS processes all scenes automatically)
    - Limits single-file (.ozx) mode to post-write archiving when HCS is enabled

    Args:
        value: True if HCS mode is enabled, False otherwise

    Note:
        Uses global max_scenes variable to determine scene selector visibility.
    """
    global max_scenes

    # Show scene selector only if NOT in HCS mode AND multiple scenes exist
    scene_selector_visible = (not value) and (max_scenes > 1)
    czi_to_omezarr_converter.scene_id.visible = scene_selector_visible

    update_use_ozx_format_enabled_state()


def on_package_choice_changed(value: omezarr_package) -> None:
    """Callback for package_choice changes.

    Manages single-file (.ozx) option availability based on selected backend:
    - ome-zarr-py: Does not support .ozx format, so option is disabled
    - ngff-zarr: Supports .ozx format, so option is enabled (unless HCS mode)

    Args:
        value: The selected OME-ZARR backend package

    Note:
        The single-file option may remain disabled if HCS mode is active.
    """
    update_use_ozx_format_enabled_state()


def on_file_changed(value: Path) -> None:
    """Callback for file selector changes.

    This function handles UI updates when a new CZI file is selected:
    1. Dynamically adjusts file selector width based on path length (600-1200px)
    2. Resets application state (clears metadata, logs, and UI displays)
    3. Disables convert button until metadata is read for the new file

    Args:
        value: Path to the newly selected CZI file

    Note:
        Uses global variables metadata and max_scenes to reset application state.
        This ensures a clean state when switching between files.
    """
    global metadata, max_scenes

    if value and value.exists():
        # Calculate width based on file path length
        # Approximate: 7 pixels per character, with min 600 and max 1200
        path_length = len(str(value))
        new_width = min(max(600, path_length * 7), 1200)
        czi_to_omezarr_converter.czi_file.min_width = new_width

        # Clear previous metadata and logs
        metadata = None
        max_scenes = 1
        info_display.value = "Select a CZI file and click 'Read Metadata' to begin."
        log_viewer.value = ""

        # Reset convert button state
        convert_button.enabled = False

        update_use_ozx_format_enabled_state()


# ============================================================================
# Widget Configuration and Callback Connections
# ============================================================================

# Set initial minimum width for file selector widget
# The @magicgui decorator creates widget attributes from the function parameters
try:
    czi_to_omezarr_converter.czi_file.min_width = 600
except AttributeError as e:
    logger.warning("Could not set file selector width: %s", e)

update_use_ozx_format_enabled_state()

# Connect callback functions to widget signals
# These callbacks handle user interactions and maintain UI state consistency
read_metadata_button.clicked.connect(on_read_metadata_clicked)
convert_button.clicked.connect(on_convert_clicked)
czi_to_omezarr_converter.write_hcs.changed.connect(on_write_hcs_changed)
czi_to_omezarr_converter.package_choice.changed.connect(on_package_choice_changed)
czi_to_omezarr_converter.use_ozx_format.changed.connect(on_use_ozx_format_changed)
czi_to_omezarr_converter.use_ozx_write_directly.changed.connect(on_use_ozx_write_directly_changed)
czi_to_omezarr_converter.use_ozx_after_writing.changed.connect(on_use_ozx_after_writing_changed)
czi_to_omezarr_converter.czi_file.changed.connect(on_file_changed)


# ============================================================================
# Main Application Container
# ============================================================================


def create_gui() -> widgets.Container:
    """Create and return the complete GUI application container.

    Assembles all widgets into a single vertical container in the following order:
    1. Version information display (top)
    2. Main conversion configuration widget (file selector, options)
    3. Read Metadata button
    4. Status/metadata information display
    5. Convert button
    6. Conversion log viewer (bottom)

    Returns:
        widgets.Container: The main application widget container with all components
    """
    # Create container with all widgets
    container = widgets.Container(
        widgets=[
            version_grid,
            czi_to_omezarr_converter,
            read_metadata_button,
            info_display,
            convert_button,
            log_viewer,
        ],
        labels=False,
    )

    return container


# ============================================================================
# Standalone Execution
# ============================================================================

# ============================================================================
# Standalone Execution
# ============================================================================


def run_gui() -> None:
    """Create and show the CZI -> OME-Zarr converter as a standalone Qt window.

    Blocks until the window is closed. This is the entry point used by the
    ``czitools-omezarr-gui`` console script and the demo launcher.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

    gui = create_gui()

    logger.info("=" * 60)
    logger.info("CZI to OME-ZARR Converter")
    logger.info("=" * 60)
    logger.info("Application started. Close the window to exit.")

    gui.native.setWindowTitle("CZI --> OME-ZARR Converter (czitools)")
    gui.show(run=True)


if __name__ == "__main__":
    run_gui()
