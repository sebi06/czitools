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

import logging
import os
import threading
from importlib.metadata import version
from pathlib import Path

# QtPy and MagicGUI select their Qt binding when they are first imported.
os.environ["QT_API"] = "pyqt6"

import ngff_zarr as nz
import ome_zarr.format
import xarray as xr
import zarr
from magicgui import magicgui, widgets
from qtpy.QtCore import QTimer
from qtpy.QtGui import QFontDatabase, QTextCursor
from qtpy.QtWidgets import QTextEdit

from czitools.metadata_tools.czi_metadata import CziMetadata
from czitools.read_tools import read_tools

# Internal imports from sibling modules (avoid lazy-loading parent package)
from ._logging import compression_type, omezarr_package, setup_logging
from .conversion import (
    convert_czi2hcs_ngff,
    convert_czi2hcs_omezarr,
    write_omezarr,
    write_omezarr_ngff,
)
from .validation import validate_ome_zarr

logger = logging.getLogger(__name__)


# ============================================================================
# Module-level Global Variables
# ============================================================================
# These variables maintain application state across callbacks and threads

# Metadata from the currently loaded CZI file
metadata: CziMetadata | None = None

# Maximum number of scenes in the current CZI file
max_scenes: int = 1

# Path to the currently selected CZI file
selected_file: Path | None = None

# Flag indicating whether a conversion is currently in progress
conversion_running: bool = False

# Path to the current conversion log file
log_file_path: Path | None = None

# Current read position in the log file for incremental updates
log_last_position: int = 0

# QTimer instance for periodic log file polling
log_timer: QTimer | None = None

# Path to napari viewer output (unused - kept for compatibility)
napari_viewer_path: str | None = None

# Default parent directory for file browser
try:
    parent_dir: Path | None = Path(__file__).resolve().parents[4] / "data"
    if not parent_dir.exists():
        parent_dir = None
except (ValueError, IndexError):
    parent_dir = None


def _scroll_log_to_end() -> None:
    """Move the log viewer cursor to the end and keep it visible."""
    cursor = log_viewer.native.textCursor()
    cursor.movePosition(QTextCursor.MoveOperation.End)
    log_viewer.native.setTextCursor(cursor)
    log_viewer.native.ensureCursorVisible()


def _append_log_content(content: str) -> None:
    """Append log content without replacing the complete text document."""
    cursor = log_viewer.native.textCursor()
    cursor.movePosition(QTextCursor.MoveOperation.End)
    cursor.insertText(content)
    log_viewer.native.setTextCursor(cursor)
    log_viewer.native.ensureCursorVisible()


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
            # Open log file and seek to last read position.
            # errors='replace' substitutes any non-UTF-8 bytes (e.g. from
            # external libraries that write cp1252 to shared log handlers)
            # with the Unicode replacement character instead of raising.
            with open(log_file_path, "r", encoding="utf-8", errors="replace") as f:
                f.seek(log_last_position)
                new_content = f.read()

                # Append new content to log viewer if any
                if new_content:
                    _append_log_content(new_content)
                    log_last_position = f.tell()
        except Exception as e:
            logger.warning("Log update error: %s", e)


def read_czi_metadata(filepath: Path) -> tuple[CziMetadata | None, int]:
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
    use_tensorstore: bool = True,
    compression_choice: compression_type | None = compression_type.BLOSC,
) -> str | None:
    """
    Perform the CZI to OME-ZARR conversion with specified parameters.

    Args:
        filepath: Path to input CZI file
        use_ozx_format: Enable single-file OME-ZARR format (.ozx)
        write_hcs: Enable HCS (multi-well plate) layout
        package_choice: Backend package (OME_ZARR or NGFF_ZARR)
        scene_id: Scene index to convert (for non-HCS mode with multiple scenes)
        use_tensorstore: Use the tensorstore backend for parallel chunk I/O in the
            ngff-zarr single-image path. Ignored by the ome-zarr-py backend and by
            the HCS ngff path.
        compression_choice: Compression type for OME-ZARR output (default: Blosc).
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
                    compression=compression_choice,
                )
            elif package_choice == omezarr_package.NGFF_ZARR:
                output_path = convert_czi2hcs_ngff(
                    czi_filepath=str(filepath),
                    overwrite=True,
                    write_ozx_directly=use_ozx_format,
                    log_file_path=str(log_file_path),
                    compression=compression_choice,
                )

            logger.info("HCS-ZARR created: %s", output_path)

        # ========== Standard OME-ZARR Conversion ==========
        else:
            logger.info(
                "Converting scene %d to OME-ZARR format using %s...",
                scene_id,
                package_choice.name,
            )

            # Read the CZI file as a 6D array
            array, mdata = read_tools.read_6darray(str(filepath), planes={"S": (scene_id, scene_id)}, use_xarray=True)

            # Extract the specified scene (remove Scene dimension to get 5D array)
            assert isinstance(array, xr.DataArray), "Expected xarray DataArray from read_6darray with use_xarray=True"
            array = array.squeeze("S")
            logger.info("Array shape: %s, dtype: %s", array.shape, array.dtype)

            if package_choice == omezarr_package.OME_ZARR:
                zarr_output_path = Path(str(filepath)[:-4] + "_zarr3.ome.zarr")

                # Write OME-ZARR using ome-zarr-py backend

                output_path = write_omezarr(
                    array,
                    zarr_path=str(zarr_output_path),
                    metadata=mdata,
                    overwrite=True,
                    log_file_path=str(log_file_path),
                )

                logger.info("OME-ZARR created: %s", output_path)

            elif package_choice == omezarr_package.NGFF_ZARR:

                if use_ozx_format:
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
                    log_file_path=str(log_file_path),
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
                logger.info(
                    "Validation skipped: .ozx archives are not validated (%s)",
                    output_path,
                )
            else:
                logger.info("Validating OME-ZARR output against OME-NGFF v0.5...")
                try:
                    is_valid = validate_ome_zarr(output_path)
                    if is_valid:
                        logger.info("Validation result: VALID [OK]")
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
        "choices": [
            ("ngff-zarr", omezarr_package.NGFF_ZARR),
            ("ome-zarr-py", omezarr_package.OME_ZARR),
        ],
        "tooltip": "Choose the backend library for OME-ZARR writing",
    },
    write_hcs={
        "label": "Write HCS Layout",
        "tooltip": "Enable HCS (High Content Screening) multi-well plate format",
    },
    use_ozx_format={
        "label": "Create Single-File OME-ZARR (.ozx)",
        "tooltip": "Create an RFC-9 single-file OZX archive (ngff-zarr only)",
    },
    compression_choice={
        "label": "Compression",
        "choices": [
            ("None", compression_type.NONE),
            ("Blosc", compression_type.BLOSC),
            ("Zstd", compression_type.ZSTD),
        ],
        "tooltip": "Choose the compression method for OME-ZARR output (default: Blosc)",
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
    package_choice: omezarr_package = omezarr_package.NGFF_ZARR,
    write_hcs: bool = False,
    use_ozx_format: bool = False,
    compression_choice: compression_type | None = compression_type.BLOSC,
    use_tensorstore: bool = False,
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

# Create info display widget
info_display = widgets.TextEdit(
    value="Select a CZI file to load its metadata.",
    label="CZI Metadata",
    enabled=True,
)
info_display.min_height = 300
info_display.read_only = True
info_display.native.setFont(QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont))
info_display.native.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)

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
    enabled=True,
)
version_grid.min_height = 80
version_grid.max_height = 120
version_grid.read_only = True


def _conversion_controls() -> tuple[widgets.Widget, ...]:
    """Return controls that require successfully loaded CZI metadata."""
    return (
        czi_to_omezarr_converter.package_choice,
        czi_to_omezarr_converter.write_hcs,
        czi_to_omezarr_converter.use_ozx_format,
        czi_to_omezarr_converter.compression_choice,
        czi_to_omezarr_converter.use_tensorstore,
        czi_to_omezarr_converter.scene_id,
        czi_to_omezarr_converter.show_napari,
        convert_button,
    )


def _set_conversion_controls_enabled(enabled: bool) -> None:
    """Set the baseline enabled state for metadata-dependent controls."""
    for control in _conversion_controls():
        control.enabled = enabled


def _format_hcs_details(mdata: CziMetadata) -> str:
    """Format concise HCS metadata as plain text for the GUI."""
    plate = mdata.hcs
    if plate is None:
        return "No HCS plate layout detected."

    lines = [
        "HCS PLATE INFORMATION",
        "",
        f"Detected: {mdata.hcs_status.detected}",
        f"Reason: {mdata.hcs_status.reason}",
        f"Plate ID: {plate.id}",
        f"Plate name: {plate.name}",
        f"Schema version: {plate.schema_version}",
        f"Declared layout: {plate.declared_rows} rows x {plate.declared_columns} columns",
        f"Observed row indices: {', '.join(map(str, plate.observed_row_indices))}",
        f"Observed column indices: {', '.join(map(str, plate.observed_column_indices))}",
        f"Total wells: {len(plate.wells)}",
        f"Total fields: {sum(len(well.fields) for well in plate.wells)}",
    ]

    sample = mdata.sample
    if sample is not None:
        lines.extend(
            [
                "",
                "SAMPLE METADATA",
                "",
                f"Scene count: {sample.scene_count}",
                f"Unique wells: {sample.well_unique_number}",
                f"Multiple positions per well: {sample.multipos_per_well}",
            ]
        )

    if plate.wells:
        well = plate.wells[0]
        lines.extend(["", f"FIELDS IN FIRST WELL ({well.canonical_name})", ""])
        for field in well.fields:
            lines.append(
                f"Field {field.field_index}: scene {field.scene_index}, "
                f"center ({field.scene_center_x:.2f}, {field.scene_center_y:.2f}) "
                f"{field.position_unit}"
            )

    return "\n".join(lines)


def load_selected_file_metadata() -> None:
    """Read metadata for the selected CZI and update the GUI.

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

    # Validate file selection
    if not filepath.is_file() or filepath.suffix.lower() != ".czi":
        metadata = None
        selected_file = None
        _set_conversion_controls_enabled(False)
        info_display.value = "Select a valid CZI file to load its metadata."
        return

    # Read metadata from CZI file
    info_display.value = "⏳ Reading metadata..."
    metadata, max_scenes = read_czi_metadata(filepath)
    selected_file = filepath if metadata is not None else None

    # Handle metadata reading failure
    if metadata is None:
        _set_conversion_controls_enabled(False)
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

    # Enable controls now that metadata is loaded, then apply capability rules.
    _set_conversion_controls_enabled(True)
    hcs_detected = mdata.hcs is not None
    czi_to_omezarr_converter.write_hcs.enabled = hcs_detected
    if not hcs_detected:
        czi_to_omezarr_converter.write_hcs.value = False
    update_use_ozx_format_enabled_state()

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
    if hcs_detected:
        info_text += f"\nHCS layout detected\n\n{_format_hcs_details(mdata)}\n"
    else:
        info_text += "\nNo HCS plate layout detected.\n"
    info_display.value = info_text


def finish_conversion(output_path: str | None, should_open_napari: bool = False) -> None:
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
            _scroll_log_to_end()
            # Re-apply after Qt has recalculated the document layout and scrollbar.
            QTimer.singleShot(0, _scroll_log_to_end)
        except Exception as e:
            _append_log_content(f"\n⚠️ Could not read log file: {e}")

    # Open napari viewer if requested (on main thread)
    if should_open_napari and output_path:
        import json

        import napari
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
    compression = czi_to_omezarr_converter.compression_choice.value
    use_tensorstore = czi_to_omezarr_converter.use_tensorstore.value

    # Validate that file exists
    if not czi_file.exists():
        info_display.value = "❌ Error: Selected file does not exist"
        return

    # Validate that metadata has been read
    if metadata is None or selected_file != czi_file:
        info_display.value = "⚠️ Select a valid CZI file and wait for its metadata to load."
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
    conversion_result = {
        "output_path": None,
        "completed": False,
        "show_napari": show_napari,
    }

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
            write_hcs=write_hcs,
            package_choice=package_choice,
            scene_id=scene_id,
            use_tensorstore=use_tensorstore,
            compression_choice=compression,
        )

        # Store result and mark as complete
        conversion_result["output_path"] = output_path
        conversion_result["completed"] = True
        conversion_running = False

    # Start conversion in a separate thread
    conversion_thread = threading.Thread(target=run_conversion, daemon=True)
    conversion_thread.start()


def update_show_napari_enabled_state() -> None:
    """Enable or disable 'Show in napari' based on whether the output will be an .ozx archive.

    napari (via napari-ome-zarr) can only open directory-based OME-ZARR stores, not
    zip-based .ozx archives. The checkbox is therefore disabled and unchecked whenever
    the conversion is configured to produce an .ozx file.
    """
    will_produce_ozx = czi_to_omezarr_converter.use_ozx_format.value

    metadata_ready = metadata is not None and selected_file is not None
    czi_to_omezarr_converter.show_napari.enabled = metadata_ready and not will_produce_ozx

    if will_produce_ozx and czi_to_omezarr_converter.show_napari.value:
        czi_to_omezarr_converter.show_napari.value = False


def update_use_ozx_format_enabled_state() -> None:
    """Enable or disable OZX controls based on backend capabilities."""

    package_choice = czi_to_omezarr_converter.package_choice.value
    metadata_ready = metadata is not None and selected_file is not None

    can_use_ozx = metadata_ready and package_choice != omezarr_package.OME_ZARR
    czi_to_omezarr_converter.use_ozx_format.enabled = can_use_ozx

    if not can_use_ozx and czi_to_omezarr_converter.use_ozx_format.value:
        czi_to_omezarr_converter.use_ozx_format.value = False

    update_show_napari_enabled_state()
    # tensorstore parallel I/O only applies to the ngff-zarr backend.
    is_ngff = czi_to_omezarr_converter.package_choice.value == omezarr_package.NGFF_ZARR
    czi_to_omezarr_converter.use_tensorstore.enabled = metadata_ready and is_ngff


def on_use_ozx_format_changed(_: bool) -> None:
    """React to OZX output changes."""
    update_show_napari_enabled_state()


def on_write_hcs_changed(value: bool) -> None:
    """Callback for write_hcs checkbox changes.

    Controls UI state based on HCS mode selection:
    - Hides scene selector in HCS mode (HCS processes all scenes automatically)
    - Keeps single-file (.ozx) output available for ngff-zarr

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
    - ngff-zarr: Supports .ozx format, so option is enabled

    Args:
        value: The selected OME-ZARR backend package

    """
    update_use_ozx_format_enabled_state()


def on_file_changed(value: Path) -> None:
    """Callback for file selector changes.

    This function handles UI updates when a new CZI file is selected:
    1. Dynamically adjusts file selector width based on path length (600-1200px)
    2. Resets application state (clears metadata, logs, and UI displays)
    3. Reads metadata immediately and enables conversion controls on success

    Args:
        value: Path to the newly selected CZI file

    Note:
        Uses global variables metadata and max_scenes to reset application state.
        This ensures a clean state when switching between files.
    """
    global metadata, max_scenes

    if value and value.is_file() and value.suffix.lower() == ".czi":
        # Calculate width based on file path length
        # Approximate: 7 pixels per character, with min 600 and max 1200
        path_length = len(str(value))
        new_width = min(max(600, path_length * 7), 1200)
        czi_to_omezarr_converter.czi_file.min_width = new_width

        # Clear previous metadata and logs
        metadata = None
        max_scenes = 1
        info_display.value = "⏳ Reading metadata..."
        log_viewer.value = ""

        _set_conversion_controls_enabled(False)
        load_selected_file_metadata()
    else:
        metadata = None
        max_scenes = 1
        info_display.value = "Select a valid CZI file to load its metadata."
        log_viewer.value = ""
        _set_conversion_controls_enabled(False)

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

_set_conversion_controls_enabled(False)

# Connect callback functions to widget signals
# These callbacks handle user interactions and maintain UI state consistency
convert_button.clicked.connect(on_convert_clicked)
czi_to_omezarr_converter.write_hcs.changed.connect(on_write_hcs_changed)
czi_to_omezarr_converter.package_choice.changed.connect(on_package_choice_changed)
czi_to_omezarr_converter.use_ozx_format.changed.connect(on_use_ozx_format_changed)
czi_to_omezarr_converter.czi_file.changed.connect(on_file_changed)


# ============================================================================
# Main Application Container
# ============================================================================


def create_gui() -> widgets.Container:
    """Create and return the complete GUI application container.

    Assembles all widgets into a single vertical container. The file and backend
    selectors appear above the scrollable metadata display. Conversion options,
    including all checkboxes and compression, appear below the metadata.

    Returns:
        widgets.Container: The main application widget container with all components
    """
    source_controls = widgets.Container(
        widgets=[
            czi_to_omezarr_converter.czi_file,
            czi_to_omezarr_converter.package_choice,
        ],
        labels=True,
    )
    conversion_options = widgets.Container(
        widgets=[
            czi_to_omezarr_converter.write_hcs,
            czi_to_omezarr_converter.scene_id,
            czi_to_omezarr_converter.use_ozx_format,
            czi_to_omezarr_converter.compression_choice,
            czi_to_omezarr_converter.use_tensorstore,
            czi_to_omezarr_converter.show_napari,
        ],
        labels=True,
    )

    container = widgets.Container(
        widgets=[
            version_grid,
            source_controls,
            info_display,
            conversion_options,
            convert_button,
            log_viewer,
        ],
        labels=False,
        scrollable=True,
    )

    return container


def _resize_gui_to_content(gui: widgets.Container) -> None:
    """Size the window to show all content, constrained to the current screen."""
    content_widget = gui.native
    window = gui.root_native_widget

    layout = content_widget.layout()
    if layout is not None:
        layout.activate()

    content_size = content_widget.sizeHint()
    frame_width = window.frameWidth() if hasattr(window, "frameWidth") else 0
    target_width = max(content_size.width() + 2 * frame_width, window.minimumWidth())
    target_height = max(content_size.height() + 2 * frame_width, window.minimumHeight())

    screen = window.screen()
    if screen is not None:
        available = screen.availableGeometry()
        desktop_margin = 40
        target_width = min(target_width, max(1, available.width() - desktop_margin))
        target_height = min(target_height, max(1, available.height() - desktop_margin))

    window.resize(target_width, target_height)


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
    setup_logging(force_reconfigure=True)

    gui = create_gui()

    logger.info("=" * 60)
    logger.info("CZI to OME-ZARR Converter")
    logger.info("=" * 60)
    logger.info("Application started. Close the window to exit.")

    gui.root_native_widget.setWindowTitle("CZI --> OME-ZARR Converter (experimental)")
    _resize_gui_to_content(gui)
    gui.show(run=True)


if __name__ == "__main__":
    run_gui()
