"""
Safe CZI Loading for Napari on Linux

This script demonstrates how to safely load CZI files with Napari on Linux
to avoid crashes related to aicspylibczi threading issues.

Author: sebi06
"""

import os

# CRITICAL: Set this BEFORE importing czitools or napari
# This prevents aicspylibczi from being used, avoiding threading conflicts
os.environ["CZITOOLS_DISABLE_AICSPYLIBCZI"] = "1"

from pathlib import Path
from czitools.read_tools import read_tools
from czitools.utils import misc
import napari

# Define base directory for test data
basedir = Path(__file__).resolve().parents[2] / "data"
os.chdir(basedir)

# Select a CZI file
filepath = misc.openfile(
    directory=str(basedir),
    title="Open CZI Image File",
    ftypename="CZI Files",
    extension="*.czi",
)

if filepath is None:
    print("No file selected. Exiting.")
    exit()

print(f"Loading: {filepath}")
print("\n⚠️  Using thread-safe mode (aicspylibczi disabled)")
print("This is the recommended approach for Napari on Linux\n")

# Read CZI with thread-safe settings
# This uses ONLY pylibCZIrw which is confirmed thread-safe
array6d, mdata = read_tools.read_6darray(
    filepath,
    use_dask=True,  # Lazy loading - efficient for large files
    use_xarray=True,  # Labeled dimensions - easy to work with
    chunk_zyx=True,  # Optimize chunking for performance
)

print(f"Image dimensions: {array6d.dims}")
print(f"Image shape: {array6d.shape}")
print(f"Number of channels: {mdata.image.SizeC}")
print(f"Pixel type: {mdata.pixeltypes}")

# Create Napari viewer
viewer = napari.Viewer()

# Add each channel to Napari
for ch in range(mdata.image.SizeC):
    # Extract channel
    channel_data = array6d.sel(C=ch)

    # Get channel name if available
    channel_name = (
        mdata.channelinfo.names[ch]
        if mdata.channelinfo.names and ch < len(mdata.channelinfo.names)
        else f"Channel {ch}"
    )

    # Get scaling information
    scale = [1.0]  # For Scene dimension if present
    if mdata.image.SizeT and mdata.image.SizeT > 1:
        scale.append(1.0)  # Time dimension
    scale.extend(
        [
            mdata.scale.Z if mdata.scale.Z else 1.0,
            mdata.scale.Y if mdata.scale.Y else 1.0,
            mdata.scale.X if mdata.scale.X else 1.0,
        ]
    )

    # Add to viewer
    viewer.add_image(
        channel_data,
        name=channel_name,
        scale=scale,
        colormap="gray" if not mdata.isRGB.get(ch, False) else "viridis",
        blending="additive",
    )

print("\n✅ CZI loaded successfully in thread-safe mode!")
print("No aicspylibczi was used - only pylibCZIrw (thread-safe)\n")

# Run Napari
napari.run()
