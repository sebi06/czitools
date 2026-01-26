#!/usr/bin/env python3
"""
Safe pattern for using get_planetable() with Napari on Linux.

The key: Extract planetable BEFORE starting Napari's event loop.
This avoids threading conflicts between aicspylibczi and PyQt.
"""

from pathlib import Path
from czitools.utils.planetable import get_planetable
from czitools.read_tools import read_tools

# Step 1: Extract planetable FIRST (before Napari)
print("Extracting planetable...")
czi_file = Path("data/CellDivision_T3_Z5_CH2_X240_Y170.czi")

# Get planetable - aicspylibczi runs here, before Napari
df_planetable, csv_path = get_planetable(czi_file, norm_time=True, save_table=False)

print(f"✅ Planetable extracted: {len(df_planetable)} rows")
print(f"   Columns: {list(df_planetable.columns)}")

# Step 2: Load image data (thread-safe, uses only pylibCZIrw)
print("\nLoading image data...")
array, metadata = read_tools.read_6darray(czi_file, use_dask=True, use_xarray=True, chunk_zyx=True)

print(f"✅ Image loaded: {array.shape}")

# Step 3: NOW start Napari (planetable already extracted)
print("\nStarting Napari...")
import napari

viewer = napari.Viewer()

# Add image
viewer.add_image(array, name="CZI Image", metadata={"czi_metadata": metadata.info})

# Use planetable data for visualization
# Example: Color timepoints differently based on planetable timestamps
if "Time[s]" in df_planetable.columns:
    time_info = df_planetable.groupby("T")["Time[s]"].first()
    print(f"\n📊 Time points from planetable:")
    for t, time_s in time_info.items():
        print(f"   T={t}: {time_s:.2f} seconds")

# Example: Show planetable as table in console
print(f"\n📊 Planetable preview:")
print(df_planetable.head(10))

print("\n✅ Safe to use Napari now - planetable extracted before event loop started")
print("   Close Napari viewer to exit.")

napari.run()
