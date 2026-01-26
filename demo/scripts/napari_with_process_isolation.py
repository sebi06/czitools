#!/usr/bin/env python3
"""
Example: Using czitools with Napari on Linux using process isolation.

This script demonstrates how to use ALL czitools functions (including read_tiles
and get_planetable) safely with Napari on Linux by enabling process isolation.

Process isolation runs aicspylibczi operations in separate processes, completely
avoiding threading conflicts with PyQt/Napari's event loop.
"""

import sys
from pathlib import Path

# CRITICAL: Enable process isolation BEFORE any czitools imports
from czitools.utils.threading_helpers import enable_process_isolation

enable_process_isolation()

print("✅ Process isolation enabled for aicspylibczi operations")

# Now safe to import all czitools modules
from czitools.read_tools import read_tools
from czitools.utils.planetable import get_planetable
from czitools.metadata_tools.czi_metadata import CziMetadata


def demo_with_tiles(filepath: Path):
    """
    Demonstrate reading tiles with process isolation.

    This would crash on Linux with Napari without process isolation!
    """
    print(f"\n{'='*70}")
    print("DEMO 1: Reading Tiles with Process Isolation")
    print(f"{'='*70}")

    # Check if file is a mosaic
    mdata = CziMetadata(filepath)
    if not mdata.ismosaic:
        print(f"⚠️  File is not a mosaic, skipping tile reading demo")
        return

    print(f"File: {filepath.name}")
    print(f"Is Mosaic: {mdata.ismosaic}")
    print(f"Scenes: {mdata.image.SizeS}, Channels: {mdata.image.SizeC}")

    # Read first tile from first scene
    # This operation now runs in a separate process - safe with Napari!
    print("\nReading tile 0 from scene 0 (in separate process)...")
    try:
        tile_stack, size = read_tools.read_tiles(filepath, scene=0, tile=0, C=0)  # First channel only
        print(f"✅ Tile read successfully!")
        print(f"   Shape: {tile_stack.shape}")
        print(f"   Dimensions: {size}")
    except Exception as e:
        print(f"❌ Error reading tile: {e}")


def demo_with_planetable(filepath: Path):
    """
    Demonstrate extracting planetable with process isolation.

    This would crash on Linux with Napari without process isolation!
    """
    print(f"\n{'='*70}")
    print("DEMO 2: Extracting Planetable with Process Isolation")
    print(f"{'='*70}")

    print(f"File: {filepath.name}")

    # Extract planetable
    # This operation now runs in a separate process - safe with Napari!
    print("\nExtracting planetable (in separate process)...")
    try:
        df, csv_path = get_planetable(filepath, norm_time=True, save_table=False)

        if not df.empty:
            print(f"✅ Planetable extracted successfully!")
            print(f"   Rows: {len(df)}, Columns: {len(df.columns)}")
            print(f"\n   First few rows:")
            print(df.head())
        else:
            print("⚠️  Empty planetable returned")
    except Exception as e:
        print(f"❌ Error extracting planetable: {e}")


def demo_with_napari(filepath: Path):
    """
    Demonstrate loading CZI in Napari with process isolation enabled.

    All czitools functions now work safely with Napari!
    """
    print(f"\n{'='*70}")
    print("DEMO 3: Loading in Napari (Process Isolation Enabled)")
    print(f"{'='*70}")

    try:
        import napari
    except ImportError:
        print("❌ Napari not installed. Install with: pip install napari[all]")
        return

    print(f"File: {filepath.name}")

    # Read using recommended method (always safe)
    print("\nReading CZI with dask...")
    array6d, mdata = read_tools.read_6darray(filepath, use_dask=True, use_xarray=True, chunk_zyx=True)

    print(f"✅ Array loaded:")
    print(f"   Shape: {array6d.shape}")
    print(f"   Dimensions: {array6d.dims}")
    print(f"   Dask chunks: {array6d.chunks if hasattr(array6d, 'chunks') else 'N/A'}")

    # Create Napari viewer and display
    print("\nOpening in Napari...")
    viewer = napari.Viewer()

    # Add each channel separately with proper metadata
    for c in range(mdata.image.SizeC):
        channel_name = mdata.channelinfo.names[c] if c < len(mdata.channelinfo.names) else f"Channel {c}"

        # Extract single channel (S, T, 1, Z, Y, X)
        channel_data = array6d.isel(C=c)

        viewer.add_image(
            channel_data,
            name=channel_name,
            scale=(
                mdata.scale.T if mdata.scale.T else 1.0,
                mdata.scale.Z if mdata.scale.Z else 1.0,
                mdata.scale.Y if mdata.scale.Y else 1.0,
                mdata.scale.X if mdata.scale.X else 1.0,
            ),
            colormap="gray" if c == 0 else "green",
        )

    print("✅ Napari viewer opened successfully!")
    print("\n📋 Summary:")
    print("   - Process isolation is enabled")
    print("   - All aicspylibczi operations run in separate processes")
    print("   - No threading conflicts with PyQt/Napari")
    print("   - ALL czitools functions work safely!")

    # Run Napari
    napari.run()


def main():
    """Main demonstration function."""
    print("=" * 70)
    print("czitools + Napari on Linux: Process Isolation Demo")
    print("=" * 70)

    # Get test file path
    basedir = Path(__file__).resolve().parents[2]

    # Use a mosaic file for tile demo
    mosaic_file = basedir / "data" / "S3_1Pos_2Mosaic_T2_Z3_CH2_sm.czi"

    # Use a standard file for main demo
    standard_file = basedir / "data" / "CellDivision_T3_Z5_CH2_X240_Y170.czi"

    if not standard_file.exists():
        print(f"❌ Test file not found: {standard_file}")
        print("   Please adjust the filepath in the script.")
        sys.exit(1)

    # Demo 1: Tiles (if mosaic file available)
    if mosaic_file.exists():
        demo_with_tiles(mosaic_file)
    else:
        print(f"\n⚠️  Mosaic file not found: {mosaic_file.name}")
        print("   Skipping tile reading demo")

    # Demo 2: Planetable
    demo_with_planetable(standard_file)

    # Demo 3: Napari integration
    print("\n" + "=" * 70)
    response = input("\nOpen in Napari? (y/n): ")
    if response.lower() == "y":
        demo_with_napari(standard_file)
    else:
        print("\nSkipping Napari demo.")

    print("\n" + "=" * 70)
    print("Demo completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
